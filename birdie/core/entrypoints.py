"""
Entrypoint resolvers for skill tools.

Each resolver implements the ``scheme:target`` contract described in SKILL.MD:
it receives the full entrypoint string plus the tool-call kwargs, and returns
a result that is sent back to the LLM as a ToolMessage.

Supported schemes
-----------------
- ``bash:``       - shell command with ``{placeholder}`` substitution
- ``http:get``    - HTTP GET; kwargs become query parameters
- ``http:post``   - HTTP POST; kwargs become the JSON body
- ``python:``     - import and call ``module.path.function(**kwargs)``
- ``grpc:``       - stub; wire up a real gRPC channel
- ``container:``  - stub; wire up Docker/Podman

Note: MCP tools are not resolved here.  MCP servers are declared via
``mcp_server`` in SKILL.MD frontmatter; their tools are loaded by
``MCPClientManager`` and injected into the graph directly as LangChain
BaseTool objects, bypassing the entrypoint system.
"""

import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import requests
import json
from typing import Callable, Any

# Timeout (seconds) for HTTP entrypoints so a hung endpoint cannot stall the
# agent turn indefinitely.
HTTP_TIMEOUT = 30.0

# Timeout (seconds) for bash entrypoints when neither the tool call nor the
# SKILL.MD sets one.  Long-running commands (builds, installs) are expected to
# raise it per call via the `timeout_s` parameter.
BASH_TIMEOUT = 120.0

# Bounds for the LLM-supplied per-call `timeout_s` value.  SKILL.MD `timeout:`
# values are author-written config and are not clamped.
MIN_CALL_TIMEOUT = 1.0
MAX_CALL_TIMEOUT = 600.0

# Name of the optional per-call timeout parameter injected into the schema of
# every timeout-capable tool (see `supports_timeout`).  Resolvers pop it from
# kwargs before argument substitution, so every call path -- LangChain
# ToolNode, ACP MCP server, direct resolver calls -- honors it.
TIMEOUT_PARAM = "timeout_s"

# Chars of each captured stream quoted back to the LLM in a timeout error.
_PARTIAL_OUTPUT_CAP = 4096

_IS_WINDOWS = sys.platform == "win32"


class ToolTimeoutError(RuntimeError):
    """A tool execution exceeded its timeout and was killed.

    Distinct from plain RuntimeError so the retry policy in
    ``adapter.skilltool_to_langchain_tool`` can skip re-attempts: retrying a
    timeout multiplies the wait without new information.
    """


def supports_timeout(entrypoint: str) -> bool:
    """True if the entrypoint's resolver enforces the per-call timeout.

    ``python:``/``grpc:``/``container:`` run in-process or are stubs, so
    advertising a timeout parameter for them would be a lie.
    """
    return entrypoint.startswith(("bash:", "http:get", "http:post"))


def timeout_param_schema(entrypoint: str) -> dict:
    """JSON Schema for the injected ``timeout_s`` property, per scheme.

    The description is LLM-facing: for skill tools the raw SKILL.MD schema is
    what the model sees, so this text is the model's only documentation of the
    timeout contract.
    """
    if entrypoint.startswith("bash:"):
        return {
            "type": "number",
            "description": (
                f"Max seconds to wait for this command (default {BASH_TIMEOUT:g}, "
                f"max {MAX_CALL_TIMEOUT:g}). On expiry the command's whole process "
                "tree is killed and the error includes any output captured so far. "
                "Set this explicitly for known-slow commands (builds, installs, "
                "test suites) instead of relying on the default."
            ),
        }
    return {
        "type": "number",
        "description": (
            f"Request timeout in seconds (default {HTTP_TIMEOUT:g}, "
            f"max {MAX_CALL_TIMEOUT:g})."
        ),
    }


def _effective_timeout(
    call_timeout: Any, config_timeout: float | None, default: float
) -> float:
    """Resolve the timeout for one execution.

    Precedence: per-call ``timeout_s`` (clamped to
    [MIN_CALL_TIMEOUT, MAX_CALL_TIMEOUT]) > SKILL.MD ``timeout:`` (unclamped;
    author config is trusted) > the scheme default.
    """
    if call_timeout is not None:
        try:
            value = float(call_timeout)
        except (TypeError, ValueError):
            raise ValueError(
                f"{TIMEOUT_PARAM} must be a number of seconds, got {call_timeout!r}"
            ) from None
        return min(max(value, MIN_CALL_TIMEOUT), MAX_CALL_TIMEOUT)
    if config_timeout is not None:
        return float(config_timeout)
    return default


def resolve_http_get(entrypoint: str, _timeout: float | None = None, **kwargs: Any) -> Any:
    """Execute an ``http:get`` entrypoint, passing kwargs as query parameters.

    Args:
        entrypoint: Full entrypoint string, e.g. ``http:get https://api.example.com/path``.
        _timeout: SKILL.MD ``timeout:`` value in seconds (default ``HTTP_TIMEOUT``).
        **kwargs: Key-value pairs appended as URL query parameters (None values
            omitted).  The reserved ``timeout_s`` kwarg is the LLM's per-call
            timeout override and is popped, never sent to the endpoint.

    Returns:
        Parsed JSON response body.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    call_timeout = kwargs.pop(TIMEOUT_PARAM, None)
    url = entrypoint.split(" ", 1)[1]
    params = {k: v for k, v in kwargs.items() if v is not None}
    response = requests.get(
        url, params=params,
        timeout=_effective_timeout(call_timeout, _timeout, HTTP_TIMEOUT),
    )
    response.raise_for_status()
    return response.json()


def resolve_http_post(entrypoint: str, _timeout: float | None = None, **kwargs: Any) -> Any:
    """Execute an ``http:post`` entrypoint, sending kwargs as a JSON body.

    Args:
        entrypoint: Full entrypoint string, e.g. ``http:post https://api.example.com/path``.
        _timeout: SKILL.MD ``timeout:`` value in seconds (default ``HTTP_TIMEOUT``).
        **kwargs: Key-value pairs serialised as the JSON request body (None
            values omitted).  The reserved ``timeout_s`` kwarg is the LLM's
            per-call timeout override and is popped, never sent in the body.

    Returns:
        Parsed JSON response body.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    call_timeout = kwargs.pop(TIMEOUT_PARAM, None)
    url = entrypoint.split(" ", 1)[1]
    data = {k: v for k, v in kwargs.items() if v is not None}
    response = requests.post(
        url, json=data,
        timeout=_effective_timeout(call_timeout, _timeout, HTTP_TIMEOUT),
    )
    response.raise_for_status()
    return response.json()


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a shell command and everything it spawned, cross-platform.

    ``subprocess``'s own timeout kill only reaches the direct child (the
    shell), leaving grandchildren -- a gradle daemon, an emulator launch --
    running and holding the output pipes open, which is exactly the hang this
    module exists to prevent.

    POSIX: the child was started in its own session (``start_new_session``),
    so its process group id is its pid and ``killpg`` reaches every
    non-daemonized descendant.  TERM first for a chance at cleanup, KILL
    after a short grace.

    Windows: there is no process group to signal through pipes; ``taskkill
    /T`` walks the parent/child tree instead.  ``proc.kill()`` afterwards
    guarantees at least the direct child dies even if taskkill is missing
    or raced.
    """
    if _IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
        )
        try:
            proc.kill()
        except OSError:
            pass
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_shell(command: str, timeout: float) -> tuple[bool, str, str, int | None]:
    """Run a shell command with a hard deadline; never blocks past it.

    Output is drained by daemon threads rather than ``communicate()``:
    after a timeout kill, ``communicate()`` blocks until pipe EOF, and a
    descendant that escaped the kill (double-fork, new session) can hold the
    pipe open indefinitely.  Here the reader threads are joined with their
    own short deadline and whatever is buffered is returned.

    Returns:
        (timed_out, stdout, stderr, returncode).
    """
    popen_kwargs: dict[str, Any] = {}
    if not _IS_WINDOWS:
        # Own session => own process group; required for _kill_process_tree.
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        command, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        # Arbitrary command output is not guaranteed UTF-8 (Windows-codepage
        # tools under WSL, binary spew); strict decoding would kill the
        # reader thread and silently truncate output.
        errors="replace",
        **popen_kwargs,
    )

    def _drain(stream, buf: list) -> None:
        # The reader thread OWNS its stream, including closing it.  The main
        # thread must never close these: file-object methods share an internal
        # lock, so close() on a stream whose reader is blocked in read() waits
        # for that read to return -- i.e. blocks until the pipe-holder exits,
        # which is the exact hang this function exists to prevent.  If a
        # descendant escapes the kill and holds the pipe, the daemon thread
        # (and the fd) lives until that process exits: a bounded leak, chosen
        # over an unbounded block.
        try:
            for chunk in iter(lambda: stream.read(8192), ""):
                buf.append(chunk)
        except (ValueError, OSError):
            pass  # pipe closed under us during kill
        finally:
            try:
                stream.close()
            except OSError:
                pass

    out_buf: list = []
    err_buf: list = []
    readers = [
        threading.Thread(target=_drain, args=(proc.stdout, out_buf), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, err_buf), daemon=True),
    ]
    for t in readers:
        t.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        try:
            proc.wait(timeout=5)  # reap; bounded in case the kill was raced
        except subprocess.TimeoutExpired:
            pass
    for t in readers:
        t.join(timeout=2)
    return timed_out, "".join(out_buf), "".join(err_buf), proc.returncode


def _tail(text: str) -> str:
    """Last _PARTIAL_OUTPUT_CAP chars of text, marked if truncated."""
    if len(text) <= _PARTIAL_OUTPUT_CAP:
        return text
    return f"[... truncated ...]\n{text[-_PARTIAL_OUTPUT_CAP:]}"


def resolve_bash(entrypoint: str, _timeout: float | None = None, **kwargs: Any) -> Any:
    """Execute a ``bash:`` entrypoint via a subprocess shell.

    The command template (everything after ``bash:``) is formatted with kwargs,
    then run through the platform shell with a hard deadline.

    Substituted values are shell-quoted (``shlex.quote``) so LLM-supplied
    arguments cannot inject extra shell commands into templates like
    ``bash:cat {path}``.  The single exception is a template that consists of
    exactly one placeholder (e.g. ``bash:{command}``): such raw-shell tools
    intentionally receive a full command string, where quoting would break it.

    Args:
        entrypoint: Full entrypoint string, e.g. ``bash:cat {path}``.
        _timeout: SKILL.MD ``timeout:`` value in seconds (None uses
            ``BASH_TIMEOUT``).
        **kwargs: Named arguments substituted into the command template.
            The reserved ``timeout_s`` kwarg is popped first: it is the
            LLM's per-call timeout override, clamped to
            [MIN_CALL_TIMEOUT, MAX_CALL_TIMEOUT].

    Returns:
        Captured stdout as a string.

    Raises:
        ToolTimeoutError: If the deadline expires.  The whole process tree is
            killed and the message quotes the output captured so far.
        RuntimeError: If the process exits with a non-zero return code.
    """
    call_timeout = kwargs.pop(TIMEOUT_PARAM, None)
    template = entrypoint.split(":", 1)[1].strip()
    if re.fullmatch(r'\{\w+\}', template):
        command = template.format(**kwargs)
    else:
        quoted = {k: shlex.quote(str(v)) for k, v in kwargs.items()}
        command = template.format(**quoted)
    timeout = _effective_timeout(call_timeout, _timeout, BASH_TIMEOUT)
    timed_out, stdout, stderr, returncode = _run_shell(command, timeout)
    if timed_out:
        raise ToolTimeoutError(
            f"Command timed out after {timeout:g}s and its process tree was "
            f"killed. If it simply needs longer, retry with a larger "
            f"{TIMEOUT_PARAM} (max {MAX_CALL_TIMEOUT:g}); if it was hung "
            f"waiting on something, fix that instead of retrying.\n"
            f"Partial stdout:\n{_tail(stdout)}\n"
            f"Partial stderr:\n{_tail(stderr)}"
        )
    if returncode != 0:
        raise RuntimeError(f"Command failed: {stderr}")
    return stdout


def resolve_python(entrypoint: str, _timeout: float | None = None, **kwargs: Any) -> Any:
    """Execute a ``python:`` entrypoint by importing and calling a function.

    Args:
        entrypoint: Full entrypoint string, e.g. ``python:birdie.skills.todo.tools.create_plan``.
        _timeout: Accepted for resolver-signature uniformity; not enforced
            for in-process calls.
        **kwargs: Passed directly as keyword arguments to the target function.

    Returns:
        Whatever the target function returns.
    """
    module_path, function_name = entrypoint.split(":", 1)[1].rsplit(".", 1)
    module = __import__(module_path, fromlist=[function_name])
    return getattr(module, function_name)(**kwargs)


def resolve_grpc(entrypoint: str, _timeout: float | None = None, **kwargs: Any) -> Any:
    """Stub for ``grpc:`` entrypoints - wire up a real gRPC channel here.

    Args:
        entrypoint: Full entrypoint string, e.g. ``grpc:package.Service/Method``.
        **kwargs: Arguments for the gRPC method.

    Returns:
        Mock response dict (replace with real gRPC stub call).
    """
    method = entrypoint.split(":", 1)[1]
    return {"grpc_method": method, "args": kwargs, "status": "mock_response"}


def resolve_container(entrypoint: str, _timeout: float | None = None, **kwargs: Any) -> Any:
    """Stub for ``container:`` entrypoints - wire up Docker/Podman here.

    Args:
        entrypoint: Full entrypoint string, e.g. ``container:image_name``.
        **kwargs: Arguments passed to the container.

    Returns:
        Mock response dict (replace with real container invocation).
    """
    image = entrypoint.split(":", 1)[1]
    return {"container": image, "args": kwargs, "status": "mock_response"}


def resolve_entrypoint(entrypoint: str) -> Callable[..., Any]:
    """Return the resolver function for the given entrypoint scheme.

    Resolver functions all share the signature
    ``resolver(entrypoint: str, **kwargs) -> Any`` and are safe to call
    multiple times with different kwargs.

    Args:
        entrypoint: A ``scheme:target`` string whose prefix determines the
            resolver (e.g. ``bash:``, ``http:get``, ``python:``).

    Returns:
        The resolver callable for the matched scheme.

    Raises:
        ValueError: If the scheme prefix is not recognised.
    """
    if entrypoint.startswith("http:get"):
        return resolve_http_get
    if entrypoint.startswith("http:post"):
        return resolve_http_post
    if entrypoint.startswith("bash:"):
        return resolve_bash
    if entrypoint.startswith("python:"):
        return resolve_python
    if entrypoint.startswith("container:"):
        return resolve_container
    if entrypoint.startswith("grpc:"):
        return resolve_grpc
    raise ValueError(f"Unsupported entrypoint scheme: {entrypoint!r}")
