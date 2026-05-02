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
- ``mcp:``        - stub; wire up a real MCP client
- ``grpc:``       - stub; wire up a real gRPC channel
- ``container:``  - stub; wire up Docker/Podman
"""

import subprocess
import requests
import json
from typing import Callable, Any


def resolve_http_get(entrypoint: str, **kwargs: Any) -> Any:
    """Execute an ``http:get`` entrypoint, passing kwargs as query parameters.

    Args:
        entrypoint: Full entrypoint string, e.g. ``http:get https://api.example.com/path``.
        **kwargs: Key-value pairs appended as URL query parameters (None values omitted).

    Returns:
        Parsed JSON response body.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    url = entrypoint.split(" ", 1)[1]
    params = {k: v for k, v in kwargs.items() if v is not None}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def resolve_http_post(entrypoint: str, **kwargs: Any) -> Any:
    """Execute an ``http:post`` entrypoint, sending kwargs as a JSON body.

    Args:
        entrypoint: Full entrypoint string, e.g. ``http:post https://api.example.com/path``.
        **kwargs: Key-value pairs serialised as the JSON request body (None values omitted).

    Returns:
        Parsed JSON response body.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    url = entrypoint.split(" ", 1)[1]
    data = {k: v for k, v in kwargs.items() if v is not None}
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json()


def resolve_bash(entrypoint: str, **kwargs: Any) -> Any:
    """Execute a ``bash:`` entrypoint via a subprocess shell.

    The command template (everything after ``bash:``) is formatted with kwargs
    using Python's ``str.format()``, then run via ``subprocess.run(shell=True)``.

    Args:
        entrypoint: Full entrypoint string, e.g. ``bash:cat {path}``.
        **kwargs: Named arguments substituted into the command template.

    Returns:
        Captured stdout as a string.

    Raises:
        RuntimeError: If the process exits with a non-zero return code.
    """
    command = entrypoint.split(":", 1)[1].strip().format(**kwargs)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {result.stderr}")
    return result.stdout


def resolve_python(entrypoint: str, **kwargs: Any) -> Any:
    """Execute a ``python:`` entrypoint by importing and calling a function.

    Args:
        entrypoint: Full entrypoint string, e.g. ``python:birdie.skills.todo.tools.create_plan``.
        **kwargs: Passed directly as keyword arguments to the target function.

    Returns:
        Whatever the target function returns.
    """
    module_path, function_name = entrypoint.split(":", 1)[1].rsplit(".", 1)
    module = __import__(module_path, fromlist=[function_name])
    return getattr(module, function_name)(**kwargs)


def resolve_mcp(entrypoint: str, **kwargs: Any) -> Any:
    """Stub for ``mcp:`` entrypoints - wire up a real MCP client here.

    Args:
        entrypoint: Full entrypoint string, e.g. ``mcp:tool_name``.
        **kwargs: Arguments for the MCP tool.

    Returns:
        Mock response dict (replace with real MCP call).
    """
    tool_name = entrypoint.split(":", 1)[1]
    return {"tool": tool_name, "args": kwargs, "status": "mock_response"}


def resolve_grpc(entrypoint: str, **kwargs: Any) -> Any:
    """Stub for ``grpc:`` entrypoints - wire up a real gRPC channel here.

    Args:
        entrypoint: Full entrypoint string, e.g. ``grpc:package.Service/Method``.
        **kwargs: Arguments for the gRPC method.

    Returns:
        Mock response dict (replace with real gRPC stub call).
    """
    method = entrypoint.split(":", 1)[1]
    return {"grpc_method": method, "args": kwargs, "status": "mock_response"}


def resolve_container(entrypoint: str, **kwargs: Any) -> Any:
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
    if entrypoint.startswith("mcp:"):
        return resolve_mcp
    if entrypoint.startswith("container:"):
        return resolve_container
    if entrypoint.startswith("grpc:"):
        return resolve_grpc
    raise ValueError(f"Unsupported entrypoint scheme: {entrypoint!r}")
