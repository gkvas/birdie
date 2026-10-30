"""
Python entrypoints for the Tasks skill: background task monitoring.

These functions are called via ``python:birdie.skills.tasks.tools.<name>``
entrypoints and their return values appear as ToolMessage content in the CLI.

Why this exists: ``run_bash`` blocks until the command finishes, and when the
timeout hits, the shell session kills the command's whole process tree - so a
long-running monitor (polling ``gh pr checks``, a build, a test suite) either
blocks the agent for the entire timeout or has to be interrupted manually.
Tasks started here are detached (their own session, output appended to a log
file), so they keep running after the tool call returns, survive birdie
restarts, and can be inspected at any time.

State lives in ``~/.birdie/tasks/<id>/`` (override with the BIRDIE_TASKS_DIR
env var): ``meta.json`` (command, pid, timestamps), ``output.log``
(stdout+stderr) and ``exitcode`` (written by the wrapper when the command
finishes).
"""

import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Live Popen handles for tasks started by this process, so their exit code
# can be reaped even if the wrapper never got to write the exitcode file.
_procs: dict = {}

# Upper bound for a single task_wait call; the caller can simply call again.
MAX_WAIT_S = 300.0
# Cap on lines returned by task_output when tail<=0 (meaning "full output").
MAX_OUTPUT_LINES = 2000


def _tasks_dir() -> Path:
    override = os.environ.get("BIRDIE_TASKS_DIR")
    return Path(override) if override else Path.home() / ".birdie" / "tasks"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _read_meta(task_dir: Path) -> dict:
    with open(task_dir / "meta.json") as f:
        return json.load(f)


def _write_meta(task_dir: Path, meta: dict) -> None:
    tmp = task_dir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(task_dir / "meta.json")


def _refresh(task_dir: Path) -> dict:
    """Recompute a task's status from disk + live pid; returns the meta dict."""
    meta = _read_meta(task_dir)
    tid = meta["id"]
    exit_file = task_dir / "exitcode"
    # Same-process reap: if we still hold the Popen and it exited, record rc.
    proc = _procs.get(tid)
    if proc is not None:
        rc = proc.poll()
        if rc is not None:
            if not exit_file.exists():
                exit_file.write_text(str(rc))
            _procs.pop(tid, None)
    if meta.get("stopped"):
        meta["status"] = "stopped"
        meta["rc"] = None
    elif exit_file.exists():
        try:
            meta["rc"] = int(exit_file.read_text().strip())
        except ValueError:
            meta["rc"] = None
        meta["status"] = "done" if meta["rc"] == 0 else "failed"
    elif _pid_alive(meta["pid"]):
        meta["status"] = "running"
        meta["rc"] = None
    else:
        # Process gone without writing exitcode (killed by a signal, or the
        # command itself called `exit`): finished, exit code unknown.
        meta["status"] = "done"
        meta["rc"] = None
    started = datetime.fromisoformat(meta["started_at"])
    meta["elapsed_s"] = (datetime.now(timezone.utc) - started).total_seconds()
    return meta


def _resolve_task_dir(task_id: str) -> Path:
    """Resolve an exact task id, or a unique prefix of one, to its directory."""
    if not task_id:
        raise ValueError("task_id must not be empty")
    base = _tasks_dir()
    exact = base / task_id
    if exact.is_dir() and (exact / "meta.json").exists():
        return exact
    matches = []
    if base.is_dir():
        matches = sorted(
            d for d in base.iterdir()
            if d.is_dir() and d.name.startswith(task_id)
            and (d / "meta.json").exists()
        )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Unknown task id: {task_id!r} (see task_list)")
    raise ValueError(
        f"Task id {task_id!r} is ambiguous: "
        + ", ".join(d.name for d in matches)
    )


def _tail_lines(path: Path, n: int) -> list:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        data = f.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[-n:] if n > 0 else lines


def task_start(command: str, name: str = "", cwd: str = "", **_) -> str:
    """Start a command as a detached background task; returns immediately.

    Args:
        command: The shell command to run in the background.
        name: Optional short label shown in task_list / task_status.
        cwd: Optional working directory (default: current directory).
        **_: Ignored extra kwargs from the StructuredTool wrapper.

    Returns:
        A confirmation with the task id, pid and log path.
    """
    if not command or not command.strip():
        raise ValueError("command must not be empty")
    shell = shutil.which("sh") or shutil.which("bash")
    if not shell:
        raise RuntimeError("background tasks need a POSIX shell (sh/bash) on PATH")
    base = _tasks_dir()
    base.mkdir(parents=True, exist_ok=True)
    tid = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    task_dir = base / tid
    task_dir.mkdir()
    log_path = task_dir / "output.log"
    exit_path = task_dir / "exitcode"
    # Wrapper: run the command, then record its exit code. With
    # `sh -c script name`, the arg after the script is $0 inside it.
    script = f'{command}\nprintf %s "$?" > "$0"\n'
    log = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            [shell, "-c", script, str(exit_path)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach: survives the shell session and birdie
            cwd=cwd or None,
        )
    finally:
        log.close()
    meta = {
        "id": tid,
        "name": name,
        "command": command,
        "pid": proc.pid,
        "cwd": os.path.abspath(cwd) if cwd else os.getcwd(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "rc": None,
    }
    _write_meta(task_dir, meta)
    _procs[tid] = proc
    label = f" ({name})" if name else ""
    return (
        f"Started task {tid}{label} (pid {proc.pid}).\n"
        f"  command: {command}\n"
        f"  output:  {log_path}\n"
        f'Check progress with task_status("{tid}") or task_list; wait with '
        f'task_wait("{tid}", timeout=...); stop with task_stop("{tid}"). '
        "The task keeps running in the background - it is not affected by "
        "run_bash timeouts."
    )


def task_list(status: str = "", limit: int = 50, **_) -> str:
    """List tasks (newest first), optionally filtered by status.

    Args:
        status: Optional status filter: running, done, failed, stopped.
        limit: Maximum number of tasks to show.
        **_: Ignored extra kwargs from the StructuredTool wrapper.
    """
    base = _tasks_dir()
    if not base.is_dir():
        return "No tasks yet (start one with task_start)."
    entries = []
    for d in sorted(base.iterdir(), reverse=True):
        if not (d.is_dir() and (d / "meta.json").exists()):
            continue
        try:
            entries.append(_refresh(d))
        except (OSError, ValueError, KeyError):
            continue
    if status:
        entries = [m for m in entries if m["status"] == status]
    if not entries:
        suffix = f" with status {status!r}" if status else ""
        return f"No tasks{suffix}."
    lines = [f"{len(entries)} task(s), newest first:"]
    for m in entries[: max(1, int(limit))]:
        label = f" [{m['name']}]" if m.get("name") else ""
        rc = "" if m["rc"] is None else f" rc={m['rc']}"
        cmd = m["command"]
        if len(cmd) > 70:
            cmd = cmd[:67] + "..."
        lines.append(
            f"  {m['id']}  {m['status']:<7}{rc:<7}"
            f"{_fmt_duration(m['elapsed_s']):>7}{label}  {cmd}"
        )
    return "\n".join(lines)


def task_status(task_id: str, **_) -> str:
    """Show one task's status, metadata and the tail of its output.

    Args:
        task_id: Exact task id, or a unique prefix of one.
        **_: Ignored extra kwargs from the StructuredTool wrapper.
    """
    task_dir = _resolve_task_dir(task_id)
    meta = _refresh(task_dir)
    rc = f" (rc={meta['rc']})" if meta["rc"] is not None else ""
    label = f" [{meta['name']}]" if meta.get("name") else ""
    lines = [
        f"Task {meta['id']}{label}",
        f"  status:  {meta['status']}{rc}",
        f"  command: {meta['command']}",
        f"  pid:     {meta['pid']}  cwd: {meta['cwd']}",
        f"  started: {meta['started_at']}  elapsed: {_fmt_duration(meta['elapsed_s'])}",
        f"  output:  {task_dir / 'output.log'}",
        "",
        "Last 15 lines of output:",
    ]
    tail = _tail_lines(task_dir / "output.log", 15)
    lines.extend(tail if tail else ["  (no output yet)"])
    return "\n".join(lines)


def task_output(task_id: str, tail: int = 100, **_) -> str:
    """Return a task's captured output (last `tail` lines; tail<=0 means all,
    capped at MAX_OUTPUT_LINES).

    Args:
        task_id: Exact task id, or a unique prefix of one.
        tail: Number of trailing lines to return (default 100; <=0 for all).
        **_: Ignored extra kwargs from the StructuredTool wrapper.
    """
    task_dir = _resolve_task_dir(task_id)
    meta = _refresh(task_dir)
    n = tail if tail and tail > 0 else MAX_OUTPUT_LINES
    lines = _tail_lines(task_dir / "output.log", n)
    header = (
        f"Task {meta['id']} ({meta['status']}), "
        f"last {len(lines)} line(s) of output:"
    )
    return header + "\n" + ("\n".join(lines) if lines else "(no output yet)")


def task_wait(task_id: str, timeout: float = 60, **_) -> str:
    """Wait up to `timeout` seconds for a task to finish; never kills it.

    Unlike a blocking run_bash loop, this returns after at most `timeout`
    seconds (capped at MAX_WAIT_S) whether or not the task finished, and the
    task keeps running in the background either way - call again later.

    Args:
        task_id: Exact task id, or a unique prefix of one.
        timeout: Seconds to wait before returning (default 60, max 300).
        **_: Ignored extra kwargs from the StructuredTool wrapper.
    """
    timeout = min(max(float(timeout), 0.0), MAX_WAIT_S)
    task_dir = _resolve_task_dir(task_id)
    deadline = time.monotonic() + timeout
    while True:
        meta = _refresh(task_dir)
        if meta["status"] != "running":
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    if meta["status"] == "running":
        return (
            f"Task {meta['id']} still running after {timeout:.0f}s "
            f"(elapsed {_fmt_duration(meta['elapsed_s'])}). It keeps running in "
            "the background - call task_wait again later, check task_status, or "
            "do other work in the meantime."
        )
    rc = f" rc={meta['rc']}" if meta["rc"] is not None else ""
    lines = [
        f"Task {meta['id']} finished: {meta['status']}{rc} "
        f"after {_fmt_duration(meta['elapsed_s'])}.",
        "",
        "Last 30 lines of output:",
    ]
    tail = _tail_lines(task_dir / "output.log", 30)
    lines.extend(tail if tail else ["  (no output)"])
    return "\n".join(lines)


def task_stop(task_id: str, **_) -> str:
    """Stop a running task (SIGTERM, then SIGKILL after a grace period).

    Args:
        task_id: Exact task id, or a unique prefix of one.
        **_: Ignored extra kwargs from the StructuredTool wrapper.
    """
    task_dir = _resolve_task_dir(task_id)
    meta = _refresh(task_dir)
    if meta["status"] != "running":
        return f"Task {meta['id']} is not running (status: {meta['status']})."
    pid = meta["pid"]
    proc = _procs.get(meta["id"])
    try:
        # start_new_session made the task its own process-group leader.
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    def _dead() -> bool:
        # poll() also reaps the child, so a zombie does not count as alive.
        if proc is not None and proc.poll() is not None:
            return True
        return not _pid_alive(pid)

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not _dead():
        time.sleep(0.1)
    if not _dead():
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if proc is not None:
            proc.poll()
    meta["stopped"] = True
    _write_meta(task_dir, meta)
    _procs.pop(meta["id"], None)
    return f"Stopped task {meta['id']} (pid {pid})."
