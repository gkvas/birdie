"""
Persistent shell session for ``bash:`` entrypoints.

One long-lived shell per birdie process, the way Claude Code's Bash tool
works: working directory, environment variables, shell functions and aliases
persist from one tool call to the next, so an agent can ``source env.sh``
once instead of threading exports through every command (and failing at it).

Protocol: each command is base64-encoded and eval'd inside the shell with
stdin redirected from /dev/null, followed by sentinel lines (a per-call UUID
plus the exit code) printed to both stdout and stderr.  Reader threads drain
the pipes continuously and a call returns when both sentinels arrive -- there
is no EOF to wait for, so a background process holding the pipes open can
never block a call.

Timeouts: the shell must survive a timeout, so the kill targets the shell's
*descendants* (TERM, then KILL) rather than the session's process group.
This necessarily includes background jobs started in earlier calls; true
daemons should be started with ``setsid <cmd> >/dev/null 2>&1 &`` so they
leave the shell's process tree.  If the sentinel still does not arrive after
the kill -- the shell is stuck in an in-shell loop, or the command consumed
the protocol -- the whole session is killed and respawned on the next call;
the caller is told, because exports and cwd are lost.

The session is shared process-wide (one CLI process == one conversation;
sub-agents and the ACP server share it).  Calls are serialized with a lock,
so concurrent tool calls queue like a human typing into one terminal.

Interrupt and shutdown: a call runs in a worker thread, and neither Ctrl+C
nor the event loop closing can reach into it, so both go through the shell
instead.  ``interrupt()`` kills the running command and keeps the session;
``close()`` kills the shell itself.  Neither waits for the call lock -- the
in-flight ``_exchange`` is exactly what would be holding it, and killing the
process is what lets it return.

Not used on Windows, where bash is not guaranteed; ``resolve_bash`` falls
back to one-shot execution there and wherever BIRDIE_PERSISTENT_SHELL=0.
"""

import atexit
import base64
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid


def _default_shell() -> list:
    # Login shell so the session starts from the user's profile, like a
    # terminal would.
    for name in ("bash", "sh"):
        path = shutil.which(name)
        if path:
            return [path, "-l"]
    raise RuntimeError("no shell found on PATH (need bash or sh)")


def _child_map() -> dict:
    """Map of ppid -> [child pids] for all processes on the system."""
    children: dict = {}
    if os.path.isdir("/proc"):
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as f:
                    stat = f.read()
                # "pid (comm) state ppid ..." -- comm may contain spaces and
                # parens, so split off everything up to the LAST ')'.
                ppid = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(ppid, []).append(int(entry))
        return children
    # macOS and other /proc-less POSIX systems
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid="],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return children
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    return children


def _kill_descendants(root_pid: int) -> None:
    """TERM then KILL every descendant of root_pid, sparing root itself."""
    children = _child_map()
    victims = []
    stack = [root_pid]
    while stack:
        for child in children.get(stack.pop(), []):
            victims.append(child)
            stack.append(child)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in victims:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        if sig == signal.SIGTERM and victims:
            time.sleep(0.3)


class ShellSession:
    """A single persistent shell process with sentinel-framed execution."""

    def __init__(self, shell_argv: list | None = None):
        self._shell_argv = shell_argv
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue | None = None

    # -- lifecycle ---------------------------------------------------------

    def _spawn(self) -> None:
        self._q = queue.Queue()
        self._proc = subprocess.Popen(
            self._shell_argv or _default_shell(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        for stream, tag in ((self._proc.stdout, "out"), (self._proc.stderr, "err")):
            threading.Thread(
                target=self._pump, args=(stream, tag, self._q), daemon=True,
            ).start()
        # Swallow login-profile output (motd, echoes from ~/.profile) so it
        # doesn't prefix the first real command's output.
        self._exchange("true", timeout=15.0)

    @staticmethod
    def _pump(stream, tag: str, q: queue.Queue) -> None:
        fd = stream.fileno()
        while True:
            try:
                chunk = os.read(fd, 65536)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            q.put((tag, chunk))
        q.put((tag, None))

    def _ensure(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._hard_reset()
            self._spawn()

    def _hard_reset(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass
        if proc.poll() is None:
            try:
                # start_new_session makes the shell its own group leader
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except OSError:
                    pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        """Kill the shell.  Never waits for an in-flight call.

        Called from atexit and from the CLI's shutdown path, where the lock
        is likely held by a worker thread parked in a long command: waiting
        for it would stall interpreter shutdown for the rest of that
        command's timeout.  Killing the shell is what unblocks that thread --
        its ``_exchange`` sees the dead process and returns.
        """
        acquired = self._lock.acquire(blocking=False)
        try:
            self._hard_reset()
        finally:
            if acquired:
                self._lock.release()

    def interrupt(self) -> None:
        """Stop whatever the shell is running now; keep the session.

        The in-flight ``_exchange`` gets its sentinel as soon as the killed
        command's ``eval`` returns, so a cancelled tool call stops within
        moments instead of running out its timeout in a background thread.
        Exports, cwd and functions survive, like Ctrl+C in a terminal.
        """
        proc = self._proc
        if proc is not None and proc.poll() is None:
            _kill_descendants(proc.pid)

    # -- execution ---------------------------------------------------------

    @staticmethod
    def _cut(buf: bytearray, idx: int) -> bytes:
        """Output up to the sentinel, minus the newline the protocol added."""
        data = bytes(buf[:idx])
        return data[:-1] if data.endswith(b"\n") else data

    def _exchange(self, command: str, timeout: float):
        """Run one command; returns (timed_out, out, err, rc, alive)."""
        proc, q = self._proc, self._q
        if proc is None:
            raise BrokenPipeError("shell session not running")
        marker = uuid.uuid4().hex
        encoded = base64.b64encode(command.encode()).decode()
        # set +e: a persisted `set -e` would make the shell exit on the
        # eval's non-zero return, before the sentinel can be printed.
        # </dev/null: the command must not read (and eat) the protocol.
        script = (
            "set +e\n"
            f"eval \"$(printf '%s' '{encoded}' | base64 -d)\" </dev/null\n"
            f"printf '\\n__birdie_{marker}__:%d\\n' \"$?\"\n"
            f"printf '\\n__birdie_{marker}__\\n' >&2\n"
        )
        proc.stdin.write(script.encode())
        proc.stdin.flush()

        out, err = bytearray(), bytearray()
        rc_re = re.compile(rb"__birdie_" + marker.encode() + rb"__:(\d+)")
        err_tag = b"__birdie_" + marker.encode() + b"__"
        deadline = time.monotonic() + timeout
        timed_out = False
        while True:
            try:
                tag, chunk = q.get(timeout=0.05)
                if chunk is not None:
                    (out if tag == "out" else err).extend(chunk)
            except queue.Empty:
                pass
            m = rc_re.search(out)
            if m is not None and err_tag in err:
                return (
                    timed_out,
                    self._cut(out, m.start()),
                    self._cut(err, err.find(err_tag)),
                    int(m.group(1)),
                    True,
                )
            if proc.poll() is not None:
                # Shell died mid-command (exit, exec, set -e...).  Flush what
                # the pumps still hold, then report the shell's exit code.
                flush_until = time.monotonic() + 0.5
                while time.monotonic() < flush_until:
                    try:
                        tag, chunk = q.get(timeout=0.05)
                        if chunk is not None:
                            (out if tag == "out" else err).extend(chunk)
                    except queue.Empty:
                        break
                return timed_out, bytes(out), bytes(err), proc.returncode, False
            if time.monotonic() >= deadline:
                if not timed_out:
                    # First expiry: kill the shell's children and give the
                    # shell a grace period to come back with the sentinel.
                    timed_out = True
                    _kill_descendants(proc.pid)
                    deadline = time.monotonic() + 5.0
                else:
                    # The shell itself is stuck (in-shell loop, or the
                    # command swallowed the protocol).  Reset it.
                    self._hard_reset()
                    return True, bytes(out), bytes(err), None, False

    def run(self, command: str, timeout: float):
        """Run a command in the session.

        Returns:
            (timed_out, stdout, stderr, returncode, session_alive).
            returncode is None only when the session was hard-reset on
            timeout; session_alive False means persisted state was lost and
            the next call starts a fresh shell.
        """
        with self._lock:
            self._ensure()
            try:
                timed_out, out, err, rc, alive = self._exchange(command, timeout)
            except BrokenPipeError:
                # The shell died between calls; one transparent respawn.
                self._ensure()
                timed_out, out, err, rc, alive = self._exchange(command, timeout)
            if not alive:
                self._hard_reset()
            return (
                timed_out,
                out.decode("utf-8", errors="replace"),
                err.decode("utf-8", errors="replace"),
                rc,
                alive,
            )


# -- process-wide default session -------------------------------------------

_default: ShellSession | None = None
_default_lock = threading.Lock()


def get_default_session() -> ShellSession:
    global _default
    with _default_lock:
        if _default is None:
            _default = ShellSession()
            atexit.register(_default.close)
        return _default


def peek_default_session() -> "ShellSession | None":
    """The default session if one was ever started, without starting one.

    Shutdown and interrupt paths use this: they must not spawn a shell just
    to discover there was nothing to kill.
    """
    with _default_lock:
        return _default


def reset_default_session() -> None:
    """Close and forget the default session (tests, /reset)."""
    global _default
    with _default_lock:
        if _default is not None:
            _default.close()
            _default = None
