"""
Tests for the persistent shell session behind bash: entrypoints.
"""

import os
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="persistent shell is POSIX-only"
)

from birdie.core import shell_session
from birdie.core.entrypoints import ToolTimeoutError, resolve_bash
from birdie.core.shell_session import ShellSession


@pytest.fixture
def session():
    s = ShellSession()
    yield s
    s.close()


class TestShellSession:
    def test_exports_persist_between_calls(self, session):
        session.run("export BIRDIE_TEST_FOO=hello", 10)
        _, out, _, rc, alive = session.run("echo $BIRDIE_TEST_FOO", 10)
        assert (rc, alive) == (0, True)
        assert out.strip() == "hello"

    def test_cwd_persists_between_calls(self, session, tmp_path):
        session.run(f"cd {tmp_path}", 10)
        _, out, _, rc, _ = session.run("pwd", 10)
        assert rc == 0
        assert os.path.realpath(out.strip()) == os.path.realpath(str(tmp_path))

    def test_functions_persist_between_calls(self, session):
        session.run("greet() { echo hi-$1; }", 10)
        _, out, _, rc, _ = session.run("greet there", 10)
        assert rc == 0
        assert out.strip() == "hi-there"

    def test_stdout_and_stderr_separated(self, session):
        _, out, err, rc, _ = session.run("echo to-out; echo to-err >&2", 10)
        assert rc == 0
        assert out.strip() == "to-out"
        assert err.strip() == "to-err"

    def test_exit_code_reported(self, session):
        _, _, _, rc, alive = session.run("(exit 7)", 10)
        assert (rc, alive) == (7, True)

    def test_no_protocol_artifacts_in_output(self, session):
        _, out, err, _, _ = session.run("echo clean", 10)
        assert "__birdie_" not in out and "__birdie_" not in err
        assert out == "clean\n"

    def test_output_without_trailing_newline(self, session):
        _, out, _, rc, _ = session.run("printf no-newline", 10)
        assert rc == 0
        assert out == "no-newline"

    def test_timeout_kills_command_but_session_survives(self, session):
        session.run("export BIRDIE_SURVIVOR=yes", 10)
        start = time.monotonic()
        timed_out, _, _, _, alive = session.run("sleep 30", 1)
        assert time.monotonic() - start < 8
        assert timed_out and alive
        _, out, _, _, _ = session.run("echo $BIRDIE_SURVIVOR", 10)
        assert out.strip() == "yes"

    def test_shell_exit_is_reported_and_recovers(self, session):
        _, _, _, rc, alive = session.run("exit 3", 10)
        assert (rc, alive) == (3, False)
        _, out, _, rc, alive = session.run("echo back", 10)
        assert (rc, alive) == (0, True)
        assert out.strip() == "back"

    def test_unbalanced_quote_does_not_wedge_protocol(self, session):
        _, _, _, rc, alive = session.run('echo "unclosed', 10)
        assert rc != 0 and alive
        _, out, _, rc, _ = session.run("echo fine", 10)
        assert rc == 0 and out.strip() == "fine"

    def test_stdin_reader_returns_immediately(self, session):
        """Commands must not be able to eat the protocol from stdin."""
        start = time.monotonic()
        _, _, _, rc, _ = session.run("cat", 10)
        assert time.monotonic() - start < 5
        assert rc == 0

    def test_persisted_set_e_does_not_break_protocol(self, session):
        session.run("set -e", 10)
        _, _, _, rc, alive = session.run("false", 10)
        assert rc != 0 and alive

    def test_background_job_does_not_block_call(self, session):
        start = time.monotonic()
        _, out, _, rc, _ = session.run("sleep 15 & echo bg=$!", 10)
        assert time.monotonic() - start < 5
        assert rc == 0 and "bg=" in out
        session.run("kill %1 2>/dev/null || true", 10)

    def test_concurrent_calls_are_serialized(self, session):
        results = {}

        def call(tag):
            _, out, _, _, _ = session.run(f"echo {tag}", 15)
            results[tag] = out.strip()

        threads = [threading.Thread(target=call, args=(t,)) for t in ("aa", "bb")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert results == {"aa": "aa", "bb": "bb"}

    def test_non_utf8_output(self, session):
        _, out, _, rc, _ = session.run(r"printf 'caf\x81 before\nafter\n'", 10)
        assert rc == 0
        assert "before" in out and "after" in out


class TestResolveBashIntegration:
    @pytest.fixture(autouse=True)
    def fresh_default_session(self):
        shell_session.reset_default_session()
        yield
        shell_session.reset_default_session()

    def test_export_visible_in_next_call(self):
        resolve_bash("bash:{command}", command="export BIRDIE_RB_STATE=42")
        out = resolve_bash("bash:{command}", command="echo $BIRDIE_RB_STATE")
        assert out.strip() == "42"

    def test_cd_visible_in_next_call(self, tmp_path):
        resolve_bash("bash:{command}", command=f"cd {tmp_path}")
        out = resolve_bash("bash:{command}", command="pwd")
        assert os.path.realpath(out.strip()) == os.path.realpath(str(tmp_path))

    def test_opt_out_env_gives_one_shot_shells(self, monkeypatch):
        monkeypatch.setenv("BIRDIE_PERSISTENT_SHELL", "0")
        resolve_bash("bash:{command}", command="export BIRDIE_RB_ONESHOT=1")
        out = resolve_bash("bash:{command}", command="echo [$BIRDIE_RB_ONESHOT]")
        assert out.strip() == "[]"

    def test_timeout_error_says_session_survived(self):
        with pytest.raises(ToolTimeoutError) as excinfo:
            resolve_bash("bash:{command}", command="echo going; sleep 30",
                         timeout_s=1)
        message = str(excinfo.value)
        assert "session itself survived" in message
        assert "going" in message
        # and it really did survive, state intact
        resolve_bash("bash:{command}", command="export BIRDIE_RB_AFTER=ok")
        assert resolve_bash(
            "bash:{command}", command="echo $BIRDIE_RB_AFTER"
        ).strip() == "ok"

    def test_shell_exit_zero_returns_output_with_reset_note(self):
        out = resolve_bash("bash:{command}", command="echo bye; exit 0")
        assert "bye" in out
        assert "reset" in out

    def test_shell_exit_nonzero_raises_with_reset_note(self):
        with pytest.raises(RuntimeError, match="reset"):
            resolve_bash("bash:{command}", command="exit 9")

    def test_quoted_template_still_injection_safe(self):
        resolve_bash("bash:{command}", command="cd /")
        out = resolve_bash("bash:echo {word}", word="hi; echo INJECTED")
        assert out.strip() == "hi; echo INJECTED"
