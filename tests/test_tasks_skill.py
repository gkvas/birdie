"""
Tests for the Tasks skill's python: entrypoints (background task monitoring).
"""

import json
import time

import pytest

from birdie.core.entrypoints import resolve_python
from birdie.skills.tasks import tools


@pytest.fixture
def tasks_dir(tmp_path, monkeypatch):
    d = tmp_path / "tasks"
    monkeypatch.setenv("BIRDIE_TASKS_DIR", str(d))
    return d


@pytest.fixture(autouse=True)
def _clean_procs():
    tools._procs.clear()
    yield
    # never leave a spawned process behind
    for tid, proc in list(tools._procs.items()):
        if proc.poll() is None:
            proc.kill()
        tools._procs.pop(tid, None)


def _only_task_id(tasks_dir):
    return next(tasks_dir.iterdir()).name


class TestTaskStart:
    def test_start_returns_id_and_writes_meta(self, tasks_dir):
        out = tools.task_start("echo hello", name="greet")
        assert "Started task" in out
        assert "pid" in out
        task_dir = next(tasks_dir.iterdir())
        meta = json.loads((task_dir / "meta.json").read_text())
        assert meta["name"] == "greet"
        assert meta["command"] == "echo hello"
        assert meta["status"] == "running"
        assert meta["pid"] > 0
        assert (task_dir / "output.log").exists()

    def test_empty_command_raises(self, tasks_dir):
        with pytest.raises(ValueError, match="must not be empty"):
            tools.task_start("   ")

    def test_callable_via_python_entrypoint(self, tasks_dir):
        out = resolve_python(
            "python:birdie.skills.tasks.tools.task_start",
            command="echo via-entrypoint",
        )
        assert "Started task" in out


class TestWaitAndStatus:
    def test_quick_command_finishes_with_rc0(self, tasks_dir):
        tools.task_start("echo hello-task")
        tid = _only_task_id(tasks_dir)
        out = tools.task_wait(tid, timeout=10)
        assert "finished: done rc=0" in out
        assert "hello-task" in out

    def test_failing_command_reports_rc(self, tasks_dir):
        tools.task_start("echo oops; exit 3")
        tid = _only_task_id(tasks_dir)
        out = tools.task_wait(tid, timeout=10)
        assert "failed" in out and "rc=3" in out
        assert "oops" in out

    def test_wait_timeout_reports_still_running_and_task_survives(self, tasks_dir):
        tools.task_start("sleep 30", name="sleeper")
        tid = _only_task_id(tasks_dir)
        out = tools.task_wait(tid, timeout=1)
        assert "still running" in out
        # the wait must not kill the task - it is still running afterwards
        assert "running" in tools.task_status(tid)
        tools.task_stop(tid)

    def test_status_shows_output_tail(self, tasks_dir):
        tools.task_start("echo line1; echo line2")
        tid = _only_task_id(tasks_dir)
        tools.task_wait(tid, timeout=10)
        status = tools.task_status(tid)
        assert "done" in status
        assert "line1" in status and "line2" in status

    def test_wait_timeout_is_capped(self, tasks_dir, monkeypatch):
        monkeypatch.setattr(tools, "MAX_WAIT_S", 1.0)
        tools.task_start("sleep 30")
        tid = _only_task_id(tasks_dir)
        start = time.monotonic()
        out = tools.task_wait(tid, timeout=99999)  # must be capped, not 99999s
        assert "still running" in out
        assert time.monotonic() - start < 10
        tools.task_stop(tid)


class TestList:
    def test_list_empty(self, tasks_dir):
        assert "No tasks" in tools.task_list()

    def test_list_shows_tasks_newest_first(self, tasks_dir):
        tools.task_start("echo a", name="a")
        time.sleep(1.1)  # distinct timestamp-based ids
        tools.task_start("echo b", name="b")
        out = tools.task_list()
        assert "2 task(s)" in out
        assert out.index("[b]") < out.index("[a]")

    def test_list_filter_by_status(self, tasks_dir):
        tools.task_start("echo quick", name="quick")
        quick_id = _only_task_id(tasks_dir)
        tools.task_wait(quick_id, timeout=10)
        tools.task_start("sleep 30", name="sleeper")
        running = tools.task_list(status="running")
        assert "sleeper" in running and "quick" not in running
        done = tools.task_list(status="done")
        assert "quick" in done and "sleeper" not in done
        sleeper_id = next(
            d.name for d in tasks_dir.iterdir() if d.name != quick_id
        )
        tools.task_stop(sleeper_id)


class TestStop:
    def test_stop_running_task(self, tasks_dir):
        tools.task_start("sleep 60", name="victim")
        tid = _only_task_id(tasks_dir)
        out = tools.task_stop(tid)
        assert "Stopped task" in out
        assert "stopped" in tools.task_status(tid)

    def test_stop_finished_task_is_noop(self, tasks_dir):
        tools.task_start("true")
        tid = _only_task_id(tasks_dir)
        tools.task_wait(tid, timeout=10)
        out = tools.task_stop(tid)
        assert "not running" in out


class TestResolve:
    def test_prefix_resolution(self, tasks_dir):
        tools.task_start("echo prefix-test")
        tid = _only_task_id(tasks_dir)
        short = tid[:13]  # timestamp part of the id
        assert tid in tools.task_status(short)

    def test_unknown_id_raises(self, tasks_dir):
        with pytest.raises(ValueError, match="Unknown task id"):
            tools.task_status("nope")

    def test_ambiguous_prefix_raises(self, tasks_dir):
        for suffix in ("aaaa", "bbbb"):
            d = tasks_dir / ("20260101-000000-" + suffix)
            d.mkdir(parents=True)
            (d / "meta.json").write_text(json.dumps({
                "id": d.name, "name": "", "command": "true", "pid": 1,
                "cwd": "/", "started_at": "2026-01-01T00:00:00+00:00",
                "status": "done", "rc": 0,
            }))
        with pytest.raises(ValueError, match="ambiguous"):
            tools.task_status("20260101-000000")


class TestOutput:
    def test_task_output_tail(self, tasks_dir):
        tools.task_start("for i in 1 2 3 4 5; do echo line$i; done")
        tid = _only_task_id(tasks_dir)
        tools.task_wait(tid, timeout=10)
        out = tools.task_output(tid, tail=2)
        assert "line4" in out and "line5" in out
        assert "line1" not in out

    def test_task_output_empty(self, tasks_dir):
        tools.task_start("sleep 30")
        tid = _only_task_id(tasks_dir)
        out = tools.task_output(tid)
        assert "no output yet" in out
        tools.task_stop(tid)


class TestOmittedOptionalArgs:
    """The StructuredTool wrapper passes None for omitted optional args;
    the tools must fall back to their documented defaults."""

    def test_list_with_limit_none(self, tasks_dir):
        tools.task_start("echo hi", name="a")
        out = tools.task_list(status=None, limit=None)
        assert "1 task(s)" in out

    def test_wait_with_timeout_none(self, tasks_dir):
        tools.task_start("echo hi; exit 0", name="b")
        tid = _only_task_id(tasks_dir)
        out = tools.task_wait(tid, timeout=None)
        assert "finished: done rc=0" in out

    def test_output_with_tail_none(self, tasks_dir):
        tools.task_start("echo hi; exit 0", name="c")
        tid = _only_task_id(tasks_dir)
        tools.task_wait(tid, timeout=10)
        out = tools.task_output(tid, tail=None)
        assert "hi" in out

    def test_start_with_name_and_cwd_none(self, tasks_dir):
        out = tools.task_start("echo hi; exit 0", name=None, cwd=None)
        assert "Started task" in out
        tid = _only_task_id(tasks_dir)
        meta = json.loads((tasks_dir / tid / "meta.json").read_text())
        assert meta["name"] == ""
        tools.task_wait(tid, timeout=10)
