"""
Tests for entrypoint resolvers, in particular bash: shell-quoting behaviour.
"""

import pytest

from birdie.core.entrypoints import resolve_bash, resolve_entrypoint


class TestResolveBashQuoting:
    def test_simple_substitution(self):
        out = resolve_bash("bash:echo {word}", word="hello")
        assert out.strip() == "hello"

    def test_injection_attempt_is_quoted(self):
        """A value containing shell metacharacters must not spawn extra commands."""
        out = resolve_bash("bash:echo {word}", word="hi; echo INJECTED")
        assert out.strip() == "hi; echo INJECTED"  # printed literally, not executed

    def test_command_substitution_is_quoted(self):
        out = resolve_bash("bash:echo {word}", word="$(echo INJECTED)")
        assert "INJECTED" in out and "$(" in out  # literal, not expanded

    def test_raw_single_placeholder_template_runs_full_command(self):
        """A template that is exactly one placeholder receives the raw command."""
        out = resolve_bash("bash:{command}", command="echo one && echo two")
        assert out.splitlines() == ["one", "two"]

    def test_nonzero_exit_raises(self):
        with pytest.raises(RuntimeError):
            resolve_bash("bash:{command}", command="false")


class TestResolveEntrypointDispatch:
    def test_bash_scheme(self):
        assert resolve_entrypoint("bash:echo hi") is resolve_bash

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError):
            resolve_entrypoint("carrier-pigeon:coo")


# ---------------------------------------------------------------------------
# Per-tool timeout and retry (F10)
# ---------------------------------------------------------------------------

_flaky_calls = {"n": 0}


def flaky_tool(**kwargs):
    """Fails on the first call, succeeds afterwards (module-level for python: entrypoint)."""
    _flaky_calls["n"] += 1
    if _flaky_calls["n"] == 1:
        raise RuntimeError("transient failure")
    return "recovered"


class TestToolTimeoutParsing:
    def test_timeout_and_retries_parsed_from_tool_block(self):
        from birdie.core.loader import parse_skill_markdown
        skill = parse_skill_markdown(
            "---\nname: T\nversion: 1.0.0\ndescription: d\n---\n\n"
            "## Tools\n\n"
            "### slow_tool\n"
            "description: d\n"
            "entrypoint: bash:sleep {seconds}\n"
            "timeout: 2.5\n"
            "retries: 3\n"
            "schema:\n"
            "  type: object\n"
            "  properties:\n"
            "    seconds:\n"
            "      type: string\n"
            "    timeout:\n"
            "      type: integer\n"
        )
        tool = skill.tools[0]
        assert tool.timeout == 2.5
        assert tool.retries == 3

    def test_schema_property_named_timeout_not_misread(self):
        from birdie.core.loader import parse_skill_markdown
        skill = parse_skill_markdown(
            "---\nname: T\nversion: 1.0.0\ndescription: d\n---\n\n"
            "## Tools\n\n"
            "### t\n"
            "description: d\n"
            "entrypoint: bash:true\n"
            "schema:\n"
            "  type: object\n"
            "  properties:\n"
            "    timeout:\n"
            "      type: integer\n"
        )
        assert skill.tools[0].timeout is None
        assert skill.tools[0].retries is None


class TestBashTimeout:
    def test_bash_timeout_raises_runtime_error(self):
        import time
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="timed out"):
            resolve_bash("bash:{command}", _timeout=0.2, command="sleep 5")
        assert time.monotonic() - start < 3


class TestAdapterRetry:
    def _make_tool(self, retries=None, entrypoint="python:tests.test_entrypoints.flaky_tool"):
        from birdie.core.adapter import skilltool_to_langchain_tool
        from birdie.core.models import SkillTool
        return skilltool_to_langchain_tool(SkillTool(
            name="flaky", description="d", entrypoint=entrypoint,
            schema={"type": "object", "properties": {}},
            retries=retries,
        ))

    def test_explicit_retries_recover_from_transient_failure(self):
        _flaky_calls["n"] = 0
        tool = self._make_tool(retries=1)
        assert tool.func() == "recovered"
        assert _flaky_calls["n"] == 2

    def test_no_retries_by_default_for_python_entrypoint(self):
        _flaky_calls["n"] = 0
        tool = self._make_tool(retries=None)
        with pytest.raises(RuntimeError, match="transient"):
            tool.func()
        assert _flaky_calls["n"] == 1

    def test_http_get_defaults_to_one_retry(self, monkeypatch):
        from birdie.core import entrypoints
        calls = {"n": 0}

        def fake_get(entrypoint, _timeout=None, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("connection reset")
            return {"ok": True}

        monkeypatch.setattr(entrypoints, "resolve_http_get", fake_get)
        tool = self._make_tool(retries=None, entrypoint="http:get https://x.test/a")
        assert tool.func() == {"ok": True}
        assert calls["n"] == 2

    def test_timeout_forwarded_to_resolver(self, monkeypatch):
        from birdie.core import entrypoints
        from birdie.core.adapter import skilltool_to_langchain_tool
        from birdie.core.models import SkillTool
        seen = {}

        def fake_get(entrypoint, _timeout=None, **kwargs):
            seen["timeout"] = _timeout
            return {}

        monkeypatch.setattr(entrypoints, "resolve_http_get", fake_get)
        tool = skilltool_to_langchain_tool(SkillTool(
            name="t", description="d", entrypoint="http:get https://x.test/a",
            schema={"type": "object", "properties": {}}, timeout=7.5,
        ))
        tool.func()
        assert seen["timeout"] == 7.5


# ---------------------------------------------------------------------------
# Per-tool-call timeout (`timeout_s`)
# ---------------------------------------------------------------------------

import os
import re as _re
import subprocess
import sys
import time

from birdie.core.entrypoints import (
    BASH_TIMEOUT,
    HTTP_TIMEOUT,
    MAX_CALL_TIMEOUT,
    MIN_CALL_TIMEOUT,
    TIMEOUT_PARAM,
    ToolTimeoutError,
    _effective_timeout,
    resolve_http_get,
)


class TestEffectiveTimeout:
    def test_default_when_nothing_set(self):
        assert _effective_timeout(None, None, BASH_TIMEOUT) == BASH_TIMEOUT

    def test_skill_config_beats_default(self):
        assert _effective_timeout(None, 42.0, BASH_TIMEOUT) == 42.0

    def test_per_call_beats_skill_config(self):
        assert _effective_timeout(5, 42.0, BASH_TIMEOUT) == 5.0

    def test_per_call_clamped_to_max(self):
        assert _effective_timeout(99999, None, BASH_TIMEOUT) == MAX_CALL_TIMEOUT

    def test_per_call_clamped_to_min(self):
        assert _effective_timeout(0.001, None, BASH_TIMEOUT) == MIN_CALL_TIMEOUT

    def test_skill_config_is_not_clamped(self):
        """Author-written SKILL.MD values are trusted config."""
        assert _effective_timeout(None, 3600.0, BASH_TIMEOUT) == 3600.0

    def test_numeric_string_accepted(self):
        """LLMs sometimes send numbers as strings."""
        assert _effective_timeout("30", None, BASH_TIMEOUT) == 30.0

    def test_non_numeric_raises_value_error(self):
        with pytest.raises(ValueError, match=TIMEOUT_PARAM):
            _effective_timeout("soon", None, BASH_TIMEOUT)


class TestBashPerCallTimeout:
    def test_timeout_s_overrides_skill_timeout(self):
        start = time.monotonic()
        with pytest.raises(ToolTimeoutError):
            resolve_bash(
                "bash:{command}", _timeout=30.0,
                command="sleep 20", timeout_s=1,
            )
        assert time.monotonic() - start < 5

    def test_timeout_error_includes_partial_output(self):
        with pytest.raises(ToolTimeoutError) as excinfo:
            resolve_bash(
                "bash:{command}",
                command="echo started; echo warn >&2; sleep 20", timeout_s=1,
            )
        message = str(excinfo.value)
        assert "timed out after 1s" in message
        assert "started" in message
        assert "warn" in message
        assert TIMEOUT_PARAM in message  # tells the LLM how to raise it

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
    def test_whole_process_tree_is_killed(self):
        """A background child must not survive the timeout kill.

        This is the emulator/daemon hang: subprocess's own kill only reaches
        the shell, leaving grandchildren running and pipes open.
        """
        with pytest.raises(ToolTimeoutError) as excinfo:
            resolve_bash(
                "bash:{command}",
                command="sleep 20 & echo pid=$!; sleep 20", timeout_s=1,
            )
        match = _re.search(r"pid=(\d+)", str(excinfo.value))
        assert match, "background pid not captured in partial output"
        pid = int(match.group(1))
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break  # dead, as it should be
            time.sleep(0.1)
        else:
            os.kill(pid, 9)  # clean up before failing
            pytest.fail(f"background child {pid} survived the timeout kill")

    def test_output_pipe_holder_does_not_hang_the_call(self):
        """A survivor holding stdout open must not stall the tool call.

        `setsid` detaches the child into its own session, out of reach of the
        group kill -- the worst case.  The call must still return promptly
        with the output captured so far.
        """
        if sys.platform == "win32":
            pytest.skip("POSIX setsid")
        start = time.monotonic()
        with pytest.raises(ToolTimeoutError):
            resolve_bash(
                "bash:{command}",
                command="setsid sleep 20 & echo held; sleep 20", timeout_s=1,
            )
        assert time.monotonic() - start < 8
        # best-effort cleanup of the deliberate escapee
        subprocess.run(["pkill", "-f", "setsid sleep 20"], capture_output=True)

    def test_fast_command_unaffected(self):
        assert resolve_bash(
            "bash:{command}", command="echo quick", timeout_s=30
        ).strip() == "quick"

    def test_non_utf8_output_does_not_crash_or_truncate(self):
        """Windows-codepage tools under WSL emit non-UTF-8 bytes; strict
        decoding would kill the reader thread and lose everything after."""
        out = resolve_bash(
            "bash:{command}", command=r"printf 'caf\x81 before\nafter\n'"
        )
        assert "before" in out and "after" in out

    def test_timeout_is_not_retried(self, monkeypatch):
        from birdie.core import entrypoints
        from birdie.core.adapter import skilltool_to_langchain_tool
        from birdie.core.models import SkillTool
        calls = {"n": 0}

        def fake_bash(entrypoint, _timeout=None, **kwargs):
            calls["n"] += 1
            raise ToolTimeoutError("timed out")

        monkeypatch.setattr(entrypoints, "resolve_bash", fake_bash)
        tool = skilltool_to_langchain_tool(SkillTool(
            name="t", description="d", entrypoint="bash:{command}",
            schema={"type": "object", "properties": {}}, retries=3,
        ))
        with pytest.raises(ToolTimeoutError):
            tool.func(command="anything")
        assert calls["n"] == 1


class TestHttpPerCallTimeout:
    def test_timeout_s_used_and_not_sent_as_query_param(self, monkeypatch):
        seen = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"ok": True}

        def fake_get(url, params=None, timeout=None):
            seen["params"] = params
            seen["timeout"] = timeout
            return _Resp()

        monkeypatch.setattr("birdie.core.entrypoints.requests.get", fake_get)
        result = resolve_http_get(
            "http:get https://x.test/a", q="1", timeout_s=5
        )
        assert result == {"ok": True}
        assert seen["params"] == {"q": "1"}
        assert seen["timeout"] == 5.0


class TestTimeoutSchemaInjection:
    def _parse(self, tool_block: str):
        from birdie.core.loader import parse_skill_markdown
        return parse_skill_markdown(
            "---\nname: T\nversion: 1.0.0\ndescription: d\n---\n\n"
            "## Tools\n\n" + tool_block
        ).tools[0]

    def test_bash_tool_gets_timeout_s(self):
        tool = self._parse(
            "### t\ndescription: d\nentrypoint: bash:{command}\n"
            "schema:\n  type: object\n  properties:\n    command:\n"
            "      type: string\n  required: [command]\n"
        )
        prop = tool.schema["properties"][TIMEOUT_PARAM]
        assert prop["type"] == "number"
        assert str(int(BASH_TIMEOUT)) in prop["description"]
        assert TIMEOUT_PARAM not in tool.schema.get("required", [])

    def test_http_tool_gets_timeout_s_with_http_default(self):
        tool = self._parse(
            "### t\ndescription: d\nentrypoint: http:get https://x.test/a\n"
            "schema:\n  type: object\n  properties: {}\n"
        )
        prop = tool.schema["properties"][TIMEOUT_PARAM]
        assert str(int(HTTP_TIMEOUT)) in prop["description"]

    def test_python_tool_is_not_injected(self):
        tool = self._parse(
            "### t\ndescription: d\nentrypoint: python:os.getcwd\n"
            "schema:\n  type: object\n  properties: {}\n"
        )
        assert TIMEOUT_PARAM not in tool.schema.get("properties", {})

    def test_skill_defined_property_wins(self):
        tool = self._parse(
            "### t\ndescription: d\nentrypoint: bash:{command}\n"
            "schema:\n  type: object\n  properties:\n    timeout_s:\n"
            "      type: string\n      description: custom\n"
        )
        assert tool.schema["properties"][TIMEOUT_PARAM] == {
            "type": "string", "description": "custom",
        }

    def test_bash_tool_without_schema_gets_one(self):
        tool = self._parse("### t\ndescription: d\nentrypoint: bash:date\n")
        assert tool.schema["type"] == "object"
        assert TIMEOUT_PARAM in tool.schema["properties"]
