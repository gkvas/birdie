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
