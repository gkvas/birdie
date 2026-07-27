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
