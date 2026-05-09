"""Tests for AgentRegistry session policy."""

import pytest
from unittest.mock import MagicMock

from birdie.core.agent_registry import AgentRegistry
from birdie.core.models import AgentDef


def _make_def(name: str, enabled_by_default: bool = False) -> AgentDef:
    return AgentDef(
        name=name,
        description=f"Agent {name}",
        enabled_by_default=enabled_by_default,
        prompt="do something",
    )


def _make_tool(name: str):
    tool = MagicMock()
    tool.name = name
    return tool


class TestAgentRegistryDefaults:
    def test_register_not_default(self):
        reg = AgentRegistry()
        reg.register(_make_def("A", enabled_by_default=False), _make_tool("A"))
        assert "A" not in reg.get_allowed_agents()

    def test_register_default(self):
        reg = AgentRegistry()
        reg.register(_make_def("A", enabled_by_default=True), _make_tool("A"))
        assert "A" in reg.get_allowed_agents()

    def test_list_agents(self):
        reg = AgentRegistry()
        reg.register(_make_def("X"), _make_tool("X"))
        reg.register(_make_def("Y"), _make_tool("Y"))
        names = {a.name for a in reg.list_agents()}
        assert names == {"X", "Y"}

    def test_get_agent(self):
        reg = AgentRegistry()
        reg.register(_make_def("Z"), _make_tool("Z"))
        assert reg.get_agent("Z").name == "Z"
        assert reg.get_agent("missing") is None


class TestAgentRegistrySessionPolicy:
    def test_session_seeds_from_defaults(self):
        reg = AgentRegistry()
        reg.register(_make_def("A", enabled_by_default=True), _make_tool("A"))
        reg.register(_make_def("B", enabled_by_default=False), _make_tool("B"))
        allowed = reg.get_allowed_agents("session1")
        assert "A" in allowed
        assert "B" not in allowed

    def test_enable_agent_for_session(self):
        reg = AgentRegistry()
        reg.register(_make_def("B", enabled_by_default=False), _make_tool("B"))
        reg.enable_agent("s1", "B")
        assert "B" in reg.get_allowed_agents("s1")

    def test_disable_agent_for_session(self):
        reg = AgentRegistry()
        reg.register(_make_def("A", enabled_by_default=True), _make_tool("A"))
        reg.disable_agent("s1", "A")
        assert "A" not in reg.get_allowed_agents("s1")

    def test_sessions_are_isolated(self):
        reg = AgentRegistry()
        reg.register(_make_def("A", enabled_by_default=False), _make_tool("A"))
        reg.enable_agent("s1", "A")
        assert "A" in reg.get_allowed_agents("s1")
        assert "A" not in reg.get_allowed_agents("s2")

    def test_no_session_returns_defaults(self):
        reg = AgentRegistry()
        reg.register(_make_def("A", enabled_by_default=True), _make_tool("A"))
        assert "A" in reg.get_allowed_agents(None)
        assert "A" in reg.get_allowed_agents("")

    def test_disable_nonexistent_is_noop(self):
        reg = AgentRegistry()
        reg.disable_agent("s1", "ghost")  # should not raise

    def test_enable_not_registered_is_harmless(self):
        reg = AgentRegistry()
        reg.enable_agent("s1", "ghost")
        assert "ghost" in reg.get_allowed_agents("s1")

    def test_enable_agents_for_session_replaces_set(self):
        reg = AgentRegistry()
        reg.register(_make_def("A", enabled_by_default=True), _make_tool("A"))
        reg.register(_make_def("B", enabled_by_default=False), _make_tool("B"))
        reg.enable_agents_for_session("s1", ["B"])
        assert "B" in reg.get_allowed_agents("s1")
        assert "A" not in reg.get_allowed_agents("s1")

    def test_enable_agents_for_session_isolates_sessions(self):
        reg = AgentRegistry()
        reg.register(_make_def("A", enabled_by_default=True), _make_tool("A"))
        reg.enable_agents_for_session("s1", [])
        assert "A" not in reg.get_allowed_agents("s1")
        assert "A" in reg.get_allowed_agents("s2")


class TestAgentRegistryGetTools:
    def test_returns_tools_for_allowed(self):
        reg = AgentRegistry()
        tool_a = _make_tool("A")
        reg.register(_make_def("A", enabled_by_default=True), tool_a)
        reg.register(_make_def("B", enabled_by_default=False), _make_tool("B"))

        tools = reg.get_tools(reg.get_allowed_agents())
        assert tool_a in tools
        assert len(tools) == 1

    def test_empty_allowed_returns_empty(self):
        reg = AgentRegistry()
        reg.register(_make_def("A", enabled_by_default=True), _make_tool("A"))
        assert reg.get_tools(set()) == []

    def test_unknown_name_in_allowed_skipped(self):
        reg = AgentRegistry()
        tools = reg.get_tools({"nonexistent"})
        assert tools == []
