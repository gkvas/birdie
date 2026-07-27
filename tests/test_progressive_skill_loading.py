"""Tests for progressive skill loading (turn-decay eviction and LRU cap)."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from birdie.agent.graph import _loaded_skills_from_history, SKILL_DECAY_TURNS, SKILL_MAX_LOADED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gs_ai(skill_name: str, tc_id: str) -> AIMessage:
    """AIMessage that calls get_skill for the given skill."""
    return AIMessage(
        content="",
        tool_calls=[{"name": "get_skill", "args": {"skill_name": skill_name}, "id": tc_id}],
    )


def _gs_result(tc_id: str) -> ToolMessage:
    return ToolMessage(content="skill body", tool_call_id=tc_id, name="get_skill")


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------

def test_empty_messages_returns_empty_set():
    assert _loaded_skills_from_history([], {"ssh"}, 5, 3) == set()


def test_no_get_skill_calls_returns_empty_set():
    messages = [
        HumanMessage(content="hello"),
        AIMessage(content="hi"),
    ]
    assert _loaded_skills_from_history(messages, {"ssh"}, 5, 3) == set()


def test_fresh_load_is_active():
    messages = [
        HumanMessage(content="help with ssh"),
        _gs_ai("ssh", "tc1"),
        _gs_result("tc1"),
        AIMessage(content="Here is SSH help..."),
    ]
    result = _loaded_skills_from_history(messages, {"ssh"}, 5, 3)
    assert "ssh" in result


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------

def test_skill_active_at_decay_boundary():
    """Skill loaded, then exactly decay_turns human turns pass: still active."""
    messages = [
        HumanMessage(content="start"),
        _gs_ai("ssh", "tc1"),
        _gs_result("tc1"),
        AIMessage(content="ok"),
    ]
    for i in range(5):  # exactly SKILL_DECAY_TURNS human turns after load
        messages.append(HumanMessage(content=f"turn {i}"))
        messages.append(AIMessage(content="ok"))

    result = _loaded_skills_from_history(messages, {"ssh"}, 5, 3)
    assert "ssh" in result


def test_skill_evicted_after_decay():
    """One turn beyond the decay window causes eviction."""
    messages = [
        HumanMessage(content="start"),
        _gs_ai("ssh", "tc1"),
        _gs_result("tc1"),
        AIMessage(content="ok"),
    ]
    for i in range(6):  # decay_turns + 1
        messages.append(HumanMessage(content=f"turn {i}"))
        messages.append(AIMessage(content="ok"))

    result = _loaded_skills_from_history(messages, {"ssh"}, 5, 3)
    assert "ssh" not in result


def test_reload_resets_decay_counter():
    """Calling get_skill again before expiry extends the lease."""
    messages = [
        HumanMessage(content="start"),
        _gs_ai("ssh", "tc1"),
        _gs_result("tc1"),
        AIMessage(content="ok"),
    ]
    # 4 turns pass (would expire after 5)
    for i in range(4):
        messages.append(HumanMessage(content=f"a{i}"))
        messages.append(AIMessage(content="ok"))
    # Reload ssh - resets counter
    messages.append(_gs_ai("ssh", "tc2"))
    messages.append(_gs_result("tc2"))
    messages.append(AIMessage(content="ok"))
    # 4 more turns (total 8 from original load, but only 4 from reload)
    for i in range(4):
        messages.append(HumanMessage(content=f"b{i}"))
        messages.append(AIMessage(content="ok"))

    result = _loaded_skills_from_history(messages, {"ssh"}, 5, 3)
    assert "ssh" in result


# ---------------------------------------------------------------------------
# LRU cap
# ---------------------------------------------------------------------------

def test_lru_cap_evicts_oldest():
    """With max_loaded=3 and 4 loaded skills, the oldest is evicted."""
    messages = []
    for name in ["a", "b", "c", "d"]:  # d loaded last
        tc_id = f"tc_{name}"
        messages.append(_gs_ai(name, tc_id))
        messages.append(_gs_result(tc_id))
        messages.append(AIMessage(content="ok"))

    result = _loaded_skills_from_history(
        messages, {"a", "b", "c", "d"}, decay_turns=10, max_loaded=3
    )
    assert len(result) == 3
    assert "d" in result
    assert "c" in result
    assert "b" in result
    assert "a" not in result  # LRU evicted


def test_lru_cap_of_one():
    """max_loaded=1 keeps only the most recently loaded skill."""
    messages = []
    for name in ["weather", "ssh"]:
        tc_id = f"tc_{name}"
        messages.append(_gs_ai(name, tc_id))
        messages.append(_gs_result(tc_id))
        messages.append(AIMessage(content="ok"))

    result = _loaded_skills_from_history(
        messages, {"weather", "ssh"}, decay_turns=10, max_loaded=1
    )
    assert result == {"ssh"}


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------

def test_skill_not_in_allowed_is_ignored():
    """get_skill calls for skills outside the allowed set are not tracked."""
    messages = [
        _gs_ai("ssh", "tc1"),
        _gs_result("tc1"),
        AIMessage(content="ok"),
    ]
    result = _loaded_skills_from_history(messages, {"weather"}, 5, 3)
    assert "ssh" not in result


def test_only_allowed_skills_tracked():
    """Multiple skills loaded; only the allowed subset is returned."""
    messages = []
    for name in ["ssh", "weather"]:
        tc_id = f"tc_{name}"
        messages.append(_gs_ai(name, tc_id))
        messages.append(_gs_result(tc_id))
        messages.append(AIMessage(content="ok"))

    result = _loaded_skills_from_history(messages, {"ssh"}, 5, 3)
    assert result == {"ssh"}
    assert "weather" not in result


# ---------------------------------------------------------------------------
# Default constant sanity
# ---------------------------------------------------------------------------

def test_default_constants():
    assert SKILL_DECAY_TURNS == 5
    assert SKILL_MAX_LOADED == 3


class TestGetSkillReturnValue:
    """get_skill acks instead of duplicating the body: the body reaches the
    LLM via the system-prompt lease, not the ToolMessage."""

    @pytest.mark.asyncio
    async def test_get_skill_tool_returns_ack_not_body(self, tmp_path):
        import os
        from birdie.agent.run import DynamicAgent

        skill_dir = tmp_path / "knowledge"
        os.makedirs(skill_dir)
        (skill_dir / "SKILL.MD").write_text(
            "---\n"
            "name: Knowledge\n"
            "version: 1.0.0\n"
            "description: A knowledge skill\n"
            "---\n\n"
            "SECRET BODY TEXT that must not appear in the tool result.\n"
        )

        call_count = 0
        captured_system_prompts = []

        class _Provider:
            def supports_tools(self):
                return True

            async def achat(self, messages, tools=None, system_prompt=None, **kw):
                nonlocal call_count
                call_count += 1
                captured_system_prompts.append(system_prompt or "")
                if call_count == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "get_skill",
                            "args": {"skill_name": "Knowledge"},
                            "id": "tc1",
                        }],
                    )
                return AIMessage(content="Done")

        from birdie.core.llm_provider import LLMProvider
        provider = _Provider()
        # Duck-typed provider: DynamicAgent wraps non-LLMProvider objects in
        # LangChainProvider, so register as a virtual subclass instead.
        LLMProvider.register(_Provider)

        agent = DynamicAgent(
            provider, skills_dir=str(tmp_path), skills_enabled=["Knowledge"]
        )
        result = await agent.invoke("load the knowledge skill", thread_id="t1")

        tool_msgs = [
            m for m in result["messages"]
            if isinstance(m, ToolMessage) and m.name == "get_skill"
        ]
        assert len(tool_msgs) == 1
        assert "SECRET BODY TEXT" not in tool_msgs[0].content
        assert "loaded" in tool_msgs[0].content.lower()
        # The body must instead be injected into the system prompt of the
        # follow-up model call.
        assert "SECRET BODY TEXT" in captured_system_prompts[-1]
