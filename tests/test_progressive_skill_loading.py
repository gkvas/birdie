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
