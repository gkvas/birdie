"""Loop-guard helpers in birdie.agent.graph."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from birdie.agent.graph import (
    LOOP_GUARD_PREFIX,
    _consecutive_call_count,
    _guard_fired_this_turn,
)


def _ai(name="t", args=None):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args or {"x": 1}, "id": "c"}])


def test_guard_not_fired_in_fresh_turn():
    msgs = [HumanMessage(content="go"), _ai(), ToolMessage(content="ok", tool_call_id="c")]
    assert _guard_fired_this_turn(msgs) is False


def test_guard_fired_is_detected_within_turn():
    msgs = [
        HumanMessage(content="go"),
        _ai(),
        ToolMessage(content=LOOP_GUARD_PREFIX + " 't' repeated", tool_call_id="c"),
        _ai(),
        ToolMessage(content="ok", tool_call_id="c"),
    ]
    assert _guard_fired_this_turn(msgs) is True


def test_guard_resets_on_new_human_turn():
    msgs = [
        HumanMessage(content="go"),
        _ai(),
        ToolMessage(content=LOOP_GUARD_PREFIX + " 't' repeated", tool_call_id="c"),
        HumanMessage(content="try again"),
        _ai(),
    ]
    assert _guard_fired_this_turn(msgs) is False


def test_consecutive_count_counts_identical_calls():
    msgs = [HumanMessage(content="go")]
    for _ in range(4):
        msgs += [_ai(), ToolMessage(content="e", tool_call_id="c")]
    msgs.append(_ai())
    assert _consecutive_call_count(msgs, "t", {"x": 1}) == 5
    assert _consecutive_call_count(msgs, "t", {"x": 2}) == 0
