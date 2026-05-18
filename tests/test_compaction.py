"""
Tests for conversation history compaction.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from birdie.agent.graph import (
    MIN_MESSAGES,
    MAX_MESSAGES,
    COMPRESSION_WINDOW,
    compact_history,
)


class _MockProvider:
    """Minimal provider stub that returns a fixed JSON compaction response."""

    _DEFAULT_JSON = (
        '{"summary": "Summary of earlier conversation.", '
        '"extracted_facts": ["fact1"], '
        '"user_preferences": [], '
        '"world_facts": [], '
        '"tool_results": [], '
        '"open_tasks": []}'
    )

    def __init__(self, response: str = _DEFAULT_JSON):
        self._response = response
        self.calls: list = []

    async def achat(self, messages, **kwargs):
        self.calls.append(messages)
        return AIMessage(content=self._response)


def _make_turn(user_text: str, assistant_text: str, *, msg_id_base: int = 0):
    """Return a (HumanMessage, AIMessage) turn with stable IDs."""
    h = HumanMessage(content=user_text, id=f"h{msg_id_base}")
    a = AIMessage(content=assistant_text, id=f"a{msg_id_base}")
    return h, a


def _build_history(n_turns: int) -> list:
    """Build a simple history of n_turns user/assistant pairs."""
    msgs = []
    for i in range(n_turns):
        h, a = _make_turn(f"User message {i}", f"Assistant response {i}", msg_id_base=i)
        msgs.append(h)
        msgs.append(a)
    return msgs


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_history_too_few_messages_no_op():
    """Below MAX_MESSAGES threshold - no compaction."""
    msgs = _build_history(MAX_MESSAGES // 2 // 2)  # well below threshold
    provider = _MockProvider()
    summary, removes = await compact_history(msgs, provider)
    assert summary == ""
    assert removes == []
    assert provider.calls == []


@pytest.mark.asyncio
async def test_compact_history_at_threshold_triggers():
    """Exactly MAX_MESSAGES messages should trigger compaction."""
    msgs = _build_history(MAX_MESSAGES // 2)  # each turn = 2 msgs
    provider = _MockProvider()
    summary, removes = await compact_history(msgs, provider)
    # Should attempt compaction (provider is called)
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_compact_history_below_threshold_no_op():
    """One message below MAX_MESSAGES - no compaction."""
    msgs = _build_history((MAX_MESSAGES - 1) // 2)
    if len(msgs) >= MAX_MESSAGES:
        msgs = msgs[:MAX_MESSAGES - 1]
    provider = _MockProvider()
    summary, removes = await compact_history(msgs, provider)
    assert removes == []
    assert provider.calls == []


# ---------------------------------------------------------------------------
# Return value shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_history_returns_summary_string():
    """compact_history returns a non-empty summary string on success."""
    msgs = _build_history(MAX_MESSAGES // 2)
    provider = _MockProvider()
    summary, removes = await compact_history(msgs, provider)
    assert isinstance(summary, str)
    assert "Summary" in summary


@pytest.mark.asyncio
async def test_compact_history_returns_only_remove_messages():
    """All state updates are RemoveMessage objects - no new messages inserted."""
    msgs = _build_history(MAX_MESSAGES // 2)
    provider = _MockProvider()
    _, removes = await compact_history(msgs, provider)
    assert len(removes) > 0
    for r in removes:
        assert isinstance(r, RemoveMessage)


@pytest.mark.asyncio
async def test_compact_history_removed_ids_are_in_original():
    """Every removed ID must exist in the original history."""
    msgs = _build_history(MAX_MESSAGES // 2)
    provider = _MockProvider()
    _, removes = await compact_history(msgs, provider)
    original_ids = {m.id for m in msgs}
    for r in removes:
        assert r.id in original_ids


# ---------------------------------------------------------------------------
# Split alignment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_history_removes_at_most_compression_window():
    """No more than COMPRESSION_WINDOW messages are removed per run."""
    msgs = _build_history(MAX_MESSAGES // 2)
    provider = _MockProvider()
    _, removes = await compact_history(msgs, provider)
    assert len(removes) <= COMPRESSION_WINDOW


@pytest.mark.asyncio
async def test_compact_history_leaves_at_least_min_messages():
    """At least MIN_MESSAGES messages must remain after compaction."""
    msgs = _build_history(MAX_MESSAGES // 2)
    provider = _MockProvider()
    _, removes = await compact_history(msgs, provider)
    remaining = len(msgs) - len(removes)
    assert remaining >= MIN_MESSAGES


# ---------------------------------------------------------------------------
# LTM integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_history_ltm_store_add_called():
    """When an LTM store is provided, add() is called with the parsed result."""
    class _MockLTM:
        def __init__(self):
            self.calls = []

        def add(self, result: dict):
            self.calls.append(result)

    msgs = _build_history(MAX_MESSAGES // 2)
    provider = _MockProvider()
    ltm = _MockLTM()
    await compact_history(msgs, provider, ltm_store=ltm)
    assert len(ltm.calls) == 1
    assert "summary" in ltm.calls[0]
    assert "extracted_facts" in ltm.calls[0]


@pytest.mark.asyncio
async def test_compact_history_ltm_store_not_called_on_no_op():
    """LTM store is not touched when compaction doesn't trigger."""

    class _MockLTM:
        def __init__(self):
            self.calls = []

        def add(self, result: dict):
            self.calls.append(result)

    msgs = _build_history(5)
    provider = _MockProvider()
    ltm = _MockLTM()
    await compact_history(msgs, provider, ltm_store=ltm)
    assert ltm.calls == []


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_history_parses_json_summary():
    """The summary field from the JSON response is extracted correctly."""
    json_response = (
        '{"summary": "Test summary text.", '
        '"extracted_facts": [], "user_preferences": [], '
        '"world_facts": [], "tool_results": [], "open_tasks": []}'
    )
    msgs = _build_history(MAX_MESSAGES // 2)
    provider = _MockProvider(response=json_response)
    summary, _ = await compact_history(msgs, provider)
    assert summary == "Test summary text."


@pytest.mark.asyncio
async def test_compact_history_handles_json_embedded_in_prose():
    """Falls back gracefully when the model wraps JSON in surrounding prose."""
    json_response = (
        'Here is the compaction result:\n'
        '{"summary": "Embedded summary.", '
        '"extracted_facts": [], "user_preferences": [], '
        '"world_facts": [], "tool_results": [], "open_tasks": []}\n'
        'End of output.'
    )
    msgs = _build_history(MAX_MESSAGES // 2)
    provider = _MockProvider(response=json_response)
    summary, _ = await compact_history(msgs, provider)
    assert summary == "Embedded summary."


# ---------------------------------------------------------------------------
# Tool messages in history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_history_with_tool_messages():
    """Tool messages in old history are included in the transcript sent for summarisation."""
    msgs = []
    for i in range(MAX_MESSAGES // 4):
        h = HumanMessage(content=f"User {i}", id=f"h{i}")
        a = AIMessage(
            content="",
            tool_calls=[{"name": "mytool", "args": {"x": i}, "id": f"tc{i}", "type": "tool_call"}],
            id=f"a{i}",
        )
        tm = ToolMessage(content=f"result {i}", tool_call_id=f"tc{i}", name="mytool", id=f"tm{i}")
        a2 = AIMessage(content=f"Done {i}", id=f"a2_{i}")
        msgs.extend([h, a, tm, a2])

    provider = _MockProvider()
    _, removes = await compact_history(msgs, provider)

    assert len(removes) > 0
    prompt_text = provider.calls[0][0].content
    assert "mytool" in prompt_text


# ---------------------------------------------------------------------------
# Provider called exactly once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_history_provider_called_once():
    """Provider.achat is called exactly once per compaction run."""
    msgs = _build_history(MAX_MESSAGES // 2)
    provider = _MockProvider()
    await compact_history(msgs, provider)
    assert len(provider.calls) == 1


# ---------------------------------------------------------------------------
# force=True bypasses threshold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_history_force_bypasses_threshold():
    """force=True compacts even when len < MAX_MESSAGES."""
    # Build a history that is above the minimum but below MAX_MESSAGES
    msgs = _build_history(30)  # 60 messages, well below MAX_MESSAGES=100
    assert len(msgs) < MAX_MESSAGES
    provider = _MockProvider()
    summary, removes = await compact_history(msgs, provider, force=True)
    # Should have compacted (provider called, removes produced)
    assert len(provider.calls) == 1
    assert len(removes) > 0


@pytest.mark.asyncio
async def test_compact_history_force_false_skips_below_threshold():
    """Without force=True, below-threshold history is not compacted."""
    msgs = _build_history(30)
    assert len(msgs) < MAX_MESSAGES
    provider = _MockProvider()
    summary, removes = await compact_history(msgs, provider, force=False)
    assert removes == []
    assert provider.calls == []


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

def test_constants_sane():
    """MAX_MESSAGES > COMPRESSION_WINDOW > MIN_MESSAGES > 0."""
    assert MAX_MESSAGES > COMPRESSION_WINDOW
    assert COMPRESSION_WINDOW > MIN_MESSAGES
    assert MIN_MESSAGES > 0
    assert MAX_MESSAGES - COMPRESSION_WINDOW >= MIN_MESSAGES
