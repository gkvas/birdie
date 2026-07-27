"""
Tests for conversation history compaction.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from birdie.agent.graph import (
    MIN_MESSAGES_AUTO,
    MIN_MESSAGES_FORCED,
    COMPRESSION_WINDOW_SIZE,
    compact_history,
)

# Auto-compaction threshold derived from the two constants.
_TRIGGER = MIN_MESSAGES_AUTO + COMPRESSION_WINDOW_SIZE


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
    """Below auto-trigger threshold - no compaction."""
    msgs = _build_history((_TRIGGER // 2) // 2)  # well below threshold
    provider = _MockProvider()
    summary, removes = await compact_history(msgs, provider)
    assert summary == ""
    assert removes == []
    assert provider.calls == []


@pytest.mark.asyncio
async def test_compact_history_at_threshold_triggers():
    """At the auto-trigger threshold, compaction fires."""
    msgs = _build_history(_TRIGGER // 2)  # each turn = 2 msgs -> exactly _TRIGGER messages
    assert len(msgs) >= _TRIGGER
    provider = _MockProvider()
    summary, removes = await compact_history(msgs, provider)
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_compact_history_below_threshold_no_op():
    """One message below auto-trigger threshold - no compaction."""
    n_turns = (_TRIGGER - 1) // 2
    msgs = _build_history(n_turns)
    if len(msgs) >= _TRIGGER:
        msgs = msgs[:_TRIGGER - 1]
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
    msgs = _build_history(_TRIGGER // 2)
    provider = _MockProvider()
    summary, removes = await compact_history(msgs, provider)
    assert isinstance(summary, str)
    assert "Summary" in summary


@pytest.mark.asyncio
async def test_compact_history_returns_only_remove_messages():
    """All state updates are RemoveMessage objects - no new messages inserted."""
    msgs = _build_history(_TRIGGER // 2)
    provider = _MockProvider()
    _, removes = await compact_history(msgs, provider)
    assert len(removes) > 0
    for r in removes:
        assert isinstance(r, RemoveMessage)


@pytest.mark.asyncio
async def test_compact_history_removed_ids_are_in_original():
    """Every removed ID must exist in the original history."""
    msgs = _build_history(_TRIGGER // 2)
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
    """No more than COMPRESSION_WINDOW_SIZE messages are removed per run."""
    msgs = _build_history(_TRIGGER // 2)
    provider = _MockProvider()
    _, removes = await compact_history(msgs, provider)
    assert len(removes) <= COMPRESSION_WINDOW_SIZE


@pytest.mark.asyncio
async def test_compact_history_leaves_at_least_min_messages():
    """At least MIN_MESSAGES_AUTO messages must remain after auto-compaction."""
    msgs = _build_history(_TRIGGER // 2)
    provider = _MockProvider()
    _, removes = await compact_history(msgs, provider)
    remaining = len(msgs) - len(removes)
    assert remaining >= MIN_MESSAGES_AUTO


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

    msgs = _build_history(_TRIGGER // 2)
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
    msgs = _build_history(_TRIGGER // 2)
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
    msgs = _build_history(_TRIGGER // 2)
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
    for i in range(_TRIGGER // 4):
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
    msgs = _build_history(_TRIGGER // 2)
    provider = _MockProvider()
    await compact_history(msgs, provider)
    assert len(provider.calls) == 1


# ---------------------------------------------------------------------------
# force=True bypasses threshold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_history_force_bypasses_threshold():
    """force=True compacts even when len < auto-trigger threshold."""
    msgs = _build_history(10)  # 20 messages, well below _TRIGGER
    assert len(msgs) < _TRIGGER
    provider = _MockProvider()
    summary, removes = await compact_history(msgs, provider, force=True)
    assert len(provider.calls) == 1
    assert len(removes) > 0


@pytest.mark.asyncio
async def test_compact_history_force_false_skips_below_threshold():
    """Without force=True, below-threshold history is not compacted."""
    msgs = _build_history(10)
    assert len(msgs) < _TRIGGER
    provider = _MockProvider()
    summary, removes = await compact_history(msgs, provider, force=False)
    assert removes == []
    assert provider.calls == []


@pytest.mark.asyncio
async def test_compact_history_force_uses_min_messages_forced():
    """force=True uses MIN_MESSAGES_FORCED floor, allowing compaction of short sessions."""
    # Build a session with only MIN_MESSAGES_FORCED + a few extra messages.
    # Auto-compaction floor (MIN_MESSAGES_AUTO) would prevent compaction here.
    msgs = _build_history((MIN_MESSAGES_FORCED + 4) // 2 + 1)
    assert len(msgs) < MIN_MESSAGES_AUTO  # confirms this is below auto floor
    provider = _MockProvider()
    _, removes = await compact_history(msgs, provider, force=True)
    remaining = len(msgs) - len(removes)
    assert remaining >= MIN_MESSAGES_FORCED


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

def test_constants_sane():
    """Verify constant relationships: trigger > window > auto_floor >= forced_floor > 0."""
    assert _TRIGGER == MIN_MESSAGES_AUTO + COMPRESSION_WINDOW_SIZE
    assert COMPRESSION_WINDOW_SIZE > MIN_MESSAGES_AUTO
    assert MIN_MESSAGES_AUTO >= MIN_MESSAGES_FORCED
    assert MIN_MESSAGES_FORCED > 0


# ---------------------------------------------------------------------------
# Rolling summary (continuity bridge)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_history_folds_prior_summary_into_prompt():
    """A prior rolling summary is included in the compaction prompt."""
    msgs = _build_history(_TRIGGER // 2 + 1)
    provider = _MockProvider()
    await compact_history(msgs, provider, prior_summary="Earlier we fixed the parser.")
    prompt_text = provider.calls[0][0].content
    assert "Earlier we fixed the parser." in prompt_text


@pytest.mark.asyncio
async def test_compact_history_no_prior_summary_marker_when_absent():
    """Without a prior summary, no already-compacted marker appears in the prompt."""
    msgs = _build_history(_TRIGGER // 2 + 1)
    provider = _MockProvider()
    await compact_history(msgs, provider)
    prompt_text = provider.calls[0][0].content
    assert "already-compacted" not in prompt_text


# ---------------------------------------------------------------------------
# Background auto-compaction through the graph
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_compaction_runs_in_background_and_applies_next_turn():
    import asyncio
    from birdie.agent.run import DynamicAgent
    from birdie.core.llm_provider import LLMProvider

    class _ChatProvider:
        """Answers normal turns with 'ok' and compaction prompts with JSON."""

        def supports_tools(self):
            return False

        async def achat(self, messages, tools=None, system_prompt=None, **kw):
            text = str(messages[0].content) if messages else ""
            if "memory compaction system" in text:
                return AIMessage(content=_MockProvider._DEFAULT_JSON)
            return AIMessage(content="ok")

    LLMProvider.register(_ChatProvider)

    agent = DynamicAgent(
        _ChatProvider(),
        min_messages_auto=4,
        compression_window_size=6,
    )

    # Trigger threshold: 4 + 6 = 10 messages. Each turn adds 2.
    for i in range(6):
        result = await agent.invoke(f"turn {i}", thread_id="bg")

    # Turn 6 saw >= 10 messages and started the background task; no removal
    # happened inside that same turn.
    assert len(result["messages"]) == 12

    # Let the background task complete, then run one more turn.
    for _ in range(10):
        await asyncio.sleep(0)
    result = await agent.invoke("turn 6", thread_id="bg")

    assert len(result["messages"]) < 14  # compacted prefix was removed
    assert result.get("summary") == "Summary of earlier conversation."


@pytest.mark.asyncio
async def test_token_threshold_triggers_compaction_below_message_count():
    import asyncio
    from birdie.agent.run import DynamicAgent
    from birdie.core.llm_provider import LLMProvider

    class _HeavyProvider:
        """Reports a huge input-token footprint on every normal reply."""

        def supports_tools(self):
            return False

        async def achat(self, messages, tools=None, system_prompt=None, **kw):
            text = str(messages[0].content) if messages else ""
            if "memory compaction system" in text:
                return AIMessage(content=_MockProvider._DEFAULT_JSON)
            return AIMessage(
                content="ok",
                usage_metadata={"input_tokens": 150_000, "output_tokens": 5,
                                "total_tokens": 150_005},
            )

    LLMProvider.register(_HeavyProvider)

    agent = DynamicAgent(_HeavyProvider(), compaction_token_threshold=100_000)

    # Far below the message-count trigger (4 turns = 8 messages < 80), but the
    # reported input tokens exceed the threshold after turn 1.
    for i in range(4):
        await agent.invoke(f"turn {i}", thread_id="heavy")
    for _ in range(10):
        await asyncio.sleep(0)
    result = await agent.invoke("final", thread_id="heavy")

    assert result.get("summary") == "Summary of earlier conversation."


@pytest.mark.asyncio
async def test_no_token_trigger_without_threshold():
    import asyncio
    from birdie.agent.run import DynamicAgent
    from birdie.core.llm_provider import LLMProvider

    class _HeavyProvider2:
        def supports_tools(self):
            return False

        async def achat(self, messages, tools=None, system_prompt=None, **kw):
            return AIMessage(
                content="ok",
                usage_metadata={"input_tokens": 150_000, "output_tokens": 5,
                                "total_tokens": 150_005},
            )

    LLMProvider.register(_HeavyProvider2)
    agent = DynamicAgent(_HeavyProvider2())

    for i in range(4):
        await agent.invoke(f"turn {i}", thread_id="light")
    for _ in range(10):
        await asyncio.sleep(0)
    result = await agent.invoke("final", thread_id="light")

    assert "summary" not in result
    assert len(result["messages"]) == 10
