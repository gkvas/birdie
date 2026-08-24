"""Placement of the ephemeral <session_context> message.

The volatile per-turn context (loaded skills, rolling summary, LTM) must be
inserted immediately BEFORE the turn's user message - never appended after
the conversation tail.  A trailing human-role message after every tool
result reads as a turn boundary to the model, which then re-orients
(re-checks state it already verified) instead of continuing its plan,
degenerating into identical-tool-call loops until the loop guard fires.
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from birdie.agent.run import DynamicAgent
from birdie.core.llm_provider import LLMProvider


def _make_knowledge_skill(tmp_path):
    skill_dir = tmp_path / "knowledge"
    os.makedirs(skill_dir)
    (skill_dir / "SKILL.MD").write_text(
        "---\n"
        "name: Knowledge\n"
        "version: 1.0.0\n"
        "description: A knowledge skill\n"
        "---\n\n"
        "SKILL BODY TEXT.\n"
    )


class _ToolCallingProvider:
    """Calls get_skill on the first round, finishes on the second."""

    def __init__(self):
        self.call_count = 0
        self.captured_requests = []

    def supports_tools(self):
        return True

    async def achat(self, messages, tools=None, system_prompt=None, **kw):
        self.call_count += 1
        self.captured_requests.append(list(messages))
        if self.call_count == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_skill",
                    "args": {"skill_name": "Knowledge"},
                    "id": "tc1",
                }],
            )
        return AIMessage(content="Done")


LLMProvider.register(_ToolCallingProvider)


def _ephemeral_indices(messages):
    return [
        i for i, m in enumerate(messages)
        if getattr(m, "additional_kwargs", {}).get("birdie_ephemeral")
    ]


@pytest.mark.asyncio
async def test_ephemeral_inserted_before_turn_user_message(tmp_path):
    """Single round: the context message precedes the user message."""
    _make_knowledge_skill(tmp_path)
    provider = _ToolCallingProvider()
    agent = DynamicAgent(
        provider, skills_dir=str(tmp_path), skills_enabled=["Knowledge"],
    )

    await agent.invoke(
        "hello", thread_id="t1", long_term_memory=["User likes tabs"],
    )

    first_request = provider.captured_requests[0]
    eph = _ephemeral_indices(first_request)
    assert len(eph) == 1
    humans = [
        i for i, m in enumerate(first_request)
        if isinstance(m, HumanMessage) and i not in eph
    ]
    # Immediately before the turn's (last real) user message.
    assert eph[0] == humans[-1] - 1


@pytest.mark.asyncio
async def test_ephemeral_not_appended_after_tool_results(tmp_path):
    """Mid-turn rounds end with the tool result, not a human-role message."""
    _make_knowledge_skill(tmp_path)
    provider = _ToolCallingProvider()
    agent = DynamicAgent(
        provider, skills_dir=str(tmp_path), skills_enabled=["Knowledge"],
    )

    await agent.invoke(
        "load the knowledge skill", thread_id="t1",
        long_term_memory=["User likes tabs"],
    )

    assert provider.call_count == 2
    second_request = provider.captured_requests[1]

    # The tool round is not interrupted: the request tail is the tool result.
    assert isinstance(second_request[-1], ToolMessage)

    # Exactly one context message, immediately before the turn's user message.
    eph = _ephemeral_indices(second_request)
    assert len(eph) == 1
    humans = [
        i for i, m in enumerate(second_request)
        if isinstance(m, HumanMessage) and i not in eph
    ]
    assert eph[0] == humans[-1] - 1
    # And the loaded skill body rides in it.
    assert "SKILL BODY TEXT" in str(second_request[eph[0]].content)
