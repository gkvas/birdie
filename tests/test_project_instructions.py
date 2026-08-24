"""Tests for project instructions (CLAUDE.md / AGENTS.md) injection.

Project instruction files are wrapped in a <system-reminder> block and merged
into the first user message of the outgoing request - never into the system
prompt, and never into the checkpoint.
"""

import pytest
from pathlib import Path
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from birdie.agent.run import DynamicAgent
from birdie.core.llm_provider import LLMProvider


class CapturingProvider(LLMProvider):
    """Records the messages and system_prompt of every achat() call."""

    def __init__(self):
        self.captured_messages: list[list] = []
        self.captured_prompts: list[str | None] = []

    def chat(self, messages, tools=None, system_prompt=None, **kwargs) -> BaseMessage:
        return AIMessage(content="ok")

    async def achat(self, messages, tools=None, system_prompt=None, **kwargs) -> BaseMessage:
        self.captured_messages.append(list(messages))
        self.captured_prompts.append(system_prompt)
        return AIMessage(content="ok")

    def stream_chat(self, messages, tools=None, system_prompt=None, **kwargs):
        yield AIMessage(content="ok")

    async def astream_chat(self, messages, tools=None, system_prompt=None, **kwargs):
        yield AIMessage(content="ok")

    def list_models(self) -> list:
        return []


def _write_skill(directory: Path, name: str) -> None:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.MD").write_text(f"""---
name: {name}
version: 1.0.0
description: Test skill {name}
---
""")


@pytest.fixture
def skills_dir(tmp_path):
    d = tmp_path / "skills"
    d.mkdir()
    _write_skill(d, "TestSkill")
    return d


def _make_agent(provider, skills_dir):
    return DynamicAgent(provider, skills_dir=str(skills_dir), skills_enabled=["TestSkill"])


def _first_human(messages):
    return next(m for m in messages if isinstance(m, HumanMessage))


@pytest.mark.asyncio
async def test_claude_md_prepended_to_first_user_message(tmp_path, monkeypatch, skills_dir):
    """CLAUDE.md content arrives wrapped in <system-reminder> before the user text."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("Always use the venv at ~/venv.")

    provider = CapturingProvider()
    await _make_agent(provider, skills_dir).invoke("Hello")

    first = _first_human(provider.captured_messages[-1])
    assert first.content.startswith("<system-reminder>\n")
    assert "Always use the venv at ~/venv." in first.content
    assert first.content.index("</system-reminder>") < first.content.index("Hello")
    # It rides in the message, not in the system prompt.
    assert "Always use the venv" not in (provider.captured_prompts[-1] or "")


@pytest.mark.asyncio
async def test_agents_md_is_the_fallback(tmp_path, monkeypatch, skills_dir):
    """AGENTS.md is picked up when no CLAUDE.md exists."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text("Run tests with make check.")

    provider = CapturingProvider()
    await _make_agent(provider, skills_dir).invoke("Hello")

    first = _first_human(provider.captured_messages[-1])
    assert "Run tests with make check." in first.content
    assert str(tmp_path / "AGENTS.md") in first.content


@pytest.mark.asyncio
async def test_claude_md_wins_over_agents_md(tmp_path, monkeypatch, skills_dir):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("claude instructions")
    (tmp_path / "AGENTS.md").write_text("agents instructions")

    provider = CapturingProvider()
    await _make_agent(provider, skills_dir).invoke("Hello")

    first = _first_human(provider.captured_messages[-1])
    assert "claude instructions" in first.content
    assert "agents instructions" not in first.content


@pytest.mark.asyncio
async def test_empty_claude_md_falls_back_to_agents_md(tmp_path, monkeypatch, skills_dir):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("   \n\t ")
    (tmp_path / "AGENTS.md").write_text("agents instructions")

    provider = CapturingProvider()
    await _make_agent(provider, skills_dir).invoke("Hello")

    first = _first_human(provider.captured_messages[-1])
    assert "agents instructions" in first.content


@pytest.mark.asyncio
async def test_no_instruction_files_leaves_message_untouched(tmp_path, monkeypatch, skills_dir):
    monkeypatch.chdir(tmp_path)

    provider = CapturingProvider()
    await _make_agent(provider, skills_dir).invoke("Hello")

    first = _first_human(provider.captured_messages[-1])
    assert first.content == "Hello"


@pytest.mark.asyncio
async def test_reminder_never_checkpointed(tmp_path, monkeypatch, skills_dir):
    """The stored history keeps the user's original message text."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("project rules")

    provider = CapturingProvider()
    result = await _make_agent(provider, skills_dir).invoke("Hello", thread_id="t")

    human = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    assert all("<system-reminder>" not in str(m.content) for m in human)
    assert human[0].content == "Hello"


@pytest.mark.asyncio
async def test_injection_stable_across_turns(tmp_path, monkeypatch, skills_dir):
    """Only the first user message carries the reminder, identically each turn."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("project rules")

    provider = CapturingProvider()
    agent = _make_agent(provider, skills_dir)
    await agent.invoke("first", thread_id="t")
    await agent.invoke("second", thread_id="t")

    turn1, turn2 = provider.captured_messages
    assert turn1[0].content == turn2[0].content  # stable cache prefix
    with_reminder = [
        m for m in turn2
        if isinstance(m, HumanMessage) and "<system-reminder>" in str(m.content)
    ]
    assert len(with_reminder) == 1
    assert with_reminder[0] is turn2[0]
