"""Tests for the configurable system prompt via .birdie/system_prompt.md."""

import pytest
from pathlib import Path
from unittest.mock import patch
from langchain_core.messages import AIMessage, BaseMessage

from birdie.agent.run import DynamicAgent
from birdie.core.llm_provider import LLMProvider


class CapturingProvider(LLMProvider):
    """Records every system_prompt value passed to achat()."""

    def __init__(self):
        self.captured: list[str | None] = []

    @property
    def last(self) -> str | None:
        return self.captured[-1] if self.captured else None

    def chat(self, messages, tools=None, system_prompt=None, **kwargs) -> BaseMessage:
        self.captured.append(system_prompt)
        return AIMessage(content="ok")

    async def achat(self, messages, tools=None, system_prompt=None, **kwargs) -> BaseMessage:
        self.captured.append(system_prompt)
        return AIMessage(content="ok")

    def stream_chat(self, messages, tools=None, system_prompt=None, **kwargs):
        self.captured.append(system_prompt)
        yield AIMessage(content="ok")

    async def astream_chat(self, messages, tools=None, system_prompt=None, **kwargs):
        self.captured.append(system_prompt)
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


def _make_agent_no_user_skills(tmp_path, provider, skills_dir, skills_enabled=None):
    """Create DynamicAgent with Path.home() pointing to an empty fake home."""
    real_path = Path
    with patch("birdie.agent.run.Path") as mock_path:
        mock_path.home.return_value = real_path(tmp_path)
        mock_path.side_effect = real_path
        return DynamicAgent(provider, skills_dir=str(skills_dir), skills_enabled=skills_enabled or [])


# ---------------------------------------------------------------------------
# Tests: with skills present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_custom_prompt_prepended_before_skills(tmp_path, monkeypatch, skills_dir):
    """Custom prompt content appears before the skills catalog in the system prompt."""
    monkeypatch.chdir(tmp_path)
    birdie_dir = tmp_path / ".birdie"
    birdie_dir.mkdir()
    (birdie_dir / "system_prompt.md").write_text("You are a helpful assistant.")

    provider = CapturingProvider()
    agent = DynamicAgent(provider, skills_dir=str(skills_dir), skills_enabled=["TestSkill"])
    await agent.invoke("Hello")

    prompt = provider.last
    assert prompt is not None
    assert "You are a helpful assistant." in prompt
    assert "You have access to the following skills:" in prompt
    assert prompt.index("You are a helpful assistant.") < prompt.index(
        "You have access to the following skills:"
    )


@pytest.mark.asyncio
async def test_no_file_does_not_add_custom_prefix(tmp_path, monkeypatch, skills_dir):
    """Without .birdie/system_prompt.md the system prompt starts with the skills catalog."""
    monkeypatch.chdir(tmp_path)  # tmp_path has no .birdie directory

    provider = CapturingProvider()
    agent = DynamicAgent(provider, skills_dir=str(skills_dir), skills_enabled=["TestSkill"])
    await agent.invoke("Hello")

    prompt = provider.last
    assert prompt is not None
    assert prompt.startswith("You have access to the following skills:")


@pytest.mark.asyncio
async def test_empty_file_is_treated_as_absent(tmp_path, monkeypatch, skills_dir):
    """An empty .birdie/system_prompt.md is ignored; the skills catalog is still returned."""
    monkeypatch.chdir(tmp_path)
    birdie_dir = tmp_path / ".birdie"
    birdie_dir.mkdir()
    (birdie_dir / "system_prompt.md").write_text("")

    provider = CapturingProvider()
    agent = DynamicAgent(provider, skills_dir=str(skills_dir), skills_enabled=["TestSkill"])
    await agent.invoke("Hello")

    prompt = provider.last
    assert prompt is not None
    assert prompt.startswith("You have access to the following skills:")


@pytest.mark.asyncio
async def test_whitespace_only_file_is_treated_as_absent(tmp_path, monkeypatch, skills_dir):
    """A whitespace-only .birdie/system_prompt.md is ignored."""
    monkeypatch.chdir(tmp_path)
    birdie_dir = tmp_path / ".birdie"
    birdie_dir.mkdir()
    (birdie_dir / "system_prompt.md").write_text("   \n\n\t  ")

    provider = CapturingProvider()
    agent = DynamicAgent(provider, skills_dir=str(skills_dir), skills_enabled=["TestSkill"])
    await agent.invoke("Hello")

    prompt = provider.last
    assert prompt is not None
    assert prompt.startswith("You have access to the following skills:")


# ---------------------------------------------------------------------------
# Tests: without skills
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_custom_prompt_alone_when_no_skills(tmp_path, monkeypatch):
    """With no skills registered the custom prompt is returned as the full system prompt."""
    monkeypatch.chdir(tmp_path)
    birdie_dir = tmp_path / ".birdie"
    birdie_dir.mkdir()
    (birdie_dir / "system_prompt.md").write_text("Custom instructions only.")

    no_skills_dir = tmp_path / "empty_skills"
    no_skills_dir.mkdir()

    provider = CapturingProvider()
    agent = _make_agent_no_user_skills(tmp_path, provider, no_skills_dir)
    await agent.invoke("Hello")

    assert provider.last == "Custom instructions only."


@pytest.mark.asyncio
async def test_no_file_no_skills_yields_none(tmp_path, monkeypatch):
    """Without a prompt file and without skills the system prompt is None."""
    monkeypatch.chdir(tmp_path)  # no .birdie directory

    no_skills_dir = tmp_path / "empty_skills"
    no_skills_dir.mkdir()

    provider = CapturingProvider()
    agent = _make_agent_no_user_skills(tmp_path, provider, no_skills_dir)
    await agent.invoke("Hello")

    assert provider.last is None
