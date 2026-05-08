"""
Integration tests for the dynamic skill system.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from birdie.agent.run import DynamicAgent
from birdie.core.models import Skill, SkillTool


@pytest.fixture
def temp_skills_dir():
    """Create a temporary skills directory with a test skill."""
    with tempfile.TemporaryDirectory() as temp_dir:
        skill_dir = os.path.join(temp_dir, "TestSkill")
        os.makedirs(skill_dir)

        skill_content = """---
name: TestSkill
version: 1.0.0
description: A test skill
tags: [test]
enabled_by_default: true
---

# Skill: TestSkill

## Tools

### echo
description: Echo a message
entrypoint: python:tests.test_integration.echo_tool
schema:
  type: object
  properties:
    message:
      type: string
  required: [message]
"""
        with open(os.path.join(skill_dir, "SKILL.MD"), "w") as f:
            f.write(skill_content)

        yield temp_dir


def echo_tool(message: str) -> str:
    return f"Echo: {message}"


@pytest.mark.asyncio
async def test_agent_integration(temp_skills_dir):
    """Full round-trip: user message → tool call → tool result → final answer."""

    call_count = 0

    class MockLLM:
        def bind_tools(self, tools):
            self.tools = tools
            return self

        async def ainvoke(self, messages, config=None):
            nonlocal call_count
            from langchain_core.messages import AIMessage, ToolCall

            call_count += 1
            if call_count == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="echo", args={"message": "test message"}, id="test_id")
                    ],
                )
            return AIMessage(content="Done")

    agent = DynamicAgent(MockLLM(), skills_dir=temp_skills_dir)
    result = await agent.invoke("Test message")

    # Expect: HumanMessage, AIMessage(tool_call), ToolMessage, AIMessage("Done")
    assert len(result["messages"]) == 4

    ai_with_tools = [
        m for m in result["messages"]
        if getattr(m, "tool_calls", None)
    ]
    assert len(ai_with_tools) == 1
    assert ai_with_tools[0].tool_calls[0]["name"] == "echo"


@pytest.mark.asyncio
async def test_dynamic_tool_binding():
    """Tools are filtered to only those allowed for the current session."""

    class MockLLM:
        def __init__(self):
            self.bound_tools = []

        def bind_tools(self, tools):
            self.bound_tools = list(tools)
            return self

        async def ainvoke(self, messages, config=None):
            from langchain_core.messages import AIMessage
            return AIMessage(content="ok")

    with tempfile.TemporaryDirectory() as temp_dir:
        for skill_name, enabled in [("PublicSkill", True), ("PrivateSkill", False)]:
            skill_dir = os.path.join(temp_dir, skill_name)
            os.makedirs(skill_dir)

            skill_content = f"""---
name: {skill_name}
version: 1.0.0
description: A {skill_name.lower()} skill
tags: [test]
enabled_by_default: {str(enabled).lower()}
---

# Skill: {skill_name}

## Tools

### {skill_name.lower()}_tool
description: A test tool
entrypoint: python:tests.test_integration.echo_tool
schema:
  type: object
  properties:
    message:
      type: string
  required: [message]
"""
            with open(os.path.join(skill_dir, "SKILL.MD"), "w") as f:
                f.write(skill_content)

        llm = MockLLM()
        agent = DynamicAgent(llm, skills_dir=temp_dir)

        # No user — only PublicSkill (enabled_by_default=true)
        await agent.invoke("Test message")
        assert len(llm.bound_tools) == 1
        assert llm.bound_tools[0].name == "publicskill_tool"

        # Enable PrivateSkill for thread user123 — both should be visible
        agent.enable_skill("user123", "PrivateSkill")
        await agent.invoke("Test message", thread_id="user123")
        assert len(llm.bound_tools) == 2
        tool_names = {t.name for t in llm.bound_tools}
        assert "publicskill_tool" in tool_names
        assert "privateskill_tool" in tool_names


def _write_skill(directory: str, name: str, enabled_by_default: bool = False) -> None:
    skill_dir = os.path.join(directory, name)
    os.makedirs(skill_dir, exist_ok=True)
    with open(os.path.join(skill_dir, "SKILL.MD"), "w") as f:
        f.write(f"""---
name: {name}
version: 1.0.0
description: Test skill {name}
enabled_by_default: {str(enabled_by_default).lower()}
---
""")


class _NoopLLM:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, config=None):
        from langchain_core.messages import AIMessage
        return AIMessage(content="ok")


def test_load_skills_registers_all_skills():
    """Skills from the primary dir are registered and policy is seeded."""
    with tempfile.TemporaryDirectory() as skills_dir:
        _write_skill(skills_dir, "SkillA", enabled_by_default=True)
        _write_skill(skills_dir, "SkillB", enabled_by_default=False)

        agent = DynamicAgent(_NoopLLM(), skills_dir=skills_dir)

        names = {s.name for s in agent.registry.list_skills()}
        assert "SkillA" in names
        assert "SkillB" in names

        # Policy seeded: SkillA enabled by default, SkillB not
        allowed = agent.policy.get_allowed_skills_for_session("default")
        assert "SkillA" in allowed
        assert "SkillB" not in allowed


def test_load_skills_merges_user_skills_dir():
    """Skills in ~/.birdie/skills/ are loaded on top of primary dir skills."""
    with tempfile.TemporaryDirectory() as primary_dir, \
         tempfile.TemporaryDirectory() as fake_home:

        _write_skill(primary_dir, "BundledSkill")
        user_skills_dir = Path(fake_home) / ".birdie" / "skills"
        user_skills_dir.mkdir(parents=True)
        _write_skill(str(user_skills_dir), "UserSkill")

        with patch("birdie.agent.run.Path") as mock_path:
            # Make Path.home() return our fake home; leave Path(...) calls intact
            real_path = Path
            mock_path.home.return_value = real_path(fake_home)
            mock_path.side_effect = real_path

            agent = DynamicAgent(_NoopLLM(), skills_dir=primary_dir)

        names = {s.name for s in agent.registry.list_skills()}
        assert "BundledSkill" in names
        assert "UserSkill" in names


def test_load_skills_no_user_dir_does_not_fail():
    """Absence of ~/.birdie/skills/ is handled silently."""
    with tempfile.TemporaryDirectory() as primary_dir, \
         tempfile.TemporaryDirectory() as fake_home:

        _write_skill(primary_dir, "BundledSkill")

        with patch("birdie.agent.run.Path") as mock_path:
            real_path = Path
            mock_path.home.return_value = real_path(fake_home)  # no .birdie/skills subdir
            mock_path.side_effect = real_path

            agent = DynamicAgent(_NoopLLM(), skills_dir=primary_dir)

        names = {s.name for s in agent.registry.list_skills()}
        assert "BundledSkill" in names
