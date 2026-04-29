"""
Integration tests for the dynamic skill system.
"""

import pytest
import tempfile
import os
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
    """Tools are filtered to only those allowed for the current user/session."""

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
        agent.enable_skill_for_user("user123", "PrivateSkill")
        await agent.invoke("Test message", thread_id="user123")
        assert len(llm.bound_tools) == 2
        tool_names = {t.name for t in llm.bound_tools}
        assert "publicskill_tool" in tool_names
        assert "privateskill_tool" in tool_names
