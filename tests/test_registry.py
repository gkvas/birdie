"""
Unit tests for the SkillRegistry.
"""

import pytest
from birdie.core.registry import SkillRegistry
from birdie.core.models import Skill, SkillTool


@pytest.fixture
def sample_skill():
    """Create a sample skill for testing."""
    tool1 = SkillTool(
        name="test_tool1",
        description="Test tool 1",
        entrypoint="bash:echo test1",
        schema={"type": "object", "properties": {"param": {"type": "string"}}}
    )
    tool2 = SkillTool(
        name="test_tool2",
        description="Test tool 2",
        entrypoint="bash:echo test2",
        schema={"type": "object", "properties": {"param": {"type": "string"}}}
    )
    
    return Skill(
        name="TestSkill",
        version="1.0.0",
        description="A test skill",
        tools=[tool1, tool2],
        tags=["test"]
    )


def test_register_skill(sample_skill):
    """Test registering a skill."""
    registry = SkillRegistry()
    registry.register_skill(sample_skill)
    
    assert len(registry.list_skills()) == 1
    assert registry.get_skill("TestSkill") == sample_skill
    assert len(registry.list_tools()) == 2
    assert registry.get_tool("test_tool1") is not None
    assert registry.get_tool("test_tool2") is not None


def test_unregister_skill(sample_skill):
    """Test unregistering a skill."""
    registry = SkillRegistry()
    registry.register_skill(sample_skill)
    registry.unregister_skill("TestSkill")
    
    assert len(registry.list_skills()) == 0
    assert len(registry.list_tools()) == 0
    assert registry.get_skill("TestSkill") is None


def test_list_tools_with_filters(sample_skill):
    """Test listing tools with filters."""
    registry = SkillRegistry()
    registry.register_skill(sample_skill)
    
    # Test tag filtering
    tools = registry.list_tools(tags=["test"])
    assert len(tools) == 2
    
    # Test skill name filtering
    tools = registry.list_tools(skill_names=["TestSkill"])
    assert len(tools) == 2
    
    # Test combined filtering
    tools = registry.list_tools(tags=["test"], skill_names=["TestSkill"])
    assert len(tools) == 2
    
    # Test empty filtering
    tools = registry.list_tools(tags=["nonexistent"])
    assert len(tools) == 0

    # Empty skill_names list must return zero tools, not all tools.
    # All built-in skills have enabled_by_default=False, so allowed is often
    # an empty set; list_tools(skill_names=[]) must honour that constraint.
    tools = registry.list_tools(skill_names=[])
    assert len(tools) == 0