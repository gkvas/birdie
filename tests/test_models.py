"""
Unit tests for core models.
"""

import pytest
from pydantic import ValidationError
from birdie.core.models import Skill, SkillTool


def test_skill_tool_validation():
    """Test SkillTool model validation."""
    # Valid tool
    tool = SkillTool(
        name="test_tool",
        description="A test tool",
        entrypoint="bash:echo test",
        schema={"type": "object", "properties": {"param": {"type": "string"}}}
    )
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    
    # Missing required field
    with pytest.raises(ValidationError):
        SkillTool(
            name="test_tool",
            description="A test tool",
            entrypoint="bash:echo test"
            # Missing schema
        )


def test_skill_validation():
    """Test Skill model validation."""
    # Valid skill
    tool = SkillTool(
        name="test_tool",
        description="A test tool",
        entrypoint="bash:echo test",
        schema={"type": "object", "properties": {"param": {"type": "string"}}}
    )
    
    skill = Skill(
        name="TestSkill",
        version="1.0.0",
        description="A test skill",
        tools=[tool]
    )
    assert skill.name == "TestSkill"
    assert skill.version == "1.0.0"
    assert len(skill.tools) == 1
    
    # Missing required field
    with pytest.raises(ValidationError):
        Skill(
            name="TestSkill",
            version="1.0.0",
            # Missing description
            tools=[tool]
        )