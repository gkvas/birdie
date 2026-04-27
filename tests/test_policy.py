"""
Unit tests for the UserSkillPolicy.
"""

import pytest
from birdie.core.policy import UserSkillPolicy
from birdie.core.models import Skill, SkillTool


@pytest.fixture
def sample_skills():
    """Create sample skills for testing."""
    tool1 = SkillTool(
        name="weather_tool",
        description="Weather tool",
        entrypoint="http:get weather.com",
        schema={"type": "object", "properties": {"city": {"type": "string"}}}
    )
    tool2 = SkillTool(
        name="fs_tool",
        description="Filesystem tool",
        entrypoint="bash:ls",
        schema={"type": "object", "properties": {"path": {"type": "string"}}}
    )
    
    weather_skill = Skill(
        name="Weather",
        version="1.0.0",
        description="Weather skill",
        tools=[tool1],
        enabled_by_default=True
    )
    
    fs_skill = Skill(
        name="Filesystem",
        version="1.0.0",
        description="Filesystem skill",
        tools=[tool2],
        enabled_by_default=False
    )
    
    return [weather_skill, fs_skill]


def test_default_skills(sample_skills):
    """Test default skill settings."""
    policy = UserSkillPolicy()
    policy.set_default_skills(sample_skills)
    
    # Weather is enabled by default, Filesystem is not
    allowed = policy.get_allowed_skills()
    assert "Weather" in allowed
    assert "Filesystem" not in allowed


def test_user_specific_skills(sample_skills):
    """Test user-specific skill settings."""
    policy = UserSkillPolicy()
    policy.set_default_skills(sample_skills)
    
    # Enable Filesystem for user123
    policy.enable_skill_for_user("user123", "Filesystem")
    
    allowed = policy.get_allowed_skills(user_id="user123")
    assert "Weather" in allowed  # Default skill
    assert "Filesystem" in allowed  # User-specific skill
    
    # Disable Weather for user123
    policy.disable_skill_for_user("user123", "Weather")
    
    allowed = policy.get_allowed_skills(user_id="user123")
    assert "Weather" not in allowed  # Disabled for user
    assert "Filesystem" in allowed  # Still enabled


def test_session_specific_skills(sample_skills):
    """Test session-specific skill settings."""
    policy = UserSkillPolicy()
    policy.set_default_skills(sample_skills)
    
    # Enable only Filesystem for session456
    policy.enable_skills_for_session("session456", ["Filesystem"])
    
    allowed = policy.get_allowed_skills(session_id="session456")
    assert "Weather" not in allowed  # Not in session skills
    assert "Filesystem" in allowed  # Session-specific skill
    
    # Test combined user and session
    policy.enable_skill_for_user("user123", "Weather")
    allowed = policy.get_allowed_skills(user_id="user123", session_id="session456")
    assert "Weather" in allowed  # User skill
    assert "Filesystem" in allowed  # Session skill