"""
Unit tests for SkillPolicy.
"""

import pytest
from birdie.core.policy import SkillPolicy
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
    policy = SkillPolicy()
    policy.set_default_skills(sample_skills)

    # Weather is enabled by default, Filesystem is not
    allowed = policy.get_allowed_skills()
    assert "Weather" in allowed
    assert "Filesystem" not in allowed


def test_session_specific_skills(sample_skills):
    """Test per-session incremental skill grants."""
    policy = SkillPolicy()
    policy.set_default_skills(sample_skills)

    # Enable Filesystem for session123
    policy.enable_skill("session123", "Filesystem")

    allowed = policy.get_allowed_skills(session_id="session123")
    assert "Weather" in allowed  # Default skill
    assert "Filesystem" in allowed  # Session-specific skill

    # Disable Weather for session123
    policy.disable_skill("session123", "Weather")

    allowed = policy.get_allowed_skills(session_id="session123")
    assert "Weather" not in allowed  # Disabled for session
    assert "Filesystem" in allowed  # Still enabled


def test_fixed_session_skills(sample_skills):
    """Test fixed-set session grants via enable_skills_for_session."""
    policy = SkillPolicy()
    policy.set_default_skills(sample_skills)

    # Grant a fixed set to session456
    policy.enable_skills_for_session("session456", ["Filesystem"])

    allowed = policy.get_allowed_skills(session_id="session456")
    # Weather is a default, Filesystem is added via fixed grant
    assert "Weather" in allowed
    assert "Filesystem" in allowed

    # Test combined incremental + fixed
    policy.enable_skill("session123", "Weather")
    allowed = policy.get_allowed_skills(session_id="session123")
    assert "Weather" in allowed  # incremental enable
