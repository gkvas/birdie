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
    """Global defaults reflect enabled_by_default flags."""
    policy = SkillPolicy()
    policy.set_default_skills(sample_skills)

    allowed = policy.get_allowed_skills()
    assert "Weather" in allowed
    assert "Filesystem" not in allowed


def test_session_seeded_from_defaults(sample_skills):
    """A new session starts with the global defaults."""
    policy = SkillPolicy()
    policy.set_default_skills(sample_skills)

    allowed = policy.get_allowed_skills("new-session")
    assert "Weather" in allowed
    assert "Filesystem" not in allowed


def test_enable_skill(sample_skills):
    """enable_skill adds a skill to the session set."""
    policy = SkillPolicy()
    policy.set_default_skills(sample_skills)

    policy.enable_skill("s1", "Filesystem")
    allowed = policy.get_allowed_skills("s1")
    assert "Weather" in allowed
    assert "Filesystem" in allowed


def test_disable_skill(sample_skills):
    """disable_skill removes a skill from the session set."""
    policy = SkillPolicy()
    policy.set_default_skills(sample_skills)

    policy.disable_skill("s1", "Weather")
    allowed = policy.get_allowed_skills("s1")
    assert "Weather" not in allowed
    assert "Filesystem" not in allowed


def test_enable_skills_for_session_replaces_defaults(sample_skills):
    """enable_skills_for_session sets an exact list, ignoring defaults."""
    policy = SkillPolicy()
    policy.set_default_skills(sample_skills)

    policy.enable_skills_for_session("s1", ["Filesystem"])
    allowed = policy.get_allowed_skills("s1")
    assert "Filesystem" in allowed
    assert "Weather" not in allowed  # default not included in explicit list


def test_sessions_are_independent(sample_skills):
    """Mutations to one session do not affect another."""
    policy = SkillPolicy()
    policy.set_default_skills(sample_skills)

    policy.enable_skill("s1", "Filesystem")
    policy.disable_skill("s2", "Weather")

    assert "Filesystem" in policy.get_allowed_skills("s1")
    assert "Filesystem" not in policy.get_allowed_skills("s2")
    assert "Weather" in policy.get_allowed_skills("s1")
    assert "Weather" not in policy.get_allowed_skills("s2")
