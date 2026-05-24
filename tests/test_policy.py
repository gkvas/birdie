"""
Unit tests for SkillPolicy.
"""

import pytest
from birdie.core.policy import SkillPolicy


def test_default_skills():
    """Global defaults reflect the configured skill names."""
    policy = SkillPolicy()
    policy.set_default_skills(["Weather"])

    allowed = policy.get_allowed_skills()
    assert "Weather" in allowed
    assert "Filesystem" not in allowed


def test_session_seeded_from_defaults():
    """A new session starts with the global defaults."""
    policy = SkillPolicy()
    policy.set_default_skills(["Weather"])

    allowed = policy.get_allowed_skills("new-session")
    assert "Weather" in allowed
    assert "Filesystem" not in allowed


def test_enable_skill():
    """enable_skill adds a skill to the session set."""
    policy = SkillPolicy()
    policy.set_default_skills(["Weather"])

    policy.enable_skill("s1", "Filesystem")
    allowed = policy.get_allowed_skills("s1")
    assert "Weather" in allowed
    assert "Filesystem" in allowed


def test_disable_skill():
    """disable_skill removes a skill from the session set."""
    policy = SkillPolicy()
    policy.set_default_skills(["Weather"])

    policy.disable_skill("s1", "Weather")
    allowed = policy.get_allowed_skills("s1")
    assert "Weather" not in allowed
    assert "Filesystem" not in allowed


def test_enable_skills_for_session_replaces_defaults():
    """enable_skills_for_session sets an exact list, ignoring defaults."""
    policy = SkillPolicy()
    policy.set_default_skills(["Weather"])

    policy.enable_skills_for_session("s1", ["Filesystem"])
    allowed = policy.get_allowed_skills("s1")
    assert "Filesystem" in allowed
    assert "Weather" not in allowed  # default not included in explicit list


def test_sessions_are_independent():
    """Mutations to one session do not affect another."""
    policy = SkillPolicy()
    policy.set_default_skills(["Weather"])

    policy.enable_skill("s1", "Filesystem")
    policy.disable_skill("s2", "Weather")

    assert "Filesystem" in policy.get_allowed_skills("s1")
    assert "Filesystem" not in policy.get_allowed_skills("s2")
    assert "Weather" in policy.get_allowed_skills("s1")
    assert "Weather" not in policy.get_allowed_skills("s2")
