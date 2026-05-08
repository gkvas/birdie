"""
Per-session skill access control.

All skills are disabled by default. Skills marked ``enabled_by_default=True``
seed the global default set, which is copied into each session on first use.
Interactive enable/disable mutates the session's live set directly.
"""

from typing import Dict, Set, Optional, List
from .models import Skill


class SkillPolicy:
    """Tracks which skills each session may access.

    A session's skill set is initialized from the global defaults on first
    access, then modified interactively via ``enable_skill`` / ``disable_skill``.
    """

    def __init__(self) -> None:
        self._default_skills: Set[str] = set()
        self._session_skills: Dict[str, Set[str]] = {}

    def set_default_skills(self, skills: List[Skill]) -> None:
        """Seed the default set from all skills with ``enabled_by_default=True``.

        Safe to call again to reset defaults (e.g. after hot-reloading skills).
        """
        self._default_skills = {s.name for s in skills if s.enabled_by_default}

    def enable_skill(self, session_id: str, skill_name: str) -> None:
        """Add a skill to a session's allowed set."""
        self._session(session_id).add(skill_name)

    def disable_skill(self, session_id: str, skill_name: str) -> None:
        """Remove a skill from a session's allowed set."""
        self._session(session_id).discard(skill_name)

    def enable_skills_for_session(self, session_id: str, skill_names: List[str]) -> None:
        """Set a session's allowed skills to an explicit list, replacing defaults."""
        self._session_skills[session_id] = set(skill_names)

    def get_allowed_skills(self, session_id: Optional[str] = None) -> Set[str]:
        """Return the allowed skill set for a session.

        Returns the global defaults when no session is specified.
        """
        if not session_id:
            return set(self._default_skills)
        return set(self._session(session_id))

    def _session(self, session_id: str) -> Set[str]:
        if session_id not in self._session_skills:
            self._session_skills[session_id] = set(self._default_skills)
        return self._session_skills[session_id]
