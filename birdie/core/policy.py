"""
Per-user and per-session skill access control.

``UserSkillPolicy`` is the single authoritative source for which skills a
given user or session may access on any turn.  The agent consults it before
building the system prompt and before executing tool calls.
"""

from typing import Dict, Set, Optional, List
from .models import Skill


class UserSkillPolicy:
    """Enforces which skills are available to which users and sessions.

    Resolution order (highest priority first):

    1. Per-user explicit disable - always blocks the skill.
    2. Per-user explicit enable ∪ session enable - union of both sets.
    3. Global defaults - skills whose ``enabled_by_default`` flag is ``True``.
    """

    def __init__(self) -> None:
        self._user_enabled: Dict[str, Set[str]] = {}
        self._user_disabled: Dict[str, Set[str]] = {}
        self._session_policies: Dict[str, Set[str]] = {}
        self._default_skills: Set[str] = set()

    def set_default_skills(self, skills: List[Skill]) -> None:
        """Seed the global default set from a list of loaded Skill objects.

        Only skills whose ``enabled_by_default`` field is ``True`` are added.
        Called once at startup after skill discovery; safe to call again to
        reset defaults (e.g. after hot-reloading skills).

        Args:
            skills: All discovered skills; those with ``enabled_by_default=True``
                become the baseline for users with no explicit grants.
        """
        self._default_skills = {s.name for s in skills if s.enabled_by_default}

    def enable_skill_for_user(self, user_id: str, skill_name: str) -> None:
        """Grant a skill to a specific user, overriding any previous disable.

        Args:
            user_id: Opaque user identifier (matches the ``user_id`` in AgentState).
            skill_name: Exact skill name as declared in its SKILL.MD frontmatter.
        """
        self._user_enabled.setdefault(user_id, set()).add(skill_name)
        self._user_disabled.get(user_id, set()).discard(skill_name)

    def disable_skill_for_user(self, user_id: str, skill_name: str) -> None:
        """Block a skill for a specific user, even if it is a global default.

        Args:
            user_id: Opaque user identifier.
            skill_name: Exact skill name to block.
        """
        self._user_disabled.setdefault(user_id, set()).add(skill_name)
        self._user_enabled.get(user_id, set()).discard(skill_name)

    def get_allowed_skills_for_user(self, user_id: str) -> Set[str]:
        """Return the set of skill names allowed for a user (ignoring session grants).

        Args:
            user_id: Opaque user identifier.

        Returns:
            Union of the user's explicit enables and global defaults, minus any
            explicit disables.
        """
        enabled = self._user_enabled.get(user_id, set()) | self._default_skills
        disabled = self._user_disabled.get(user_id, set())
        return enabled - disabled

    def enable_skills_for_session(self, session_id: str, skill_names: List[str]) -> None:
        """Grant a fixed set of skills for the duration of a session.

        Session grants are additive with user grants when both are provided to
        ``get_allowed_skills``.  They are not affected by user-level disables.

        Args:
            session_id: Opaque session identifier (matches ``session_id`` in AgentState).
            skill_names: Skills to enable for this session (replaces any prior session grant).
        """
        self._session_policies[session_id] = set(skill_names)

    def get_allowed_skills_for_session(self, session_id: str) -> Set[str]:
        """Return the set of skill names granted to a session.

        Args:
            session_id: Opaque session identifier.

        Returns:
            The session's skill grant set, or an empty set if none was registered.
        """
        return self._session_policies.get(session_id, set())

    def get_allowed_skills(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Set[str]:
        """Return the full set of skill names allowed for a user/session combination.

        This is the method called by the agent on every turn.  It applies the
        priority rules described in the class docstring.

        Args:
            user_id: Optional user identifier.  ``None`` means anonymous - only
                global defaults apply.
            session_id: Optional session identifier.  Session grants are unioned
                with user grants when both are provided.

        Returns:
            Set of skill names the user/session is permitted to use this turn.
        """
        if not user_id and not session_id:
            return set(self._default_skills)

        allowed: Set[str] = set()
        if user_id:
            allowed.update(self.get_allowed_skills_for_user(user_id))
        if session_id:
            allowed.update(self.get_allowed_skills_for_session(session_id))
        return allowed
