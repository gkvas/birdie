"""
Per-session skill access control.

``SkillPolicy`` is the single authoritative source for which skills a given
session may access on any turn.  The agent consults it before building the
system prompt and before executing tool calls.
"""

from typing import Dict, Set, Optional, List
from .models import Skill


class SkillPolicy:
    """Enforces which skills are available to which sessions.

    Resolution order (highest priority first):

    1. Per-session explicit disable - always blocks the skill.
    2. Per-session explicit enable - overrides defaults.
    3. Global defaults - skills whose ``enabled_by_default`` flag is ``True``.
    4. Fixed session grant (``set_session_skills``) - unioned with the above.
    """

    def __init__(self) -> None:
        self._session_enabled: Dict[str, Set[str]] = {}
        self._session_disabled: Dict[str, Set[str]] = {}
        self._session_fixed: Dict[str, Set[str]] = {}
        self._default_skills: Set[str] = set()

    def set_default_skills(self, skills: List[Skill]) -> None:
        """Seed the global default set from a list of loaded Skill objects.

        Only skills whose ``enabled_by_default`` field is ``True`` are added.
        Called once at startup after skill discovery; safe to call again to
        reset defaults (e.g. after hot-reloading skills).

        Args:
            skills: All discovered skills; those with ``enabled_by_default=True``
                become the baseline for sessions with no explicit grants.
        """
        self._default_skills = {s.name for s in skills if s.enabled_by_default}

    def enable_skill(self, session_id: str, skill_name: str) -> None:
        """Grant a skill for a session, overriding any previous disable.

        Args:
            session_id: Session identifier (matches the ``thread_id`` passed to invoke).
            skill_name: Exact skill name as declared in its SKILL.MD frontmatter.
        """
        self._session_enabled.setdefault(session_id, set()).add(skill_name)
        self._session_disabled.get(session_id, set()).discard(skill_name)

    def disable_skill(self, session_id: str, skill_name: str) -> None:
        """Block a skill for a session, even if it is a global default.

        Args:
            session_id: Session identifier.
            skill_name: Exact skill name to block.
        """
        self._session_disabled.setdefault(session_id, set()).add(skill_name)
        self._session_enabled.get(session_id, set()).discard(skill_name)

    def enable_skills_for_session(self, session_id: str, skill_names: List[str]) -> None:
        """Grant a fixed set of skills for the duration of a session.

        Fixed grants are additive with per-session enables and are not affected
        by per-session disables.

        Args:
            session_id: Session identifier.
            skill_names: Skills to enable for this session (replaces any prior fixed grant).
        """
        self._session_fixed[session_id] = set(skill_names)

    def get_allowed_skills_for_session(self, session_id: str) -> Set[str]:
        """Return the full set of skill names allowed for a session.

        Merges per-session incremental grants/blocks with the global defaults
        and any fixed grant set via ``enable_skills_for_session``.

        Args:
            session_id: Session identifier.

        Returns:
            Set of skill names the session is permitted to use.
        """
        enabled = self._session_enabled.get(session_id, set()) | self._default_skills
        disabled = self._session_disabled.get(session_id, set())
        base = enabled - disabled
        fixed = self._session_fixed.get(session_id, set())
        return base | fixed

    def get_allowed_skills(self, session_id: Optional[str] = None) -> Set[str]:
        """Return the full set of skill names allowed for a session.

        This is the method called by the agent on every turn.

        Args:
            session_id: Session identifier.  ``None`` means no session context -
                only global defaults apply.

        Returns:
            Set of skill names the session is permitted to use this turn.
        """
        if not session_id:
            return set(self._default_skills)
        return self.get_allowed_skills_for_session(session_id)
