"""
In-memory skill and tool registry.

``SkillRegistry`` is the single source of truth for all loaded skills and their
tools.  It maintains three secondary indexes (by name, by tag, and tool-to-skill
ownership) that let the agent resolve tools in O(1) without scanning all skills.
"""

from typing import List, Dict, Optional, Set
from .models import Skill, SkillTool


class SkillRegistry:
    """Index of all loaded skills and their tools.

    Skills are registered once at startup.  The registry never touches disk;
    ``loader.discover_skills_from_directory`` feeds parsed ``Skill`` objects in.

    Indexes maintained
    ------------------
    ``_skills``         name → Skill
    ``_tools``          name → SkillTool
    ``_tags_index``     tag  → set of tool names
    ``_tool_to_skill``  tool name → owning skill name
    """

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self._tools: Dict[str, SkillTool] = {}
        self._tags_index: Dict[str, Set[str]] = {}
        self._tool_to_skill: Dict[str, str] = {}

    def register_skill(self, skill: Skill) -> None:
        """Add a skill and all its tools to the registry.

        Skill-level tags from the frontmatter ``tags`` field are propagated to
        every tool in the skill so tag-based lookup works without per-tool tags.

        Args:
            skill: A fully parsed ``Skill`` object.

        Raises:
            ValueError: If a skill or any of its tools is already registered.
        """
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' already registered")

        self._skills[skill.name] = skill

        for tool in skill.tools:
            if tool.name in self._tools:
                raise ValueError(
                    f"Tool '{tool.name}' already exists in skill "
                    f"'{self._tool_to_skill[tool.name]}'"
                )
            self._tools[tool.name] = tool
            self._tool_to_skill[tool.name] = skill.name

            for tag in tool.tags:
                self._tags_index.setdefault(tag, set()).add(tool.name)

        for tag in skill.tags:
            tag_set = self._tags_index.setdefault(tag, set())
            for tool in skill.tools:
                tag_set.add(tool.name)

    def unregister_skill(self, name: str) -> None:
        """Remove a skill and all its tools from the registry.

        Args:
            name: Exact skill name as used in ``register_skill``.

        Raises:
            ValueError: If the skill is not registered.
        """
        if name not in self._skills:
            raise ValueError(f"Skill '{name}' not found")

        skill = self._skills[name]

        for tool in skill.tools:
            del self._tools[tool.name]
            del self._tool_to_skill[tool.name]

            for tag in tool.tags:
                if tag in self._tags_index:
                    self._tags_index[tag].discard(tool.name)

        for tag in skill.tags:
            if tag in self._tags_index:
                for tool in skill.tools:
                    self._tags_index[tag].discard(tool.name)

        del self._skills[name]

    def list_skills(self) -> List[Skill]:
        """Return all registered skills in insertion order."""
        return list(self._skills.values())

    def get_skill(self, name: str) -> Optional[Skill]:
        """Look up a skill by name.

        Args:
            name: Exact skill name.

        Returns:
            The ``Skill`` object, or ``None`` if not registered.
        """
        return self._skills.get(name)

    def list_tools(
        self,
        tags: Optional[List[str]] = None,
        skill_names: Optional[List[str]] = None,
    ) -> List[SkillTool]:
        """Return tools, optionally filtered by tag and/or owning skill.

        Filters are AND-ed: when both ``tags`` and ``skill_names`` are provided,
        only tools that satisfy both constraints are returned.

        Args:
            tags: If given, only tools carrying at least one of these tags are
                included.  Tags are resolved through the tag index built at
                registration time.
            skill_names: If given, only tools owned by one of these skills are
                included.

        Returns:
            List of matching ``SkillTool`` objects.
        """
        tool_names = set(self._tools.keys())

        if tags:
            tag_filtered: Set[str] = set()
            for tag in tags:
                if tag in self._tags_index:
                    tag_filtered.update(self._tags_index[tag])
            tool_names.intersection_update(tag_filtered)

        if skill_names:
            skill_filtered: Set[str] = set()
            for skill_name in skill_names:
                if skill_name in self._skills:
                    for tool in self._skills[skill_name].tools:
                        skill_filtered.add(tool.name)
            tool_names.intersection_update(skill_filtered)

        return [self._tools[name] for name in tool_names]

    def get_tool(self, name: str) -> Optional[SkillTool]:
        """Look up a tool by name.

        Args:
            name: Exact tool name.

        Returns:
            The ``SkillTool`` object, or ``None`` if not registered.
        """
        return self._tools.get(name)

    def is_tool_allowed(
        self,
        tool_name: str,
        allowed_skill_names: Optional[Set[str]] = None,
    ) -> bool:
        """Check whether a tool is accessible under the given skill allow-set.

        Args:
            tool_name: Exact tool name to check.
            allowed_skill_names: Set of allowed skill names from ``UserSkillPolicy``.
                Pass ``None`` to skip the skill check (tool is allowed if it exists).

        Returns:
            ``True`` if the tool exists and its owning skill is in the allow-set.
        """
        if tool_name not in self._tools:
            return False
        if allowed_skill_names is None:
            return True
        return self._tool_to_skill.get(tool_name) in allowed_skill_names

    def find_skills_by_trigger(
        self,
        text: str,
        allowed_skill_names: Optional[Set[str]] = None,
    ) -> List[Skill]:
        """Return freetext skills whose trigger keywords appear in *text*.

        Only skills without tools (freetext skills) participate - structured
        skills are never matched here.  Matching is case-insensitive substring.

        Args:
            text: The user's message text to search within.
            allowed_skill_names: Optional allow-set; skills outside this set
                are ignored even if their triggers match.

        Returns:
            List of matching freetext ``Skill`` objects.
        """
        lower = text.lower()
        matched = []
        for skill in self._skills.values():
            if allowed_skill_names is not None and skill.name not in allowed_skill_names:
                continue
            if not skill.triggers or skill.tools:
                continue
            if any(trigger.lower() in lower for trigger in skill.triggers):
                matched.append(skill)
        return matched
