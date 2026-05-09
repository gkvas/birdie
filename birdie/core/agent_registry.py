"""
AgentRegistry: stores AgentDef objects and their pre-built async tools,
with built-in per-session enable/disable logic.

Completely separate from SkillRegistry and SkillPolicy.  Skills and agents
only meet in graph.py when assembling the combined tool list for the LLM.
"""

from typing import Dict, List, Optional, Set

from langchain_core.tools import StructuredTool

from .models import AgentDef


class AgentRegistry:
    """Holds all discovered AgentDefs and their executable StructuredTools.

    Session policy mirrors SkillPolicy's single-set model: each session starts
    with the global defaults and can be mutated via enable/disable.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, AgentDef] = {}
        self._tools: Dict[str, StructuredTool] = {}
        self._default_agents: Set[str] = set()
        self._session_agents: Dict[str, Set[str]] = {}

    # -- registration ----------------------------------------------------------

    def register(self, agent_def: AgentDef, tool: StructuredTool) -> None:
        """Register an agent and its pre-built async tool."""
        self._agents[agent_def.name] = agent_def
        self._tools[agent_def.name] = tool
        if agent_def.enabled_by_default:
            self._default_agents.add(agent_def.name)

    def list_agents(self) -> List[AgentDef]:
        return list(self._agents.values())

    def get_agent(self, name: str) -> Optional[AgentDef]:
        return self._agents.get(name)

    # -- session policy --------------------------------------------------------

    def _session(self, session_id: str) -> Set[str]:
        if session_id not in self._session_agents:
            self._session_agents[session_id] = set(self._default_agents)
        return self._session_agents[session_id]

    def enable_agent(self, session_id: str, name: str) -> None:
        """Add an agent to a session's allowed set."""
        self._session(session_id).add(name)

    def disable_agent(self, session_id: str, name: str) -> None:
        """Remove an agent from a session's allowed set."""
        self._session(session_id).discard(name)

    def enable_agents_for_session(self, session_id: str, agent_names: List[str]) -> None:
        """Replace a session's allowed set with an explicit list."""
        self._session_agents[session_id] = set(agent_names)

    def get_allowed_agents(self, session_id: Optional[str] = None) -> Set[str]:
        """Return allowed agent names for a session (defaults if no session)."""
        if not session_id:
            return set(self._default_agents)
        return set(self._session(session_id))

    # -- tool access (intersection point with graph.py) ------------------------

    def get_tools(self, allowed: Set[str]) -> List[StructuredTool]:
        """Return pre-built tools whose agent names are in the allowed set."""
        return [self._tools[name] for name in allowed if name in self._tools]
