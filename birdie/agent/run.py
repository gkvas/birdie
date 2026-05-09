"""
Agent entrypoint and runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.checkpoint.memory import MemorySaver

from ..core.registry import SkillRegistry
from ..core.loader import discover_skills_from_directory
from ..core.policy import SkillPolicy
from ..core.mcp_client import MCPClientManager
from ..core.agent_loader import discover_agents_from_directory
from ..core.agent_registry import AgentRegistry
from ..core.agent_runner import agentdef_to_langchain_tool
from ..core.llm_provider import (
    LLMProvider,
    LangChainProvider,
    ProviderConfig,
    get_llm_provider,
    get_llm_provider_from_json,
    get_llm_provider_from_file,
)
from .graph import create_agent_graph, AgentState


class DynamicAgent:
    """
    Orchestrates the SKILL.MD skill system with a vendor-agnostic LLMProvider.

    Construction options
    --------------------
    1. Pass a pre-built LLMProvider directly::

         provider = AnthropicProvider(model="claude-sonnet-4-6")
         agent = DynamicAgent(provider, skills_dir="birdie/skills")

    2. Pass any LangChain BaseChatModel for backward compatibility::

         from langchain_openai import ChatOpenAI
         agent = DynamicAgent(ChatOpenAI(model="gpt-4o"), skills_dir="...")

    3. Build from a config dict (drives vendor selection via env var or config)::

         agent = DynamicAgent.from_config(
             {"vendor": "mistral", "model": "mistral-large-latest"},
             skills_dir="birdie/skills",
             db_path=Path("~/.birdie/sessions/alice/checkpoints.db"),
         )

    Persistence
    -----------
    Pass a pre-built ``checkpointer`` (e.g. ``AsyncSqliteSaver``) for durable
    message history across restarts.  When ``checkpointer`` is ``None``
    (default), a ``MemorySaver`` is used - state is in-memory only, suitable
    for tests or one-shot scripts.

    The ``thread_id`` passed to ``astream``/``invoke`` acts as the session
    identifier: each unique ``thread_id`` has its own independent message
    history and is also used as the policy key for skill access control.
    """

    def __init__(
        self,
        llm_or_provider: Any,
        skills_dir: str = "skills",
        agents_dir: Optional[str] = None,
        checkpointer=None,
    ) -> None:
        # Accept either a native LLMProvider or any LangChain BaseChatModel
        if isinstance(llm_or_provider, LLMProvider):
            self.provider = llm_or_provider
        else:
            self.provider = LangChainProvider(llm_or_provider)

        self.skills_dir = skills_dir
        self.agents_dir = agents_dir
        self.registry = SkillRegistry()
        self.policy = SkillPolicy()
        self.mcp_manager = MCPClientManager()
        self.agent_registry = AgentRegistry()

        self._load_skills()
        self._load_agents()

        graph = create_agent_graph(
            self.provider, self.registry, self.policy, self.mcp_manager,
            self.agent_registry,
        )
        self.app = graph.compile(checkpointer=checkpointer or MemorySaver())

    @classmethod
    def from_config(
        cls,
        provider_config: "Dict[str, Any] | str | Path | ProviderConfig | None" = None,
        skills_dir: str = "skills",
        agents_dir: Optional[str] = None,
        checkpointer=None,
    ) -> "DynamicAgent":
        """
        Create a DynamicAgent from a provider configuration.

        ``provider_config`` accepts the same types as :func:`get_llm_provider`:
        a ``dict``, a JSON string, a ``Path`` to a ``.json`` file, or a
        ``ProviderConfig`` instance.  Pass ``None`` to rely entirely on
        environment variables.

        ``checkpointer`` is forwarded to ``__init__``.  Pass an
        ``AsyncSqliteSaver`` (or any ``BaseCheckpointSaver``) for durable
        history.  Defaults to ``MemorySaver`` (in-memory).

        Environment variable overrides (checked in priority order)
        ----------------------------------------------------------
        ``LLM_PROVIDER_CONFIG``
            Full JSON config blob - overrides everything else.

        ``LLM_VENDOR``
            Override just the vendor, keeping all other config fields.

        ``LLM_MODEL``
            Override just the model.
        """
        env_full = os.environ.get("LLM_PROVIDER_CONFIG")
        if env_full:
            provider_config = env_full
        else:
            if provider_config is None:
                provider_config = {}
            if isinstance(provider_config, dict):
                provider_config = dict(provider_config)
                if os.environ.get("LLM_VENDOR"):
                    provider_config["vendor"] = os.environ["LLM_VENDOR"]
                if os.environ.get("LLM_MODEL"):
                    provider_config["model"] = os.environ["LLM_MODEL"]

        provider = get_llm_provider(provider_config)
        return cls(provider, skills_dir=skills_dir, agents_dir=agents_dir, checkpointer=checkpointer)

    # -- skill management ---------------------------------------------------

    def _load_skills(self) -> None:
        """Discover SKILL.MD files from all skill dirs and register them."""
        dirs = [self.skills_dir]
        user_skills_dir = Path.home() / ".birdie" / "skills"
        if user_skills_dir.is_dir():
            dirs.append(str(user_skills_dir))

        for d in dirs:
            for skill in discover_skills_from_directory(d):
                self.registry.register_skill(skill)
                if skill.mcp_server is not None:
                    self.mcp_manager.register_server(skill.name, skill.mcp_server)
        self.policy.set_default_skills(self.registry.list_skills())

    def _load_agents(self) -> None:
        """Discover AGENTS.MD files from all agent dirs and register them."""
        dirs = []
        if self.agents_dir:
            dirs.append(self.agents_dir)
        user_agents_dir = Path.home() / ".birdie" / "agents"
        if user_agents_dir.is_dir():
            dirs.append(str(user_agents_dir))

        if not dirs:
            return

        vendor = getattr(self.provider, 'vendor_name', None)
        model = getattr(self.provider, 'model_name', None)

        for d in dirs:
            for agent_def in discover_agents_from_directory(d):
                tool = agentdef_to_langchain_tool(
                    agent_def,
                    skills_dir=self.skills_dir,
                    agents_dir=self.agents_dir,
                    fallback_vendor=vendor,
                    fallback_model=model,
                )
                self.agent_registry.register(agent_def, tool)

    async def shutdown(self) -> None:
        """Release resources - call when the agent is no longer needed."""
        pass  # MCPClientManager uses per-call sessions; nothing to tear down

    def enable_skill(self, session_id: str, skill_name: str) -> None:
        """Grant a skill for a session. Takes effect on the next turn.

        Args:
            session_id: The ``thread_id`` used when invoking the agent.
            skill_name: Exact skill name as declared in its SKILL.MD frontmatter.
        """
        self.policy.enable_skill(session_id, skill_name)

    def disable_skill(self, session_id: str, skill_name: str) -> None:
        """Block a skill for a session, overriding global defaults.

        Args:
            session_id: The ``thread_id`` used when invoking the agent.
            skill_name: Exact skill name to block.
        """
        self.policy.disable_skill(session_id, skill_name)

    def enable_skills_for_session(self, session_id: str, skill_names: List[str]) -> None:
        """Grant a fixed skill set for the lifetime of a session."""
        self.policy.enable_skills_for_session(session_id, skill_names)

    # -- invocation ---------------------------------------------------------

    async def invoke(
        self,
        message: str,
        thread_id: str = "default",
        long_term_memory: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        # Legacy keyword-only aliases kept for backward compatibility
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AgentState:
        """Run the agent to completion and return the final state.

        Args:
            message: The user's input text.
            thread_id: Session identifier - used by the checkpointer to load
                prior history and by the policy engine to resolve skill grants.
                Defaults to ``"default"`` (a single shared session).
            long_term_memory: Strings injected as Tier 3 in the system prompt
                on this turn.  Sourced from the user's ``memory.json``; not
                stored in the checkpoint.
            config: Optional extra LangGraph run config merged into the
                ``configurable`` dict.

        Returns:
            The final ``AgentState`` dict (``messages`` key holds full history).
        """
        # Legacy: if only user_id/session_id is supplied, use it as thread_id
        effective_thread = thread_id
        if effective_thread == "default" and (user_id or session_id):
            effective_thread = user_id or session_id or "default"

        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
        }
        run_config: Dict[str, Any] = {"configurable": {
            "thread_id": effective_thread,
            "long_term_memory": long_term_memory or [],
        }}
        if config:
            run_config.setdefault("configurable", {}).update(
                config.get("configurable", {})
            )
        return await self.app.ainvoke(initial_state, run_config)

    async def astream(
        self,
        message: str,
        thread_id: str = "default",
        long_term_memory: Optional[List[str]] = None,
        # Legacy keyword-only aliases
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Yield LangGraph node update dicts for streaming CLI display.

        Args:
            message: The user's input text for this turn.
            thread_id: Session identifier (see ``invoke`` for full semantics).
            long_term_memory: LTM strings injected into the system prompt.
        """
        effective_thread = thread_id
        if effective_thread == "default" and (user_id or session_id):
            effective_thread = user_id or session_id or "default"

        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
        }
        run_config: Dict[str, Any] = {"configurable": {
            "thread_id": effective_thread,
            "long_term_memory": long_term_memory or [],
        }}
        async for update in self.app.astream(initial_state, run_config, stream_mode="updates"):
            yield update
