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
from ..core.ltm import LTMStore
from .graph import create_agent_graph, compact_history, AgentState


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
        agent_console=None,
        checkpointer=None,
        provider_config: Optional[Dict[str, Any]] = None,
        ltm_store_factory=None,
    ) -> None:
        # Accept either a native LLMProvider or any LangChain BaseChatModel
        if isinstance(llm_or_provider, LLMProvider):
            self.provider = llm_or_provider
        else:
            self.provider = LangChainProvider(llm_or_provider)
        self._provider_config = provider_config
        self._ltm_store_factory = ltm_store_factory

        self.skills_dir = skills_dir
        self.agents_dir = agents_dir
        self._agent_console = agent_console
        self.agent_output_mode: str = "off"
        self.registry = SkillRegistry()
        self.policy = SkillPolicy()
        self.mcp_manager = MCPClientManager()
        self.agent_registry = AgentRegistry()

        self._load_skills()
        self._load_agents()

        graph = create_agent_graph(
            self.provider, self.registry, self.policy, self.mcp_manager,
            self.agent_registry,
            ltm_factory=self._ltm_store_factory,
        )
        self.app = graph.compile(checkpointer=checkpointer or MemorySaver())

    @classmethod
    def from_config(
        cls,
        provider_config: "Dict[str, Any] | str | Path | ProviderConfig | None" = None,
        skills_dir: str = "skills",
        agents_dir: Optional[str] = None,
        agent_console=None,
        checkpointer=None,
        ltm_store_factory=None,
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

        # Normalize to a dict so sub-agents can inherit the full provider config
        if isinstance(provider_config, ProviderConfig):
            config_dict: Dict[str, Any] = provider_config.model_dump(exclude_none=True)
        elif isinstance(provider_config, dict):
            config_dict = dict(provider_config)
        elif isinstance(provider_config, str):
            config_dict = ProviderConfig.model_validate_json(provider_config).model_dump(exclude_none=True)
        elif isinstance(provider_config, Path):
            config_dict = ProviderConfig.from_file(provider_config).model_dump(exclude_none=True)
        else:
            config_dict = {}

        provider = get_llm_provider(provider_config)
        if ltm_store_factory is None:
            def ltm_store_factory(uid: str) -> LTMStore:
                return LTMStore(uid)
        return cls(provider, skills_dir=skills_dir, agents_dir=agents_dir,
                   agent_console=agent_console, checkpointer=checkpointer,
                   provider_config=config_dict, ltm_store_factory=ltm_store_factory)

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
        """Discover AGENT.MD files from all agent dirs and register them."""
        _bundled = Path(__file__).parent.parent / "agents"
        primary = self.agents_dir or (str(_bundled) if _bundled.is_dir() else None)

        dirs = []
        if primary:
            dirs.append(primary)
        user_agents_dir = Path.home() / ".birdie" / "agents"
        if user_agents_dir.is_dir():
            dirs.append(str(user_agents_dir))

        if not dirs:
            return

        for d in dirs:
            for agent_def in discover_agents_from_directory(d):
                tool = agentdef_to_langchain_tool(
                    agent_def,
                    skills_dir=self.skills_dir,
                    agents_dir=self.agents_dir,
                    fallback_provider_config=self._provider_config,
                    console=self._agent_console,
                    get_tool_output_mode=lambda: self.agent_output_mode,
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

    # -- agent management ---------------------------------------------------

    def enable_agent(self, session_id: str, agent_name: str) -> None:
        """Grant an agent for a session. Takes effect on the next turn.

        Args:
            session_id: The ``thread_id`` used when invoking the agent.
            agent_name: Exact agent name as declared in its AGENT.MD frontmatter.
        """
        self.agent_registry.enable_agent(session_id, agent_name)

    def disable_agent(self, session_id: str, agent_name: str) -> None:
        """Block an agent for a session, overriding global defaults.

        Args:
            session_id: The ``thread_id`` used when invoking the agent.
            agent_name: Exact agent name to block.
        """
        self.agent_registry.disable_agent(session_id, agent_name)

    def enable_agents_for_session(self, session_id: str, agent_names: List[str]) -> None:
        """Grant a fixed agent set for the lifetime of a session."""
        self.agent_registry.enable_agents_for_session(session_id, agent_names)

    # -- invocation ---------------------------------------------------------

    async def invoke(
        self,
        message: str,
        thread_id: str = "default",
        long_term_memory: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
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
            user_id: User identity for LTM lookup.  When provided it is put
                into ``configurable["user_id"]``.  If ``thread_id`` is
                ``"default"`` this also becomes the ``thread_id`` (legacy
                behaviour).

        Returns:
            The final ``AgentState`` dict (``messages`` key holds full history).
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
        if user_id:
            run_config["configurable"]["user_id"] = user_id
        if config:
            for k, v in config.items():
                if k == "configurable":
                    run_config.setdefault("configurable", {}).update(v)
                else:
                    run_config[k] = v
        return await self.app.ainvoke(initial_state, run_config)

    async def astream(
        self,
        message: str,
        thread_id: str = "default",
        long_term_memory: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Yield LangGraph node update dicts for streaming CLI display.

        Args:
            message: The user's input text for this turn.
            thread_id: Session identifier (see ``invoke`` for full semantics).
            long_term_memory: LTM strings injected into the system prompt.
            config: Optional extra LangGraph run config (e.g. recursion_limit).
            user_id: User identity for LTM lookup (see ``invoke``).
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
        if user_id:
            run_config["configurable"]["user_id"] = user_id
        if config:
            for k, v in config.items():
                if k == "configurable":
                    run_config.setdefault("configurable", {}).update(v)
                else:
                    run_config[k] = v
        async for update in self.app.astream(initial_state, run_config, stream_mode="updates"):
            yield update

    async def compact_session(
        self,
        thread_id: str,
        user_id: str = "",
    ) -> tuple[int, str]:
        """Force-compact the stored history for the given session.

        Reads the checkpoint, runs compaction regardless of history length,
        stores the result in LTM (when a factory and user_id are available),
        and writes the RemoveMessage deletions back to the checkpoint.

        Returns ``(messages_removed, summary_text)``.
        ``messages_removed == 0`` means there was nothing to compact.
        """
        run_config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        snapshot = await self.app.aget_state(run_config)
        all_messages = list((snapshot.values or {}).get("messages", []))

        ltm_store = None
        if self._ltm_store_factory and user_id:
            ltm_store = self._ltm_store_factory(user_id)

        summary, removes = await compact_history(
            all_messages, self.provider, ltm_store=ltm_store, force=True,
        )

        if removes:
            await self.app.aupdate_state(run_config, {"messages": removes})

        return len(removes), summary
