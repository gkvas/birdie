"""
Agent entrypoint and runtime.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import MemorySaver

from ..core.registry import SkillRegistry
from ..core.loader import discover_skills_from_directory
from ..core.policy import UserSkillPolicy
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
         )
    """

    def __init__(
        self,
        llm_or_provider: Any,
        skills_dir: str = "skills",
        use_memory: bool = True,
    ) -> None:
        # Accept either a native LLMProvider or any LangChain BaseChatModel
        if isinstance(llm_or_provider, LLMProvider):
            self.provider = llm_or_provider
        else:
            # Wrap a LangChain BaseChatModel for backward compatibility
            self.provider = LangChainProvider(llm_or_provider)

        self.skills_dir = skills_dir
        self.registry = SkillRegistry()
        self.policy = UserSkillPolicy()

        self._load_skills()

        graph = create_agent_graph(self.provider, self.registry, self.policy)
        checkpointer = MemorySaver() if use_memory else None
        self.app = graph.compile(checkpointer=checkpointer)
        self._use_memory = use_memory

    @classmethod
    def from_config(
        cls,
        provider_config: "Dict[str, Any] | str | ProviderConfig | None" = None,
        skills_dir: str = "skills",
    ) -> "DynamicAgent":
        """
        Create a DynamicAgent from a provider configuration.

        ``provider_config`` accepts the same types as :func:`get_llm_provider`:
        a ``dict``, a JSON string, a ``Path`` to a ``.json`` file, or a
        ``ProviderConfig`` instance.  Pass ``None`` to rely entirely on
        environment variables.

        Environment variable overrides (checked in priority order)
        ----------------------------------------------------------
        ``LLM_PROVIDER_CONFIG``
            Full JSON config blob — overrides everything else::

                export LLM_PROVIDER_CONFIG='{"vendor":"mistral","model":"mistral-large-latest"}'

        ``LLM_VENDOR``
            Override just the vendor, keeping all other config fields::

                LLM_VENDOR=anthropic python -m birdie.main

        ``LLM_MODEL``
            Override just the model::

                LLM_MODEL=gpt-4o-mini python -m birdie.main
        """
        # Highest priority: a full JSON config from the environment
        env_full = os.environ.get("LLM_PROVIDER_CONFIG")
        if env_full:
            provider_config = env_full  # JSON string → get_llm_provider handles it
        else:
            # Start from whatever was passed in
            if provider_config is None:
                provider_config = {}

            # Allow individual env vars to patch specific fields when the
            # caller passes a dict (or nothing).
            if isinstance(provider_config, dict):
                provider_config = dict(provider_config)  # don't mutate caller's dict
                if os.environ.get("LLM_VENDOR"):
                    provider_config["vendor"] = os.environ["LLM_VENDOR"]
                if os.environ.get("LLM_MODEL"):
                    provider_config["model"] = os.environ["LLM_MODEL"]

        provider = get_llm_provider(provider_config)
        return cls(provider, skills_dir=skills_dir)

    # -- skill management ---------------------------------------------------

    def _load_skills(self) -> None:
        """Discover SKILL.MD files, register them, and seed default policy."""
        skills = discover_skills_from_directory(self.skills_dir)
        for skill in skills:
            self.registry.register_skill(skill)
        self.policy.set_default_skills(skills)

    def enable_skill_for_user(self, user_id: str, skill_name: str) -> None:
        """Grant a skill to a user; takes effect on the next turn.

        Args:
            user_id: Opaque user identifier.
            skill_name: Exact skill name as declared in its SKILL.MD frontmatter.
        """
        self.policy.enable_skill_for_user(user_id, skill_name)

    def disable_skill_for_user(self, user_id: str, skill_name: str) -> None:
        """Block a skill for a user; overrides global defaults.

        Args:
            user_id: Opaque user identifier.
            skill_name: Exact skill name to block.
        """
        self.policy.disable_skill_for_user(user_id, skill_name)

    def enable_skills_for_session(self, session_id: str, skill_names: List[str]) -> None:
        """Grant a fixed skill set for the lifetime of a session.

        Args:
            session_id: Opaque session identifier.
            skill_names: Skills to enable; replaces any prior session grant.
        """
        self.policy.enable_skills_for_session(session_id, skill_names)

    # -- invocation ---------------------------------------------------------

    async def invoke(
        self,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        """Run the agent to completion and return the final state.

        Args:
            message: The user's input text.
            user_id: Optional user identifier used for skill policy resolution.
            session_id: Optional session identifier used for session-scoped grants.
            config: Optional LangGraph run config (e.g. custom ``thread_id`` under
                ``configurable``).  When ``use_memory=True`` and no ``thread_id``
                is provided, ``"default"`` is used.

        Returns:
            The final ``AgentState`` dict, which includes the full message history
            under the ``"messages"`` key.
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "session_id": session_id,
            "active_skill_names": None,
        }
        run_config = dict(config or {})
        if self._use_memory:
            run_config.setdefault("configurable", {})["thread_id"] = (
                run_config.get("configurable", {}).get("thread_id", "default")
            )
        return await self.app.ainvoke(initial_state, run_config)

    async def astream(
        self,
        message: str,
        thread_id: str = "default",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Yield LangGraph node update dicts for streaming CLI display."""
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "session_id": session_id,
            "active_skill_names": None,
        }
        run_config: Dict[str, Any] = {}
        if self._use_memory:
            run_config["configurable"] = {"thread_id": thread_id}
        async for update in self.app.astream(initial_state, run_config, stream_mode="updates"):
            yield update
