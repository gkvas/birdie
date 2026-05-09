"""
LangGraph state machine for the Birdie agent.

The graph has two nodes - ``agent`` (calls the LLM) and ``tools`` (executes
tool calls) - connected by a conditional edge that loops until the model
produces a message with no tool calls.

Node functions are closures over the shared ``provider``, ``registry``, and
``policy`` objects so the graph stays stateless and testable.

Per-invocation context (session identity, long-term memory) is passed via
``config["configurable"]`` rather than stored in AgentState:

* ``config["configurable"]["thread_id"]``  - session ID, used as policy key
* ``config["configurable"]["long_term_memory"]``  - list of LTM strings

This keeps AgentState minimal (just ``messages``) so LangGraph's checkpointer
owns and reconstructs it without any application-level serialization.
"""

import asyncio
import logging
from pathlib import Path
from typing import TypedDict, List, Annotated, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from ..core.registry import SkillRegistry
from ..core.adapter import skilltool_to_langchain_tool
from ..core.policy import SkillPolicy
from ..core.llm_provider import LLMProvider, skilltool_to_normalized_def, lc_tool_to_normalized_def
from ..core.mcp_client import MCPClientManager
from ..core.agent_registry import AgentRegistry

log = logging.getLogger(__name__)

# Maximum number of messages forwarded to the LLM per turn.
# The full history is stored by the checkpointer; this window controls cost.
MAX_CONTEXT_MESSAGES = 20

# Delays (seconds) between successive retries when the provider returns 429.
_RATE_LIMIT_RETRY_DELAYS = (5, 15, 45)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _consecutive_call_count(messages: list, name: str, args: dict) -> int:
    """Count how many consecutive recent agent cycles included (name, args) in their tool calls.

    Walks backward through the message list, counting AIMessages that contain
    the target (name, args) pair and skipping ToolMessages between them.
    Stops as soon as an AIMessage without the target call is found.
    """
    count = 0
    i = len(messages) - 1
    while i >= 0:
        msg = messages[i]
        if isinstance(msg, AIMessage):
            if any(tc["name"] == name and tc["args"] == args
                   for tc in getattr(msg, "tool_calls", [])):
                count += 1
                i -= 1
            else:
                break
        elif isinstance(msg, ToolMessage):
            i -= 1
        else:
            break
    return count


def create_agent_graph(
    provider: LLMProvider,
    registry: SkillRegistry,
    policy: SkillPolicy,
    mcp_manager: MCPClientManager | None = None,
    agent_registry: AgentRegistry | None = None,
) -> StateGraph:
    """
    Build a LangGraph workflow that routes through the LLMProvider.

    On every agent-node invocation:
      1. The full message history is trimmed to MAX_CONTEXT_MESSAGES.
      2. Any dangling tool calls (interrupted prior turn) are repaired.
      3. The currently-allowed SkillTools are fetched from the registry.
      4. provider.achat() is called with the clean context window.
      5. The response (an AIMessage) is appended to state.

    On every tool-node invocation:
      1. A fresh ToolNode is created with LangChain-executable tools.
      2. Tool results are appended to state and the loop continues.
    """

    def _session_id(config: RunnableConfig) -> str:
        return config.get("configurable", {}).get("thread_id", "") or ""

    def _get_allowed(config: RunnableConfig) -> set:
        """Resolve the allowed skill name set for the current session."""
        return policy.get_allowed_skills(session_id=_session_id(config))

    def _get_allowed_agents(config: RunnableConfig) -> set:
        """Resolve the allowed agent name set for the current session."""
        if agent_registry is None:
            return set()
        return agent_registry.get_allowed_agents(session_id=_session_id(config))

    def _last_user_text(state: AgentState) -> str:
        """Return the content of the most recent HumanMessage in state."""
        for msg in reversed(list(state["messages"])):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        return ""

    def _triggered_freetext(state: AgentState, allowed: set) -> list:
        """Return freetext skills whose triggers appear in the latest user message."""
        return registry.find_skills_by_trigger(_last_user_text(state), allowed)

    def _get_skill_tools(config: RunnableConfig) -> list:
        """Return the callable SkillTools for all allowed skills this turn."""
        return list(registry.list_tools(skill_names=list(_get_allowed(config))))

    def _load_custom_system_prompt() -> str | None:
        """Load a custom system prompt from .birdie/system_prompt.md if present."""
        path = Path(".birdie") / "system_prompt.md"
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            return text if text else None
        return None

    def _build_system_prompt(state: AgentState, config: RunnableConfig) -> str | None:
        """Assemble the system prompt for this turn from in-memory skill objects.

        Four tiers:

        - **Tier 0** - custom instructions from ``.birdie/system_prompt.md`` (if present).
        - **Tier 1** - compact bullet list of all allowed skills (always sent).
        - **Tier 2a** - full prose body for ``always_inject`` skills (always sent).
        - **Tier 2b** - full prose body for freetext skills whose triggers fired.
        - **Tier 3** - long-term memory strings from the user's memory store.

        Returns ``None`` when no skills are allowed and no custom prompt exists.
        """
        custom = _load_custom_system_prompt()

        allowed = _get_allowed(config)
        all_skills = [s for s in registry.list_skills() if s.name in allowed]
        if not all_skills:
            return custom  # may be None

        # Tier 0 - custom project/user instructions
        system = ""
        if custom:
            system = custom + "\n\n"

        # Tier 1 - compact skill catalog (always included)
        lines = ["You have access to the following skills:\n"]
        for skill in all_skills:
            trigger_hint = (
                f"  triggers: {', '.join(skill.triggers)}" if skill.triggers else ""
            )
            lines.append(f"- **{skill.name}**: {skill.description}{trigger_hint}")
        system += "\n".join(lines)

        # Tier 2a - always-inject skill bodies (e.g. planning/meta skills)
        for skill in all_skills:
            if skill.always_inject and skill.body:
                system += f"\n\n--- {skill.name} instructions ---\n{skill.body}"

        # Tier 2b - full body for triggered freetext skills
        for skill in _triggered_freetext(state, allowed):
            if skill.body:
                system += f"\n\n--- {skill.name} skill context ---\n{skill.body}"

        # Tier 3 - user long-term memory injected via config
        ltm = config.get("configurable", {}).get("long_term_memory") or []
        if ltm:
            system += "\n\n--- Long-term memory ---\n"
            system += "\n".join(f"- {entry}" for entry in ltm)

        return system

    def _repair_dangling_tool_calls(
        messages: List[BaseMessage],
    ) -> Tuple[List[ToolMessage], List[BaseMessage]]:
        """
        Scan for AIMessages whose tool_calls have no matching ToolMessage and
        inject placeholder ToolMessages for each orphan.

        This handles the case where the process was interrupted between the LLM
        response (checkpointed) and the tool execution, leaving the checkpoint
        in a state that most providers reject as a protocol violation.

        Returns (repair_msgs, patched_messages).  ``repair_msgs`` is included
        in the node's return value so it is written back to the checkpoint,
        healing the state permanently.
        """
        tool_call_ids_answered: set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_call_ids_answered.add(msg.tool_call_id)

        repair_msgs: List[ToolMessage] = []
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            for tc in getattr(msg, "tool_calls", []):
                if tc["id"] not in tool_call_ids_answered:
                    repair_msgs.append(ToolMessage(
                        content="Tool execution was interrupted before a result was produced.",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    ))

        if not repair_msgs:
            return [], messages

        # Insert repairs immediately after their originating AIMessage
        patched: List[BaseMessage] = []
        repair_by_call_id = {m.tool_call_id: m for m in repair_msgs}
        for msg in messages:
            patched.append(msg)
            if isinstance(msg, AIMessage):
                for tc in getattr(msg, "tool_calls", []):
                    if tc["id"] in repair_by_call_id:
                        patched.append(repair_by_call_id[tc["id"]])

        return repair_msgs, patched

    async def call_model(state: AgentState, config: RunnableConfig) -> dict:
        all_messages = list(state["messages"])

        # Build a context window that never splits a turn.  Walk backward
        # through HumanMessage boundaries, accumulating complete turns until
        # the window contains at least MAX_CONTEXT_MESSAGES messages.  This
        # guarantees the context always starts at a HumanMessage (required by
        # Mistral) and that a long tool chain in the previous turn is never
        # evicted mid-sequence, which would cause the LLM to lose its own
        # prior response on a follow-up.
        human_indices = [
            i for i, m in enumerate(all_messages)
            if isinstance(m, HumanMessage)
        ]
        if not human_indices:
            # No HumanMessage at all - fall back to a plain tail slice.
            context_msgs = all_messages[-MAX_CONTEXT_MESSAGES:]
        else:
            anchor = human_indices[-1]
            for j in range(len(human_indices) - 2, -1, -1):
                if len(all_messages) - anchor >= MAX_CONTEXT_MESSAGES:
                    break
                anchor = human_indices[j]
            context_msgs = list(all_messages[anchor:])

        # Repair any dangling tool calls within the context window.
        # Repairs are returned to state so the checkpoint heals permanently.
        repair_msgs, clean_messages = _repair_dangling_tool_calls(context_msgs)

        allowed = _get_allowed(config)
        skill_tools = list(registry.list_tools(skill_names=list(allowed)))
        mcp_tools = await mcp_manager.get_tools(allowed) if mcp_manager else []
        # intersection point: agent tools join skill/mcp tools only here
        agent_tools = agent_registry.get_tools(_get_allowed_agents(config)) if agent_registry else []

        if provider.supports_tools() and (skill_tools or mcp_tools or agent_tools):
            normalized_tools = (
                [skilltool_to_normalized_def(t) for t in skill_tools]
                + [lc_tool_to_normalized_def(t) for t in mcp_tools + agent_tools]
            )
        else:
            normalized_tools = None

        system_prompt = _build_system_prompt(state, config)

        # Retry on HTTP 429 (rate limit) with exponential back-off.
        response: BaseMessage | None = None
        for attempt in range(len(_RATE_LIMIT_RETRY_DELAYS) + 1):
            if attempt > 0:
                delay = _RATE_LIMIT_RETRY_DELAYS[attempt - 1]
                log.warning(
                    "Provider rate-limited (429); retrying in %ds (attempt %d/%d)",
                    delay, attempt + 1, len(_RATE_LIMIT_RETRY_DELAYS) + 1,
                )
                await asyncio.sleep(delay)
            try:
                response = await provider.achat(
                    messages=clean_messages,
                    tools=normalized_tools,
                    system_prompt=system_prompt,
                )
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < len(_RATE_LIMIT_RETRY_DELAYS):
                    continue
                raise

        return {"messages": repair_msgs + [response]}

    async def execute_tools(state: AgentState, config: RunnableConfig) -> dict:
        # Infinite loop guard: block any tool call that has appeared consecutively
        # more than max_tool_repetitions times with identical parameters.
        max_reps = config.get("configurable", {}).get("max_tool_repetitions", 3)
        all_messages = list(state["messages"])
        last_ai = all_messages[-1] if all_messages else None
        if isinstance(last_ai, AIMessage):
            for tc in getattr(last_ai, "tool_calls", []):
                if _consecutive_call_count(all_messages, tc["name"], tc["args"]) > max_reps:
                    return {"messages": [
                        ToolMessage(
                            content=(
                                f"Error: '{tc['name']}' has been called more than {max_reps} "
                                f"times consecutively with identical parameters. "
                                f"Breaking the loop to prevent infinite repetition."
                            ),
                            tool_call_id=t["id"],
                            name=t["name"],
                        )
                        for t in getattr(last_ai, "tool_calls", [])
                    ]}

        allowed = _get_allowed(config)
        skill_tools = list(registry.list_tools(skill_names=list(allowed)))
        mcp_tools = await mcp_manager.get_tools(allowed) if mcp_manager else []
        # intersection point: agent tools added here for execution
        agent_tools = agent_registry.get_tools(_get_allowed_agents(config)) if agent_registry else []
        langchain_tools = (
            [skilltool_to_langchain_tool(t) for t in skill_tools] + mcp_tools + agent_tools
        )
        tool_node = ToolNode(langchain_tools)
        try:
            return await tool_node.ainvoke(state, config)
        except Exception as exc:
            # If the ToolNode itself raises before producing output, synthesize
            # error ToolMessages so the state stays balanced and the LLM can
            # respond to the failure.
            last = state["messages"][-1]
            return {"messages": [
                ToolMessage(
                    content=f"Error: {exc}",
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
                for tc in getattr(last, "tool_calls", [])
            ]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    return workflow
