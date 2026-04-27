"""
LangGraph state machine for the Birdie agent.

The graph has two nodes — ``agent`` (calls the LLM) and ``tools`` (executes
tool calls) — connected by a conditional edge that loops until the model
produces a message with no tool calls.

Node functions are closures over the shared ``provider``, ``registry``, and
``policy`` objects so the graph stays stateless and testable.
"""

from typing import TypedDict, List, Optional, Annotated, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from ..core.registry import SkillRegistry
from ..core.adapter import skilltool_to_langchain_tool
from ..core.policy import UserSkillPolicy
from ..core.llm_provider import LLMProvider, skilltool_to_normalized_def


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: Optional[str]
    session_id: Optional[str]
    active_skill_names: Optional[List[str]]


def create_agent_graph(
    provider: LLMProvider,
    registry: SkillRegistry,
    policy: UserSkillPolicy,
) -> StateGraph:
    """
    Build a LangGraph workflow that routes through the LLMProvider.

    On every agent-node invocation:
      1. The currently-allowed SkillTools are fetched from the registry.
      2. They are converted to NormalizedToolDef dicts and forwarded to
         provider.achat() — the provider handles all vendor-specific formatting.
      3. The response (an AIMessage) is appended to state.

    On every tool-node invocation:
      1. A fresh ToolNode is created with LangChain-executable tools so that
         the entrypoint resolvers (bash, http, python…) are wired up.
      2. Tool results are appended to state and the loop continues.
    """

    def _get_allowed(state: AgentState) -> set:
        """Resolve the allowed skill name set for the current user/session."""
        return policy.get_allowed_skills(
            user_id=state.get("user_id"),
            session_id=state.get("session_id"),
        )

    def _last_user_text(state: AgentState) -> str:
        """Return the content of the most recent HumanMessage in state."""
        for msg in reversed(list(state["messages"])):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        return ""

    def _triggered_freetext(state: AgentState, allowed: set) -> list:
        """Return freetext skills whose triggers appear in the latest user message."""
        return registry.find_skills_by_trigger(_last_user_text(state), allowed)

    def _get_skill_tools(state: AgentState) -> list:
        """Return the callable SkillTools for all allowed skills this turn."""
        return list(registry.list_tools(skill_names=list(_get_allowed(state))))

    def _build_system_prompt(state: AgentState) -> str | None:
        """Assemble the system prompt for this turn from in-memory skill objects.

        Two tiers:

        - **Tier 1** — compact bullet list of all allowed skills (always sent).
        - **Tier 2a** — full prose body for ``always_inject`` skills (always sent).
        - **Tier 2b** — full prose body for freetext skills whose triggers fired
          this turn.

        Returns ``None`` when no skills are allowed for the user/session.
        """
        allowed = _get_allowed(state)
        all_skills = [s for s in registry.list_skills() if s.name in allowed]
        if not all_skills:
            return None

        # Tier 1 — compact skill catalog (always included)
        lines = ["You have access to the following skills:\n"]
        for skill in all_skills:
            trigger_hint = (
                f"  triggers: {', '.join(skill.triggers)}" if skill.triggers else ""
            )
            lines.append(f"- **{skill.name}**: {skill.description}{trigger_hint}")
        system = "\n".join(lines)

        # Tier 2a — always-inject skill bodies (e.g. planning/meta skills)
        for skill in all_skills:
            if skill.always_inject and skill.body:
                system += f"\n\n--- {skill.name} instructions ---\n{skill.body}"

        # Tier 2b — full body for triggered freetext skills
        for skill in _triggered_freetext(state, allowed):
            if skill.body:
                system += f"\n\n--- {skill.name} skill context ---\n{skill.body}"

        return system

    def _repair_dangling_tool_calls(
        messages: List[BaseMessage],
    ) -> Tuple[List[ToolMessage], List[BaseMessage]]:
        """
        Scan the message list for AIMessages whose tool_calls have no matching
        ToolMessage and inject placeholder ToolMessages for each missing one.

        Returns (repair_msgs, patched_messages) so callers can both persist the
        repairs to state and send the clean history to the provider.
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
        # Repair any broken history caused by interrupted tool executions.
        # Repairs are included in the return value so they are persisted to the
        # checkpoint — the state heals permanently on the next save.
        repair_msgs, clean_messages = _repair_dangling_tool_calls(list(state["messages"]))

        skill_tools = _get_skill_tools(state)

        normalized_tools = (
            [skilltool_to_normalized_def(t) for t in skill_tools]
            if skill_tools and provider.supports_tools()
            else None
        )

        system_prompt = _build_system_prompt(state)

        response = await provider.achat(
            messages=clean_messages,
            tools=normalized_tools,
            system_prompt=system_prompt,
        )
        return {"messages": repair_msgs + [response]}

    async def execute_tools(state: AgentState) -> dict:
        skill_tools = _get_skill_tools(state)
        langchain_tools = [skilltool_to_langchain_tool(t) for t in skill_tools]
        tool_node = ToolNode(langchain_tools)
        try:
            return await tool_node.ainvoke(state)
        except Exception as exc:
            # If the ToolNode itself raises (e.g. the entrypoint throws before
            # producing output), create error ToolMessages for every pending call
            # so the state stays balanced and the LLM can respond to the failure.
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
