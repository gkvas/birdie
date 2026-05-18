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
* ``config["configurable"]["user_id"]``    - user ID for LTM lookup
* ``config["configurable"]["long_term_memory"]``  - list of manual LTM strings
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import TypedDict, List, Annotated, Sequence, Tuple, Optional, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
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
from ..core.ltm import LTMStore

log = logging.getLogger(__name__)

# Compaction thresholds.
MIN_MESSAGES = 20       # minimum messages to retain after compaction
MAX_MESSAGES = 100      # trigger compaction when stored history reaches this
COMPRESSION_WINDOW = 60  # maximum number of oldest messages to compress per run

# Delays (seconds) between successive retries when the provider returns 429.
_RATE_LIMIT_RETRY_DELAYS = (5, 15, 45)

_COMPACTION_PROMPT = """\
You are a memory compaction system. Analyse the following conversation \
transcript and extract structured information for long-term storage.

Output a single JSON object with exactly these six fields:
{{
  "summary": "<2-4 sentence narrative of what happened in this segment>",
  "extracted_facts": ["<specific fact, decision, or named value>", ...],
  "user_preferences": ["<how the user likes things done or styled>", ...],
  "world_facts": ["<factual observation about the external world>", ...],
  "tool_results": ["<key finding or outcome from a tool call>", ...],
  "open_tasks": ["<task mentioned but not yet completed>", ...]
}}

Rules:
- summary: plain narrative, 2-4 sentences, no bullet points.
- Lists may be empty ([]) when nothing fits that category.
- Output ONLY the JSON object, no other text.

<conversation>
{history}
</conversation>"""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _msg_text(msg: BaseMessage) -> str:
    """Extract plain text from a message whose content may be a list of blocks."""
    content = msg.content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def _parse_compaction_json(text: str) -> dict:
    """Parse the structured JSON produced by the compaction prompt.

    Falls back gracefully when the model includes surrounding prose.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {
        "summary": text,
        "extracted_facts": [],
        "user_preferences": [],
        "world_facts": [],
        "tool_results": [],
        "open_tasks": [],
    }


async def compact_history(
    all_messages: List[BaseMessage],
    provider: LLMProvider,
    ltm_store: Optional[LTMStore] = None,
    force: bool = False,
    min_messages: int = MIN_MESSAGES,
    max_messages: int = MAX_MESSAGES,
    compression_window: int = COMPRESSION_WINDOW,
) -> Tuple[str, List[RemoveMessage]]:
    """Summarize the oldest conversation segment and store it in LTM.

    Finds the largest HumanMessage-aligned split point within
    ``compression_window`` messages from the start, such that at least
    ``min_messages`` messages remain after the split.  The messages before
    the split are summarised with a structured JSON prompt; the result is
    stored in ``ltm_store`` (when provided) and the old messages are
    returned as RemoveMessage deletions for the LangGraph checkpointer.

    When ``force`` is ``True`` the ``max_messages`` threshold is bypassed so
    compaction runs regardless of history length (used by ``/compact``).

    Returns ``("", [])`` when there is nothing worth compacting.
    """
    if not force and len(all_messages) < max_messages:
        return "", []

    human_indices = [i for i, m in enumerate(all_messages) if isinstance(m, HumanMessage)]
    if len(human_indices) < 2:
        return "", []

    # Find the largest HumanMessage index that:
    #   - is at least 1 (don't compress zero messages)
    #   - keeps at least min_messages messages after the split
    #   - does not exceed compression_window (don't over-compress)
    max_split = min(compression_window, len(all_messages) - min_messages)
    split_at = None
    for idx in reversed(human_indices):
        if 0 < idx <= max_split:
            split_at = idx
            break

    if split_at is None:
        return "", []

    old_msgs = all_messages[:split_at]
    if len(old_msgs) < min_messages // 2:
        return "", []

    # Build a readable transcript of the messages to be summarised.
    lines: List[str] = []
    for msg in old_msgs:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {_msg_text(msg)}")
        elif isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                names = ", ".join(tc["name"] for tc in msg.tool_calls)
                prefix = f"Assistant (used tools: {names})"
                text = _msg_text(msg)
                lines.append(f"{prefix}: {text}" if text else prefix)
            else:
                lines.append(f"Assistant: {_msg_text(msg)}")
        elif isinstance(msg, ToolMessage):
            excerpt = _msg_text(msg)
            if len(excerpt) > 300:
                excerpt = excerpt[:300] + "..."
            lines.append(f"Tool result ({msg.name}): {excerpt}")

    history_text = "\n".join(lines)
    prompt = _COMPACTION_PROMPT.format(history=history_text)

    response = await provider.achat(messages=[HumanMessage(content=prompt)])
    compaction_result = _parse_compaction_json(_msg_text(response))
    summary_text = compaction_result.get("summary", "")

    if ltm_store is not None:
        ltm_store.add(compaction_result)

    remove_msgs: List[RemoveMessage] = [
        RemoveMessage(id=m.id)  # type: ignore[misc]
        for m in old_msgs
        if m.id is not None
    ]

    log.info(
        "Compaction: summarised %d messages into LTM, kept %d (threshold=%d)",
        len(old_msgs), len(all_messages) - len(old_msgs), max_messages,
    )
    return summary_text, remove_msgs


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
    ltm_factory: Optional[Callable[[str], LTMStore]] = None,
    min_messages: int = MIN_MESSAGES,
    max_messages: int = MAX_MESSAGES,
    compression_window: int = COMPRESSION_WINDOW,
) -> StateGraph:
    """
    Build a LangGraph workflow that routes through the LLMProvider.

    On every agent-node invocation:
      1. LTM is queried with the current user message and injected into the
         system prompt (when ``ltm_factory`` is provided and a user_id is known).
      2. The full message history is compacted when it exceeds MAX_MESSAGES.
      3. The full non-compacted history is sent to the LLM, starting from
         the first HumanMessage (required by some providers, e.g. Mistral).
      4. Any dangling tool calls (interrupted prior turn) are repaired.
      5. The currently-allowed SkillTools are fetched from the registry.
      6. provider.achat() is called with the clean context window.
      7. The response (an AIMessage) is appended to state.

    On every tool-node invocation:
      1. A fresh ToolNode is created with LangChain-executable tools.
      2. Tool results are appended to state and the loop continues.
    """
    # LTM store cache: one LTMStore instance per user_id for the lifetime of
    # this agent graph so we avoid reloading the JSON file on every turn.
    _ltm_cache: dict[str, LTMStore] = {}

    def _get_ltm_store(user_id: str) -> Optional[LTMStore]:
        if not ltm_factory or not user_id:
            return None
        if user_id not in _ltm_cache:
            _ltm_cache[user_id] = ltm_factory(user_id)
        return _ltm_cache[user_id]

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

    def _build_system_prompt(
        state: AgentState,
        config: RunnableConfig,
        ltm_context: str = "",
    ) -> str | None:
        """Assemble the system prompt for this turn from in-memory skill objects.

        Five tiers, joined with double newlines:

        - **Tier 0** - custom instructions from ``.birdie/system_prompt.md``.
        - **Tier 1** - compact bullet list of all allowed skills.
        - **Tier 2a** - full prose body for ``always_inject`` skills.
        - **Tier 2b** - full prose body for freetext skills whose triggers fired.
        - **Tier 3** - long-term memory: manual entries from config plus
          semantically retrieved compaction entries (``ltm_context``).

        Returns ``None`` when nothing would be included.
        """
        custom = _load_custom_system_prompt()
        allowed = _get_allowed(config)
        all_skills = [s for s in registry.list_skills() if s.name in allowed]
        manual_ltm = config.get("configurable", {}).get("long_term_memory") or []

        parts: List[str] = []

        # Tier 0
        if custom:
            parts.append(custom)

        # Tiers 1 + 2
        if all_skills:
            lines = ["You have access to the following skills:\n"]
            for skill in all_skills:
                trigger_hint = (
                    f"  triggers: {', '.join(skill.triggers)}" if skill.triggers else ""
                )
                lines.append(f"- **{skill.name}**: {skill.description}{trigger_hint}")
            skill_block = "\n".join(lines)

            for skill in all_skills:
                if skill.always_inject and skill.body:
                    skill_block += f"\n\n--- {skill.name} instructions ---\n{skill.body}"
            for skill in _triggered_freetext(state, allowed):
                if skill.body:
                    skill_block += f"\n\n--- {skill.name} skill context ---\n{skill.body}"

            parts.append(skill_block)

        # Tier 3
        if manual_ltm or ltm_context:
            ltm_lines = ["--- Long-term memory ---"]
            if manual_ltm:
                ltm_lines.append("\n".join(f"- {entry}" for entry in manual_ltm))
            if ltm_context:
                ltm_lines.append(ltm_context)
            parts.append("\n".join(ltm_lines))

        return "\n\n".join(parts) or None

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
        user_id = config.get("configurable", {}).get("user_id") or ""

        # Retrieve semantically relevant LTM entries for the current user message.
        ltm_context = ""
        ltm_store = _get_ltm_store(user_id)
        if ltm_store is not None:
            user_text = _last_user_text(state)
            if user_text:
                entries = ltm_store.query(user_text, k=5)
                if entries:
                    ltm_context = ltm_store.format_for_prompt(entries)

        # Compact history when it has grown beyond the threshold.
        compaction_removes: List[BaseMessage] = []
        if len(all_messages) >= max_messages:
            _, compaction_removes = await compact_history(
                all_messages, provider, ltm_store=ltm_store,
                min_messages=min_messages,
                max_messages=max_messages,
                compression_window=compression_window,
            )
            if compaction_removes:
                # Remove compacted messages from the local working copy.
                remove_ids = {r.id for r in compaction_removes}
                all_messages = [m for m in all_messages if m.id not in remove_ids]

        # Send the full non-compacted history to the LLM.  Compaction already
        # bounds the checkpoint size; trimming further would discard context
        # that the LTM system deliberately preserved.
        # Start from the first HumanMessage so providers that require a
        # HumanMessage-first context (e.g. Mistral) are always satisfied.
        human_indices = [
            i for i, m in enumerate(all_messages)
            if isinstance(m, HumanMessage)
        ]
        context_msgs = (
            all_messages[human_indices[0]:] if human_indices else all_messages
        )

        # Repair any dangling tool calls within the context window.
        repair_msgs, clean_messages = _repair_dangling_tool_calls(context_msgs)

        allowed = _get_allowed(config)
        skill_tools = list(registry.list_tools(skill_names=list(allowed)))
        mcp_tools = await mcp_manager.get_tools(allowed) if mcp_manager else []
        agent_tools = agent_registry.get_tools(_get_allowed_agents(config)) if agent_registry else []

        if provider.supports_tools() and (skill_tools or mcp_tools or agent_tools):
            normalized_tools = (
                [skilltool_to_normalized_def(t) for t in skill_tools]
                + [lc_tool_to_normalized_def(t) for t in mcp_tools + agent_tools]
            )
        else:
            normalized_tools = None

        system_prompt = _build_system_prompt(state, config, ltm_context=ltm_context)

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

        return {"messages": compaction_removes + repair_msgs + [response]}

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
        agent_tools = agent_registry.get_tools(_get_allowed_agents(config)) if agent_registry else []
        langchain_tools = (
            [skilltool_to_langchain_tool(t) for t in skill_tools] + mcp_tools + agent_tools
        )
        tool_node = ToolNode(langchain_tools)
        try:
            return await tool_node.ainvoke(state, config)
        except Exception as exc:
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
