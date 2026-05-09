"""
Agent runner: converts an AgentDef into an executable async LangChain tool.

At call time:
  1. Input values are substituted into the prompt template ({{ param }} syntax).
  2. An ephemeral DynamicAgent is created with the agent's vendor/model config.
  3. The agent runs with the allowed_skills fixed-list; no session is persisted.
  4. The final AIMessage content is returned as the tool result.
"""

import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from .models import AgentDef, AgentParam

# Indentation levels mirror the CLI's tool output conventions:
#   _IH  = 3 spaces  = same level as regular tool output
#   _IC  = 6 spaces  = sub-agent content (tool output + 3)
#   _IC2 = 9 spaces  = sub-agent arg/result content (sub-agent content + 3)
_IH  = "   "
_IC  = "      "
_IC2 = "         "


def _substitute(template: str, params: Dict[str, Any]) -> str:
    """Replace {{ name }} placeholders with param values."""
    def replace(m: re.Match) -> str:
        key = m.group(1).strip()
        return str(params.get(key, m.group(0)))
    return re.sub(r'\{\{\s*(\w+)\s*\}\}', replace, template)


def _input_schema(params: List[AgentParam]) -> dict:
    """Build a JSON Schema object from a list of AgentParam objects."""
    _TYPE_MAP = {
        "string": "string", "integer": "integer", "number": "number",
        "boolean": "boolean", "array": "array", "object": "object",
    }
    properties = {}
    required = []
    for p in params:
        properties[p.name] = {
            "type": _TYPE_MAP.get(p.type, "string"),
            "description": p.description,
        }
        if p.required:
            required.append(p.name)
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _extract_text(content: Any) -> str:
    """Normalise AIMessage/ToolMessage content to a plain string."""
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return str(content)


def _render_block_content(console: Any, text: str, mode: str) -> None:
    """Render body text at _IC2 indent following off|short|full rules."""
    lines = text.splitlines() or [""]
    n = len(lines)

    if mode == "off":
        console.print(f"{_IC2}[dim]{n} line{'s' if n != 1 else ''}[/dim]")
        return

    if mode == "short":
        limit = 1000
        display = text[:limit]
        remaining = len(text) - limit
        display_lines = display.splitlines() or [""]
    else:  # full
        display_lines = lines
        remaining = 0

    for line in display_lines:
        console.print(f"{_IC2}[dim]{line}[/dim]", highlight=False)
    if remaining > 0:
        console.print(
            f"{_IC2}[dim]... {remaining} more character{'s' if remaining != 1 else ''}[/dim]"
        )


def _render_tool_call(console: Any, tc: dict, mode: str) -> None:
    """Render a sub-agent tool call at _IC indent."""
    console.print(f"{_IC}[bold cyan]→[/bold cyan] [bold]{tc['name']}[/bold]")
    if mode == "off":
        n = len(tc["args"])
        console.print(f"{_IC2}[dim]{n} arg{'s' if n != 1 else ''}[/dim]")
        return
    for k, v in tc["args"].items():
        v_str = v if isinstance(v, str) else repr(v)
        if mode == "short":
            flat = v_str.replace("\n", "↵ ")
            if len(flat) > 120:
                flat = flat[:120] + "…"
            console.print(f"{_IC2}[dim]{k}:[/dim] {flat}", highlight=False)
        else:  # full
            lines = v_str.splitlines()
            if len(lines) > 1:
                console.print(f"{_IC2}[dim]{k}:[/dim]", highlight=False)
                for line in lines:
                    console.print(f"{_IC2}  {line}", highlight=False)
            else:
                console.print(f"{_IC2}[dim]{k}:[/dim] {v_str}", highlight=False)


def _render_tool_result(console: Any, text: str, mode: str) -> None:
    """Render a sub-agent tool result at _IC indent."""
    console.print(f"{_IC}[dim cyan]←[/dim cyan]")
    _render_block_content(console, text, mode)


def _render_ai_message(console: Any, text: str, mode: str) -> None:
    """Render a sub-agent AI message at _IC indent."""
    lines = text.splitlines() or [""]
    console.print(f"{_IC}🐦 {lines[0]}")
    if len(lines) > 1:
        _render_block_content(console, "\n".join(lines[1:]), mode)


def _print_agent_transcript(
    console: Any,
    run_id: str,
    transcript: List[Tuple[str, Any]],
    mode: str,
) -> None:
    """Print the full buffered sub-agent transcript as one block.

    Prints a single header line at the tool-output indent level (_IH), then
    all tool calls, tool results, and AI messages at _IC indent.
    """
    console.print(f"{_IH}[dim]\\[{run_id}][/dim]")
    for kind, payload in transcript:
        if kind == "tc":
            _render_tool_call(console, payload, mode)
        elif kind == "tr":
            _render_tool_result(console, payload, mode)
        elif kind == "ai":
            _render_ai_message(console, payload, mode)
    console.print()


def agentdef_to_langchain_tool(
    agent_def: AgentDef,
    skills_dir: str,
    agents_dir: Optional[str] = None,
    fallback_vendor: Optional[str] = None,
    fallback_model: Optional[str] = None,
    console: Optional[Any] = None,
    get_tool_output_mode: Optional[Callable[[], str]] = None,
) -> StructuredTool:
    """Wrap an AgentDef as an async LangChain StructuredTool.

    Args:
        agent_def: The parsed AGENT.MD definition.
        skills_dir: Skills directory passed to the ephemeral DynamicAgent.
        agents_dir: Agents directory passed to the ephemeral DynamicAgent.
        fallback_vendor: Vendor to use if agent_def.vendor is unset.
        fallback_model: Model to use if agent_def.model is unset.
        console: Optional rich Console. When provided the sub-agent transcript
            is printed as a single block after the sub-agent completes.
        get_tool_output_mode: Callable returning the current output mode
            (``"off"``, ``"short"``, or ``"full"``). Called at invocation time
            so live mode changes take effect. Defaults to ``"short"``.

    Returns:
        An async StructuredTool the calling agent can invoke as a regular tool.
    """
    from ..agent.run import DynamicAgent

    vendor = agent_def.vendor or fallback_vendor
    model = agent_def.model or fallback_model

    async def _run(**kwargs: Any) -> str:
        config: dict = {}
        if vendor:
            config["vendor"] = vendor
        if model:
            config["model"] = model

        prompt = _substitute(agent_def.prompt, kwargs)

        sub_agent = DynamicAgent.from_config(
            provider_config=config or None,
            skills_dir=skills_dir,
            agents_dir=agents_dir,
        )
        sub_agent.enable_skills_for_session("_run", agent_def.allowed_skills)

        run_id = f"{agent_def.name}#{uuid.uuid4().hex[:4]}"
        invoke_config = {"recursion_limit": agent_def.recursion_limit}

        if console is None:
            # Silent path: run to completion and return the last message.
            result = await sub_agent.invoke(
                prompt, thread_id="_run", config=invoke_config,
            )
            last = result["messages"][-1]
            return _extract_text(last.content)

        # Streaming path: collect messages, then print as one block.
        mode = get_tool_output_mode() if get_tool_output_mode else "short"
        final_content = ""
        transcript: List[Tuple[str, Any]] = []

        async for update in sub_agent.astream(prompt, thread_id="_run", config=invoke_config):
            for _node, data in update.items():
                for msg in data.get("messages", []):
                    if isinstance(msg, AIMessage):
                        if getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                transcript.append(("tc", tc))
                        elif msg.content:
                            text = _extract_text(msg.content)
                            final_content = text
                            transcript.append(("ai", text))
                    elif isinstance(msg, ToolMessage):
                        transcript.append(("tr", _extract_text(msg.content)))

        if transcript:
            _print_agent_transcript(console, run_id, transcript, mode)

        return final_content

    from .adapter import create_args_schema
    schema = _input_schema(agent_def.input_params)

    return StructuredTool.from_function(
        coroutine=_run,
        name=agent_def.name,
        description=agent_def.description,
        args_schema=create_args_schema(schema),
    )
