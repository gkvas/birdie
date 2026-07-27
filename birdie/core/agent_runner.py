"""
Agent runner: converts an AgentDef into an executable async LangChain tool.

At call time:
  1. Input values are substituted into the prompt template ({{ param }} syntax).
  2. An ephemeral DynamicAgent is created with the agent's vendor/model config.
  3. The agent runs with the allowed_skills fixed-list; no session is persisted.
  4. The final AIMessage content is returned as the tool result.
"""

import json
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


def render_agent_prompt(agent_def: AgentDef, params: Dict[str, Any]) -> str:
    """Render the final sub-agent prompt from an AgentDef and input params.

    Substitutes ``{{ param }}`` placeholders and, when the agent declares
    ``output_params``, appends explicit output-format instructions so the
    declared output schema actually shapes the reply.
    """
    prompt = _substitute(agent_def.prompt, params)
    if agent_def.output_params:
        lines = [
            "",
            "Return your final answer as a single JSON object with exactly "
            "these fields:",
        ]
        for p in agent_def.output_params:
            desc = f": {p.description}" if p.description else ""
            lines.append(f'- "{p.name}" ({p.type}){desc}')
        lines.append("Output only the JSON object, no other text.")
        prompt += "\n".join(lines)
    return prompt


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


_JSON_TYPE_CHECKS: Dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def parse_agent_output(
    agent_def: AgentDef, text: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Validate a sub-agent reply against its declared ``output_params``.

    Extracts a JSON object from *text* (tolerating surrounding prose), checks
    that every required declared field is present with the declared type, and
    returns ``(canonical_json, None)`` on success or ``(None, error)`` with a
    human-readable error the sub-agent can act on.
    """
    stripped = text.strip()
    data: Any = None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', stripped, re.DOTALL)
        if m is None:
            return None, "the reply contains no JSON object"
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None, "the reply is not valid JSON"

    if not isinstance(data, dict):
        return None, "the top-level JSON value must be an object"

    errors: List[str] = []
    for p in agent_def.output_params:
        if p.name not in data:
            if p.required:
                errors.append(f"missing required field '{p.name}'")
            continue
        expected = _JSON_TYPE_CHECKS.get(p.type)
        if expected is not None and not isinstance(data[p.name], expected):
            errors.append(
                f"field '{p.name}' must be of type {p.type}, "
                f"got {type(data[p.name]).__name__}"
            )
    if errors:
        return None, "; ".join(errors)
    return json.dumps(data, ensure_ascii=False), None


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
    fallback_provider_config: Optional[Dict[str, Any]] = None,
    console: Optional[Any] = None,
    get_tool_output_mode: Optional[Callable[[], str]] = None,
) -> StructuredTool:
    """Wrap an AgentDef as an async LangChain StructuredTool.

    Args:
        agent_def: The parsed AGENT.MD definition.
        skills_dir: Skills directory passed to the ephemeral DynamicAgent.
        agents_dir: Agents directory passed to the ephemeral DynamicAgent.
        fallback_provider_config: Full provider config dict from the parent agent
            (vendor, api_key, base_url, etc.). AGENT.MD may only override model,
            temperature, and max_tokens.
        console: Optional rich Console. When provided the sub-agent transcript
            is printed as a single block after the sub-agent completes.
        get_tool_output_mode: Callable returning the current output mode
            (``"off"``, ``"short"``, or ``"full"``). Called at invocation time
            so live mode changes take effect. Defaults to ``"short"``.

    Returns:
        An async StructuredTool the calling agent can invoke as a regular tool.
    """
    from ..agent.run import DynamicAgent

    # Inherit the full parent provider config; AGENT.MD may only override
    # model, temperature, and max_tokens - never vendor or api_key.
    config = dict(fallback_provider_config) if fallback_provider_config else {}

    if agent_def.model:
        config["model"] = agent_def.model
    if agent_def.temperature is not None:
        config["temperature"] = agent_def.temperature
    if agent_def.max_tokens is not None:
        config["max_tokens"] = agent_def.max_tokens

    if agent_def.vendor and config.get("vendor") and agent_def.vendor != config["vendor"]:
        raise ValueError(
            f"Vendor cannot be overridden in AGENT.MD. "
            f"Parent vendor: {config['vendor']}, AGENT.MD vendor: {agent_def.vendor}"
        )

    # The DynamicAgent is built once per tool and reused across invocations -
    # construction re-discovers and re-parses every SKILL.MD/AGENT.MD on disk,
    # which is pure overhead per call.  Each invocation gets a fresh thread_id
    # so histories never bleed between runs.
    _cache: Dict[str, Any] = {}

    def _get_sub_agent():
        if "agent" not in _cache:
            _cache["agent"] = DynamicAgent.from_config(
                provider_config=config or None,
                skills_dir=skills_dir,
                agents_dir=agents_dir,
            )
        return _cache["agent"]

    async def _run(**kwargs: Any) -> str:

        prompt = render_agent_prompt(agent_def, kwargs)

        sub_agent = _get_sub_agent()
        run_id = f"{agent_def.name}#{uuid.uuid4().hex[:4]}"
        thread = f"_run-{run_id}"
        sub_agent.enable_skills_for_session(thread, agent_def.allowed_skills)

        invoke_config = {
            "recursion_limit": agent_def.recursion_limit,
            "configurable": {"max_tool_repetitions": agent_def.max_tool_repetitions},
        }

        async def _finalize(text: str) -> str:
            """Validate against output_params, retrying once on mismatch."""
            if not agent_def.output_params:
                return text
            cleaned, err = parse_agent_output(agent_def, text)
            if err is None:
                return cleaned
            retry = await sub_agent.invoke(
                f"Your previous reply was invalid: {err}. Return ONLY a "
                "single JSON object with exactly the required fields.",
                thread_id=thread, config=invoke_config,
            )
            text = _extract_text(retry["messages"][-1].content)
            cleaned, err = parse_agent_output(agent_def, text)
            return cleaned if err is None else text

        if console is None:
            # Silent path: run to completion and return the last message.
            result = await sub_agent.invoke(
                prompt, thread_id=thread, config=invoke_config,
            )
            last = result["messages"][-1]
            return await _finalize(_extract_text(last.content))

        # Streaming path: collect messages, then print as one block.
        mode = get_tool_output_mode() if get_tool_output_mode else "off"
        final_content = ""
        transcript: List[Tuple[str, Any]] = []

        async for update in sub_agent.astream(prompt, thread_id=thread, config=invoke_config):
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

        if transcript and mode != "off":
            _print_agent_transcript(console, run_id, transcript, mode)

        return await _finalize(final_content)

    from .adapter import create_args_schema
    schema = _input_schema(agent_def.input_params)

    return StructuredTool.from_function(
        coroutine=_run,
        name=agent_def.name,
        description=agent_def.description,
        args_schema=create_args_schema(schema),
    )
