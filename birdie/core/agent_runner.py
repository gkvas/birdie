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
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from .models import AgentDef, AgentParam


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


def _args_preview(args: Dict[str, Any], max_val: int = 60) -> str:
    """Format tool-call args dict for one-line display, truncating long values."""
    parts = []
    for k, v in args.items():
        s = repr(v)
        if len(s) > max_val:
            s = s[:max_val] + "…'"
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def agentdef_to_langchain_tool(
    agent_def: AgentDef,
    skills_dir: str,
    agents_dir: Optional[str] = None,
    fallback_vendor: Optional[str] = None,
    fallback_model: Optional[str] = None,
    console: Optional[Any] = None,
) -> StructuredTool:
    """Wrap an AgentDef as an async LangChain StructuredTool.

    Args:
        agent_def: The parsed AGENT.MD definition.
        skills_dir: Skills directory passed to the ephemeral DynamicAgent.
        agents_dir: Agents directory passed to the ephemeral DynamicAgent.
        fallback_vendor: Vendor to use if agent_def.vendor is unset.
        fallback_model: Model to use if agent_def.model is unset.
        console: Optional rich Console. When provided the sub-agent streams its
            work to the console with a unique ``[Name#xxxx]`` prefix so multiple
            agents can be distinguished.

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

        # Streaming path: print each node update with the run_id prefix.
        prefix = f"[dim]\\[{run_id}][/dim]"
        final_content = ""
        async for update in sub_agent.astream(prompt, thread_id="_run", config=invoke_config):
            for _node, data in update.items():
                for msg in data.get("messages", []):
                    if isinstance(msg, AIMessage):
                        if getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                preview = _args_preview(tc["args"])
                                console.print(
                                    f"{prefix} 🐦 [bold]{tc['name']}[/bold]({preview})"
                                )
                        elif msg.content:
                            text = _extract_text(msg.content)
                            final_content = text
                            lines = text.splitlines()
                            if lines:
                                console.print(f"{prefix} 🐦 {lines[0]}")
                            for line in lines[1:]:
                                console.print(f"{prefix}    {line}")
                    elif isinstance(msg, ToolMessage):
                        text = _extract_text(msg.content)
                        first = text.splitlines()[0][:200] if text else ""
                        console.print(f"{prefix}    {first}")

        return final_content

    from .adapter import create_args_schema
    schema = _input_schema(agent_def.input_params)

    return StructuredTool.from_function(
        coroutine=_run,
        name=agent_def.name,
        description=agent_def.description,
        args_schema=create_args_schema(schema),
    )
