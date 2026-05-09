"""
Agent runner: converts an AgentDef into an executable async LangChain tool.

At call time:
  1. Input values are substituted into the prompt template ({{ param }} syntax).
  2. An ephemeral DynamicAgent is created with the agent's vendor/model config.
  3. The agent runs with the allowed_skills fixed-list; no session is persisted.
  4. The final AIMessage content is returned as the tool result.
"""

import re
from typing import Any, Dict, List, Optional

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


def agentdef_to_langchain_tool(
    agent_def: AgentDef,
    skills_dir: str,
    agents_dir: Optional[str] = None,
    fallback_vendor: Optional[str] = None,
    fallback_model: Optional[str] = None,
) -> StructuredTool:
    """Wrap an AgentDef as an async LangChain StructuredTool.

    Args:
        agent_def: The parsed AGENTS.MD definition.
        skills_dir: Skills directory passed to the ephemeral DynamicAgent.
        agents_dir: Agents directory passed to the ephemeral DynamicAgent.
        fallback_vendor: Vendor to use if agent_def.vendor is unset.
        fallback_model: Model to use if agent_def.model is unset.

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

        result = await sub_agent.invoke(prompt, thread_id="_run")
        last = result["messages"][-1]
        content = last.content
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        return str(content)

    from .adapter import create_args_schema
    schema = _input_schema(agent_def.input_params)

    return StructuredTool.from_function(
        coroutine=_run,
        name=agent_def.name,
        description=agent_def.description,
        args_schema=create_args_schema(schema),
    )
