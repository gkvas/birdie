"""
Standalone MCP server that exposes Birdie skill tools and agents over stdio.

ACPProvider spawns this as a subprocess and passes its config via the
mcpServers field of session/new.  The ACP agent (e.g. claude-agent-acp)
then connects to it via the MCP stdio transport and can call Birdie's
skill tools and sub-agents directly.

Tool definitions (serialised NormalizedToolDef dicts that include an
``entrypoint`` key) are read from the BIRDIE_TOOLS_JSON environment
variable set by ACPProvider.

Agent definitions (serialised AgentDef dicts) are read from the
BIRDIE_AGENTS_JSON environment variable set by ACPProvider.  Each agent
is exposed as an MCP tool whose execution spins up an ephemeral
DynamicAgent in-process.
"""

import asyncio
import json
import os

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from birdie.core.entrypoints import resolve_entrypoint


def _build_server(tool_defs: list[dict], agent_defs: list[dict]) -> Server:
    server = Server("birdie-tools")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        tools = [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t.get("parameters", {"type": "object", "properties": {}}),
            )
            for t in tool_defs
        ]
        for a in agent_defs:
            tools.append(
                types.Tool(
                    name=a["name"],
                    description=a["description"],
                    inputSchema=a.get("parameters", {"type": "object", "properties": {}}),
                )
            )
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        # --- skill tool path ---
        tool = next((t for t in tool_defs if t["name"] == name), None)
        if tool is not None:
            entrypoint = tool["entrypoint"]
            resolver = resolve_entrypoint(entrypoint)
            result = await asyncio.to_thread(resolver, entrypoint, **(arguments or {}))
            return [types.TextContent(type="text", text=str(result))]

        # --- agent tool path ---
        agent_raw = next((a for a in agent_defs if a["name"] == name), None)
        if agent_raw is not None:
            result = await _invoke_agent(agent_raw, arguments or {})
            return [types.TextContent(type="text", text=str(result))]

        raise ValueError(f"Unknown tool: {name!r}")

    return server


async def _invoke_agent(agent_raw: dict, arguments: dict) -> str:
    """Spin up an ephemeral DynamicAgent and run the agent prompt."""
    # Import here to avoid circular imports at module load time.
    from birdie.core.models import AgentDef, AgentParam
    from birdie.agent.run import DynamicAgent

    # Reconstruct the AgentDef from the serialised dict stored in BIRDIE_AGENTS_JSON.
    agent_def = AgentDef.model_validate(agent_raw["_agent_def"])

    # Provider config is forwarded so the sub-agent uses the same vendor/model
    # as the parent (unless the AGENT.MD overrides model/temperature/max_tokens).
    provider_config: dict = agent_raw.get("_provider_config") or {}
    skills_dir: str = agent_raw.get("_skills_dir", "skills")
    agents_dir: str | None = agent_raw.get("_agents_dir")

    # Build the prompt by substituting {{ param }} placeholders.
    import re

    def _substitute(template: str, params: dict) -> str:
        def replace(m: re.Match) -> str:
            key = m.group(1).strip()
            return str(params.get(key, m.group(0)))
        return re.sub(r'\{\{\s*(\w+)\s*\}\}', replace, template)

    prompt = _substitute(agent_def.prompt, arguments)

    sub_agent = DynamicAgent.from_config(
        provider_config=provider_config or None,
        skills_dir=skills_dir,
        agents_dir=agents_dir,
    )
    sub_agent.enable_skills_for_session("_acp_run", agent_def.allowed_skills)

    invoke_config = {
        "recursion_limit": agent_def.recursion_limit,
        "configurable": {"max_tool_repetitions": agent_def.max_tool_repetitions},
    }
    result = await sub_agent.invoke(prompt, thread_id="_acp_run", config=invoke_config)
    last = result["messages"][-1]
    content = last.content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


async def _run() -> None:
    tool_defs = json.loads(os.environ.get("BIRDIE_TOOLS_JSON", "[]"))
    agent_defs = json.loads(os.environ.get("BIRDIE_AGENTS_JSON", "[]"))
    server = _build_server(tool_defs, agent_defs)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(_run())
