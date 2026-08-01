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

Compatible with both mcp 1.x (decorator-based Server handlers) and
mcp 2.x (constructor-callback handlers).
"""

import asyncio
import json
import os
import sys

try:
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import Server
except ImportError:  # pragma: no cover - exercised only in bare installs
    if __name__ == "__main__":
        print(
            "birdie: MCP support is not installed. "
            "Install the optional extra: pip install 'birdie-agent[mcp]'",
            file=sys.stderr,
        )
        sys.exit(1)
    raise

from birdie.core.entrypoints import resolve_entrypoint


# True when running against mcp 1.x, whose Server exposes decorator-based
# handler registration.  mcp 2.0 moved handlers to constructor callbacks.
_MCP_V1 = hasattr(Server, "list_tools")


def _tool_listing(tool_defs: list[dict], agent_defs: list[dict]) -> list:
    """Build the types.Tool listing for all skill tools and agents.

    ``inputSchema`` is the 1.x field name and the 2.x alias, so one spelling
    works on both versions.
    """
    return [
        types.Tool(
            name=entry["name"],
            description=entry["description"],
            inputSchema=entry.get("parameters", {"type": "object", "properties": {}}),
        )
        for entry in [*tool_defs, *agent_defs]
    ]


async def _execute_tool(
    name: str,
    arguments: dict,
    tool_defs: list[dict],
    agent_defs: list[dict],
) -> str:
    """Run a skill tool or agent by name and return its text result."""
    # --- skill tool path ---
    tool = next((t for t in tool_defs if t["name"] == name), None)
    if tool is not None:
        entrypoint = tool["entrypoint"]
        resolver = resolve_entrypoint(entrypoint)
        result = await asyncio.to_thread(resolver, entrypoint, **arguments)
        return str(result)

    # --- agent tool path ---
    agent_raw = next((a for a in agent_defs if a["name"] == name), None)
    if agent_raw is not None:
        return str(await _invoke_agent(agent_raw, arguments))

    raise ValueError(f"Unknown tool: {name!r}")


def _build_server(tool_defs: list[dict], agent_defs: list[dict]) -> Server:
    if _MCP_V1:
        server = Server("birdie-tools")

        @server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return _tool_listing(tool_defs, agent_defs)

        @server.call_tool()
        async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
            text = await _execute_tool(name, arguments or {}, tool_defs, agent_defs)
            return [types.TextContent(type="text", text=text)]

        return server

    # mcp 2.x: handlers are constructor callbacks that receive a request
    # context + params object and return result models directly.
    async def on_list_tools(ctx, params):
        return types.ListToolsResult(tools=_tool_listing(tool_defs, agent_defs))

    async def on_call_tool(ctx, params):
        try:
            text = await _execute_tool(
                params.name, params.arguments or {}, tool_defs, agent_defs,
            )
        except Exception as exc:
            # Mirror the 1.x framework behaviour of turning handler
            # exceptions into an error result the model can read.
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(exc))],
                is_error=True,
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
        )

    return Server(
        "birdie-tools",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


# Sub-agent instances reused across call_tool invocations (construction
# re-parses all skills/agents from disk).  Keyed by agent name; each run uses
# a fresh thread_id so histories never bleed between calls.
_agent_cache: dict = {}


async def _invoke_agent(agent_raw: dict, arguments: dict) -> str:
    """Run the agent prompt on a cached ephemeral DynamicAgent."""
    # Import here to avoid circular imports at module load time.
    import uuid

    from birdie.core.models import AgentDef
    from birdie.agent.run import DynamicAgent

    # Reconstruct the AgentDef from the serialised dict stored in BIRDIE_AGENTS_JSON.
    agent_def = AgentDef.model_validate(agent_raw["_agent_def"])

    # Provider config is forwarded so the sub-agent uses the same vendor/model
    # as the parent (unless the AGENT.MD overrides model/temperature/max_tokens).
    provider_config: dict = agent_raw.get("_provider_config") or {}
    skills_dir: str = agent_raw.get("_skills_dir", "skills")
    agents_dir: str | None = agent_raw.get("_agents_dir")

    # Build the prompt: {{ param }} substitution + output-format instructions.
    from birdie.core.agent_runner import render_agent_prompt

    prompt = render_agent_prompt(agent_def, arguments)

    sub_agent = _agent_cache.get(agent_def.name)
    if sub_agent is None:
        sub_agent = DynamicAgent.from_config(
            provider_config=provider_config or None,
            skills_dir=skills_dir,
            agents_dir=agents_dir,
        )
        _agent_cache[agent_def.name] = sub_agent

    thread = f"_acp_run-{uuid.uuid4().hex[:8]}"
    sub_agent.enable_skills_for_session(thread, agent_def.allowed_skills)

    invoke_config = {
        "recursion_limit": agent_def.recursion_limit,
        "configurable": {"max_tool_repetitions": agent_def.max_tool_repetitions},
    }
    result = await sub_agent.invoke(prompt, thread_id=thread, config=invoke_config)

    def _text(content) -> str:
        if isinstance(content, list):
            return "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        return str(content)

    text = _text(result["messages"][-1].content)

    # Validate declared structured output, retrying once on mismatch.
    from birdie.core.agent_runner import parse_agent_output

    if agent_def.output_params:
        cleaned, err = parse_agent_output(agent_def, text)
        if err is not None:
            retry = await sub_agent.invoke(
                f"Your previous reply was invalid: {err}. Return ONLY a "
                "single JSON object with exactly the required fields.",
                thread_id=thread, config=invoke_config,
            )
            text = _text(retry["messages"][-1].content)
            cleaned, err = parse_agent_output(agent_def, text)
        if err is None:
            return cleaned
    return text


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
