"""
Standalone MCP server that exposes Birdie skill tools over stdio.

ACPProvider spawns this as a subprocess and passes its config via the
mcpServers field of session/new.  The ACP agent (e.g. claude-agent-acp)
then connects to it via the MCP stdio transport and can call Birdie's
skill tools directly.

Tool definitions (serialised NormalizedToolDef dicts that include an
``entrypoint`` key) are read from the BIRDIE_TOOLS_JSON environment
variable set by ACPProvider.
"""

import asyncio
import json
import os

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from birdie.core.entrypoints import resolve_entrypoint


def _build_server(tool_defs: list[dict]) -> Server:
    server = Server("birdie-tools")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t.get("parameters", {"type": "object", "properties": {}}),
            )
            for t in tool_defs
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        tool = next((t for t in tool_defs if t["name"] == name), None)
        if tool is None:
            raise ValueError(f"Unknown tool: {name!r}")
        entrypoint = tool["entrypoint"]
        resolver = resolve_entrypoint(entrypoint)
        result = await asyncio.to_thread(resolver, entrypoint, **(arguments or {}))
        return [types.TextContent(type="text", text=str(result))]

    return server


async def _run() -> None:
    tool_defs = json.loads(os.environ.get("BIRDIE_TOOLS_JSON", "[]"))
    server = _build_server(tool_defs)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(_run())
