"""
Minimal MCP demo server exposing two tools: echo and reverse_string.

Uses the low-level Server API so it runs under both mcp 1.x and mcp 2.x
(FastMCP left the mcp package in 2.0).  The file is self-contained - it is
spawned as a subprocess by the mcp_demo skill and must not import birdie.

Run directly to test outside of Birdie:
    python birdie/skills/mcp_demo/server.py
"""

import asyncio

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

_TOOLS = {
    "echo": {
        "description": "Return the message unchanged.",
        "schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        "fn": lambda message: message,
    },
    "reverse_string": {
        "description": "Return the text with characters in reverse order.",
        "schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "fn": lambda text: text[::-1],
    },
}


def _listing() -> list:
    return [
        types.Tool(name=name, description=t["description"], inputSchema=t["schema"])
        for name, t in _TOOLS.items()
    ]


def _execute(name: str, arguments: dict | None) -> str:
    if name not in _TOOLS:
        raise ValueError(f"Unknown tool: {name!r}")
    return str(_TOOLS[name]["fn"](**(arguments or {})))


def _build_server() -> Server:
    if hasattr(Server, "list_tools"):  # mcp 1.x: decorator-based handlers
        server = Server("mcp_demo")

        @server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return _listing()

        @server.call_tool()
        async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
            return [types.TextContent(type="text", text=_execute(name, arguments))]

        return server

    # mcp 2.x: handlers are constructor callbacks returning result models.
    async def on_list_tools(ctx, params):
        return types.ListToolsResult(tools=_listing())

    async def on_call_tool(ctx, params):
        try:
            text = _execute(params.name, params.arguments)
        except Exception as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(exc))],
                is_error=True,
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
        )

    return Server("mcp_demo", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def _run() -> None:
    server = _build_server()
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(_run())
