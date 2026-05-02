"""
Minimal MCP server for testing - exposes two tools via stdio transport.

Run directly to test outside of Birdie:
    python birdie/skills/mcp_demo/server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp_demo")


@mcp.tool()
def echo(message: str) -> str:
    """Return the message unchanged."""
    return message


@mcp.tool()
def reverse_string(text: str) -> str:
    """Return the text with characters in reverse order."""
    return text[::-1]


if __name__ == "__main__":
    mcp.run(transport="stdio")
