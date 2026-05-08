"""
Example 10 – MCP-Backed Skill

Demonstrates a skill whose tools come from a Model Context Protocol (MCP)
server process rather than from SKILL.MD entrypoints.  The agent spawns the
server as a subprocess, discovers its tools dynamically, and calls them.

The bundled mcp_demo server exposes two tools:
  • echo(message)         — returns the message unchanged
  • reverse_string(text)  — returns the characters in reverse order

This pattern is useful for integrating external capability providers: the
skill declaration only needs a connection config; tool schemas are discovered
at runtime from the server.

Prerequisites
─────────────
Install the MCP optional extra (adds `mcp` SDK and `langchain-mcp-adapters`):

    pip install -e ".[mcp]"

Then set your LLM credentials:

    export LLM_VENDOR=anthropic
    export LLM_MODEL=claude-sonnet-4-6
    export ANTHROPIC_API_KEY=sk-ant-...

    # Or use LLM_PROVIDER_CONFIG for a single JSON blob (overrides the above):
    export LLM_PROVIDER_CONFIG='{"vendor":"anthropic","model":"claude-sonnet-4-6","api_key":"sk-ant-..."}'

Run
───
    python examples/10_mcp_tool.py
"""

import asyncio
import sys
from pathlib import Path

from birdie.agent.run import DynamicAgent
from birdie.core.models import MCPServerConfig

SKILLS_DIR = Path(__file__).parent.parent / "birdie" / "skills"

# Absolute path to the bundled demo MCP server script.
MCP_SERVER = Path(__file__).parent.parent / "birdie" / "skills" / "mcp_demo" / "server.py"

SESSION_ID = "mcp-demo"


def _text(content) -> str:
    """Return message content as a plain string (handles list-of-blocks format)."""
    if isinstance(content, list):
        return "\n".join(b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in content)
    return str(content)


async def main() -> None:
    agent = DynamicAgent.from_config(skills_dir=str(SKILLS_DIR))

    # The bundled mcp_demo SKILL.MD uses the relative path "server.py" which
    # would only work from inside the mcp_demo directory.  Re-register the
    # server with the absolute path so it works from any working directory.
    agent.mcp_manager.register_server(
        "mcp_demo",
        MCPServerConfig(
            transport="stdio",
            command=sys.executable,
            args=[str(MCP_SERVER)],
        ),
    )

    agent.enable_skill(SESSION_ID, "mcp_demo")

    # ── Discover tools from the MCP server ───────────────────────────────────
    print("=== Discovering MCP tools ===")
    try:
        mcp_tools = await agent.mcp_manager.get_tools(allowed={"mcp_demo"})
    except ImportError as exc:
        print(f"\nMCP extra not installed: {exc}")
        print("Run:  pip install -e \".[mcp]\"")
        return

    for tool in mcp_tools:
        print(f"  {tool.name}: {tool.description}")
    print()

    # ── Use the MCP tools via the agent ──────────────────────────────────────
    queries = [
        "Use the echo tool to send the message 'birdie-agent works with MCP!'",
        "Reverse the string 'Hello, World!' using the reverse_string tool.",
    ]

    for q in queries:
        print(f"User:  {q}")
        result = await agent.invoke(q, thread_id=SESSION_ID)

        for msg in result["messages"][1:]:
            kind = type(msg).__name__
            if kind == "AIMessage" and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    print(f"  → [MCP] {tc['name']}({tc['args']})")
            elif kind == "ToolMessage":
                print(f"  ← [MCP result] {_text(msg.content).strip()}")
            elif kind == "AIMessage" and msg.content:
                print(f"Agent: {_text(msg.content)}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
