"""
Example 03 – Web Search with DuckDuckGo

Enables the built-in DuckDuckGo skill and asks the agent to search the web.
The DuckDuckGo skill requires no extra API key — it uses the `ddgs` library
which ships as a core dependency of birdie-agent.

The example also shows how to inspect the message history to see exactly
which tools were called and what they returned.

Prerequisites
─────────────
    export LLM_VENDOR=anthropic
    export LLM_MODEL=claude-sonnet-4-6
    export ANTHROPIC_API_KEY=sk-ant-...

    # Or use LLM_PROVIDER_CONFIG for a single JSON blob (overrides the above):
    export LLM_PROVIDER_CONFIG='{"vendor":"anthropic","model":"claude-sonnet-4-6","api_key":"sk-ant-..."}'

Run
───
    python examples/03_web_search.py
"""

import asyncio
from pathlib import Path

from birdie.agent.run import DynamicAgent

SKILLS_DIR = Path(__file__).parent.parent / "birdie" / "skills"
SESSION_ID = "web-search-demo"


def print_trace(messages) -> None:
    """Walk the message list and pretty-print what the agent did."""
    for msg in messages[1:]:   # skip the opening HumanMessage
        kind = type(msg).__name__
        if kind == "AIMessage" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"  → [tool call] {tc['name']}(query={tc['args'].get('query', tc['args'])})")
        elif kind == "ToolMessage":
            preview = msg.content[:300]
            if len(msg.content) > 300:
                preview += f"\n  … ({len(msg.content) - 300} more chars)"
            print(f"  ← [tool result]\n{preview}")
        elif kind == "AIMessage" and msg.content:
            print(f"\nAgent: {msg.content}")


async def main() -> None:
    agent = DynamicAgent.from_config(skills_dir=str(SKILLS_DIR))

    # Grant the DuckDuckGo skill to this session.
    agent.enable_skill(SESSION_ID, "DuckDuckGo")

    enabled = sorted(agent.policy.get_allowed_skills(session_id=SESSION_ID))
    print(f"Enabled skills: {enabled}\n")

    query = "What is LangGraph and what problems does it solve for AI agent development?"
    print(f"User: {query}\n")

    result = await agent.invoke(query, thread_id=SESSION_ID)
    print_trace(result["messages"])


if __name__ == "__main__":
    asyncio.run(main())
