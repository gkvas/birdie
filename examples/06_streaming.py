"""
Example 06 – Streaming Output

Uses `agent.astream()` to receive LangGraph node-level updates as they are
produced, rather than waiting for the full turn to finish.

Each update yielded by `astream()` is a dict keyed by node name:
  • "agent" node  — the LLM's reply (may contain tool_calls)
  • "tools" node  — tool execution results

This is the same pattern used by the birdie CLI to display tool calls and
responses in real time.

Prerequisites
─────────────
    export LLM_VENDOR=anthropic
    export LLM_MODEL=claude-sonnet-4-6
    export ANTHROPIC_API_KEY=sk-ant-...

    # Or use LLM_PROVIDER_CONFIG for a single JSON blob (overrides the above):
    export LLM_PROVIDER_CONFIG='{"vendor":"anthropic","model":"claude-sonnet-4-6","api_key":"sk-ant-..."}'

Run
───
    python examples/06_streaming.py
"""

import asyncio
from pathlib import Path

from birdie.agent.run import DynamicAgent

SKILLS_DIR = Path(__file__).parent.parent / "birdie" / "skills"
SESSION_ID = "streaming-demo"


async def main() -> None:
    agent = DynamicAgent.from_config(skills_dir=str(SKILLS_DIR))
    agent.enable_skill_for_user(SESSION_ID, "Shell")

    message = "List the Python files in the current directory, then tell me how many there are."
    print(f"User: {message}\n")

    async for update in agent.astream(message, thread_id=SESSION_ID):
        # update is e.g. {"agent": {"messages": [AIMessage(...)]}}
        #            or  {"tools": {"messages": [ToolMessage(...)]}}
        node_name = next(iter(update))
        new_messages = update[node_name]["messages"]

        for msg in new_messages:
            kind = type(msg).__name__

            if kind == "AIMessage" and getattr(msg, "tool_calls", None):
                # LLM decided to call one or more tools
                for tc in msg.tool_calls:
                    print(f"  → [tool call — {node_name}] {tc['name']}({tc['args']})")

            elif kind == "ToolMessage":
                # Tool ran and returned a result
                preview = msg.content.strip()[:300]
                suffix = f"\n     … ({len(msg.content) - 300} more chars)" if len(msg.content) > 300 else ""
                print(f"  ← [tool result — {node_name}]\n{preview}{suffix}")

            elif kind == "AIMessage" and msg.content:
                # Final text response from the LLM
                print(f"\nAgent [{node_name}]: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
