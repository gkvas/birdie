"""
Example 04 – Shell Commands

Enables the Shell skill and demonstrates the agent executing local shell
commands to answer questions about the environment.

The Shell skill exposes a single `run_bash` tool whose entrypoint is
`bash:{command}` — the LLM decides what command to run, and the framework
executes it and returns stdout.

⚠️  Security notice: the Shell skill can run ANY command the current user
    is permitted to run.  Only enable it in environments you trust.

Prerequisites
─────────────
    export LLM_VENDOR=anthropic
    export LLM_MODEL=claude-sonnet-4-6
    export ANTHROPIC_API_KEY=sk-ant-...

    # Or use LLM_PROVIDER_CONFIG for a single JSON blob (overrides the above):
    export LLM_PROVIDER_CONFIG='{"vendor":"anthropic","model":"claude-sonnet-4-6","api_key":"sk-ant-..."}'

Run
───
    python examples/04_shell_commands.py
"""

import asyncio
from pathlib import Path

from birdie.agent.run import DynamicAgent

SKILLS_DIR = Path(__file__).parent.parent / "birdie" / "skills"
SESSION_ID = "shell-demo"


async def ask(agent: DynamicAgent, question: str) -> None:
    """Send one question and print the agent's reply plus any tool calls made."""
    print(f"User: {question}")
    result = await agent.invoke(question, thread_id=SESSION_ID)

    for msg in result["messages"][1:]:
        kind = type(msg).__name__
        if kind == "AIMessage" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"  → run_bash({tc['args'].get('command', '')})")
        elif kind == "ToolMessage":
            preview = msg.content.strip()[:200]
            print(f"  ← {preview}")
        elif kind == "AIMessage" and msg.content:
            print(f"Agent: {msg.content}")
    print()


async def main() -> None:
    agent = DynamicAgent.from_config(skills_dir=str(SKILLS_DIR))
    agent.enable_skill(SESSION_ID, "Shell")

    # Three independent questions — each is a fresh invoke() call on the same
    # thread, so the agent accumulates context across them.
    await ask(agent, "What operating system and kernel version am I running?")
    await ask(agent, "How many Python files are in the current directory tree?")
    await ask(agent, "What is the total disk usage of the current directory?")


if __name__ == "__main__":
    asyncio.run(main())
