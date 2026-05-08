"""
Example 07 – Long-Term Memory

Shows how to pass `long_term_memory` to `invoke()`.  The list of strings is
injected verbatim into the system prompt as "Tier 3" context — the agent reads
them as persistent background knowledge about the user on every turn.

In the birdie CLI this data is stored in ~/.birdie/sessions/<user>/memory.json
and loaded automatically.  Here we pass the list directly to keep the script
self-contained.

The example runs the same question twice — once without memory and once with —
so you can compare how the response changes.

Prerequisites
─────────────
    export LLM_VENDOR=anthropic
    export LLM_MODEL=claude-sonnet-4-6
    export ANTHROPIC_API_KEY=sk-ant-...

Run
───
    python examples/07_long_term_memory.py
"""

import asyncio
from pathlib import Path

from birdie.agent.run import DynamicAgent

SKILLS_DIR = Path(__file__).parent.parent / "birdie" / "skills"

# Facts that would normally be persisted in memory.json via the /remember command.
USER_MEMORY = [
    "Prefers concise answers — no preamble or unnecessary padding",
    "Is an expert Python developer; no need to explain basic syntax",
    "Working on a project called 'Luminary' — a distributed task queue backed by Redis Streams",
    "Prefers British English spelling (e.g. 'optimise', 'serialise', 'behaviour')",
]

QUESTION = (
    "I need to choose between Redis Pub/Sub and Redis Streams as the messaging "
    "layer for my project. What would you recommend and why?"
)


async def ask(agent: DynamicAgent, question: str, memory: list, thread_id: str) -> str:
    result = await agent.invoke(question, thread_id=thread_id, long_term_memory=memory)
    return result["messages"][-1].content


async def main() -> None:
    agent = DynamicAgent.from_config(skills_dir=str(SKILLS_DIR))

    print("=" * 60)
    print("WITHOUT long-term memory")
    print("=" * 60)
    print(f"User: {QUESTION}\n")
    without = await ask(agent, QUESTION, memory=[], thread_id="no-memory")
    print(f"Agent: {without}\n")

    print("=" * 60)
    print("WITH long-term memory")
    print("=" * 60)
    print("Memory injected into system prompt:")
    for entry in USER_MEMORY:
        print(f"  - {entry}")
    print(f"\nUser: {QUESTION}\n")
    with_mem = await ask(agent, QUESTION, memory=USER_MEMORY, thread_id="with-memory")
    print(f"Agent: {with_mem}")


if __name__ == "__main__":
    asyncio.run(main())
