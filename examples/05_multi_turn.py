"""
Example 05 – Multi-Turn Conversation

Demonstrates that the same `thread_id` accumulates history across multiple
`invoke()` calls.  The agent remembers what was said in earlier turns and
uses that context in later replies.

Under the hood, the default `MemorySaver` checkpointer keeps history in
process memory.  See example 08_sqlite_persistence.py for durable history
that survives restarts.

Prerequisites
─────────────
    export LLM_VENDOR=anthropic
    export LLM_MODEL=claude-sonnet-4-6
    export ANTHROPIC_API_KEY=sk-ant-...

    # Or use LLM_PROVIDER_CONFIG for a single JSON blob (overrides the above):
    export LLM_PROVIDER_CONFIG='{"vendor":"anthropic","model":"claude-sonnet-4-6","api_key":"sk-ant-..."}'

Run
───
    python examples/05_multi_turn.py
"""

import asyncio
from pathlib import Path

from birdie.agent.run import DynamicAgent

SKILLS_DIR = Path(__file__).parent.parent / "birdie" / "skills"
THREAD_ID = "multi-turn-demo"


async def chat(agent: DynamicAgent, message: str) -> str:
    """Send a message on THREAD_ID and return the agent's reply."""
    result = await agent.invoke(message, thread_id=THREAD_ID)
    return result["messages"][-1].content


async def main() -> None:
    agent = DynamicAgent.from_config(skills_dir=str(SKILLS_DIR))

    print(f"Thread: {THREAD_ID}\n")

    # Turn 1 & 2: establish facts
    turns = [
        "My name is Alice and I'm a Python developer.",
        "I'm building a distributed task queue called Luminary that uses Redis Streams.",
        # Turn 3: test whether the agent recalls earlier context
        "What is my name and what am I building?",
        # Turn 4: follow-up that requires prior context to answer well
        "What Redis Streams feature should I explore first for Luminary?",
    ]

    for message in turns:
        print(f"User:  {message}")
        reply = await chat(agent, message)
        print(f"Agent: {reply}\n")

    # Inspect the history accumulated in the in-memory checkpointer.
    state = await agent.app.aget_state({"configurable": {"thread_id": THREAD_ID}})
    messages = state.values.get("messages", []) if state.values else []
    print(f"─── conversation history: {len(messages)} messages stored in checkpointer ───")


if __name__ == "__main__":
    asyncio.run(main())
