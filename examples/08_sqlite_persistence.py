"""
Example 08 – SQLite Persistence

Demonstrates durable session history using `AsyncSqliteSaver` as the
LangGraph checkpointer.  Conversation history is written to a SQLite database
after every turn and reloaded on the next run — the agent picks up exactly
where it left off.

Run this script twice to observe persistence:
  • 1st run: the agent has no prior history; introduce yourself
  • 2nd run: the agent recalls the previous conversation

The database is stored at examples/persistence_demo.db.  Delete it to reset.

Prerequisites
─────────────
    export LLM_VENDOR=anthropic
    export LLM_MODEL=claude-sonnet-4-6
    export ANTHROPIC_API_KEY=sk-ant-...

    # Or use LLM_PROVIDER_CONFIG for a single JSON blob (overrides the above):
    export LLM_PROVIDER_CONFIG='{"vendor":"anthropic","model":"claude-sonnet-4-6","api_key":"sk-ant-..."}'

Run
───
    python examples/08_sqlite_persistence.py
    python examples/08_sqlite_persistence.py   # run a second time to see recall
"""

import asyncio
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from birdie.agent.run import DynamicAgent

SKILLS_DIR = Path(__file__).parent.parent / "birdie" / "skills"

# The database file persists between runs.  Delete it to start fresh.
DB_PATH = Path(__file__).parent / "persistence_demo.db"
THREAD_ID = "persistent-session"


async def main() -> None:
    print(f"Database : {DB_PATH}")
    print(f"Thread   : {THREAD_ID}\n")

    # AsyncSqliteSaver is used as an async context manager.
    # The database is created automatically if it does not exist yet.
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        agent = DynamicAgent.from_config(
            skills_dir=str(SKILLS_DIR),
            checkpointer=checkpointer,
        )

        # Check whether this thread already has history in the database.
        state = await agent.app.aget_state({"configurable": {"thread_id": THREAD_ID}})
        prior_messages = state.values.get("messages", []) if state.values else []

        if not prior_messages:
            print("=== First run — no prior history found ===\n")
            messages_to_send = [
                "Hi! My name is Alex. I'm building a recommendation engine in Python.",
                "I'm using matrix factorisation with Alternating Least Squares (ALS). Remember this.",
            ]
        else:
            print(f"=== Subsequent run — {len(prior_messages)} messages already in the database ===\n")
            messages_to_send = [
                "Do you remember who I am and what algorithm I mentioned?",
                "What are the main hyperparameters I should tune for ALS?",
            ]

        for message in messages_to_send:
            print(f"User:  {message}")
            result = await agent.invoke(message, thread_id=THREAD_ID)
            print(f"Agent: {result['messages'][-1].content}\n")

    # The 'async with' block has now closed the checkpointer.
    # All history is safely written to DB_PATH.
    print(f"History written to {DB_PATH}")
    print("Run the script again to continue the conversation from where it left off.")


if __name__ == "__main__":
    asyncio.run(main())
