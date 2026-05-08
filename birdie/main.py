"""
Demonstration entrypoint for the Birdie dynamic skill system.

Runs three hardcoded example interactions that exercise default skills, per-user
skill grants, and session-scoped grants.  For interactive use, see ``cli.py``.

Vendor selection via environment variables::

    LLM_VENDOR=openai    LLM_MODEL=gpt-4o                 python -m birdie.main
    LLM_VENDOR=anthropic LLM_MODEL=claude-sonnet-4-6       python -m birdie.main
    LLM_VENDOR=mistral   LLM_MODEL=mistral-large-latest    python -m birdie.main
    LLM_VENDOR=gemini    LLM_MODEL=gemini-2.0-flash        python -m birdie.main
"""

import asyncio
import os

from birdie.agent.run import DynamicAgent

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


def build_agent() -> DynamicAgent:
    """Construct a DynamicAgent from ``LLM_VENDOR`` / ``LLM_MODEL`` env vars."""
    config = {
        "vendor": os.environ.get("LLM_VENDOR", "openai"),
        "model": os.environ.get("LLM_MODEL", "gpt-4o"),
    }
    return DynamicAgent.from_config(config, skills_dir=SKILLS_DIR)


def main() -> None:
    agent = build_agent()
    vendor = os.environ.get("LLM_VENDOR", "openai")
    print(f"Provider: {vendor}  |  Skills loaded: {[s.name for s in agent.registry.list_skills()]}\n")

    print("=== Example 1: Basic Weather Query (default skills) ===")
    asyncio.run(run_example(agent, "What's the weather in Paris?"))

    print("\n=== Example 2: Session with Filesystem Access ===")
    agent.enable_skill("session123", "Filesystem")
    asyncio.run(run_example(
        agent,
        "List files in the current directory and show me README.md",
        thread_id="session123",
    ))

    print("\n=== Example 3: Session with Weather Forecast ===")
    agent.enable_skills_for_session("session456", ["Weather"])
    asyncio.run(run_example(
        agent,
        "What's the 5-day forecast for Berlin?",
        thread_id="session456",
    ))


async def run_example(
    agent: DynamicAgent,
    message: str,
    thread_id: str = "default",
) -> None:
    """Invoke the agent with *message* and print the last message content."""
    print(f"User: {message}")
    result = await agent.invoke(message, thread_id=thread_id)
    last = result["messages"][-1]
    print(f"Agent: {last.content if hasattr(last, 'content') else last}")


if __name__ == "__main__":
    main()
