"""
Example 01 – Hello World

The simplest possible birdie-agent script.  Creates an agent from environment
variables, sends one message, and prints the reply.

All built-in skills are disabled by default, so the agent answers entirely
from the model's own knowledge — no external tool calls are made.

Prerequisites
─────────────
Set your LLM provider credentials, for example:

    export LLM_VENDOR=anthropic
    export LLM_MODEL=claude-sonnet-4-6
    export ANTHROPIC_API_KEY=sk-ant-...

    # Or with OpenAI:
    export LLM_VENDOR=openai
    export LLM_MODEL=gpt-4o
    export OPENAI_API_KEY=sk-...

    # Or use LLM_PROVIDER_CONFIG for a single JSON blob (overrides the above):
    export LLM_PROVIDER_CONFIG='{"vendor":"anthropic","model":"claude-sonnet-4-6","api_key":"sk-ant-..."}'

Run
───
    python examples/01_hello_world.py
"""

import asyncio
from pathlib import Path

from birdie.agent.run import DynamicAgent

# Resolve the bundled skills directory relative to this file.
SKILLS_DIR = Path(__file__).parent.parent / "birdie" / "skills"


async def main() -> None:
    # Build the agent.  Provider and model are read from LLM_VENDOR / LLM_MODEL
    # environment variables (see module docstring for options).
    agent = DynamicAgent.from_config(skills_dir=str(SKILLS_DIR))

    print(f"Provider : {agent.provider.vendor_name} / {agent.provider.model_name}")
    print(f"Skills   : {[s.name for s in agent.registry.list_skills()]} (all disabled)\n")

    message = "In one sentence, what is the LangGraph framework?"
    print(f"User : {message}")

    # invoke() runs the agent to completion and returns the final AgentState.
    # result["messages"] holds the full conversation: HumanMessage → AIMessage.
    result = await agent.invoke(message)

    reply = result["messages"][-1].content
    print(f"Agent: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
