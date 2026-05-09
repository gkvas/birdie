"""
Example 11: Sub-agents via AGENTS.MD

Sub-agents are independent agents defined in AGENTS.MD files that the main
agent can call as tools.  Each sub-agent runs in an ephemeral DynamicAgent
with its own skills and LLM config, then returns its result as a tool response.

Directory layout
----------------
~/.birdie/agents/
    summarizer/
        AGENTS.MD       # defines the Summarizer sub-agent
    translator/
        AGENTS.MD       # defines a Translator sub-agent
    ...

The main agent discovers all AGENTS.MD files under ~/.birdie/agents/ and
registers each one as an async tool.  The calling LLM sees these tools
alongside regular skills and decides when to use them.

Running this example
--------------------
1. Copy the example agent definition to your user agents directory:

       mkdir -p ~/.birdie/agents/summarizer
       cp examples/agents/summarizer/AGENTS.MD ~/.birdie/agents/summarizer/

2. Set your LLM credentials (e.g. ANTHROPIC_API_KEY) and run:

       python examples/11_sub_agent.py
"""

import asyncio
from birdie.agent.run import DynamicAgent


TEXT = """
Large language models (LLMs) are a type of artificial intelligence that can
understand and generate human language. They are trained on vast amounts of
text data using a technique called self-supervised learning, where the model
learns to predict missing or next words in a sequence. The resulting models
can perform a wide range of tasks, including answering questions, writing code,
translating languages, and summarizing documents. Recent advances have made
LLMs significantly more capable, leading to their widespread adoption in
applications like chatbots, code assistants, and content generation tools.
"""


async def main() -> None:
    agent = DynamicAgent.from_config(skills_dir="skills")

    print("Asking the agent to summarize a passage (will call the Summarizer sub-agent):")
    print("-" * 60)

    async for update in agent.astream(
        f"Please summarize this text in 3 bullet points:\n\n{TEXT.strip()}",
        thread_id="example-11",
    ):
        for node, data in update.items():
            for msg in data.get("messages", []):
                if hasattr(msg, "content") and msg.content:
                    content = msg.content
                    if isinstance(content, list):
                        content = "\n".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        )
                    print(f"[{node}] {content}")


if __name__ == "__main__":
    asyncio.run(main())
