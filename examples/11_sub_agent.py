"""
Example 11 - Sub-agents via AGENT.MD

Shows how to define a sub-agent at runtime using an AGENT.MD file written to
a temporary directory.  The main agent discovers the sub-agent and can call it
as a regular tool.

A custom "Summarizer" sub-agent is created in a temp directory, loaded via
agents_dir, and the main agent is asked to summarize a passage - triggering
the sub-agent tool call automatically.

Agent defined in this example
──────────────────────────────
  name: Summarizer
  input: text (string), max_points (integer, optional)
  prompt: summarize {{ text }} in at most {{ max_points }} bullet points

Prerequisites
─────────────
    export LLM_VENDOR=mistral
    export LLM_MODEL=mistral-large-latest
    export MISTRAL_API_KEY=...

    # Or point at a JSON config file:
    export LLM_PROVIDER_CONFIG='{"vendor":"mistral","model":"mistral-large-latest",...}'

Run
───
    python examples/11_sub_agent.py
"""

import asyncio
import tempfile
from pathlib import Path

from birdie.agent.run import DynamicAgent
from birdie.core.agent_loader import discover_agents_from_directory

SKILLS_DIR = Path(__file__).parent.parent / "birdie" / "skills"

SUMMARIZER_AGENTS_MD = """\
---
name: Summarizer
version: 1.0.0
description: Summarize a piece of text into concise bullet points
enabled_by_default: true
allowed_skills: []
---

## Input

### text
type: string
description: The text to summarize
required: true

### max_points
type: integer
description: Maximum number of bullet points (default 5)
required: false

## Output

### summary
type: string
description: Bullet-point summary of the text

## Prompt

Summarize the following text concisely. Return at most {{ max_points }} bullet points (default to 5 if not specified). Use plain bullet points starting with "- ".

Text to summarize:

{{ text }}
"""

TEXT = """\
Large language models (LLMs) are a type of artificial intelligence that can
understand and generate human language. They are trained on vast amounts of
text data using a technique called self-supervised learning, where the model
learns to predict missing or next words in a sequence. The resulting models
can perform a wide range of tasks, including answering questions, writing code,
translating languages, and summarizing documents. Recent advances have made
LLMs significantly more capable, leading to their widespread adoption in
applications like chatbots, code assistants, and content generation tools.\
"""


async def main() -> None:
    with tempfile.TemporaryDirectory() as agents_dir:
        # Write the AGENT.MD to a subdirectory (one subdirectory = one agent).
        agent_dir = Path(agents_dir) / "Summarizer"
        agent_dir.mkdir()
        (agent_dir / "AGENT.MD").write_text(SUMMARIZER_AGENTS_MD)

        # Parse and inspect the agent definition before wiring it in.
        agent_defs = discover_agents_from_directory(agents_dir)
        summarizer = agent_defs[0]
        print("=== Sub-agent parsed from AGENT.MD ===")
        print(f"  name        : {summarizer.name}")
        print(f"  description : {summarizer.description}")
        print(f"  input params: {[p.name for p in summarizer.input_params]}")
        print(f"  enabled     : {summarizer.enabled_by_default}\n")

        # Build the agent with the temp agents dir.
        agent = DynamicAgent.from_config(
            skills_dir=str(SKILLS_DIR),
            agents_dir=agents_dir,
        )

        # Confirm the sub-agent is in the allowed set.
        allowed = agent.agent_registry.get_allowed_agents("default")
        print(f"=== Summarizer in default allowed set: {'Summarizer' in allowed} ===\n")

        # Ask the main agent to summarize - it should delegate to the Summarizer.
        query = f"Please summarize this text in 3 bullet points:\n\n{TEXT}"
        print(f"User: {query[:80]}...\n")

        result = await agent.invoke(query)

        for msg in result["messages"][1:]:
            kind = type(msg).__name__
            if kind == "AIMessage" and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    preview = str(tc["args"].get("text", ""))[:60].replace("\n", " ")
                    print(f"  -> {tc['name']}(text=\"{preview}...\", max_points={tc['args'].get('max_points', '?')})")
            elif kind == "ToolMessage":
                print(f"  <- {msg.content.strip()}")
            elif kind == "AIMessage" and msg.content:
                print(f"Agent: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
