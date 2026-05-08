"""
Example 09 – Writing a Custom Skill

Shows that adding a new capability is just writing a SKILL.MD file.
A custom "Dice" skill is created at runtime in a temporary directory,
loaded into the agent's registry, and then used to answer user queries.

The Dice skill uses a `bash:` entrypoint — the tool executes a one-liner
Python command to generate a random number.  No additional Python code or
imports are needed in the skill declaration itself.

Skill defined in this example
──────────────────────────────
  name: Dice
  tool: roll_dice(sides)  →  bash:python3 -c "import random; print(random.randint(1, {sides}))"

Prerequisites
─────────────
    export LLM_VENDOR=anthropic
    export LLM_MODEL=claude-sonnet-4-6
    export ANTHROPIC_API_KEY=sk-ant-...

    # Or use LLM_PROVIDER_CONFIG for a single JSON blob (overrides the above):
    export LLM_PROVIDER_CONFIG='{"vendor":"anthropic","model":"claude-sonnet-4-6","api_key":"sk-ant-..."}'

Run
───
    python examples/09_custom_skill.py
"""

import asyncio
import tempfile
from pathlib import Path

from birdie.agent.run import DynamicAgent
from birdie.core.loader import discover_skills_from_directory

SKILLS_DIR = Path(__file__).parent.parent / "birdie" / "skills"

# The full SKILL.MD content for our custom Dice skill.
DICE_SKILL_MD = """\
---
name: Dice
version: 1.0.0
description: Roll dice with any number of sides and return the result.
tags: [dice, random, game]
enabled_by_default: true
---

# Skill: Dice

## Tools

### roll_dice
description: Roll a single die with the given number of sides and return the result. Use whenever the user asks to roll dice or needs a random number within a range.
entrypoint: bash:python3 -c "import random; print(random.randint(1, {sides}))"
schema:
  type: object
  properties:
    sides:
      type: integer
      description: Number of sides on the die (e.g. 6 for a D6, 20 for a D20, 100 for a percentile die)
      default: 6
  required: [sides]
"""


async def main() -> None:
    with tempfile.TemporaryDirectory() as custom_dir:
        # Write the SKILL.MD to a subdirectory (one subdirectory = one skill).
        skill_dir = Path(custom_dir) / "Dice"
        skill_dir.mkdir()
        (skill_dir / "SKILL.MD").write_text(DICE_SKILL_MD)

        # Parse and inspect the skill before wiring it into the agent.
        custom_skills = discover_skills_from_directory(custom_dir)
        dice = custom_skills[0]
        print("=== Custom skill parsed from SKILL.MD ===")
        print(f"  name       : {dice.name}")
        print(f"  description: {dice.description}")
        print(f"  tags       : {dice.tags}")
        print(f"  tool       : {dice.tools[0].name} — entrypoint: {dice.tools[0].entrypoint}")
        print(f"  enabled    : {dice.enabled_by_default}\n")

        # Build the agent with the bundled skills, then register the custom skill.
        agent = DynamicAgent.from_config(skills_dir=str(SKILLS_DIR))
        for skill in custom_skills:
            agent.registry.register_skill(skill)
        # Refresh the policy so it picks up the new default-enabled skill.
        agent.policy.set_default_skills(agent.registry.list_skills())

        # Confirm the skill is live.
        allowed = agent.policy.get_allowed_skills_for_session("default")
        print(f"=== Dice in default allowed set: {'Dice' in allowed} ===\n")

        # Ask the agent to use the new skill.
        queries = [
            "Roll a standard six-sided die.",
            "Roll a D20 for my dungeon-crawl game.",
            "Roll three D6 dice and give me the total.",
        ]

        for q in queries:
            print(f"User:  {q}")
            result = await agent.invoke(q)

            for msg in result["messages"][1:]:
                kind = type(msg).__name__
                if kind == "AIMessage" and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        print(f"  → roll_dice(sides={tc['args'].get('sides', '?')})")
                elif kind == "ToolMessage":
                    print(f"  ← result: {msg.content.strip()}")
                elif kind == "AIMessage" and msg.content:
                    print(f"Agent: {msg.content}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
