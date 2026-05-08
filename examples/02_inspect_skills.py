"""
Example 02 – Inspect the Skill Registry

Walks through the registry and policy APIs without making any LLM calls.
Use this to understand what skills are loaded, what tools they expose, and
how skill access control works.

No API key required — a mock LLM is used so the agent can be constructed
without any provider credentials.

Run
───
    python examples/02_inspect_skills.py
"""

from pathlib import Path

from langchain_core.messages import AIMessage

from birdie.agent.run import DynamicAgent

SKILLS_DIR = Path(__file__).parent.parent / "birdie" / "skills"


class _MockLLM:
    """Minimal stand-in that never makes network calls."""

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, config=None):
        return AIMessage(content="(mock response)")


def main() -> None:
    agent = DynamicAgent(_MockLLM(), skills_dir=str(SKILLS_DIR))

    # ── 1. All registered skills ──────────────────────────────────────────────
    print("=== Loaded Skills ===")
    for skill in agent.registry.list_skills():
        default = "on " if skill.enabled_by_default else "off"
        kind = "[MCP]" if skill.mcp_server else f"{len(skill.tools)} tool(s)" if skill.tools else "knowledge"
        print(f"  [{default}] {skill.name:<15}  {kind:<12}  {skill.description[:55]}")
    print()

    # ── 2. Tools inside a specific skill ─────────────────────────────────────
    shell = agent.registry.get_skill("Shell")
    if shell:
        print("=== Shell Skill — Tools ===")
        for tool in shell.tools:
            print(f"  {tool.name}")
            print(f"    description : {tool.description}")
            print(f"    entrypoint  : {tool.entrypoint}")
            print(f"    schema      : {list(tool.schema.get('properties', {}).keys())}")
        print()

    # ── 3. Freetext knowledge skill — trigger keywords ────────────────────────
    ssh = agent.registry.get_skill("ssh")
    if ssh:
        print("=== ssh Skill — Knowledge (no tools) ===")
        print(f"  triggers : {ssh.triggers}")
        print(f"  body     : {len(ssh.body or '')} chars (injected into system prompt on trigger)")
        print()

    # ── 4. Default policy — which skills are on without explicit grants ───────
    default_allowed = agent.policy.get_allowed_skills()
    print("=== Default Allowed Skills ===")
    print(f"  {sorted(default_allowed) or '(none - all skills are disabled by default)'}")
    print()

    # ── 5. Enable skills for a session ───────────────────────────────────────
    SESSION = "demo-session"
    agent.enable_skill(SESSION, "Shell")
    agent.enable_skill(SESSION, "DuckDuckGo")

    session_allowed = agent.policy.get_allowed_skills(session_id=SESSION)
    print(f"=== Allowed for session '{SESSION}' (after enables) ===")
    print(f"  {sorted(session_allowed)}")
    print()

    # ── 6. Disable one of the enabled skills ─────────────────────────────────
    agent.disable_skill(SESSION, "DuckDuckGo")
    after_disable = agent.policy.get_allowed_skills(session_id=SESSION)
    print(f"=== After disabling DuckDuckGo ===")
    print(f"  {sorted(after_disable)}")
    print()

    # ── 7. Session with an explicit fixed skill set ───────────────────────────
    # enable_skills_for_session() replaces the default seed with an exact list.
    agent.enable_skills_for_session("restricted", ["Filesystem"])
    restricted = agent.policy.get_allowed_skills("restricted")
    print("=== Session-scoped skills for 'restricted' ===")
    print(f"  {sorted(restricted)}")
    print()

    # ── 8. Tool lookup by name ────────────────────────────────────────────────
    tool = agent.registry.get_tool("run_bash")
    if tool:
        print(f"=== Tool lookup: 'run_bash' ===")
        print(f"  name        : {tool.name}")
        print(f"  entrypoint  : {tool.entrypoint}")
        print(f"  tags        : {tool.tags}")


if __name__ == "__main__":
    main()
