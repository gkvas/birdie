# Birdie - Claude Code Instructions

## Codebase overview

Birdie is a LangGraph-based AI agent framework with a dynamic skill system. Key concepts:

- **Skills**: Markdown files (`SKILL.MD`) with YAML frontmatter. The LLM sees compact listings with `[load: <name>]` hints and calls the `get_skill` tool to fetch full skill bodies on demand (progressive loading with turn-decay eviction and LRU cap).
- **Agents**: Configured via YAML/JSON (`AgentDef`). Agents define which skills are allowed, which LLM provider to use, and runtime settings.
- **LTM**: Per-user long-term memory stored in `~/.birdie/ltm/<user_id>.json`. Compacted conversation summaries are embedded and retrieved by cosine similarity at the start of each session.

Key source locations:

| Path | Purpose |
|------|---------|
| `birdie/agent/graph.py` | LangGraph state machine, skill loading logic |
| `birdie/agent/run.py` | `DynamicAgent` - main entry point, config parsing |
| `birdie/core/models.py` | `Skill`, `SkillTool`, `AgentDef` data models |
| `birdie/core/loader.py` | `SKILL.MD` parser (`parse_skill_markdown`) |
| `birdie/core/registry.py` | `SkillRegistry` - loads and indexes skills |
| `birdie/core/policy.py` | `SkillPolicy` - per-session skill access control |
| `birdie/core/ltm.py` | `LTMStore` - persistent long-term memory |
| `birdie/core/retrieval.py` | Hash-trick bag-of-ngrams embedder (512-dim) |
| `birdie/core/llm_provider.py` | Vendor-agnostic LLM interface (OpenAI, Anthropic, Mistral) |
| `birdie/skills/` | Built-in skill definitions |
| `tests/` | pytest test suite |

Agent-level config keys (extracted from JSON provider config, not forwarded to vendor):
`tool_output_cap`, `skill_decay_turns`, `skill_max_loaded`, `ltm_max_age_days`, `ltm_max_entries`, `ltm_min_score`

## Python environment

Always use the venv at `~/venv` for all Python commands in this project:

- Run tests: `~/venv/bin/python -m pytest tests/ -q`
- Run scripts: `~/venv/bin/python <script>`
- Install packages: `~/venv/bin/pip install <package>`

The system `python` is missing `mcp` and other dependencies declared in `pyproject.toml`.

## Release process

Follow this exact workflow for every release. Never push directly to main.

1. **Create a feature/release branch** - e.g. `bump-0.x.y`
2. **Update `CHANGELOG.md`** - add a new `## [x.y.z] - YYYY-MM-DD` entry at the top (below the header), listing all changes since the last release
3. **Bump version in `pyproject.toml`** - update the `version = "x.y.z"` field
4. **Run tests** - `~/venv/bin/python -m pytest tests/ -q` - all must pass
5. **Commit both files** - stage `CHANGELOG.md` and `pyproject.toml` explicitly (never `git add -A`)
6. **Push the branch** and **open a PR**
7. **Wait for all CI checks to pass** before merging
8. **Squash merge** the PR (delete the branch after merging)
9. **Tag the merge commit** on main with a `v` prefix:
   ```
   git fetch origin
   git tag v{version} origin/main
   git push origin v{version}
   ```

Tag naming: always `v0.x.y` (with `v` prefix). Never create bare numeric tags like `0.x.y`.
