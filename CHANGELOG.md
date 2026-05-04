# Changelog

All notable changes to this project are documented here.

## [0.2.5] - 2026-05-04

### Changed
- Rework `ACPProvider` to use JSON-RPC 2.0 over stdio instead of HTTP - birdie
  now spawns the ACP adapter binary directly (e.g. `claude-agent-acp`) and
  communicates via stdin/stdout pipes; no local server needs to be started
- Remove `httpx` dependency (was only used by the HTTP-based ACP implementation)
- Config field for ACP is now `model` (binary name) instead of `base_url` +
  `agent_name`

## [0.2.4] - 2026-05-04

### Fixed
- `list_tools(skill_names=[])` was skipping the skill filter (empty list is
  falsy in Python) and returning all registered tools - with all built-in skills
  set to `enabled_by_default: false`, any session with no explicitly enabled
  skills exposed every tool to the LLM

## [0.2.3] - 2026-05-04

### Fixed
- `AzureOpenAIProvider`: switch from `openai.AzureOpenAI` to `AzureChatOpenAI`
  from `langchain-openai` - the raw `AzureOpenAI` client rejected the `tools`
  parameter on some Azure deployments; `AzureChatOpenAI` routes tool definitions
  through `bind_tools()` which works correctly

### Added
- `ACPProvider`: connect birdie to any ACP-compatible agent (e.g.
  `claude-agent-acp`, `codex-acp`) - the inner agent runs its own tool loop,
  allowing use of existing Claude Code or ChatGPT Plus subscriptions without
  per-token API costs

## [0.2.2] - 2026-05-02

### Fixed
- MCP servers from disabled skills no longer connect - the MCP client manager
  now filters servers against the allowed skill set before establishing
  connections

## [0.2.1] - 2026-05-02

### Fixed
- User skills with `python:` entrypoints now resolve correctly - the skills root
  is added to `sys.path` so local module imports work

## [0.2.0] - 2026-05-02

### Added
- Test suite for `_load_skills` and related loader behaviour
- Pre-commit hook running flake8 with `.flake8` exclude config

### Fixed
- Undefined `skills` variable in `_load_skills` when the user skills directory
  is absent

## [0.1.9] - 2026-04-30

### Added
- User skill directory: skills placed in `~/.birdie/skills/` are loaded on top
  of the bundled skills every startup, enabling persistent personal skills
  without modifying the package

## [0.1.8] - 2026-04-29

### Added
- DuckDuckGo search skill (opt-in)
- Version number shown in the startup banner
- Skill name suggestions when the user types an unrecognised `/skill` name

### Changed
- All built-in skills now default to `enabled_by_default: false` - skills must
  be explicitly enabled per session via `/skill enable <name>`

## [0.1.7] - 2026-04-28

### Fixed
- `mcp_demo` server path is now resolved relative to its skill directory,
  fixing launch failures when birdie is run from a different working directory

## [0.1.6] - 2026-04-27

### Fixed
- MCP server path resolution corrected for packaged installs

## [0.1.5] - 2026-04-26

### Changed
- `mcp`, `mistralai`, and `anthropic` promoted from optional to core
  dependencies so all supported providers work out of the box after `pip install`

## [0.1.4] - 2026-04-25

### Added
- Skills are now bundled inside the package, eliminating the need to copy skill
  files manually after installation

### Changed
- Welcome banner simplified

## [0.1.3] - 2026-04-24

### Fixed
- License corrected to Apache-2.0 in `pyproject.toml`

## [0.1.2] - 2026-04-23

### Fixed
- Fail fast with a readable help message when no LLM vendor is configured,
  instead of an opaque error

## [0.1.1] - 2026-04-22

### Added
- PyPI metadata (`classifiers`, `urls`, `keywords`)
- Package renamed to `birdie-agent` on PyPI to avoid conflicts

## [0.1.0] - 2026-04-21

Initial public release.

- Vendor-agnostic `LLMProvider` interface with built-in support for OpenAI,
  Azure OpenAI, Anthropic, Mistral, Google Gemini, Ollama, and any LangChain
  `BaseChatModel`
- SKILL.MD skill system: define tools, triggers, and instructions in Markdown
  frontmatter
- Per-session skill access control via `UserSkillPolicy`
- MCP (Model Context Protocol) client integration for external tool servers
- LangGraph-based agent loop with tool execution, checkpoint repair, and
  rate-limit retry
- SQLite-backed session persistence with rolling context window
- Interactive CLI with slash commands, streaming output, and token counters
