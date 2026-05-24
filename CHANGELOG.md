## [0.3.1] - 2026-05-24

### Changed
- Skills and agents no longer declare `enabled_by_default` in their SKILL.MD/AGENT.MD frontmatter. Deployers now pass explicit `skills_enabled` and `agents_enabled` lists to `DynamicAgent` (or via `ProviderConfig` JSON) to control which capabilities are active by default.

### Fixed
- Async ACP stdout reader mock updated to use `read()` instead of `readline()`, matching the chunked-read implementation introduced in 0.3.0.

## [0.3.0] - 2026-05-23

### Added
- Skill directory loading is now additive: skills from `--skills-dir`, `~/.birdie/skills`, and bundled skills are all loaded, with higher priority sources overriding lower priority ones for skills with the same name.
- Weather skill now reads the API key from the `WEATHERAPI_KEY` environment variable, eliminating the need to pass it in the conversation.
- ACP provider now exposes enabled Birdie agents to the ACP agent via a stdio MCP server, allowing the ACP agent to use Birdie's agent tools.

### Changed
- Weather skill entrypoints switched from `http:get` to `python:` to securely inject the API key server-side.
- Tool descriptions in skill SKILL.MD files updated to include actionable guidance previously buried in the Markdown body.

### Fixed
- Skill directory loading now respects priority order: `--skills-dir` (highest), `~/.birdie/skills` (medium), bundled skills (lowest).
- Weather API queries now use the correct parameter `q` instead of `city`.
- ACP provider correctly forwards agent tools alongside skill tools to the MCP server.
# Changelog

All notable changes to this project are documented here.

## [0.2.14] - 2026-05-23

### Added
- ACP provider now exposes enabled Birdie skills to the ACP agent via a
  stdio MCP server (`birdie.core.acp_mcp_server`). When at least one skill
  with a local entrypoint is enabled, the MCP server entry is passed in
  `session/new` so the underlying model (e.g. Claude Code) can call skill
  tools directly through MCP.
- Built-in ACP callbacks (`terminal/create`, `fs/read_text_file`,
  `fs/write_text_file`) are disabled when an MCP server is active, so the
  model uses Birdie's skill tools exclusively.

### Fixed
- ACP `session/request_permission` response now uses the correct nested
  format `{"outcome": {"outcome": "selected", "optionId": "allow"}}`;
  the previous flat `{"optionId": "allow"}` caused every MCP tool call to
  be silently denied.
- Claude Code built-in tools (Read, Bash, Write, etc.) are now suppressed
  when MCP mode is active by setting `disableBuiltInTools: true` in
  `session/new`; previously the model could use all built-in tools
  regardless of which Birdie skills were enabled.
- ACP provider sends the full conversation history on every turn instead
  of only the last user message, giving the model the same context as
  native providers.
- Tool calls made by the ACP agent are now visible in the CLI output.

## [0.2.13] - 2026-05-18

### Added
- Configurable compaction thresholds: `min_messages`, `max_messages`, and
  `compression_window` can now be set in the JSON provider config file
  (e.g. `{"vendor": "anthropic", "model": "...", "min_messages": 10,
  "max_messages": 40}`); the three fields are extracted before the config is
  forwarded to the vendor SDK so they are safe to include alongside vendor-
  specific fields; wired through `ProviderConfig`, `DynamicAgent.from_config()`,
  `DynamicAgent.__init__()`, `create_agent_graph()`, and `compact_history()`
  so all compaction paths - automatic and manual - honour the same settings

### Changed
- `MIN_MESSAGES` lowered from 40 to 20: the previous value left a large dead
  zone between the compaction floor and the context window; 20 is more
  aggressive while still leaving enough tail for meaningful context
- `MAX_CONTEXT_MESSAGES` constant removed: the full non-compacted checkpoint is
  now forwarded to the LLM on every turn (compaction itself keeps the checkpoint
  bounded); a Mistral-compatible one-liner ensures the context always starts at
  a `HumanMessage` boundary

## [0.2.12] - 2026-05-18

### Added
- Automatic conversation history compaction: when a session's stored message
  count reaches `MAX_MESSAGES` (100), the oldest segment is summarised by the
  LLM and the raw messages are permanently removed from the LangGraph checkpoint
  via `RemoveMessage`; the compaction threshold, minimum retained messages
  (`MIN_MESSAGES = 40`), and maximum compressed window (`COMPRESSION_WINDOW = 60`)
  are tunable constants in `birdie/agent/graph.py`
- `compact_history()` coroutine (`birdie/agent/graph.py`): finds the largest
  HumanMessage-aligned split point within the compression window, renders the
  segment as a readable transcript, sends it to the LLM with a structured JSON
  prompt extracting six categories (summary, facts, preferences, world knowledge,
  tool outcomes, open tasks), and returns `RemoveMessage` deletions for the
  checkpointer; `force=True` bypasses the automatic threshold for manual use
- `/compact` slash command: force-compacts the current session regardless of
  history length, displays the generated summary and the number of messages
  removed; implemented in `birdie/cli.py`, backed by `DynamicAgent.compact_session()`
- `DynamicAgent.compact_session(thread_id, user_id)` method (`birdie/agent/run.py`):
  reads the checkpoint, runs compaction with `force=True`, writes `RemoveMessage`
  entries back, returns `(n_removed, summary_text)`
- Long-term memory (LTM) store (`birdie/core/ltm.py`): per-user JSON file at
  `~/.birdie/ltm/<user_id>.json`; each compaction appends a structured `LTMEntry`
  with an embedding vector; `LTMStore` loads lazily, writes atomically
  (write-then-rename), and exposes `query(text, k=5)` for cosine-similarity
  retrieval and `format_for_prompt(entries)` for system-prompt injection
- Retrieval primitives (`birdie/core/retrieval.py`): public API `embed(text)`,
  `cosine_similarity(a, b)`, `EMBED_DIM`; dependency-free hash-trick
  bag-of-ngrams embedding (unigrams + bigrams, SHA-256, L2-normalised) so that
  dot-product equals cosine similarity; no model downloads required
- Per-turn semantic LTM retrieval: on every `call_model()` invocation the top-5
  most relevant `LTMEntry` objects are retrieved by cosine similarity on the
  current user message and injected into system-prompt Tier 3 alongside manual
  `/remember` entries; the `LTMStore` is cached per `user_id` for the lifetime
  of the graph to avoid repeated disk reads
- `ltm_store_factory` parameter on `DynamicAgent` and `DynamicAgent.from_config()`:
  callable `(user_id: str) -> LTMStore`; defaults to `lambda uid: LTMStore(uid)`;
  pass `None` to disable the LTM store entirely
- `user_id` parameter on `DynamicAgent.invoke()` and `DynamicAgent.astream()`:
  stored in `config["configurable"]["user_id"]` so the graph can look up the
  correct LTM store; when omitted, LTM retrieval and compaction storage are
  silently skipped

### Changed
- System-prompt Tier 3 (long-term memory) now merges two sources: manual entries
  from `/remember` (forwarded via `config["configurable"]["long_term_memory"]`)
  and semantically retrieved compaction entries from `LTMStore`; both are rendered
  under a single `--- Long-term memory ---` block

### Tests
- `tests/test_compaction.py` (new, 302 lines): 18 async tests covering threshold
  behaviour, split alignment, `RemoveMessage` shape, LTM integration, JSON
  parsing edge cases (prose-wrapped JSON), `force=True` mode, and tool messages
  in history
- `tests/test_ltm.py` (new, 175 lines): 20 tests covering `LTMStore` persistence,
  atomic writes, user isolation, `query()` relevance ordering, and
  `format_for_prompt()` rendering
- `tests/test_retrieval.py` (new, 116 lines): 18 tests covering `embed()` and
  `cosine_similarity()` - dimension, normalisation, determinism, case folding,
  symmetry, range bounds, and semantic discrimination including bigram-specific
  phrase handling

## [0.2.11] - 2026-05-09

### Added
- `max_tool_repetitions` guard: blocks any tool call that appears more than N
  times consecutively with identical parameters; injects an error `ToolMessage`
  so the LLM can recover instead of looping forever; configurable per sub-agent
  via `max_tool_repetitions` in AGENT.MD (default 3)
- `/agent output off|short|full` command to control sub-agent transcript
  verbosity independently from `/tool output`; default is `off` (transcript
  hidden, only the final reply shown as a tool result)
- Sub-agent output rendered as a buffered, indented transcript block printed
  after the agent completes: `[AgentName#xxxx]` header at 3-space indent,
  tool calls and AI messages at 6-space indent, args/results at 9-space indent

### Fixed
- `recursion_limit` was not forwarded on the streaming path (`astream()`
  lacked a `config` parameter, so the inner agent always used the LangGraph
  default of 25); now forwarded correctly via the same config-merging logic
  used in `invoke()`

### Changed
- README split into focused documentation files under `doc/`: `cli.md`,
  `skills.md`, `agents.md`, `mcp.md`, `architecture.md`; README is now a
  concise entry point with links to each file

## [0.2.10] - 2026-05-08

### Changed
- `UserSkillPolicy` renamed to `SkillPolicy`; the per-"user" enable/disable
  tracking now uses session IDs consistently throughout - the old name was
  misleading because the policy was always keyed by session/thread ID, not by
  a distinct user identity
- Policy internals simplified: three separate dicts replaced by a single
  `_session_skills` dict seeded from `enabled_by_default` on first access and
  mutated directly by `enable_skill` / `disable_skill`
- `DynamicAgent.enable_skill_for_user` / `disable_skill_for_user` renamed to
  `enable_skill` / `disable_skill`

### Added
- 10 runnable example scripts in `examples/` covering hello world, skill
  inspection, web search, shell commands, multi-turn conversation, streaming,
  long-term memory, SQLite persistence, custom skills, and MCP-backed skills
- `LLM_PROVIDER_CONFIG` environment variable: pass a full JSON provider config
  as a single variable, overriding all other provider env vars; accepts a JSON
  string or a path to a `.json` file
- Azure OpenAI env var documentation and examples (`AZURE_OPENAI_API_KEY`,
  `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`)

## [0.2.9] - 2026-05-07

### Added
- Working directory shown in the status bar (`~/...` notation when under home,
  absolute path otherwise)
- `/cd <path>` command to change the working directory; no argument goes to
  home; supports `~` expansion and relative paths; errors reported inline
- Tab completion for `/cd` path arguments - only directories, `~` expansion,
  activated by Tab only (not while typing)

## [0.2.8] - 2026-05-07

### Added
- Custom system prompt via `.birdie/system_prompt.md` - if the file exists in
  the current working directory its contents are prepended to the system prompt
  on every turn (Tier 0), before the skill catalog; re-read each turn so
  changes take effect immediately without restarting Birdie

### Fixed
- Pass `config` to `ToolNode.ainvoke()` for correct LangGraph compatibility;
  the previous call without `config` broke checkpointer context propagation in
  newer LangGraph versions

## [0.2.7] - 2026-05-05

### Changed
- Ctrl+C behaviour overhauled:
  - Ctrl+C with text in the input line clears the line (standard shell
    behaviour) instead of exiting
  - Ctrl+C on an empty line shows a grey inline hint ("Press Ctrl+C again to
    exit, or type new instructions to continue"); typing any character dismisses
    it; a second Ctrl+C exits
  - Ctrl+C while the agent is thinking or executing a tool cancels the active
    `asyncio.Task` and returns immediately to the `you>` prompt, printing
    "Interrupted."

### Added
- `/log llm on|off` - attach a file handler to the `birdie.core.llm_provider`
  logger; every request (model, message count, last user text) and response
  (content, tool calls) is written to `~/.birdie/llm.log`
- `/log http on|off` - monkey-patch `httpx.AsyncClient.send` /
  `httpx.Client.send` to capture full JSON request and response bodies to
  `~/.birdie/http.log`; streaming responses are noted but not reassembled
  (ACP traffic uses stdio, so use `/log llm` for that provider instead)    

## [0.2.6] - 2026-05-05

### Fixed
- `ACPProvider`: corrected wire format - `session/prompt` now sends `prompt` as
  a flat array of ContentBlocks instead of a `message` role/content wrapper
- `ACPProvider`: streaming update parsing now uses the `sessionUpdate:
  "agent_message_chunk"` discriminator from the real schema (docs were wrong)
- `ACPProvider`: response text is accumulated from chunk notifications; the
  final `PromptResponse` carries only `stopReason`, not content

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
