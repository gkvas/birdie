## [Unreleased]

### Added
- Non-interactive mode via `-p`/`--prompt` CLI argument: allows passing a prompt directly to birdie, which then runs once and outputs the AI response to stdout (similar to Claude Code's `-p` flag). Works with all existing CLI options like `--config`, `--user`, and `--session-id`.

## [0.8.1] - 2026-08-22

### Fixed
- Documentation: Added ACP to the list of supported vendors in the CLI help text (`/help`).

## [0.8.0] - 2026-08-21

## [0.8.0] - 2026-08-21

### Added
- ACP tool-call streaming: `tool_call` / `tool_call_update` session updates
  from the ACP agent subprocess are rendered live in the CLI while the turn
  is streaming, instead of the agent's tool activity being invisible.
- ACP usage reporting: `usage_update` notifications (context tokens used,
  context window size, cumulative cost) are captured by `ACPProvider` and
  attached to the returned message as `usage_metadata` /
  `response_metadata`. The CLI status bar shows context as used/window,
  `/cost` prefers the agent-reported cost over the pricing-table estimate,
  and cumulative cost is persisted per session as `total_cost_usd`. As a
  side effect, `compaction_token_threshold` now also works for ACP
  sessions.
- ACP permission gate: `session/request_permission` is routed through an
  optional `permission_callback` (sync or async, returning
  allow / allow_always / deny; errors fail closed) instead of being
  auto-allowed. The CLI prompts interactively and persists "always"
  approvals per session in `approved_acp_tools`, keyed by the agent's
  allow_always option label, so they survive the per-turn subprocess
  respawn. Library users without a callback keep the auto-allow behaviour.
- The CLI status bar shows the current git branch and working-tree status
  on the right.

### Changed
- Documentation synced with the current code.

### Fixed
- CLI crashed at startup while rendering the status bar.
- The status bar displayed "unknown" instead of the ACP agent's active
  model; it now shows the model reported by the agent session.
- Pinned the `mcp` dependency to `<2.0` to stay compatible with the
  current API.

## [0.7.0] - 2026-07-27

### Added
- Skill permission enforcement: tools of skills that declare a
  `## Permissions` section now require approval before executing, via a
  pluggable `tool_approval_callback` (sync or async) returning
  allow / always / deny. The CLI prompts interactively ("always" is
  persisted per session in `approved_skills` and re-applied on load);
  library users without a callback keep the previous allow-all behaviour.
- Structured sub-agent outputs: replies of agents that declare
  `output_params` are parsed as JSON (tolerating surrounding prose),
  validated against the declared fields and types, and retried once with a
  corrective follow-up on mismatch. Valid replies are returned as canonical
  JSON on both invocation paths.
- Token/cost accounting: per-model pricing table
  (`birdie.core.pricing`), cumulative per-session token totals persisted in
  the session JSON, and a `/cost` command showing usage with an estimated
  cost (upper bound; cache discounts are not modelled).
- History replay: resuming a session (startup with prior turns,
  `/session switch`) replays the tail of the checkpointed conversation;
  `/history [N]` prints the last N messages on demand.
- `/skill reload` re-discovers SKILL.MD files from disk in place
  (`DynamicAgent.reload_skills()`); `/skill new <name>` scaffolds a starter
  SKILL.MD under `~/.birdie/skills/<name>/`.
- Per-tool resilience settings in SKILL.MD tool blocks: `timeout:` bounds a
  single execution (enforced via the bash/HTTP resolvers) and `retries:`
  re-attempts failed executions; idempotent `http:get` entrypoints default
  to one automatic retry.

### Changed
- Prompt-cache-friendly context layout: the system prompt now carries only
  stable content (custom instructions, skill listing, always_inject
  bodies); volatile per-turn context (loaded skill bodies, rolling
  summary, LTM) is delivered as an ephemeral trailing message that is never
  checkpointed. AnthropicProvider places `cache_control` breakpoints on the
  last tool, the system block, and the last stable message block, so tools,
  system prompt, and the growing conversation history are served from the
  provider prompt cache across turns. Disable with `"prompt_cache": false`
  in the provider config.
- The `[loaded]` hint was removed from the skill listing (it was the one
  volatile byte in the stable prefix); loaded skills are visible through
  their injected "skill context" blocks instead.

## [0.6.0] - 2026-07-27

### Added
- `enabled_by_default` frontmatter in SKILL.MD and AGENT.MD is now honoured:
  flagged skills/agents are granted to every session alongside the
  `skills_enabled` / `agents_enabled` config lists (previously the field was
  parsed by neither loader and silently ignored).
- Sub-agent `output_params` (AGENT.MD `## Output` section) are rendered into
  the sub-agent prompt as explicit JSON output-format instructions on both
  invocation paths (previously parsed but unused).
- New agent-level config key `compaction_token_threshold`: additionally
  triggers auto-compaction when the last model call's reported input tokens
  reach the threshold, independent of message count.

### Changed
- Auto-compaction runs as a background task per thread; its removals and
  summary are applied on the first turn after it completes, so the triggering
  turn no longer pays an extra LLM round trip.
- The compaction summary is kept as a rolling `summary` state channel and
  injected into the system prompt every turn ("Earlier conversation
  (compacted)"), so continuity survives even without an LTM store; `/compact`
  writes it back to the checkpoint and later compactions fold it in.
- `get_skill` returns a short acknowledgment instead of the skill body; the
  body reaches the model solely via the system-prompt lease, so it is no
  longer billed twice for the whole decay window.
- `SkillRegistry.list_tools` returns tools in deterministic registration
  order (stable prompts, provider prompt-cache friendly).
- Sub-agent `DynamicAgent` instances are cached across invocations (skills
  and agents are no longer re-parsed from disk per call); each run uses a
  unique thread id so histories never bleed.
- MCP tools are cached per server-name set instead of a single slot, so
  sessions with different allowed skills no longer evict each other.
- The Anthropic model catalog in `list_models` reflects the current
  generation (Fable 5, Opus 5/4.8/4.7, Sonnet 5/4.6 at 1M context tokens,
  Haiku 4.5 at 200K); transient 500/502/503/504 and connection errors are
  retried with backoff alongside 429/529.
- The deprecated trigger-matching API (`find_skills_by_trigger`) was removed;
  the `triggers` frontmatter field remains parseable for backward
  compatibility. `Skill.permissions` is explicitly documented as
  informational-only.

### Fixed
- The documented config keys `skill_decay_turns` and `skill_max_loaded` are
  now extracted from the provider config and forwarded to the graph;
  previously they were silently ignored and leaked into vendor SDK kwargs.
- `get_skill` accepts a skill's `location` as well as its name, matching the
  `[load: <location>]` hint shown in the system prompt; skills with a custom
  location were previously impossible to load.
- OpenAI/Mistral history conversion no longer drops non-empty assistant text
  when tool calls are present (only empty content is omitted).
- `bash:` entrypoint arguments are shell-quoted with `shlex.quote`, closing a
  shell-injection hole in templated tools like `bash:cat {path}`; a template
  consisting of a single placeholder (raw-shell tools like `bash:{command}`)
  still receives the full command string.
- `api_key` is never serialised into `BIRDIE_AGENTS_JSON` (the MCP subprocess
  inherits the vendor environment variable instead), and
  `ProviderConfig.to_json` now honours its documented api_key exclusion.
- `DynamicAgent` no longer implicitly loads a `skills/` directory from the
  current working directory (and no longer puts it on `sys.path`);
  `skills_dir` defaults to None, meaning user + bundled directories only.
- Agent directories load with first-wins precedence matching skills (explicit
  `agents_dir` before `~/.birdie/agents`); a broken AGENT.MD is skipped with
  a warning instead of crashing startup.
- Session IDs sort numerically by suffix (`_10` after `_2`); `/session
  delete` also removes the thread's history from `checkpoints.db`.
- HTTP entrypoints and the weather skill send requests with a timeout.
- Malformed tool-call argument JSON from a model degrades to an empty-args
  call (recoverable validation error) instead of crashing the turn; the
  Anthropic history path truncates oversized tool results like the OpenAI
  path.
- `LTMStore.add` re-reads the store file before appending so concurrent
  sessions of the same user do not clobber each other's entries.

## [0.5.4] - 2026-07-27

### Fixed
- The Anthropic provider no longer sends `temperature` to model families that
  removed the sampling parameters (Opus 4.7/4.8, Opus 5, Sonnet 5, Fable 5,
  Mythos). Those models reject the parameter with HTTP 400
  (`` `temperature` is deprecated for this model. ``), which made every request
  fail. The provider now decides up front from the model name whether to send
  `temperature`, and if the API rejects it at runtime anyway the parameter is
  dropped and the request retried once - the provider remembers the outcome so
  later calls skip it entirely. Applies to `chat`, `achat`, `stream_chat`, and
  `astream_chat`; a streaming retry is only attempted when nothing has been
  yielded yet. Documented the behaviour in `doc/cli.md`.

## [0.5.3] - 2026-06-21

### Fixed
- The `allowed_objects` warning still surfaced because `langchain_core`, when
  imported (after birdie), calls `surface_langchain_deprecation_warnings()`,
  which prepends a `"default"` filter ahead of ours. Birdie now imports
  `langchain_core` first and registers its ignore filter afterwards, so the
  filter stays in front and the warning is suppressed for real. Added a
  subprocess regression test that reproduces the langchain import ordering.

## [0.5.2] - 2026-06-21

### Fixed
- The `allowed_objects` warning suppression added in 0.5.1 did not take effect
  on environments where langchain's `LangChainPendingDeprecationWarning`
  descends from `DeprecationWarning` rather than `PendingDeprecationWarning`.
  The filter now matches on the message only (against the base `Warning`
  class), so it suppresses the warning regardless of the concrete subclass.

## [0.5.1] - 2026-06-21

### Fixed
- Suppress the `LangChainPendingDeprecationWarning` about the future default
  of `allowed_objects` emitted by newer langgraph versions when they build a
  `JsonPlusSerializer` inside every checkpointer. Birdie never constructs that
  serializer itself and the parameter does not exist on the langgraph floor it
  declares, so the warning is filtered at package import.

## [0.5.0] - 2026-06-17

### Added
- MCP Streamable HTTP transport: `mcp_server` blocks now accept
  `transport: streamable_http` (with `http` as a friendly alias) to connect
  to remote MCP servers over the current MCP standard transport.
- New optional `timeout` and `sse_read_timeout` fields (seconds) for the
  `sse` and `streamable_http` transports. Streamable HTTP timeouts are
  converted to `timedelta` as the adapter expects; SSE keeps floats.
- `MCPServerConfig` now validates required fields per transport (`command`
  for stdio, `url` for sse/streamable_http), failing fast at config time
  instead of deep inside the adapter at connection time.

## [0.4.1] - 2026-05-31

### Added
- LTM score threshold: `query()` now filters out entries whose cosine
  similarity to the current message is below `min_score` (default 0.05),
  preventing low-signal queries like greetings from injecting irrelevant
  context on every turn.
- LTM TTL eviction: entries older than `max_age_days` (default 90) are
  dropped on store load and after each `add()`.
- LTM entry cap: after TTL, if more than `max_entries` (default 100)
  remain, the oldest are dropped to keep the newest N.
- All three limits are configurable via `LTMStore` constructor params and
  via JSON provider config keys `ltm_min_score`, `ltm_max_age_days`,
  `ltm_max_entries`.

### Fixed
- `v0.4.0` git tag was created without the `v` prefix; corrected to
  `v0.4.0` for consistency with all prior release tags.

## [0.4.0] - 2026-05-28

### Changed
- Freetext (knowledge) skills are no longer auto-injected based on keyword
  triggers. The LLM now sees a compact `[load: <name>]` hint per skill and
  calls the new `get_skill` tool when it needs the full body. This eliminates
  false-positive injections and keeps the system prompt small by default.
- `Skill.triggers` is retained for backward compatibility with existing
  SKILL.MD files but no longer drives injection.

### Added
- `get_skill` built-in tool: returns the prose body of any allowed freetext
  skill by name. Exposed automatically when at least one loadable skill is
  enabled; no SKILL.MD changes required.
- Progressive skill loading with turn-decay eviction: a loaded skill's body is
  injected into the system prompt for `skill_decay_turns` human turns (default
  5) after the last `get_skill` call; an LRU cap of `skill_max_loaded`
  (default 3) limits simultaneous loaded skills. Both limits are overrideable
  via `config["configurable"]`.
- `Skill.location` field: identifier used in the `[load: <name>]` hint.
  Defaults to the skill name; reserved for future remote skill references.

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
