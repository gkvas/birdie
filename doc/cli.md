# CLI reference

## Invocation

```
birdie [--user USER_ID] [--session-id SESSION_ID] [--skills-dir PATH]
       [--agents-dir PATH] [--config FILE]
```

| Flag | Description |
|---|---|
| `--user USER_ID` | Filesystem namespace for sessions (defaults to `$USER`) |
| `--session-id SESSION_ID` | Resume a specific session (e.g. `2026-04-28_1`) |
| `--skills-dir PATH` | Replace the built-in skills directory. `~/.birdie/skills/` is always also loaded on top. |
| `--agents-dir PATH` | Replace the built-in agents directory. `~/.birdie/agents/` is always also loaded on top. |
| `--config FILE` | Path to a JSON provider config file |

---

## Provider configuration

Birdie resolves the LLM provider in this priority order:

1. `LLM_PROVIDER_CONFIG` environment variable (full JSON blob - overrides everything)
2. `--config FILE` flag
3. `LLM_VENDOR` + `LLM_MODEL` + vendor API key env vars

### Environment variables per vendor

| Vendor | Required variables |
|---|---|
| Anthropic | `LLM_VENDOR=anthropic`, `LLM_MODEL=claude-sonnet-4-6`, `ANTHROPIC_API_KEY` |
| OpenAI | `LLM_VENDOR=openai`, `LLM_MODEL=gpt-4o`, `OPENAI_API_KEY` |
| Azure OpenAI | `LLM_VENDOR=azure`, `LLM_MODEL=<deployment-name>`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| Mistral | `LLM_VENDOR=mistral`, `LLM_MODEL=mistral-large-latest`, `MISTRAL_API_KEY` |
| Google Gemini | `LLM_VENDOR=gemini`, `LLM_MODEL=gemini-2.5-pro`, `GEMINI_API_KEY` |
| Ollama | `LLM_VENDOR=ollama`, `LLM_MODEL=llama3` (no key; local server must be running) |

### JSON config file

Pass via `--config FILE` or by setting `LLM_PROVIDER_CONFIG="$(cat file.json)"`:

**Anthropic**
```json
{
  "vendor": "anthropic",
  "model": "claude-sonnet-4-6",
  "api_key": "sk-ant-...",
  "temperature": 0.3
}
```

**OpenAI**
```json
{
  "vendor": "openai",
  "model": "gpt-4o",
  "api_key": "sk-...",
  "max_tokens": 4096
}
```

**Azure OpenAI**
```json
{
  "vendor": "azure",
  "model": "my-gpt4o-deployment",
  "api_key": "...",
  "base_url": "https://my-resource.openai.azure.com/",
  "api_version": "2024-02-01"
}
```

**Google Gemini**
```json
{
  "vendor": "gemini",
  "model": "gemini-2.5-pro",
  "api_key": "AIza..."
}
```

**Mistral**
```json
{
  "vendor": "mistral",
  "model": "mistral-large-latest",
  "api_key": "..."
}
```

**Ollama** (local, no key needed)
```json
{
  "vendor": "ollama",
  "model": "llama3",
  "base_url": "http://localhost:11434/v1"
}
```

**ACP agent** (e.g. Claude Code via `claude-agent-acp`)
```json
{
  "vendor": "acp",
  "model": "claude-agent-acp"
}
```

The `model` field is the binary name to spawn. Birdie starts it as a child process and communicates via stdin/stdout (JSON-RPC 2.0 over stdio). The binary must be on PATH.

**Skills with ACP**: Skills enabled via `/skill enable` are available to the ACP agent. When at least one skill is enabled, Birdie starts an MCP server (stdio transport) alongside the ACP subprocess and passes it in the `session/new` handshake. The ACP agent's underlying model can then call skill tools through MCP, and the built-in ACP callbacks (`terminal/create`, `fs/read_text_file`, `fs/write_text_file`) are disabled in favour of the MCP tools. Tool calls are visible in the CLI output.

**Conversation history**: The full conversation history is sent to the ACP subprocess on every turn as a formatted dialogue, giving the model the same context as native providers.

### Config fields

| Field | Type | Default | Description |
|---|---|---|---|
| `vendor` | string | `openai` | `openai` \| `azure` \| `anthropic` \| `mistral` \| `gemini` \| `ollama` \| `langchain` \| `acp` |
| `model` | string | provider default | Model identifier |
| `api_key` | string | from env var | API key (omit to use env var) |
| `base_url` | string | - | Override API endpoint (proxy, local server) |
| `temperature` | float | `0.0` | Sampling temperature (0.0 - 2.0). Ignored for Anthropic models that removed sampling parameters (Opus 4.7+, Opus 5, Sonnet 5, Fable 5, ...) |
| `max_tokens` | int | - | Max completion tokens |
| `api_version` | string | `2024-02-01` | Azure OpenAI API version |
| `timeout` | float | `120.0` | Request timeout in seconds (Mistral) |
| `prompt_cache` | bool | `true` | Anthropic only: place `cache_control` breakpoints on tools, system prompt, and conversation history |

### Agent-level config fields

These fields are extracted from the same JSON config before it is forwarded to the vendor SDK, so they are safe to mix with the provider fields above:

| Field | Default | Description |
|---|---|---|
| `skills_enabled` | `[]` | Skill names granted to every session |
| `agents_enabled` | `[]` | Agent names granted to every session |
| `tool_output_cap` | `20000` | Max characters of tool output stored in history per call |
| `skill_decay_turns` | `5` | Human turns a loaded knowledge skill stays injected |
| `skill_max_loaded` | `3` | Max simultaneously loaded knowledge skills (LRU) |
| `min_messages_auto` | `20` | Messages retained after automatic compaction |
| `min_messages_forced` | `4` | Messages retained after `/compact` |
| `compression_window_size` | `60` | Max oldest messages compressed per run |
| `compaction_token_threshold` | - | Also compact when the last call used at least this many input tokens |
| `ltm_max_age_days` | `90` | Drop LTM entries older than this |
| `ltm_max_entries` | `100` | Keep at most this many LTM entries |
| `ltm_min_score` | `0.05` | Minimum similarity for LTM retrieval |

---

## Slash commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/remember <text>` | Save a note to long-term memory |
| `/compact` | Force-compact the current session's history into long-term memory now |
| `/cost` | Show token usage and estimated cost for this session |
| `/history [N]` | Replay the last N messages of this session (default 10) |
| `/info` | Show user, session ID, turn count, and provider |
| `/cd <path>` | Change working directory (Tab cycles directory completions) |
| `/clear` | Clear the screen |
| `/quit` | Exit |
| **Tools** | |
| `/tool list` | List callable tools for the current session |
| `/tool output full` | Show complete tool output |
| `/tool output short` | Show first 1000 characters + remaining count (default) |
| `/tool output off` | Show only line count, no content |
| `/tool cap <N>` | Cap tool output stored in history at N characters (`off` disables) |
| **Skills** | |
| `/skill list` | List all loaded skills with enabled/disabled status |
| `/skill enable <name>` | Enable a skill (persisted to session). Suggests closest match on miss. |
| `/skill disable <name>` | Disable a skill (persisted to session). |
| `/skill reload` | Re-discover skills from disk (picks up SKILL.MD edits) |
| `/skill new <name>` | Scaffold `~/.birdie/skills/<name>/SKILL.MD` from a template |
| **Agents** | |
| `/agent list` | List all loaded agents with enabled/disabled status |
| `/agent enable <name>` | Enable a sub-agent (persisted to session). Suggests closest match on miss. |
| `/agent disable <name>` | Disable a sub-agent (persisted to session). |
| `/agent output full` | Show complete sub-agent transcript after each invocation |
| `/agent output short` | Show sub-agent transcript truncated to 1000 chars |
| `/agent output off` | Hide sub-agent transcript - show only the final answer (default) |
| **Sessions** | |
| `/session new` | Create a new session and switch to it |
| `/session switch <id>` | Resume an existing session |
| `/session delete <id>` | Delete a session (creates a new one if current) |
| `/session list` | List all sessions for this user |
| `/session info` | Show session metadata (created, turns, memory, enabled skills/agents) |
| `/new` | Alias for `/session new` |
| **Logging** | |
| `/log llm on\|off` | Enable/disable LLM request/response logging to `~/.birdie/llm.log` |
| `/log http on\|off` | Enable/disable raw HTTP body logging to `~/.birdie/http.log` |

---

## Key bindings

| Key | Action |
|---|---|
| `Enter` | Submit message |
| `Ctrl+J` | Insert newline (multi-line message) |
| `Tab` (after `/cd `) | Cycle through directory completions |
| `Ctrl+C` (non-empty input) | Clear the current input line |
| `Ctrl+C` (empty input, first press) | Show hint: "Press Ctrl+C again to exit" |
| `Ctrl+C` (empty input, second press) | Exit |
| `Ctrl+C` (while agent is running) | Cancel the current turn and return to the prompt |

---

## Status bar

```
 anthropic · claude-sonnet-4-6   │   ~/projects/demo   │   session: 2026-04-29_1   │   ctx: 1,234 tok   │   spent: ↑5,678  ↓1,234 tok
```

| Field | Meaning |
|---|---|
| second column | Current working directory (changed with `/cd`) |
| `session` | Active session ID |
| `ctx` | Input tokens in the most recent API call |
| `↑` / `↓` | Cumulative input / output tokens this process run |

---

## Permission prompts

Tools that belong to a skill with a `## Permissions` section require approval before they run. The CLI pauses the turn and asks:

```
Skill MySkill requests permissions: network, filesystem
Tool call: fetch_data(url='https://example.com')
Allow? [y]es once / [a]lways this session / [N]o:
```

`a` stores a standing grant in the session file (`approved_skills`), so the skill is not asked about again when the session is resumed. `n` (or Enter) denies the call; the model receives an error tool result and can adjust.

---

## Sub-agent output

When a sub-agent is invoked, its transcript (tool calls, results, LLM messages) can be displayed at different verbosity levels, controlled independently from regular tool output:

- `/agent output off` (default) - silent; the sub-agent's final answer is shown as a normal tool result
- `/agent output short` - transcript printed as a single block after the sub-agent completes, content truncated to 1000 chars
- `/agent output full` - full untruncated transcript

The transcript is indented to distinguish it from top-level tool output:

```
🐦 CVulnAnalyst(code='...', filename='null_pointer.c')
   [CVulnAnalyst#d67c]
      → run_bash
         command: cat null_pointer.c
      ←
         #include <stdio.h>
         ...
      🐦 Finding 1: Null pointer dereference...

   Finding 1: Null pointer dereference...
```

The header line (`[CVulnAnalyst#d67c]`) is at the same indent as regular tool output (3 spaces). Sub-agent content is at 6 spaces, args/results at 9 spaces.

---

## Conversation compaction

### How automatic compaction works

Birdie stores every message in a SQLite checkpoint (`~/.birdie/sessions/<user>/checkpoints.db`). As sessions grow long, the checkpoint accumulates messages that are loaded and forwarded to the LLM on every turn, increasing both cost and latency. Automatic compaction fires when the stored history reaches **80 messages** (`min_messages_auto` + `compression_window_size`, both configurable) or, when `compaction_token_threshold` is set, when the last model call consumed at least that many input tokens. It runs as a **background task** so the current turn is never blocked; the results are applied on the first turn after it finishes:

1. Finds the largest group of complete turns at the start of the history that can be summarised while leaving at least `min_messages_auto` (default 20) messages behind.
2. Sends that group to the LLM with a structured prompt that extracts six categories: a narrative summary, specific facts, user preferences, world knowledge, tool outcomes, and open tasks.
3. Stores the result as a new entry in `~/.birdie/ltm/<user>.json`, and keeps the narrative summary with the session as a rolling continuity bridge that is injected into the model's context every turn (later compactions fold the previous summary in).
4. Permanently removes the summarised messages from the checkpoint via LangGraph's `RemoveMessage` mechanism.

The result is that very long sessions stay responsive and cheap while key information is preserved both in the session (rolling summary) and in the LTM store, where it can be retrieved by semantic similarity on future turns.

### `/compact` - manual compaction

Run `/compact` at any time to trigger compaction regardless of history length - useful at the natural end of a working session to capture everything before starting fresh:

```
you> /compact
```

Example output when compaction succeeds:

```
Compacted 38 messages into LTM.
The user spent the session debugging an async Python service that hung on
startup. Root cause was identified as a blocking call inside the asyncio
event loop during initialisation. The fix was to move the call to a
thread pool executor.
```

Example output when history is too short:

```
Nothing to compact - history is too short.
```

`/compact` calls `DynamicAgent.compact_session(thread_id, user_id)` (implemented in `birdie/agent/run.py`), which reads the current checkpoint, calls `compact_history(..., force=True)` to bypass the automatic threshold, and writes the resulting `RemoveMessage` entries back to the checkpoint.

### What gets stored in LTM

Each compaction creates one `LTMEntry` (defined in `birdie/core/ltm.py`) with these fields:

| Field | Contents |
|---|---|
| `summary` | 2-4 sentence narrative of the compacted segment |
| `extracted_facts` | Named values, decisions, configuration details |
| `user_preferences` | How the user likes things done |
| `world_facts` | Factual observations about the external environment |
| `tool_results` | Key findings from tool calls (e.g. strace output, test results) |
| `open_tasks` | Tasks mentioned but not completed |

Each entry also stores a 512-dimensional embedding vector (computed by `birdie/core/retrieval.py`) so that future turns can retrieve it by semantic similarity.

### Viewing and managing LTM

The LTM file for your user is at `~/.birdie/ltm/<user>.json`. It is plain JSON and can be inspected or manually edited if needed. There is currently no CLI command to list or delete individual LTM entries.

### Configuring compaction thresholds

The thresholds that control when and how much is compacted can be set in the provider JSON config file (see [Provider configuration](#provider-configuration)):

| Field | Default | Description |
|---|---|---|
| `min_messages_auto` | `20` | Minimum messages to keep in the checkpoint after automatic compaction |
| `min_messages_forced` | `4` | Minimum messages to keep after a forced `/compact` |
| `compression_window_size` | `60` | Maximum number of oldest messages to compress per run |
| `compaction_token_threshold` | - | Also trigger compaction when the last model call consumed at least this many input tokens (disabled when unset) |

Automatic compaction triggers when the history reaches `min_messages_auto + compression_window_size` messages (80 by default), or when the token threshold fires.

Example - a config that compacts more aggressively:

```json
{
  "vendor": "anthropic",
  "model": "claude-sonnet-4-6",
  "min_messages_auto": 10,
  "compression_window_size": 25,
  "compaction_token_threshold": 60000
}
```

These fields are stripped from the config before it is forwarded to the vendor SDK, so they are safe to include alongside vendor-specific fields like `api_key` and `temperature`.
