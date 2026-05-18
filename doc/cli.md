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

### Config fields

| Field | Type | Default | Description |
|---|---|---|---|
| `vendor` | string | `openai` | `openai` \| `azure` \| `anthropic` \| `mistral` \| `gemini` \| `ollama` \| `langchain` \| `acp` |
| `model` | string | provider default | Model identifier |
| `api_key` | string | from env var | API key (omit to use env var) |
| `base_url` | string | - | Override API endpoint (proxy, local server) |
| `temperature` | float | `0.0` | Sampling temperature (0.0 - 2.0) |
| `max_tokens` | int | - | Max completion tokens |
| `api_version` | string | `2024-02-01` | Azure OpenAI API version |
| `timeout` | float | `120.0` | Request timeout in seconds (Mistral) |

---

## Slash commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/remember <text>` | Save a note to long-term memory |
| `/compact` | Force-compact the current session's history into long-term memory now |
| `/info` | Show user, session ID, turn count, and provider |
| `/clear` | Clear the screen |
| `/quit` | Exit |
| **Tools** | |
| `/tool list` | List callable tools for the current session |
| `/tool output full` | Show complete tool output |
| `/tool output short` | Show first 1000 characters + remaining count (default) |
| `/tool output off` | Show only line count, no content |
| **Skills** | |
| `/skill list` | List all loaded skills with enabled/disabled status |
| `/skill enable <name>` | Enable a skill (persisted to session). Suggests closest match on miss. |
| `/skill disable <name>` | Disable a skill (persisted to session). |
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
| `Ctrl+C` (non-empty input) | Clear the current input line |
| `Ctrl+C` (empty input, first press) | Show hint: "Press Ctrl+C again to exit" |
| `Ctrl+C` (empty input, second press) | Exit |
| `Ctrl+C` (while agent is running) | Cancel the current turn and return to the prompt |

---

## Status bar

```
 anthropic · claude-sonnet-4-6   │   session: 2026-04-29_1   │   ctx: 1,234 tok   │   spent: ↑5,678  ↓1,234 tok
```

| Field | Meaning |
|---|---|
| `session` | Active session ID |
| `ctx` | Input tokens in the most recent API call |
| `↑` / `↓` | Cumulative input / output tokens this process run |

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

Birdie stores every message in a SQLite checkpoint (`~/.birdie/sessions/<user>/checkpoints.db`). As sessions grow long, the checkpoint accumulates messages that will never be sent to the LLM again (the live context window is capped at 20 messages). Automatic compaction fires when the stored history reaches **100 messages** and silently:

1. Finds the largest group of complete turns at the start of the history that can be summarised without leaving fewer than 40 messages behind.
2. Sends that group to the LLM with a structured prompt that extracts six categories: a narrative summary, specific facts, user preferences, world knowledge, tool outcomes, and open tasks.
3. Stores the result as a new entry in `~/.birdie/ltm/<user>.json`.
4. Permanently removes the summarised messages from the checkpoint via LangGraph's `RemoveMessage` mechanism.

The result is that very long sessions stay responsive and cheap while key information is preserved in the LTM store, where it can be retrieved by semantic similarity on future turns.

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
