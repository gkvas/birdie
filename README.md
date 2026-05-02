# Birdie

```
      ___                       ___           ___                       ___     
    /\  \          ___        /\  \         /\  \          ___        /\  \    
   /::\  \        /\  \      /::\  \       /::\  \        /\  \      /::\  \   
  /:/\:\  \       \:\  \    /:/\:\  \     /:/\:\  \       \:\  \    /:/\:\  \  
 /::\~\:\__\      /::\__\  /::\~\:\  \   /:/  \:\__\      /::\__\  /::\~\:\  \ 
/:/\:\ \:|__|  __/:/\/__/ /:/\:\ \:\__\ /:/__/ \:|__|  __/:/\/__/ /:/\:\ \:\__\
\:\~\:\/:/  / /\/:/  /    \/_|::\/:/  / \:\  \ /:/  / /\/:/  /    \:\~\:\ \/__/
 \:\ \::/  /  \::/__/        |:|::/  /   \:\  /:/  /  \::/__/      \:\ \:\__\  
  \:\/:/  /    \:\__\        |:|\/__/     \:\/:/  /    \:\__\       \:\ \/__/  
   \::/__/      \/__/        |:|  |        \::/__/      \/__/        \:\__\    
    ~~                        \|__|         ~~                        \/__/    
```

A LangGraph-based agent that discovers capabilities at runtime from **SKILL.MD** files. Skills, tools, and their execution entrypoints are all declared in plain Markdown - no code changes required to add new capabilities.

---

## Quick start

```bash
pip install -e .

# Mistral
LLM_VENDOR=mistral LLM_MODEL=mistral-large-latest MISTRAL_API_KEY=... birdie

# OpenAI
LLM_VENDOR=openai LLM_MODEL=gpt-4o OPENAI_API_KEY=... birdie

# Anthropic
LLM_VENDOR=anthropic LLM_MODEL=claude-sonnet-4-6 ANTHROPIC_API_KEY=... birdie
```

Or use a full JSON config:

```bash
export LLM_PROVIDER_CONFIG='{"vendor":"mistral","model":"mistral-large-latest","api_key":"...","temperature":0.3}'
birdie
```

---

## Project layout

```
birdie/
├── agent/
│   ├── graph.py      # LangGraph state machine (agent loop)
│   └── run.py        # DynamicAgent - public API
├── cli.py            # Interactive REPL
├── core/
│   ├── models.py     # Skill, SkillTool data models
│   ├── loader.py     # SKILL.MD parser
│   ├── registry.py   # In-memory skill/tool index
│   ├── policy.py     # Per-user/session access control
│   ├── session.py    # Session persistence (history, LTM, skill grants)
│   ├── adapter.py    # SkillTool → LangChain StructuredTool
│   ├── entrypoints.py# bash / http / python / mcp / grpc resolvers
│   └── llm_provider.py # Vendor-agnostic LLM abstraction
└── skills/
    ├── weather/SKILL.MD
    ├── filesystem/SKILL.MD
    ├── shell/SKILL.MD
    └── ssh/SKILL.MD
```

---

## Skill system

The skill system is built in three layers, each sitting on top of the one below:

```
┌──────────────────────────────────────────────────────┐
│  Knowledge skills  (freetext SKILL.MD)               │
│  Domain know-how injected on trigger.                │
│  The LLM uses whatever tool skills are enabled.      │
├──────────────────────────────────────────────────────┤
│  Tool skills  (structured SKILL.MD)                  │
│  Named, schema-typed tools the LLM can call.         │
│  Each tool is wired to exactly one entrypoint.       │
├──────────────────────────────────────────────────────┤
│  Entrypoints  (hardcoded in core/entrypoints.py)     │
│  Fixed execution primitives: bash, http, python, …   │
│  Not declared in SKILL.MD - part of the framework.   │
└──────────────────────────────────────────────────────┘
```

Entrypoints answer *how* to run something. Tool skills answer *what* can be run. Knowledge skills answer *when and why* - and delegate the actual execution back down to a tool skill.

---

### Layer 1 - Entrypoints

Entrypoints are the fixed execution mechanisms built into `core/entrypoints.py`. They are **not declared in SKILL.MD** - they are the substrate that tool skills are built on. A tool skill picks one entrypoint scheme per tool; the framework resolves it at call time.

| Scheme | Format | Behaviour |
|---|---|---|
| `bash:` | `bash:{command template}` | Shell command via `subprocess`. `{arg}` placeholders are substituted from tool-call arguments. Non-zero exit raises `RuntimeError`. |
| `http:get` | `http:get https://host/path` | HTTP GET; kwargs become query parameters. |
| `http:post` | `http:post https://host/path` | HTTP POST; kwargs become the JSON body. |
| `python:` | `python:module.path.function` | Imports the module and calls the function with kwargs. |
| `mcp:` | `mcp:tool_name` | Stub - wire up a real MCP client. |
| `grpc:` | `grpc:package.Service/Method` | Stub - wire up a real gRPC channel. |
| `container:` | `container:image_name` | Stub - wire up Docker/Podman. |

`bash:` is the workhorse for local skills. Arguments are injected via `str.format()`, so `bash:cat {path}` called with `path="/etc/hosts"` becomes `cat /etc/hosts`.

---

### Layer 2 - Tool skills

Tool skills expose callable tools to the LLM. Each tool declares an entrypoint that the framework executes when the LLM calls it. Every skill lives in its own subdirectory as a `SKILL.MD` file.

```markdown
---
name: Shell
version: 1.0.0
description: Execute arbitrary shell commands on the local machine.
tags: [shell, local, system]
enabled_by_default: false
---

## Tools

### run_bash
description: Run a shell command and return its output.
entrypoint: bash:{command}
schema:
  type: object
  properties:
    command:
      type: string
  required: [command]
```

Tags declared on the skill (`tags: [shell, local, system]`) propagate to every tool in the skill. This is what allows knowledge skills to find executor tools without knowing their names - they ask for tools by tag, not by name.

**Frontmatter fields**

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique skill identifier |
| `version` | no | Semver string (default `1.0.0`) |
| `description` | yes | One-line description shown to the LLM |
| `tags` | no | Propagated to all tools; used by knowledge skills for tag-based lookup |
| `enabled_by_default` | no | Whether all users get this skill without an explicit grant (default `true`) |
| `always_inject` | no | Inject the skill's prose body into the system prompt on every turn |

**Tool block fields**

| Field | Required | Description |
|---|---|---|
| `description` | yes | Shown to the LLM as the tool description |
| `entrypoint` | yes | `scheme:target` - see Layer 1 |
| `schema` | yes | JSON Schema object describing the tool's arguments |

---

### Layer 3 - Knowledge skills

Knowledge skills carry no tools of their own. Their SKILL.MD contains only frontmatter and free-form Markdown prose. The prose is injected into the system prompt when a trigger keyword appears in the user's message - it is never sent otherwise.

```markdown
---
name: ssh
description: Establish and manage SSH connections to remote machines.
triggers:
  - ssh
  - remote server
  - remote connection
  - secure shell
---

# SSH Skill

## Capabilities
- Establish SSH connections using password or key-based authentication
- Execute remote commands over SSH
...
```

When the user says anything containing "ssh" or "remote server", the full Markdown body is appended to the system prompt. The LLM then uses whatever tool skills the user has enabled - typically the Shell skill's `run_bash` - to construct and execute the SSH command. No wiring between skills is needed; the LLM connects the knowledge to the available tools itself.

> **Note:** Ensure the Shell skill (or another skill that provides execution tools) is enabled when using knowledge skills that require command execution.

**Frontmatter fields**

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique skill identifier |
| `description` | yes | Compact description included in the Tier 1 skill catalog |
| `triggers` | yes | Keyword phrases; any substring match in the user's message fires the skill |

---

## Skill loading

Skills are loaded **eagerly at startup** from the skills directory via `discover_skills_from_directory`. Every subdirectory containing a `SKILL.MD` is parsed into a `Skill` object and registered in the `SkillRegistry`.

What is lazy is **what the agent does with a skill on any given turn**:

- The skill object (metadata + tool definitions) lives in memory from startup.
- The tool's **executable wrapper** (a LangChain `StructuredTool` backed by the entrypoint resolver) is created fresh on every agent turn in `execute_tools`, because the active tool set can change between turns as users enable/disable skills.
- The skill's **body** (freetext documentation) is read from the `Skill` object in memory but only appended to the system prompt when a trigger keyword fires - it is never sent to the LLM otherwise.

The loading sequence:

```
startup
  └─ discover_skills_from_directory(skills_dir)
       └─ for each subdir/SKILL.MD
            └─ parse_skill_markdown()          ← YAML frontmatter + body parsed once
                 ├─ Skill.tools  populated      (structured skills)
                 └─ Skill.body   populated      (freetext skills)
  └─ SkillRegistry.register_skill(skill)        ← adds to name, tool, and tag indexes
  └─ UserSkillPolicy.set_default_skills(skills) ← seeds enabled_by_default set

per turn (inside LangGraph nodes)
  └─ _get_skill_tools(state)                    ← reads policy + registry (no disk I/O)
  └─ _build_system_prompt(state)                ← reads Skill.body from memory
  └─ ToolNode([fresh StructuredTool, ...])      ← wires entrypoint resolvers
```

---

## Agent loop

The agent is a LangGraph `StateGraph` with two nodes and a conditional edge.

```
START
  │
  ▼
┌─────────────────┐
│   agent node    │  call_model()
│                 │  1. repair dangling tool calls in checkpoint
│                 │  2. resolve allowed tools for this user/session
│                 │  3. build system prompt (Tier 1 + Tier 2)
│                 │  4. call provider.achat()
│                 │  5. append AIMessage (+ any repairs) to state
└────────┬────────┘
         │
         ▼ should_continue()
    last message
    has tool_calls?
         │
    yes  │  no
         │  └──► END
         ▼
┌─────────────────┐
│   tools node    │  execute_tools()
│                 │  1. resolve allowed tools (same logic as agent node)
│                 │  2. build fresh LangChain ToolNode
│                 │  3. execute; on error return error ToolMessage
│                 │  4. append ToolMessage(s) to state
└────────┬────────┘
         │
         └──────────────────────────► agent node (loop)
```

The loop continues until the model returns a message with no `tool_calls`. There is no hard cap on iterations - it is the model's decision to stop.

### State

`AgentState` is a LangGraph `TypedDict`:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

`messages` uses the `add_messages` reducer, which appends new messages and deduplicates by message ID. The state is intentionally minimal: the session ID, long-term memory, and policy key all flow through `config["configurable"]` rather than being stored in the graph state.

History persistence is handled by LangGraph's `SqliteSaver` checkpointer keyed on `config["configurable"]["thread_id"]` (the session ID). When a turn starts, the checkpointer loads the full prior history automatically - the application only passes the new `HumanMessage`. After each node, the checkpointer writes the delta back to disk.

### Checkpoint repair

If a tool execution is interrupted before its `ToolMessage` is written (e.g. process killed, exception before ToolNode finishes), the checkpoint ends up with an `AIMessage` whose `tool_calls` have no matching `ToolMessage`. Providers like Mistral reject this as a protocol violation.

At the start of every `call_model` invocation, `_repair_dangling_tool_calls` scans the loaded message list, finds any unanswered tool calls, and inserts placeholder `ToolMessage`s immediately after the offending `AIMessage`. The repair messages are included in the state delta returned by `call_model`, so they are written to the checkpoint and the conversation history heals permanently on the next save.

---

## Context window and system prompt assembly

Every call to `call_model` constructs a fresh system prompt from the in-memory skill objects. Nothing is read from disk.

### Tier 1 - skill catalog (always present)

A compact bullet list of every skill currently allowed for the user/session:

```
You have access to the following skills:

- **Filesystem**: Local file operations using shell commands.
- **ssh**: Establish and manage SSH connections...  triggers: ssh, remote server, ...
```

This is sent on every turn regardless of what the user said. It is intentionally small - only `name`, `description`, and `triggers` from the frontmatter, not the skill body.

### Tier 2 - freetext skill body (on trigger only)

When the most recent `HumanMessage` contains any of a freetext skill's trigger keywords (case-insensitive substring match), the skill's full Markdown body is appended:

```
--- ssh skill context ---
# SSH Skill

This skill provides capabilities for establishing and managing SSH connections...
[full body]
```

### Tier 3 - long-term memory (on every turn)

Notes stored via `/remember` are injected as a third tier at the end of the system prompt:

```
--- Long-term memory ---
- prefers concise answers without bullet points
- working on a Python project called Birdie
```

These are stored in the user-scoped `memory.json` and survive both restarts and session switches. They are always present once added, regardless of what the user says.

### Full context structure sent to the LLM

```
[system prompt]
  Tier 1: skill catalog
  Tier 2: freetext skill body (if triggered)
  Tier 3: long-term memory notes (if any)

[message history - rolling window, last 20 messages from checkpointer]
  HumanMessage  (turn N-4)
  AIMessage     (tool_calls)
  ToolMessage   (tool result)
  AIMessage     (text response)
  HumanMessage  (turn N)
  ...
  HumanMessage  (current turn)
```

The full history is stored in the checkpointer's SQLite database; the last 20 messages are forwarded to the LLM each turn. Tool results longer than 32,000 characters are truncated with a count of dropped bytes to keep payloads within model limits.

The provider layer converts this LangChain message list into the wire format expected by each vendor (OpenAI, Mistral, Anthropic each have different conventions for tool messages and system prompts).

---

## Access control

`UserSkillPolicy` enforces which skills a user may access. Resolution order (highest priority first):

1. **Explicit disable** - always blocks the skill for that key
2. **Explicit enable** - grants the skill for that key
3. **Global defaults** - skills with `enabled_by_default: true`

In the CLI, the **session ID** is used as the policy key (not the filesystem `--user` value). This means each session has fully independent skill grants: `/enable` and `/disable` affect only the current session and are persisted to the session JSON, so they are restored on resume.

```python
# CLI uses session.id as the key
agent.enable_skill_for_user(session.id, "Filesystem")
agent.disable_skill_for_user(session.id, "Weather")
```

The policy is consulted on every `call_model` and `execute_tools` invocation so changes take effect immediately on the next turn without restarting the agent.

---

## Providers

Birdie wraps each vendor SDK behind a common `LLMProvider` interface:

```python
class LLMProvider(ABC):
    def achat(messages, tools, system_prompt, ...) -> AIMessage: ...
    def supports_tools() -> bool: ...
    def vendor_name -> str: ...
    def model_name -> str: ...
```

| Vendor | Class | Install |
|---|---|---|
| OpenAI | `OpenAIProvider` | `pip install openai` |
| Azure OpenAI | `AzureOpenAIProvider` | no extra dep - set `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |
| Anthropic | `AnthropicProvider` | `pip install anthropic` |
| Mistral | `MistralProvider` | `pip install mistralai>=1.0` |
| Google Gemini | `GeminiProvider` | no extra dep - set `GEMINI_API_KEY` |
| Ollama | `OllamaProvider` | local server required |
| Any LangChain model | `LangChainProvider` | existing `BaseChatModel` |

Returned `AIMessage` objects carry `usage_metadata` with `input_tokens`, `output_tokens`, and `total_tokens` from the API response. The CLI status bar reads these to display live context and spend counters.

---

## CLI

```
birdie [--user USER_ID] [--session-id SESSION_ID] [--skills-dir PATH] [--config FILE]
```

| Flag | Description |
|---|---|
| `--user USER_ID` | Filesystem namespace for sessions (defaults to `$USER`) |
| `--session-id SESSION_ID` | Resume a specific session (e.g. `2026-04-28_1`) |
| `--skills-dir PATH` | Override the default skills directory |
| `--config FILE` | Path to a JSON provider config file |

### Provider configuration

By default Birdie reads `LLM_VENDOR`, `LLM_MODEL`, and the vendor API key from environment variables. The `--config` flag lets you store all of that in a file instead:

```bash
birdie --config ~/.birdie/anthropic.json
```

**Example config files**

Anthropic:
```json
{
  "vendor": "anthropic",
  "model": "claude-sonnet-4-6",
  "api_key": "sk-ant-...",
  "temperature": 0.3
}
```

OpenAI:
```json
{
  "vendor": "openai",
  "model": "gpt-4o",
  "api_key": "sk-...",
  "max_tokens": 4096
}
```

Azure OpenAI:
```json
{
  "vendor": "azure",
  "model": "my-gpt4o-deployment",
  "api_key": "...",
  "base_url": "https://my-resource.openai.azure.com/",
  "api_version": "2024-02-01"
}
```

Google Gemini:
```json
{
  "vendor": "gemini",
  "model": "gemini-2.5-pro",
  "api_key": "AIza..."
}
```

Mistral:
```json
{
  "vendor": "mistral",
  "model": "mistral-large-latest",
  "api_key": "..."
}
```

Ollama (local, no key needed):
```json
{
  "vendor": "ollama",
  "model": "llama3",
  "base_url": "http://localhost:11434/v1"
}
```

**Config fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `vendor` | string | `openai` | `openai` \| `azure` \| `anthropic` \| `mistral` \| `gemini` \| `ollama` \| `langchain` |
| `model` | string | provider default | Model identifier |
| `api_key` | string | from env var | API key (omit to use env var) |
| `base_url` | string | - | Override API endpoint (proxy, local server) |
| `temperature` | float | `0.0` | Sampling temperature (0.0 – 2.0) |
| `max_tokens` | int | - | Max completion tokens |
| `api_version` | string | `2024-02-01` | Azure OpenAI API version |
| `timeout` | float | `120.0` | Request timeout in seconds (Mistral) |

The `LLM_PROVIDER_CONFIG` environment variable accepts an inline JSON string and takes precedence over everything else:

```bash
export LLM_PROVIDER_CONFIG='{"vendor":"anthropic","model":"claude-sonnet-4-6","api_key":"sk-ant-..."}'
birdie
```

**Key bindings**

| Key | Action |
|---|---|
| `Enter` | Submit message |
| `Ctrl+J` | Insert newline (multi-line message) |
| `Ctrl+C` | Quit |

**Slash commands**

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/skills` | List loaded skills with enable/disable status |
| `/tools` | List callable tools for the current session |
| `/enable <Skill>` | Enable a skill (persisted to session) |
| `/disable <Skill>` | Disable a skill (persisted to session) |
| `/remember <text>` | Save a note to long-term memory |
| `/info` | Show user, session ID, turn count, and provider |
| `/session new` | Create a new session and switch to it |
| `/session switch <id>` | Resume an existing session |
| `/session delete <id>` | Delete a session (creates a new one if current) |
| `/session list` | List all sessions for this user |
| `/session info` | Show session metadata (created, turns, memory) |
| `/new` | Alias for `/session new` |
| `/clear` | Clear the screen |
| `/quit` | Exit |

---

## Memory and sessions

### How agents use memory

An LLM has no persistent state of its own - every API call is stateless. Building a useful assistant means giving it two distinct kinds of memory:

**Short-term memory** is the message history forwarded with each request. It contains the conversation so far: the user's messages, the model's replies, tool calls, and tool results. The model uses it to maintain continuity, reference earlier messages, and understand what tools it has already called in this turn. Short-term memory is bounded by the model's context window and must be actively managed for cost.

**Long-term memory** is facts that should survive across conversations and restarts. Rather than accumulating in the message list, they are stored in a separate layer and *injected into the system prompt* on every request. The model sees them as persistent background knowledge. Long-term memory requires an explicit write operation - it does not grow automatically from conversation content.

### Memory in the agentic loop

Each call to `call_model` assembles everything the LLM will see:

```
┌──────────────────────────────────────────────────────────────────────┐
│  What the LLM receives on every call_model() invocation              │
│                                                                      │
│  system prompt (rebuilt from in-memory skill objects, no disk I/O)  │
│    Tier 1  skill catalog      - what tools are available this turn   │
│    Tier 2  knowledge context  - freetext skill body (if triggered)   │
│    Tier 3  long-term memory   - user facts (always present)          │
│                                                                      │
│  message window  (last 20 from the full checkpointed history)        │
│    HumanMessage   "list files in current dir"                        │
│    AIMessage      → calling list_files(path=".")                     │
│    ToolMessage    "main.py\nREADME.md\n..."                          │
│    AIMessage      "Here are the files: ..."                          │
│    HumanMessage   "which is the largest?"   ← current turn          │
└──────────────────────────────────────────────────────────────────────┘
```

Short-term memory is the message window. Long-term memory is the Tier 3 block injected into the system prompt. Both are assembled per-turn at call time from their respective stores.

The agent loop (`START → agent → tools → agent → … → END`) can execute multiple `call_model` invocations per user turn - once for each iteration around the tool loop. Each one receives the same system prompt and the same growing message window, with each completed tool call appended as a new `ToolMessage`.

### Birdie's implementation

#### Short-term memory - LangGraph's `SqliteSaver` checkpointer

Birdie delegates all message persistence to LangGraph's `SqliteSaver`. After every graph node, the checkpointer writes the updated `AgentState` to a per-user SQLite database:

```
~/.birdie/sessions/<user_id>/checkpoints.db
```

Each session is a LangGraph **thread**, identified by the session ID (`2026-04-29_1`). When a turn starts, the checkpointer loads the full prior history for that thread automatically - the application only passes the new `HumanMessage`. The graph receives a complete, accurately-typed message list without any application-level serialization or deserialization.

The context window trim happens inside `call_model`, not at the storage layer:

```python
# graph.py - inside call_model()
all_messages = list(state["messages"])          # full history from checkpointer
context_msgs = all_messages[-MAX_CONTEXT_MESSAGES:]  # last 20 forwarded to LLM
```

The checkpointer retains unlimited history; the 20-message cap controls only what is sent to the provider per turn.

**Checkpoint repair.** If the process dies after the LLM responds (its `AIMessage` is already written to the checkpoint) but before tool execution completes (no `ToolMessage` follows), the checkpoint is left in a state that providers reject as a protocol violation. On the next invocation, `_repair_dangling_tool_calls` detects orphaned tool calls within the context window, synthesises placeholder `ToolMessage`s, and returns them alongside the real response so the checkpoint heals permanently on write.

#### Long-term memory - user-scoped `memory.json`

Notes added via `/remember` are written to a JSON file alongside the session files:

```
~/.birdie/sessions/<user_id>/memory.json
```

This store is **user-scoped**, not session-scoped. A fact added during session `2026-04-29_1` is still present when you run `/session new` or resume `2026-04-29_2` the next day. The file is a flat list of timestamped entries:

```json
{
  "user_id": "alice",
  "entries": [
    {"id": "a3f1c2b0", "timestamp": "2026-04-29T09:12:00+00:00", "content": "prefers concise answers"},
    {"id": "b7e4d9f1", "timestamp": "2026-04-29T10:00:00+00:00", "content": "working on project Birdie"}
  ]
}
```

At the start of each turn the CLI reads `memory.json` and forwards the contents as `long_term_memory` through `config["configurable"]`. The graph reads it from config - not from state - so LTM is never written into the checkpoint and never mixed with message history:

```python
# graph.py - inside _build_system_prompt()
ltm = config.get("configurable", {}).get("long_term_memory") or []
if ltm:
    system += "\n\n--- Long-term memory ---\n"
    system += "\n".join(f"- {entry}" for entry in ltm)
```

Birdie uses **explicit-only** long-term memory: nothing is extracted or summarised from the conversation automatically. Only `/remember` writes to the store, so nothing is recorded without your knowledge.

#### Session files - lightweight metadata

The session JSON files store only what neither the checkpointer nor the memory file can represent: skill grants and administrative metadata. They are small and fully human-readable:

```json
{
  "id": "2026-04-29_1",
  "user_id": "alice",
  "created_at": "2026-04-29T08:00:00+00:00",
  "updated_at": "2026-04-29T11:30:00+00:00",
  "turns": 12,
  "enabled_skills": ["Shell", "Filesystem"],
  "disabled_skills": ["Weather"]
}
```

There is no `messages` field - history lives in `checkpoints.db`. There is no `memory` field - facts live in `memory.json`. The session file is the glue that ties a human-readable ID to these two backing stores.

#### File layout

```
~/.birdie/sessions/
  alice/
    checkpoints.db        ← LangGraph SqliteSaver (all sessions as LG threads)
    memory.json           ← user-scoped long-term memory
    2026-04-29_1.json     ← session metadata (skill grants, turn count)
    2026-04-29_2.json
  bob/
    checkpoints.db
    memory.json
    2026-04-29_1.json
```

#### The session ID as a shared key

The session ID (`2026-04-29_1`) plays three roles simultaneously:

| Role | Where used | Effect |
|---|---|---|
| LangGraph `thread_id` | `SqliteSaver` checkpointer | Loads and saves this session's message history |
| Skill policy key | `UserSkillPolicy` | Determines which skills are active this turn |
| Filename | `<session_id>.json` | Links to the metadata JSON on disk |

Switching sessions changes all three at once - history, skill grants, and metadata - by changing one string.

### Session ID format and lifecycle

Session IDs use `YYYY-MM-DD_N` where `N` increments from 1 for each new session on that calendar day:

```
2026-04-29_1   ← first session on 29 April
2026-04-29_2   ← second session that day
2026-04-30_1   ← first session on 30 April
```

```bash
# Start a new session (default on first run)
birdie --user alice

# Resume a specific session
birdie --user alice --session-id 2026-04-29_1
```

### Using long-term memory

```
you> /remember prefers concise answers without bullet points
you> /remember working on a Python project called Birdie
```

After these commands, every subsequent turn in every session includes:

```
--- Long-term memory ---
- prefers concise answers without bullet points
- working on a Python project called Birdie
```

in the system prompt, for the lifetime of the user (until explicitly removed from `memory.json`).

**Status bar** (bottom of terminal)

```
 anthropic · claude-sonnet-4-6   │   session: 2026-04-29_1   │   ctx: 1,234 tok   │   spent: ↑5,678  ↓1,234 tok
```

- `session` - the active session ID
- `ctx` - input tokens in the most recent API call
- `↑` / `↓` - cumulative input / output tokens this process run

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
