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

A LangGraph-based agent that discovers capabilities at runtime from **SKILL.MD** files. Skills, tools, and their execution entrypoints are all declared in plain Markdown — no code changes required to add new capabilities.

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
│   └── run.py        # DynamicAgent — public API
├── cli.py            # Interactive REPL
├── core/
│   ├── models.py     # Skill, SkillTool data models
│   ├── loader.py     # SKILL.MD parser
│   ├── registry.py   # In-memory skill/tool index
│   ├── policy.py     # Per-user/session access control
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
│  Not declared in SKILL.MD — part of the framework.   │
└──────────────────────────────────────────────────────┘
```

Entrypoints answer *how* to run something. Tool skills answer *what* can be run. Knowledge skills answer *when and why* — and delegate the actual execution back down to a tool skill.

---

### Layer 1 — Entrypoints

Entrypoints are the fixed execution mechanisms built into `core/entrypoints.py`. They are **not declared in SKILL.MD** — they are the substrate that tool skills are built on. A tool skill picks one entrypoint scheme per tool; the framework resolves it at call time.

| Scheme | Format | Behaviour |
|---|---|---|
| `bash:` | `bash:{command template}` | Shell command via `subprocess`. `{arg}` placeholders are substituted from tool-call arguments. Non-zero exit raises `RuntimeError`. |
| `http:get` | `http:get https://host/path` | HTTP GET; kwargs become query parameters. |
| `http:post` | `http:post https://host/path` | HTTP POST; kwargs become the JSON body. |
| `python:` | `python:module.path.function` | Imports the module and calls the function with kwargs. |
| `mcp:` | `mcp:tool_name` | Stub — wire up a real MCP client. |
| `grpc:` | `grpc:package.Service/Method` | Stub — wire up a real gRPC channel. |
| `container:` | `container:image_name` | Stub — wire up Docker/Podman. |

`bash:` is the workhorse for local skills. Arguments are injected via `str.format()`, so `bash:cat {path}` called with `path="/etc/hosts"` becomes `cat /etc/hosts`.

---

### Layer 2 — Tool skills

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

Tags declared on the skill (`tags: [shell, local, system]`) propagate to every tool in the skill. This is what allows knowledge skills to find executor tools without knowing their names — they ask for tools by tag, not by name.

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
| `entrypoint` | yes | `scheme:target` — see Layer 1 |
| `schema` | yes | JSON Schema object describing the tool's arguments |

---

### Layer 3 — Knowledge skills

Knowledge skills carry no tools of their own. Their SKILL.MD contains only frontmatter and free-form Markdown prose. The prose is injected into the system prompt when a trigger keyword appears in the user's message — it is never sent otherwise.

Execution capability is borrowed from a tool skill via `requires_tags`. When the knowledge skill fires, the framework resolves all tools whose tags match `requires_tags` through the tag index and injects them into the tool list for that turn. No tool or skill names are hardcoded anywhere.

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

When the user says anything containing "ssh" or "remote server", the full Markdown body is appended to the system prompt. The LLM then uses whatever tool skills the user has enabled — typically the Shell skill's `run_bash` — to construct and execute the SSH command. No wiring between skills is needed; the LLM connects the knowledge to the available tools itself.

> **Note:** Ensure the Shell skill is enabled when using SSH or similar knowledge skills that require command execution.

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
- The skill's **body** (freetext documentation) is read from the `Skill` object in memory but only appended to the system prompt when a trigger keyword fires — it is never sent to the LLM otherwise.

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

The loop continues until the model returns a message with no `tool_calls`. There is no hard cap on iterations — it is the model's decision to stop.

### State

`AgentState` is a LangGraph `TypedDict`:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id:  Optional[str]
    session_id: Optional[str]
    active_skill_names: Optional[List[str]]  # reserved for future use
```

`messages` uses the `add_messages` reducer, which appends new messages and can update existing ones by ID. This means the full conversation history accumulates in state across turns.

With `use_memory=True` (default), the graph is compiled with a `MemorySaver` checkpointer. Each call to `astream` or `invoke` must pass a `thread_id` in `configurable`. The checkpointer loads the previous state for that thread, the new `HumanMessage` is appended via the reducer, and the updated state is saved after each node completes.

### Checkpoint repair

If a tool execution is interrupted before its `ToolMessage` is written (e.g. process killed, exception before ToolNode finishes), the checkpoint ends up with an `AIMessage` whose `tool_calls` have no matching `ToolMessage`. Providers like Mistral reject this as a protocol violation.

At the start of every `call_model` invocation, `_repair_dangling_tool_calls` scans the loaded message list, finds any unanswered tool calls, and inserts placeholder `ToolMessage`s immediately after the offending `AIMessage`. The repair messages are included in the state delta returned by `call_model`, so they are written to the checkpoint and the conversation history heals permanently on the next save.

---

## Context window and system prompt assembly

Every call to `call_model` constructs a fresh system prompt from the in-memory skill objects. Nothing is read from disk.

### Tier 1 — skill catalog (always present)

A compact bullet list of every skill currently allowed for the user/session:

```
You have access to the following skills:

- **Filesystem**: Local file operations using shell commands.
- **ssh**: Establish and manage SSH connections...  triggers: ssh, remote server, ...
```

This is sent on every turn regardless of what the user said. It is intentionally small — only `name`, `description`, and `triggers` from the frontmatter, not the skill body.

### Tier 2 — freetext skill body (on trigger only)

When the most recent `HumanMessage` contains any of a freetext skill's trigger keywords (case-insensitive substring match), the skill's full Markdown body is appended:

```
--- ssh skill context ---
# SSH Skill

This skill provides capabilities for establishing and managing SSH connections...
[full body]

To execute commands from the ssh context above, use the available tool(s): `run_bash`.
```

The tool hint at the end is built dynamically: Birdie resolves `skill.requires_tags` through the tag index to find the actual tool names at runtime — no names are hardcoded.

### Full context structure sent to the LLM

```
[system prompt]
  Tier 1: skill catalog
  Tier 2: freetext skill body (if triggered)

[message history — full thread from MemorySaver]
  HumanMessage  (turn 1)
  AIMessage     (tool_calls)
  ToolMessage   (tool result)
  AIMessage     (text response)
  HumanMessage  (turn 2)
  ...
  HumanMessage  (current turn)
```

The provider layer converts this LangChain message list into the wire format expected by each vendor (OpenAI, Mistral, Anthropic each have different conventions for tool messages and system prompts).

---

## Access control

`UserSkillPolicy` enforces which skills a user may access. Resolution order (highest priority first):

1. **Per-user explicit disable** — always blocks the skill
2. **Per-user explicit enable** ∪ **session enable** — union of both
3. **Global defaults** — skills with `enabled_by_default: true`

```python
agent.enable_skill_for_user("alice", "Filesystem")   # add to alice's allow-list
agent.disable_skill_for_user("alice", "Weather")     # block even if default-on
agent.enable_skills_for_session("sess42", ["Shell"]) # grant for one session
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
| Anthropic | `AnthropicProvider` | `pip install anthropic` |
| Mistral | `MistralProvider` | `pip install mistralai>=1.0` |
| Google Gemini | `GeminiProvider` | no extra dep — set `GEMINI_API_KEY` |
| Ollama | `OllamaProvider` | local server required |
| Any LangChain model | `LangChainProvider` | existing `BaseChatModel` |

Returned `AIMessage` objects carry `usage_metadata` with `input_tokens`, `output_tokens`, and `total_tokens` from the API response. The CLI status bar reads these to display live context and spend counters.

---

## CLI

```
birdie [--user USER_ID] [--skills-dir PATH]
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
| `/tools` | List callable tools for the current user |
| `/enable <Skill>` | Enable a skill for the current user |
| `/disable <Skill>` | Disable a skill for the current user |
| `/user <id>` | Switch user identity |
| `/new` | Start a fresh conversation thread |
| `/clear` | Clear the screen |
| `/info` | Show current user, thread ID, and provider |
| `/quit` | Exit |

**Status bar** (bottom of terminal)

```
 mistral · mistral-large-latest   │   ctx: 1,234 tok   │   spent: ↑5,678  ↓1,234 tok
```

- `ctx` — input tokens in the most recent API request (current context window usage)
- `↑` — cumulative input tokens sent this session
- `↓` — cumulative output tokens received this session

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
