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
│   ├── graph.py        # LangGraph state machine (agent loop)
│   └── run.py          # DynamicAgent - public API
├── cli.py              # Interactive REPL
├── core/
│   ├── models.py       # Skill, SkillTool, MCPServerConfig data models
│   ├── loader.py       # SKILL.MD parser
│   ├── registry.py     # In-memory skill/tool index
│   ├── policy.py       # Per-user/session access control
│   ├── session.py      # Session persistence (history, LTM, skill grants)
│   ├── adapter.py      # SkillTool → LangChain StructuredTool
│   ├── entrypoints.py  # bash / http / python / grpc resolvers
│   ├── mcp_client.py   # MCP client manager (wraps langchain-mcp-adapters)
│   └── llm_provider.py # Vendor-agnostic LLM abstraction
└── skills/
    ├── weather/SKILL.MD
    ├── filesystem/SKILL.MD
    ├── shell/SKILL.MD
    ├── ssh/SKILL.MD
    └── mcp_demo/
        ├── SKILL.MD    # MCP skill declaration
        └── server.py   # Example stdio MCP server
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
│  Fixed execution primitives: bash, http, python, ...  │
│  Not declared in SKILL.MD - part of the framework.   │
└──────────────────────────────────────────────────────┘
```

Entrypoints answer *how* to run something. Tool skills answer *what* can be run. Knowledge skills answer *when and why* - and delegate the actual execution back down to a tool skill.

### The frontmatter / body boundary

Every SKILL.MD file is split into two distinct parts by the parser in `core/loader.py`. Understanding this boundary is fundamental to understanding how the skill system works.

**Frontmatter** is the YAML block between the opening and closing `---` delimiters. It is parsed at startup into typed Python fields on the `Skill` Pydantic model. Frontmatter fields are structural metadata consumed by application code - the registry, the policy engine, the adapter, and the system prompt builder all read from these fields. Nothing in the frontmatter is ever sent verbatim to the LLM.

**Body** is everything in the file after the closing `---`. It is stored as a single raw Markdown string in `Skill.body`. The body is not parsed, indexed, or processed in any way at load time. It is kept in memory and may be appended verbatim to the system prompt at turn time - but only under specific conditions (see below). The body is the text the LLM reads, not the text the application reads.

```
birdie/skills/ssh/SKILL.MD
─────────────────────────────────────────────────────────
---                          ← frontmatter start
name: ssh                    ← parsed into Skill.name       (used by code)
description: SSH connections ← parsed into Skill.description (used by code)
triggers:                    ← parsed into Skill.triggers    (used by code)
  - ssh
  - remote server
---                          ← frontmatter end / body start
# SSH Skill                  ←
                             ←  stored verbatim in Skill.body
## Capabilities              ←  appended to system prompt
- Establish SSH connections  ←  only when triggered
...                          ←
─────────────────────────────────────────────────────────
```

This boundary is what allows the skill system to be both **efficient** (only send what is needed to the LLM) and **dynamic** (add new skills without changing code - the parser handles any valid SKILL.MD).

---

### Layer 1 - Entrypoints

Entrypoints are the fixed execution mechanisms built into `core/entrypoints.py`. They are **not declared in SKILL.MD** - they are the substrate that tool skills are built on. A tool skill picks one entrypoint scheme per tool; the framework resolves it at call time.

| Scheme | Format | Behaviour |
|---|---|---|
| `bash:` | `bash:{command template}` | Shell command via `subprocess`. `{arg}` placeholders are substituted from tool-call arguments. Non-zero exit raises `RuntimeError`. |
| `http:get` | `http:get https://host/path` | HTTP GET; kwargs become query parameters. |
| `http:post` | `http:post https://host/path` | HTTP POST; kwargs become the JSON body. |
| `python:` | `python:module.path.function` | Imports the module and calls the function with kwargs. |
| `grpc:` | `grpc:package.Service/Method` | Stub - wire up a real gRPC channel. |
| `container:` | `container:image_name` | Stub - wire up Docker/Podman. |

> **MCP tools do not use an entrypoint scheme.** They are declared via `mcp_server` in the SKILL.MD frontmatter and loaded as native LangChain tools by `MCPClientManager`, bypassing the entrypoint resolver entirely. See the [MCP integration](#mcp-integration) section.

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

For a structured tool skill, the parser does the following:

- **Frontmatter** is parsed into `Skill` model fields. `name`, `description`, `tags`, `enabled_by_default`, `always_inject` all become typed attributes.
- **`## Tools` section** is extracted from the body and parsed into a list of `SkillTool` objects, each holding `name`, `description`, `entrypoint`, and `schema`. These are stored in `Skill.tools`.
- **`Skill.body`** is set to `None`. A plain structured skill has no prose to inject - its tools alone are its contribution. No body is stored or ever sent to the LLM.

```
After parsing a structured skill:

Skill.name        = "Shell"
Skill.description = "Execute arbitrary shell commands..."
Skill.tags        = ["shell", "local", "system"]
Skill.tools       = [SkillTool(name="run_bash", entrypoint="bash:{command}", schema={...})]
Skill.body        = None   ← no prose body; nothing to inject
```

The `## Tools` section is deliberately excluded from `Skill.body`. If it were included, it would be sent as raw Markdown to the LLM on injection, which would be confusing and wasteful - the LLM already receives structured tool schemas through the function-calling API.

Tags declared on the skill (`tags: [shell, local, system]`) are propagated to every tool in the skill at registration time. This is what allows knowledge skills to find executor tools without knowing their names - they ask for tools by tag, not by name.

**Frontmatter fields**

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique skill identifier |
| `version` | no | Semver string (default `1.0.0`) |
| `description` | yes | One-line summary - appears in the Tier 1 skill catalog sent to the LLM every turn |
| `tags` | no | Propagated to all tools at registration time; used for tag-based lookup |
| `enabled_by_default` | no | If `true`, all users get this skill without an explicit grant (default `true`) |
| `always_inject` | no | See below - the exception that allows a structured skill to also have a prose body |

**Tool block fields**

| Field | Required | Description |
|---|---|---|
| `description` | yes | Shown to the LLM as the tool's purpose; the LLM uses this to decide when to call the tool |
| `entrypoint` | yes | `scheme:target` - see Layer 1 |
| `schema` | yes | JSON Schema object describing the tool's arguments; used to build the Pydantic `args_schema` |

**The `always_inject` exception**

A structured skill can optionally carry a prose body alongside its tools by setting `always_inject: true`. In this case the parser stores the prose that appears *before* the `## Tools` section in `Skill.body`, and that prose is sent to the LLM on **every single turn**, regardless of what the user said. This is for skills whose instructions are permanently relevant - for example a planning skill that tells the agent how to reason step-by-step, or a persona skill that defines communication style. Such instructions need to be present on every turn, not just when a trigger keyword fires.

```
always_inject structured skill - what gets stored:

Skill.tools = [SkillTool(...), ...]    ← from ## Tools section
Skill.body  = "Always reason step by ← prose before ## Tools
               step before answering"   injected every turn
```

The `## Tools` section itself is never included in `Skill.body` in any case.

---

### Layer 3 - Knowledge skills

Knowledge skills carry no tools of their own. Their SKILL.MD consists of frontmatter plus free-form Markdown prose - no `## Tools` section.

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

For a knowledge skill, the parser does the following:

- **Frontmatter** is parsed into `Skill` model fields as usual.
- **`## Tools` section** does not exist, so `Skill.tools` is an empty list. No `SkillTool` objects are created. Nothing is registered in the tool index.
- **`Skill.body`** is set to the **entire Markdown body** - every character after the closing `---`. This is the full prose text, stored as-is, ready to be appended to the system prompt.

```
After parsing a knowledge skill:

Skill.name     = "ssh"
Skill.triggers = ["ssh", "remote server", "remote connection", "secure shell"]
Skill.tools    = []     ← no tools; nothing registered in the tool index
Skill.body     = "# SSH Skill\n\n## Capabilities\n- Establish SSH..."
                         ← full prose body stored verbatim
```

**Why the body is not always sent**

At startup, every enabled knowledge skill's body is sitting in memory. If all of them were appended to the system prompt on every turn, the prompt would grow with every skill added - potentially thousands of tokens of context that has nothing to do with what the user asked. This is wasteful and quickly becomes the dominant cost driver for API usage.

The trigger mechanism is the solution: `Skill.triggers` is a list of keyword phrases. At the start of every `call_model` invocation, the agent checks whether any trigger phrase appears as a substring of the most recent `HumanMessage` (case-insensitive). If none match, the body is not sent. If any match, the full `Skill.body` is appended to the system prompt for that turn only.

This keeps the baseline system prompt small - just Tier 1 (the compact skill catalog, roughly 50-100 tokens) - while making the full knowledge available on-demand. A session with 10 knowledge skills enabled pays the prose cost only for the specific skills relevant to each turn.

The body injection happens entirely in `_build_system_prompt` in `graph.py`, inside the `call_model` node. It reads from in-memory `Skill` objects - there is no disk I/O:

```python
# graph.py - _build_system_prompt()
for skill in _triggered_freetext(state, allowed):  # trigger matching
    if skill.body:
        system += f"\n\n--- {skill.name} skill context ---\n{skill.body}"
```

**How the knowledge reaches the tools**

The LLM is now responsible for the connection. When the ssh body is injected, the LLM reads it and understands what an SSH connection requires. It then looks at the available tools (the Shell skill's `run_bash`, for example) and constructs the appropriate command. No explicit wiring between the knowledge skill and the tool skill exists in the application code - the LLM makes the connection itself based on the context.

> **Note:** Ensure the Shell skill (or another skill that provides execution tools) is enabled when using knowledge skills that require command execution.

**Frontmatter fields**

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique skill identifier |
| `description` | yes | One-line summary included in the Tier 1 skill catalog - always sent to the LLM |
| `triggers` | yes | Keyword phrases; any case-insensitive substring match in the user's message injects the body |

---

## Skill loading

### What is eager and what is lazy

Understanding which parts of the skill system are initialised at startup versus resolved at turn time is essential to understanding the design.

**Eager - happens once at startup:**

`discover_skills_from_directory` scans every subdirectory of the skills directory, finds `SKILL.MD` files, and calls `parse_skill_markdown` on each. This produces a fully populated `Skill` Pydantic object whose fields are derived entirely from the file:

- `Skill.name`, `Skill.description`, `Skill.tags`, `Skill.triggers`, `Skill.enabled_by_default`, `Skill.always_inject` - from YAML frontmatter
- `Skill.tools` - list of `SkillTool` objects parsed from the `## Tools` section (empty for knowledge skills)
- `Skill.body` - raw Markdown string of the prose body (empty for plain structured skills, full body for knowledge skills, pre-`## Tools` prose for `always_inject` skills)
- `Skill.mcp_server` - parsed from the `mcp_server` frontmatter key if present

After parsing, each `Skill` is registered in the `SkillRegistry` (which builds name, tag, and tool-ownership indexes) and its `mcp_server` config (if any) is registered with `MCPClientManager`. Finally, `UserSkillPolicy.set_default_skills` seeds which skills are on by default.

**After startup, no SKILL.MD file is ever read again.** All turn-time decisions are made from in-memory `Skill` objects.

**Lazy - happens on every agent turn:**

Three things are deliberately deferred to turn time:

1. **Tool schema and execution wiring.** `SkillTool` objects store only the entrypoint string and JSON Schema. The `StructuredTool` wrapper that LangChain's `ToolNode` can actually execute is created fresh in `execute_tools` on every turn. This is necessary because the set of allowed skills is resolved per-turn from the policy engine - a skill can be enabled or disabled between turns, so the tool list cannot be fixed at startup.

2. **Trigger matching and body injection.** The prose in `Skill.body` is never sent to the LLM automatically. On each `call_model` invocation, `_build_system_prompt` compares the most recent `HumanMessage` against the `triggers` list of every allowed knowledge skill. Only matching skills have their body appended to the system prompt for that turn. This is the primary token cost-control mechanism: skills whose knowledge is not relevant to the current message contribute nothing to the prompt.

3. **MCP tool discovery.** `MCPClientManager.get_tools()` is called on every turn. The first call connects to the MCP server and caches the tool list; subsequent calls return the cache. The deferral means the MCP subprocess is not spawned until it is actually needed.

**The loading sequence end to end:**

```
startup
  └─ discover_skills_from_directory(skills_dir)
       └─ for each subdir/SKILL.MD
            └─ parse_skill_markdown()
                 ├─ YAML frontmatter   → Skill model fields (name, tags, triggers, ...)
                 ├─ ## Tools section   → Skill.tools = [SkillTool(...), ...]
                 │                       (empty list for knowledge skills)
                 └─ Markdown body      → Skill.body = "raw prose string"
                                         (None for plain structured skills)
  └─ SkillRegistry.register_skill(skill)
       ├─ _skills[name]          = skill
       ├─ _tools[tool.name]      = tool     (for each tool in skill.tools)
       ├─ _tags_index[tag]      += tool.name (tag→tool mapping)
       └─ _tool_to_skill[tool]   = skill.name
  └─ MCPClientManager.register_server(skill.name, skill.mcp_server)
       (only for skills where mcp_server is set)
  └─ UserSkillPolicy.set_default_skills(skills)
       └─ seeds the enabled_by_default allow-set

per turn (inside LangGraph call_model and execute_tools nodes)
  └─ _get_allowed(config)
       └─ UserSkillPolicy.get_allowed_skills(thread_id)   ← policy lookup, no disk I/O
  └─ _build_system_prompt(state, config)
       ├─ Tier 1: iterate Skill.description for all allowed skills (always)
       ├─ Tier 2a: append Skill.body for always_inject skills (always)
       ├─ Tier 2b: match HumanMessage against Skill.triggers,
       │           append Skill.body for matches only    ← lazy injection
       └─ Tier 3: append long_term_memory from config["configurable"]
  └─ execute_tools()
       ├─ skilltool_to_langchain_tool(t) for each allowed SkillTool
       │   └─ StructuredTool.from_function(...)          ← created fresh each turn
       ├─ MCPClientManager.get_tools()                   ← lazy MCP connection
       └─ ToolNode(all_tools).ainvoke(state)
```

**Why fresh StructuredTool wrappers each turn**

A `StructuredTool` is a LangChain object that bundles a callable, a name, a description, and a Pydantic schema. It could in principle be created once at startup and reused. The reason it is not: the set of tools passed to `ToolNode` must exactly reflect the skills enabled for the current session at the current turn. If a user runs `/disable Shell` mid-session, the next turn's `ToolNode` must not include `run_bash`. Building the list fresh from the policy on every turn is the simplest way to guarantee this without any invalidation logic.

### LangChain API: StructuredTool and Pydantic schema generation

LangChain's `StructuredTool` is the bridge between a declarative `SkillTool` (a name, description, and JSON Schema) and something a `ToolNode` can actually execute. The conversion happens in `core/adapter.py`:

```python
from langchain_core.tools import StructuredTool

tool = StructuredTool.from_function(
    func=_wrapped,            # the callable that runs the entrypoint
    name=skill_tool.name,
    description=skill_tool.description,
    args_schema=create_args_schema(skill_tool.schema),  # Pydantic model
)
```

`StructuredTool.from_function` is the standard factory for tools whose arguments need schema validation. The three things it needs are:

- **`func`** - the Python callable to invoke when the tool is called. Birdie wraps the entrypoint resolver here.
- **`description`** - shown to the LLM as the tool's purpose. The LLM uses this to decide when to call the tool.
- **`args_schema`** - a Pydantic `BaseModel` subclass. LangChain uses this to validate incoming arguments and to generate the JSON Schema it sends to the LLM as the function signature.

The `args_schema` Pydantic class is built dynamically from the JSON Schema in the SKILL.MD using Pydantic's `create_model`:

```python
from pydantic import BaseModel, create_model

# schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

fields = {}
for field_name, field_schema in schema["properties"].items():
    python_type = _TYPE_MAP.get(field_schema.get("type", ""), Any)
    if field_name in schema.get("required", []):
        fields[field_name] = (python_type, ...)      # required: Ellipsis as default
    else:
        fields[field_name] = (Optional[python_type], None)  # optional: None default

DynamicModel = create_model("ToolArgs", **fields)
```

`create_model` is Pydantic's factory for building model classes at runtime without writing class definitions. The resulting class behaves exactly like a hand-written `BaseModel` - LangChain can call `.model_json_schema()` on it and validate tool call arguments against it before execution.

---

## Agent loop

The agent is a LangGraph `StateGraph` with two nodes and a conditional edge.

### LangGraph API: building the graph

The entire agent loop is defined using LangGraph's `StateGraph` builder. The full definition lives in `agent/graph.py` and `agent/run.py`:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# 1. Declare the graph with its state type
workflow = StateGraph(AgentState)

# 2. Register node functions (sync or async callables)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)

# 3. Wire the edges
workflow.add_edge(START, "agent")           # always start at "agent"
workflow.add_conditional_edges(             # after "agent", decide what's next
    "agent",
    should_continue,                        # routing function
    {"tools": "tools", END: END},           # return value -> destination map
)
workflow.add_edge("tools", "agent")         # after tools, always go back to "agent"

# 4. Compile into a runnable with optional persistence
app = workflow.compile(checkpointer=checkpointer)
```

Key concepts:

- **`StateGraph(AgentState)`** - the state type is declared once at construction. Every node receives the full current state and returns a dict of updates (not the full state). LangGraph merges the updates using the field reducers.
- **`add_node(name, func)`** - registers a function as a graph node. The function signature must be `(state, config) -> dict` (or the async equivalent). LangGraph calls it with the current state and run config automatically.
- **`add_conditional_edges(source, router, map)`** - after `source` runs, LangGraph calls `router(state)` and uses the return value as a key into `map` to determine the next node. Returning `END` from the router (or mapping a value to `END`) terminates the graph.
- **`add_edge(a, b)`** - unconditional transition: after `a` always go to `b`.
- **`compile(checkpointer=...)`** - produces a `CompiledGraph` (also a `Runnable`) that can be invoked via `.ainvoke()`, `.astream()`, etc. The checkpointer is wired in here; LangGraph calls it automatically before and after each node.

```
START
  │
  ▼
┌──────────────────────────────────────┐
│   agent node    call_model()         │
│                                      │
│   1. repair dangling tool calls      │
│   2. resolve skill tools             │
│      (registry + policy)             │
│   3. fetch MCP tools                 │
│      (MCPClientManager)              │
│   4. build system prompt             │
│      (Tier 1 + Tier 2 + Tier 3)      │
│   5. call provider.achat()           │
│      with all tools merged           │
│   6. append AIMessage to state       │
└──────────────────┬───────────────────┘
                   │
                   ▼ should_continue()
              last message
              has tool_calls?
                   │
              yes  │  no
                   │  └──► END
                   ▼
┌──────────────────────────────────────┐
│   tools node    execute_tools()      │
│                                      │
│   1. resolve skill tools             │
│   2. fetch MCP tools                 │
│   3. build fresh LangChain ToolNode  │
│      with all tools merged           │
│   4. execute called tool             │
│   5. on error: return error          │
│      ToolMessage so state stays      │
│      balanced and LLM can recover    │
│   6. append ToolMessage(s) to state  │
└──────────────────┬───────────────────┘
                   │
                   └─────────────────► agent node (loop)
```

The loop continues until the model returns a message with no `tool_calls`. There is no hard cap on iterations - it is the model's decision to stop.

### State and the add_messages reducer

`AgentState` is a LangGraph `TypedDict`:

```python
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

The `Annotated[..., add_messages]` syntax is LangGraph's **reducer** pattern. When a node returns `{"messages": [new_msg]}`, LangGraph does not replace the `messages` list - it calls `add_messages(current, [new_msg])` to merge the update. `add_messages` appends new messages and deduplicates by message ID, which means re-delivering the same message (e.g. after a retry) does not create duplicates.

Without a reducer, every node would have to return the complete messages list. With it, each node only returns the delta - new messages to append - and LangGraph handles the merge.

The state is intentionally minimal: only `messages`. The session ID, long-term memory, and policy key all flow through `config["configurable"]` rather than being stored in state, because they are per-invocation context, not persistent conversation data.

### LangGraph API: RunnableConfig and per-turn context

Every node function receives a second argument, `config: RunnableConfig`, alongside state:

```python
from langchain_core.runnables import RunnableConfig

async def call_model(state: AgentState, config: RunnableConfig) -> dict:
    thread_id = config.get("configurable", {}).get("thread_id", "")
    ltm       = config.get("configurable", {}).get("long_term_memory") or []
```

`RunnableConfig` is LangChain's standard carrier for execution-time metadata. The `"configurable"` key is the designated slot for application-defined values that need to flow into graph nodes without being stored in state. LangGraph propagates the same config dict to every node in the graph for a given invocation.

The caller sets these values when invoking the graph:

```python
run_config = {"configurable": {
    "thread_id": "2026-04-29_1",          # identifies the session / checkpoint
    "long_term_memory": ["user fact 1"],  # injected into system prompt, not stored
}}
await app.ainvoke(initial_state, run_config)
```

`thread_id` in `config["configurable"]` is the LangGraph convention for identifying which checkpoint to load and save. Every checkpointer implementation reads this key automatically - the application does not need to pass it separately to the checkpointer.

History persistence is handled by LangGraph's checkpointer keyed on `config["configurable"]["thread_id"]` (the session ID). When a turn starts, the checkpointer loads the full prior history automatically - the application only passes the new `HumanMessage`. After each node, the checkpointer writes the delta back to disk.

### LangGraph API: ToolNode

`ToolNode` is a prebuilt LangGraph node that handles tool execution. Rather than writing the dispatch loop manually, the graph hands a list of LangChain tools to `ToolNode` and delegates the entire execution step to it:

```python
from langgraph.prebuilt import ToolNode

async def execute_tools(state: AgentState, config: RunnableConfig) -> dict:
    langchain_tools = (
        [skilltool_to_langchain_tool(t) for t in skill_tools] + mcp_tools
    )
    tool_node = ToolNode(langchain_tools)
    return await tool_node.ainvoke(state)
```

What `ToolNode` does internally:

1. Reads `state["messages"][-1]` and extracts its `tool_calls` list (each entry has `id`, `name`, and `args`).
2. Looks up each tool call by `name` in the provided tools list.
3. Calls the matched tools (in parallel when multiple tool calls exist in a single `AIMessage`).
4. Wraps each result in a `ToolMessage` with `tool_call_id` matching the original `tool_calls` entry.
5. Returns `{"messages": [ToolMessage, ...]}` - the delta that LangGraph appends to state via the `add_messages` reducer.

The `tool_call_id` pairing is how the LLM knows which result belongs to which call. Providers like Mistral and Anthropic validate that every `tool_calls` entry in an `AIMessage` has exactly one matching `ToolMessage` before they accept the message history - this is why the checkpoint repair step is necessary.

A fresh `ToolNode` is created on every invocation of `execute_tools` rather than once at startup. This is deliberate: the set of allowed tools can change between turns as the user enables or disables skills, so the tool list must be resolved at call time.

### Checkpoint repair

If a tool execution is interrupted before its `ToolMessage` is written (e.g. process killed, exception before ToolNode finishes), the checkpoint ends up with an `AIMessage` whose `tool_calls` have no matching `ToolMessage`. Providers like Mistral reject this as a protocol violation.

At the start of every `call_model` invocation, `_repair_dangling_tool_calls` scans the loaded message list, finds any unanswered tool calls, and inserts placeholder `ToolMessage`s immediately after the offending `AIMessage`. The repair messages are included in the state delta returned by `call_model`, so they are written to the checkpoint and the conversation history heals permanently on the next save.

---

## MCP integration

[Model Context Protocol](https://modelcontextprotocol.io) (MCP) is an open standard that lets a server expose a set of tools over a well-defined wire protocol. Instead of writing a Python function and wiring it into an entrypoint, you run a separate process that speaks MCP - the agent connects to it, discovers the available tools, and calls them exactly as it would call any other tool.

This is useful for showcasing how agents can consume external capability providers: the agent does not need to know how a tool is implemented or where it runs. It only needs to know how to reach the server.

Birdie integrates MCP through [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters), which converts MCP tool definitions into native LangChain `BaseTool` objects. These are merged with the skill tools before each LLM call and each ToolNode invocation, so the model sees them alongside any `bash:` or `python:` tools without any special handling.

### LangChain API: MultiServerMCPClient

MCP integration uses `langchain-mcp-adapters`, a first-party LangChain library that converts MCP tool definitions into native LangChain `BaseTool` objects. The central class is `MultiServerMCPClient`:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "mcp_demo": {
        "transport": "stdio",
        "command": "python",
        "args": ["birdie/skills/mcp_demo/server.py"],
    }
})

# Connects to each server, calls tools/list, returns List[BaseTool]
tools: list[BaseTool] = await client.get_tools()
```

Each call to `get_tools()` opens a fresh session to each server, fetches the tool list, and closes the session. The `BaseTool` objects it returns are fully callable - invoking them opens another fresh session, sends a `tools/call` request, and returns the result. There are no persistent connections to manage.

Because `BaseTool` is the same interface used by `StructuredTool` (and everything else in LangChain's tool ecosystem), the MCP tools slot directly into `ToolNode` alongside the skill tools:

```python
langchain_tools = (
    [skilltool_to_langchain_tool(t) for t in skill_tools]  # StructuredTool
    + mcp_tools                                             # BaseTool from MCP
)
tool_node = ToolNode(langchain_tools)  # ToolNode doesn't care which type
```

For the LLM's tool schema (sent in the API request), Birdie converts `BaseTool` objects to `NormalizedToolDef` dicts using `lc_tool_to_normalized_def`, which reads `tool.args_schema`. MCP tools expose `args_schema` as a plain JSON Schema dict (not a Pydantic class), so the conversion handles both cases:

```python
def lc_tool_to_normalized_def(tool: BaseTool) -> NormalizedToolDef:
    args_schema = tool.args_schema
    if isinstance(args_schema, dict):
        schema = dict(args_schema)          # MCP: already a JSON Schema dict
    else:
        schema = args_schema.model_json_schema()  # StructuredTool: Pydantic model
    return {"name": tool.name, "description": tool.description, "parameters": schema}
```

### How it works end to end

```
Startup
  └─ loader discovers SKILL.MD with mcp_server: frontmatter key
       └─ MCPClientManager.register_server(name, MCPServerConfig)

First tool call (lazy connection)
  └─ MCPClientManager.get_tools()
       └─ MultiServerMCPClient.get_tools()
            └─ spawns server process (stdio) or connects (SSE/HTTP)
            └─ calls tools/list  → gets tool names + schemas
            └─ returns List[BaseTool]  (cached for process lifetime)

Every call_model() invocation
  └─ skill tools  (SkillTool objects from registry)  → NormalizedToolDef list
  └─ MCP tools    (BaseTool objects from manager)    → NormalizedToolDef list
  └─ merged list sent to provider.achat() so LLM sees all tools

Every execute_tools() invocation
  └─ skill tools  → LangChain StructuredTool list
  └─ MCP tools    → BaseTool list (already LangChain-compatible)
  └─ merged list passed to ToolNode for execution
```

The MCP client opens a fresh session for each tool invocation. This is the design pattern recommended by `langchain-mcp-adapters` - it keeps the client stateless and avoids managing long-lived connections.

### Declaring an MCP server in SKILL.MD

Add an `mcp_server` block to the frontmatter. No `## Tools` section is needed - the tools are discovered dynamically from the server at runtime.

**stdio transport** (server runs as a subprocess):

```yaml
---
name: my_tools
version: 1.0.0
description: Tools provided by my MCP server
enabled_by_default: false
mcp_server:
  transport: stdio
  command: python
  args: ["path/to/server.py"]
---
```

**SSE / HTTP transport** (server is a running process you connect to):

```yaml
---
name: remote_tools
version: 1.0.0
description: Tools from a remote MCP server
enabled_by_default: false
mcp_server:
  transport: sse
  url: http://localhost:8080/sse
---
```

**`mcp_server` fields**

| Field | Required | Description |
|---|---|---|
| `transport` | yes | `stdio` or `sse` |
| `command` | stdio only | Executable to launch (e.g. `python`, `node`) |
| `args` | stdio only | List of arguments passed to the command |
| `env` | no | Extra environment variables for the subprocess |
| `cwd` | no | Working directory for the subprocess |
| `url` | sse only | URL of the SSE endpoint |
| `headers` | sse only | HTTP headers to send with the connection |

### Writing an MCP server

An MCP server can be written in any language that has an MCP SDK. The Python SDK makes it very compact using `FastMCP`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my_server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

The function's name, docstring, and type annotations become the tool name, description, and argument schema automatically. The LLM sees exactly what is reflected from the function signature.

### The demo server

`birdie/skills/mcp_demo/` contains a minimal working example:

```python
# birdie/skills/mcp_demo/server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp_demo")

@mcp.tool()
def echo(message: str) -> str:
    """Return the message unchanged."""
    return message

@mcp.tool()
def reverse_string(text: str) -> str:
    """Return the text with characters in reverse order."""
    return text[::-1]

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

The matching `SKILL.MD` points the agent at this server:

```yaml
---
name: mcp_demo
version: 1.0.0
description: Demo tools served via MCP (echo and reverse_string)
enabled_by_default: false
mcp_server:
  transport: stdio
  command: python
  args: ["birdie/skills/mcp_demo/server.py"]
---
```

To try it, enable the skill for your session and call one of the tools:

```
you> /enable mcp_demo
you> reverse "hello world"
```

The agent will call `reverse_string` via MCP and show you `dlrow olleh`.

### Install the MCP extra

MCP support is an optional dependency. Install it alongside Birdie:

```bash
pip install -e ".[mcp]"
```

This adds `mcp` (the official Python SDK) and `langchain-mcp-adapters`. If `mcp_server` is declared in a SKILL.MD but the extra is not installed, `MCPClientManager.get_tools()` raises an `ImportError` with a clear message on first use.

### Where MCP fits in the architecture

The entrypoint resolver (`entrypoints.py`) handles synchronous, stateless execution: it receives a string like `bash:cat {path}` and returns a callable. MCP does not fit this model because it requires an async connection lifecycle and returns pre-built LangChain tool objects rather than raw callables.

Instead, MCP tools are handled by `MCPClientManager` (`core/mcp_client.py`) and merged directly into the graph's tool pools at the point where they are used:

```
core/mcp_client.py     MCPClientManager
                            register_server()  ← called by DynamicAgent at startup
                            get_tools()        ← called by graph.py on every turn

agent/graph.py         call_model()
                            skill tools (from registry) → NormalizedToolDef
                            MCP tools   (from manager)  → NormalizedToolDef
                            all merged → provider.achat()

                       execute_tools()
                            skill tools → LangChain StructuredTool
                            MCP tools   → LangChain BaseTool (already usable)
                            all merged → ToolNode
```

This keeps `MCPClientManager` as a thin, focused module and avoids coupling the entrypoint system to async connection management.

---



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

### LangChain API: the message types

The entire conversation history is a list of `BaseMessage` subclasses from `langchain_core.messages`. These are the same types used across all LangChain integrations and are what every provider converts to and from:

```python
from langchain_core.messages import (
    HumanMessage,    # role: "user"    - the human's input
    AIMessage,       # role: "assistant" - the model's reply or tool call request
    ToolMessage,     # role: "tool"    - the result of a tool invocation
    SystemMessage,   # role: "system"  - injected instructions (Birdie builds this each turn)
    AIMessageChunk,  # streaming delta - accumulated into AIMessage for history
)
```

Each message type maps directly to a role in the provider's wire format. Birdie's `_lc_to_openai_messages` and equivalent functions in each provider translate the LangChain message list to the vendor-specific JSON format before the API call.

`AIMessage` carries `tool_calls` when the model requests a tool. This is a list of dicts with `id`, `name`, and `args`:

```python
AIMessage(
    content="",
    tool_calls=[
        {"id": "call_abc123", "name": "run_bash", "args": {"command": "ls -la"}}
    ]
)
```

`ToolMessage` closes the loop by referencing that same `id`:

```python
ToolMessage(
    content="main.py\nREADME.md",
    tool_call_id="call_abc123",
    name="run_bash",
)
```

This paired `id` relationship is enforced by every provider. `ToolNode` creates `ToolMessage` objects with the correct `tool_call_id` automatically. The checkpoint repair code fills in placeholder `ToolMessage`s when the process was interrupted before `ToolNode` could do so.

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

### LangGraph API: invoking the graph and streaming

`DynamicAgent` exposes two invocation methods that map directly to LangGraph's compiled graph API:

```python
# Run to completion - returns final AgentState
result = await agent.invoke("list my files", thread_id="session-1")
final_messages = result["messages"]

# Stream node-level updates - yields one dict per node execution
async for update in agent.astream("list my files", thread_id="session-1"):
    # update is e.g. {"agent": {"messages": [AIMessage(...)]}}
    # or             {"tools": {"messages": [ToolMessage(...)]}}
    node_name = list(update.keys())[0]
    new_messages = update[node_name]["messages"]
```

Under the hood, `astream` calls `app.astream(initial_state, run_config, stream_mode="updates")`.

**`stream_mode`** controls the granularity of what is yielded:

| Mode | Yields | Use case |
|---|---|---|
| `"updates"` | Dict of `{node_name: state_delta}` after each node | CLI display - know which node ran and what it produced |
| `"values"` | Full `AgentState` after each node | Inspection - always see the complete current state |
| `"messages"` | Individual LLM token chunks during streaming | Token-level streaming to the user |

Birdie uses `"updates"` so the CLI can display tool call names and results as they happen, without waiting for the full turn to complete.

The `initial_state` passed to `astream` or `ainvoke` contains only the new `HumanMessage`. LangGraph loads the existing thread history from the checkpointer and appends to it - the application never manages the full history:

```python
initial_state = {"messages": [HumanMessage(content=message)]}
run_config    = {"configurable": {"thread_id": thread_id, "long_term_memory": ltm}}
async for update in app.astream(initial_state, run_config, stream_mode="updates"):
    ...
```

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

### LangChain API: LangChainProvider and bind_tools

For callers that already have a LangChain `BaseChatModel` (e.g. `ChatOpenAI`, `ChatAnthropic`), `LangChainProvider` wraps it without any native SDK dependency:

```python
from langchain_openai import ChatOpenAI
agent = DynamicAgent(ChatOpenAI(model="gpt-4o"), skills_dir="birdie/skills")
```

Internally, `LangChainProvider` uses two key LangChain patterns:

**`bind_tools`** - attaches a tool schema to the model so the LLM knows what tools are available. It returns a new runnable (the original model is not mutated):

```python
def _with_tools(self, tools: list[NormalizedToolDef] | None):
    if not tools:
        return self._llm
    lc_tools = [_normalized_tool_to_lc_schema(t) for t in tools]
    return self._llm.bind_tools(lc_tools)   # returns Runnable, not BaseChatModel
```

`bind_tools` is the LangChain standard for attaching tool definitions to any chat model. The tools are passed in the API request as the `tools` or `functions` parameter (depending on the provider). When the model decides to call a tool, the response comes back as an `AIMessage` with a populated `tool_calls` field.

**`ainvoke` and `astream`** - the standard `Runnable` interface:

```python
async def achat(self, messages, tools, system_prompt, ...) -> BaseMessage:
    msgs = self._inject_system(messages, system_prompt)
    return await self._with_tools(tools).ainvoke(msgs)

async def astream_chat(self, messages, tools, system_prompt, ...):
    msgs = self._inject_system(messages, system_prompt)
    async for chunk in self._with_tools(tools).astream(msgs):
        yield chunk  # yields AIMessageChunk objects
```

`ainvoke` returns a complete `AIMessage`. `astream` yields `AIMessageChunk` objects - partial tokens that accumulate into the full message. Both are part of LangChain's `Runnable` interface, implemented by every `BaseChatModel`.

The native providers (e.g. `MistralProvider`, `AnthropicProvider`) call their vendor SDKs directly and convert to/from LangChain message types manually, rather than going through `BaseChatModel`. This gives finer control over vendor-specific features (e.g. Anthropic's native system prompt parameter, Mistral's tool_call content handling) while still producing the same `AIMessage` output the graph expects.

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

#### Short-term memory - LangGraph's checkpointer

Birdie delegates all message persistence to LangGraph's checkpointer. After every graph node, the checkpointer writes the updated `AgentState` to a per-user SQLite database:

```
~/.birdie/sessions/<user_id>/checkpoints.db
```

**LangGraph API: checkpointer setup**

LangGraph ships two checkpointer implementations out of the box:

```python
from langgraph.checkpoint.memory import MemorySaver          # in-process, no persistence
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver # SQLite, survives restarts

# In-memory - suitable for tests and one-shot scripts
app = workflow.compile(checkpointer=MemorySaver())

# SQLite - Birdie's production default
async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)
```

The checkpointer is passed to `compile()`. From that point, LangGraph handles all reads and writes automatically:

- **Before the first node**: the checkpointer loads the latest snapshot for the current `thread_id`. The graph sees the restored `AgentState` as if nothing had happened.
- **After each node**: the checkpointer saves the state delta (not the full state). Deltas are merged on read, so the storage is append-only and efficient.
- **No application code needed**: the graph never calls the checkpointer directly. LangGraph's runtime wires it in.

Each session is a LangGraph **thread**, identified by the session ID (`2026-04-29_1`). The thread concept is LangGraph's unit of isolated history: two different `thread_id` values produce two completely independent conversation histories stored in the same database.

When a turn starts, the checkpointer loads the full prior history for that thread automatically - the application only passes the new `HumanMessage`. The graph receives a complete, accurately-typed message list without any application-level serialization or deserialization.

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
pip install -e ".[dev,mcp]"
pytest
```
