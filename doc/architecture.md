# Architecture

## Project layout

```
birdie/
├── agent/
│   ├── graph.py        # LangGraph state machine (agent loop)
│   └── run.py          # DynamicAgent - public API
├── agents/
│   └── summarizer/AGENT.MD   # bundled sub-agent
├── cli.py              # Interactive REPL
├── core/
│   ├── models.py       # Skill, SkillTool, AgentDef, MCPServerConfig data models
│   ├── loader.py       # SKILL.MD parser
│   ├── agent_loader.py # AGENT.MD parser
│   ├── agent_registry.py # In-memory agent index and session policy
│   ├── agent_runner.py # AgentDef → async LangChain StructuredTool
│   ├── registry.py     # In-memory skill/tool index
│   ├── policy.py       # Per-session skill access control
│   ├── session.py      # Session persistence (history, LTM, skill/agent grants)
│   ├── adapter.py      # SkillTool → LangChain StructuredTool
│   ├── entrypoints.py  # bash / http / python / grpc resolvers
│   ├── mcp_client.py   # MCP client manager (wraps langchain-mcp-adapters)
│   └── llm_provider.py # Vendor-agnostic LLM abstraction
└── skills/
    ├── weather/SKILL.MD
    ├── filesystem/SKILL.MD
    ├── shell/SKILL.MD
    ├── ssh/SKILL.MD
    ├── todo/SKILL.MD
    ├── duckduckgo/SKILL.MD
    └── mcp_demo/
        ├── SKILL.MD
        └── server.py
```

---

## Agent loop

The agent is a LangGraph `StateGraph` with two nodes and a conditional edge.

```
START
  │
  ▼
┌──────────────────────────────────────┐
│   agent node    call_model()         │
│                                      │
│   1. repair dangling tool calls      │
│   2. resolve skill + agent tools     │
│      (registry + policy)             │
│   3. fetch MCP tools                 │
│   4. build system prompt             │
│      (Tier 0 + 1 + 2 + 3)           │
│   5. call provider.achat()           │
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
│   1. check for repeated tool calls   │
│      (max_tool_repetitions guard)    │
│   2. resolve skill + agent tools     │
│   3. fetch MCP tools                 │
│   4. build LangChain ToolNode        │
│   5. execute called tool(s)          │
│   6. append ToolMessage(s) to state  │
└──────────────────┬───────────────────┘
                   │
                   └─────────────────► agent node (loop)
```

The loop continues until the model returns a message with no `tool_calls`.

### State

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

`Annotated[..., add_messages]` is LangGraph's reducer pattern. When a node returns `{"messages": [new_msg]}`, LangGraph calls `add_messages(current, [new_msg])` to append (not replace). The state is intentionally minimal: only `messages`. Session ID, long-term memory, and skill policy key all flow through `config["configurable"]`.

### Per-turn context via RunnableConfig

Every node function receives `config: RunnableConfig` alongside state:

```python
async def call_model(state: AgentState, config: RunnableConfig) -> dict:
    thread_id = config.get("configurable", {}).get("thread_id", "")
    ltm       = config.get("configurable", {}).get("long_term_memory") or []
    max_reps  = config.get("configurable", {}).get("max_tool_repetitions", 3)
```

The caller sets these when invoking the graph:

```python
run_config = {"configurable": {
    "thread_id": "2026-04-29_1",
    "long_term_memory": ["prefers concise answers"],
    "max_tool_repetitions": 3,
}}
await app.ainvoke(initial_state, run_config)
```

### Infinite loop guard

Before executing tool calls, `execute_tools` walks backward through the message history and counts how many consecutive agent cycles contained the same `(tool_name, args)` pair. If the count exceeds `max_tool_repetitions`, error `ToolMessage`s are injected for all tool calls in the current batch, so the LLM receives the failure and can adjust its strategy.

### Checkpoint repair

If a tool execution is interrupted (process killed, exception) before its `ToolMessage` is written, the checkpoint ends up with an `AIMessage` whose `tool_calls` have no matching `ToolMessage`. Providers like Mistral reject this as a protocol violation.

At the start of every `call_model` invocation, `_repair_dangling_tool_calls` scans the loaded message list, finds any unanswered tool calls, and inserts placeholder `ToolMessage`s immediately after the offending `AIMessage`. The repair messages are returned as part of the state delta and written back to the checkpoint, healing it permanently.

### ToolNode

`ToolNode` is a prebuilt LangGraph node that handles tool dispatch:

1. Reads `state["messages"][-1]` and extracts its `tool_calls`.
2. Looks up each tool call by `name` in the provided tools list.
3. Calls the matched tools (in parallel for multiple tool calls).
4. Wraps each result in a `ToolMessage` with the matching `tool_call_id`.
5. Returns `{"messages": [ToolMessage, ...]}`.

A fresh `ToolNode` is created on every `execute_tools` invocation because the allowed tool set can change between turns.

---

## System prompt

Each `call_model` invocation assembles the system prompt in four tiers.

### Tier 0 - custom instructions

If `.birdie/system_prompt.md` exists in the current working directory, its contents are prepended before all skill context. Re-read on every turn, so changes take effect immediately.

```bash
cat > .birdie/system_prompt.md << 'EOF'
You are a senior Python engineer working on this codebase.
Always prefer readability over cleverness.
EOF
```

### Tier 1 - skill catalog (always present)

A compact bullet list of every skill currently allowed for the session:

```
You have access to the following skills:

- **Shell**: Execute arbitrary shell commands on the local machine.
- **ssh**: Establish and manage SSH connections...  triggers: ssh, remote server, ...
```

### Tier 2 - freetext skill body (on trigger only)

When the most recent `HumanMessage` contains any of a knowledge skill's trigger keywords (case-insensitive substring match), the skill's full Markdown body is appended for that turn only.

### Tier 3 - long-term memory (always present if any)

Notes stored via `/remember` are injected at the end of the system prompt:

```
--- Long-term memory ---
- prefers concise answers without bullet points
- working on a Python project called Birdie
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

| Vendor | Class | Notes |
|---|---|---|
| OpenAI | `OpenAIProvider` | `pip install openai` |
| Azure OpenAI | `AzureOpenAIProvider` | Set `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |
| Anthropic | `AnthropicProvider` | `pip install anthropic` |
| Mistral | `MistralProvider` | `pip install mistralai>=1.0` |
| Google Gemini | `GeminiProvider` | Set `GEMINI_API_KEY` |
| Ollama | `OllamaProvider` | Local server required |
| Any LangChain model | `LangChainProvider` | Wraps any `BaseChatModel` |
| ACP agent | `ACPProvider` | ACP adapter binary on PATH |

Returned `AIMessage` objects carry `usage_metadata` with `input_tokens`, `output_tokens`, and `total_tokens`. The CLI status bar reads these to display live context and spend counters.

### LangChainProvider

For callers with an existing `BaseChatModel`:

```python
from langchain_openai import ChatOpenAI
agent = DynamicAgent(ChatOpenAI(model="gpt-4o"), skills_dir="birdie/skills")
```

Internally, `LangChainProvider` uses `bind_tools` to attach tool schemas and `ainvoke`/`astream` from the standard `Runnable` interface. The native providers (e.g. `MistralProvider`, `AnthropicProvider`) call their vendor SDKs directly and convert to/from LangChain message types manually.

---

## Access control

`SkillPolicy` tracks which skills each session may access. All skills are disabled by default; skills with `enabled_by_default: true` seed the initial set. The policy is consulted on every `call_model` and `execute_tools` invocation so changes take effect immediately on the next turn.

`AgentRegistry` provides the equivalent for sub-agents: session-scoped enable/disable sets, identical semantics to `SkillPolicy`.

In the CLI, the **session ID** is used as the policy key. Each session has fully independent skill and agent grants, persisted to the session JSON and restored on resume.

---

## Memory and sessions

### Short-term memory - LangGraph's checkpointer

All message persistence is delegated to LangGraph's SQLite checkpointer:

```
~/.birdie/sessions/<user_id>/checkpoints.db
```

Before the first node of each turn, the checkpointer loads the full prior history for the current `thread_id`. After each node, the checkpointer saves the state delta. The application only passes the new `HumanMessage`; LangGraph handles the rest.

Each session is a LangGraph **thread**, identified by the session ID. Two different session IDs produce completely independent conversation histories in the same database.

The context window trim happens inside `call_model`, not at the storage layer:

```python
all_messages = list(state["messages"])           # full history from checkpointer
context_msgs = all_messages[-MAX_CONTEXT_MESSAGES:]   # last 20 forwarded to LLM
```

### Long-term memory - user-scoped `memory.json`

Notes added via `/remember` are written to:

```
~/.birdie/sessions/<user_id>/memory.json
```

This store is **user-scoped**, not session-scoped. A fact added during one session is present in all future sessions. The file is a flat list of timestamped entries:

```json
{
  "user_id": "alice",
  "entries": [
    {"id": "a3f1c2b0", "timestamp": "2026-04-29T09:12:00+00:00", "content": "prefers concise answers"}
  ]
}
```

At the start of each turn the CLI reads `memory.json` and forwards the contents as `long_term_memory` through `config["configurable"]`. Long-term memory is never written into the checkpoint and never mixed with message history.

Birdie uses **explicit-only** long-term memory: nothing is extracted or summarised from conversation automatically. Only `/remember` writes to the store.

### Session files - lightweight metadata

Session JSON files store what neither the checkpointer nor the memory file can represent: skill/agent grants and administrative metadata.

```json
{
  "id": "2026-04-29_1",
  "user_id": "alice",
  "created_at": "2026-04-29T08:00:00+00:00",
  "updated_at": "2026-04-29T11:30:00+00:00",
  "turns": 12,
  "enabled_skills": ["Shell", "Filesystem"],
  "disabled_skills": [],
  "enabled_agents": ["CVulnAnalyst"],
  "disabled_agents": []
}
```

There is no `messages` field - history lives in `checkpoints.db`. There is no `memory` field - facts live in `memory.json`.

### File layout

```
~/.birdie/sessions/
  alice/
    checkpoints.db        ← LangGraph SqliteSaver (all sessions as LG threads)
    memory.json           ← user-scoped long-term memory
    2026-04-29_1.json     ← session metadata (skill/agent grants, turn count)
    2026-04-29_2.json
  bob/
    checkpoints.db
    memory.json
    2026-04-29_1.json
```

### The session ID as a shared key

The session ID (`2026-04-29_1`) plays three roles simultaneously:

| Role | Where used | Effect |
|---|---|---|
| LangGraph `thread_id` | `SqliteSaver` checkpointer | Loads and saves this session's message history |
| Policy key | `SkillPolicy`, `AgentRegistry` | Determines which skills and agents are active |
| Filename | `<session_id>.json` | Links to the metadata JSON on disk |

Switching sessions changes all three at once by changing one string.

Session IDs use `YYYY-MM-DD_N` where `N` increments from 1 for each new session on that calendar day.

---

## DynamicAgent API

`DynamicAgent` is the main public API for embedding Birdie in other applications.

```python
from birdie.agent.run import DynamicAgent

# From a provider config dict (or JSON file path, or None to use env vars)
agent = DynamicAgent.from_config(
    {"vendor": "anthropic", "model": "claude-sonnet-4-6"},
    skills_dir="birdie/skills",
)

# Or pass any LangChain BaseChatModel directly
from langchain_openai import ChatOpenAI
agent = DynamicAgent(ChatOpenAI(model="gpt-4o"), skills_dir="birdie/skills")

# Run to completion
result = await agent.invoke("list my files", thread_id="session-1")

# Stream node-level updates
async for update in agent.astream("list my files", thread_id="session-1"):
    node_name = list(update.keys())[0]     # "agent" or "tools"
    new_messages = update[node_name]["messages"]

# Skill and agent management
agent.enable_skill("session-1", "Shell")
agent.disable_skill("session-1", "Weather")
agent.enable_agent("session-1", "Summarizer")
```

For durable history across restarts, pass an `AsyncSqliteSaver` as `checkpointer`:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as cp:
    agent = DynamicAgent.from_config(config, checkpointer=cp)
    result = await agent.invoke("hello", thread_id="my-session")
```
