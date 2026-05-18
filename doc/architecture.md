# Architecture

This document is the primary technical reference for the Birdie codebase. It covers the full
execution path from a user's typed message to the agent's reply, explains every major subsystem,
and cross-references the source files where each piece is implemented. Read it top-to-bottom
the first time; use the section headings as an index when returning to a specific area.

---

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
│   ├── llm_provider.py # Vendor-agnostic LLM abstraction
│   ├── ltm.py          # LTM store: per-user JSON persistence + semantic query
│   └── retrieval.py    # Embedding and cosine similarity primitives
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

### What lives where

- **`birdie/agent/`** - the LangGraph state machine and the high-level `DynamicAgent` wrapper. Most
  readers should start here: `graph.py` contains essentially all of the interesting runtime logic.
- **`birdie/core/`** - every supporting subsystem: loading skills from disk, resolving tool calls,
  managing sessions, speaking to LLM providers, and the new LTM pipeline.
- **`birdie/skills/`** - plain-text `SKILL.MD` files that declare what the agent can do. No Python
  here; the loader in `core/loader.py` parses them into `Skill` objects at startup.
- **`birdie/agents/`** - bundled sub-agents, each a self-contained `AGENT.MD` file. Sub-agents are
  turned into callable LangChain tools at runtime by `core/agent_runner.py`.
- **`birdie/cli.py`** - the interactive REPL. Nothing here is needed to understand the core
  architecture; it is a consumer of `DynamicAgent` and the session/memory utilities.

---

## Agent loop

The agent is a LangGraph `StateGraph` with two nodes and a conditional edge. Both nodes are defined
in `birdie/agent/graph.py`; the graph is compiled once and reused for every turn of every session.

### Overview

```
START
  │
  ▼
┌──────────────────────────────────────────┐
│   agent node    call_model()             │
│                                          │
│   1. query LTM for semantic context      │
│   2. compact history (if ≥ MAX_MESSAGES) │
│   3. build context window                │
│      (full history, starts at HumanMsg) │
│   4. repair dangling tool calls          │
│   5. resolve skill + agent tools         │
│   6. fetch MCP tools                     │
│   7. build system prompt (Tiers 0-3)     │
│   8. call provider.achat()               │
│   9. append removes + AIMessage to state │
└──────────────────┬───────────────────────┘
                   │
                   ▼ should_continue()
              last message
              has tool_calls?
                   │
              yes  │  no
                   │  └──► END
                   ▼
┌──────────────────────────────────────────┐
│   tools node    execute_tools()          │
│                                          │
│   1. check for repeated tool calls       │
│      (max_tool_repetitions guard)        │
│   2. resolve skill + agent tools         │
│   3. fetch MCP tools                     │
│   4. build LangChain ToolNode            │
│   5. execute called tool(s)              │
│   6. append ToolMessage(s) to state      │
└──────────────────┬───────────────────────┘
                   │
                   └─────────────────► agent node (loop)
```

The loop continues until the model returns a message with no `tool_calls`. The conditional edge
`should_continue()` inspects only the last message in state to make this decision.

### State

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

`Annotated[..., add_messages]` is LangGraph's reducer pattern. When a node returns
`{"messages": [new_msg]}`, LangGraph calls `add_messages(current, [new_msg])` to **append**, not
replace. If the node returns `RemoveMessage` objects (see *Conversation compaction* below),
LangGraph instead **deletes** the matching rows from the checkpoint before appending anything else.

The state is intentionally minimal: only `messages`. Session ID, long-term memory, user identity,
and skill policy key all flow through `config["configurable"]`, not through state.

### Per-turn context via RunnableConfig

Every node function receives `config: RunnableConfig` alongside state. This is how session-specific
data reaches the nodes without polluting the persistent message log:

```python
async def call_model(state: AgentState, config: RunnableConfig) -> dict:
    thread_id = config.get("configurable", {}).get("thread_id", "")
    user_id   = config.get("configurable", {}).get("user_id", "")
    ltm       = config.get("configurable", {}).get("long_term_memory") or []
    max_reps  = config.get("configurable", {}).get("max_tool_repetitions", 3)
```

The caller sets these when invoking the graph:

```python
run_config = {"configurable": {
    "thread_id": "2026-04-29_1",
    "user_id": "alice",
    "long_term_memory": ["prefers concise answers"],
    "max_tool_repetitions": 3,
}}
await app.ainvoke(initial_state, run_config)
```

- `thread_id` is the key into the SQLite checkpointer; it determines which conversation is loaded.
- `user_id` is used to locate the user's `LTMStore` (see *Long-term memory store* below).
- `long_term_memory` carries strings from the user's `memory.json` (the `/remember` store).
- `max_tool_repetitions` caps the infinite-loop guard.

### Step-by-step: what happens inside `call_model()`

This is the most important function in the codebase. Understanding it in order makes everything
else easier to follow.

**Step 1 - Query LTM for semantic context.**
If `user_id` is set in `config["configurable"]`, the node looks up (or creates) the user's
`LTMStore` in a per-graph `_ltm_cache` dictionary. It then extracts the text of the most recent
`HumanMessage` and calls `ltm_store.query(user_text, k=5)` to retrieve the five most relevant
prior compaction entries by cosine similarity. These are formatted with
`ltm_store.format_for_prompt()` and held in a local variable for Tier 3 injection. If `user_id`
is empty, this step is skipped entirely.

**Step 2 - Compact history if needed.**
The full message list is loaded from the LangGraph checkpointer into `all_messages`. If
`len(all_messages) >= MAX_MESSAGES` (100), `compact_history()` is called. This function summarises
the oldest segment of the conversation, stores the result in the LTM store, and returns a list of
`RemoveMessage` objects identifying which messages to delete. The removes are collected and will be
included in the node's return value at the end of this function. See *Conversation compaction*
for the full algorithm.

**Step 3 - Build the context window.**
The full non-compacted checkpoint is forwarded to the LLM. Because compaction already trims old
messages (see *Conversation compaction* below), the live checkpoint only contains the recent
history that fits within the active context. One constraint is applied: the context must start at
a `HumanMessage`. This keeps the sequence well-formed for providers like Mistral that reject
contexts starting with an `AIMessage` or `ToolMessage`. The one-liner that enforces this is:

```python
human_indices = [i for i, m in enumerate(all_messages) if isinstance(m, HumanMessage)]
context_msgs = all_messages[human_indices[0]:] if human_indices else all_messages
```

In practice this slice is almost always a no-op: compaction always splits at `HumanMessage`
boundaries, so the first message retained in the checkpoint after any compaction run is already a
`HumanMessage`.

**Step 4 - Repair dangling tool calls.**
If the process was killed or crashed after an `AIMessage` with `tool_calls` was written to the
checkpoint but before the corresponding `ToolMessage` was written, the checkpoint is in a state
that providers reject. The `_repair_dangling_tool_calls()` helper scans the context window for
exactly this condition and inserts placeholder `ToolMessage`s. These repair messages are included
in the node's return value so the checkpoint is healed permanently.

**Step 5 - Resolve skill and agent tools.**
The node calls into `core/registry.py` and `core/agent_registry.py` to collect all tools that the
current session's skill policy and agent policy allow. This is done on every turn so that enabling
or disabling a skill mid-session takes effect immediately.

**Step 6 - Fetch MCP tools.**
The `MCPClientManager` in `core/mcp_client.py` connects to any configured MCP servers and returns
their tool schemas as LangChain-compatible objects.

**Step 7 - Build the system prompt.**
The five-tier prompt assembly (see *System prompt* below) happens here, incorporating skill
catalog, always-injected skill bodies, triggered skill bodies, manual LTM, and the semantic LTM
entries retrieved in Step 1.

**Step 8 - Call the provider.**
`provider.achat()` receives the trimmed context window, the full tool list, and the assembled
system prompt. If the provider returns HTTP 429 (rate limit), the call is retried with exponential
backoff: 5 seconds, then 15 seconds, then 45 seconds, then a hard failure.

**Step 9 - Return the state delta.**
The node returns `{"messages": compaction_removes + repair_msgs + [response]}`. LangGraph processes
`RemoveMessage` entries first (deleting rows from SQLite), then appends the new messages. This is
atomic from the application's perspective: the checkpoint either contains the full delta or none
of it.

### Infinite loop guard

Before executing tool calls, `execute_tools()` in `birdie/agent/graph.py` walks backward through
the message history and counts how many consecutive agent cycles contained the same
`(tool_name, args)` pair. If the count exceeds `max_tool_repetitions`, error `ToolMessage`s are
injected for all tool calls in the current batch, so the LLM receives the failure and can adjust
its strategy. This prevents runaway loops where the model repeatedly calls the same tool with the
same arguments because it misread the output.

### Checkpoint repair

If a tool execution is interrupted (process killed, exception) before its `ToolMessage` is written,
the checkpoint ends up with an `AIMessage` whose `tool_calls` have no matching `ToolMessage`.
Providers like Mistral reject this as a protocol violation.

At the start of every `call_model` invocation, `_repair_dangling_tool_calls()` in
`birdie/agent/graph.py` scans the loaded message list, finds any unanswered tool calls, and inserts
placeholder `ToolMessage`s immediately after the offending `AIMessage`. The repair messages are
returned as part of the state delta and written back to the checkpoint, healing it permanently.

### ToolNode

`ToolNode` is a prebuilt LangGraph node that handles tool dispatch in `execute_tools()`:

1. Reads `state["messages"][-1]` and extracts its `tool_calls`.
2. Looks up each tool call by `name` in the provided tools list.
3. Calls the matched tools (in parallel for multiple tool calls).
4. Wraps each result in a `ToolMessage` with the matching `tool_call_id`.
5. Returns `{"messages": [ToolMessage, ...]}`.

A fresh `ToolNode` is created on every `execute_tools` invocation because the allowed tool set can
change between turns (the user may enable or disable a skill during a session).

---

## Conversation compaction

This section explains the compaction pipeline in depth. Compaction is one of the more complex
features in Birdie; understanding it requires following the interaction between `birdie/agent/graph.py`,
`birdie/core/ltm.py`, and `birdie/core/retrieval.py`.

### Why compaction exists

LangGraph's SQLite checkpointer stores every message in the conversation history forever. This is
great for durability but causes two problems as sessions grow long:

1. **Growth**: Without pruning, the checkpoint grows without bound. A long-running session
   accumulates hundreds or thousands of messages that are loaded from the database on every turn
   and forwarded to the LLM, increasing both latency and cost.
2. **Cost**: Sending the entire unbounded history to the LLM on every turn is expensive.
   Compaction keeps the checkpoint at a manageable size so that each turn only sends a recent,
   relevant slice of the conversation.

Compaction solves both problems by replacing old messages with a structured summary, then
permanently removing them from the checkpoint. The summary is stored in the LTM store so the
information is not simply discarded - it remains available for semantic retrieval on future turns.

### Thresholds

Three constants in `birdie/agent/graph.py` control when and how much is compacted:

```python
MIN_MESSAGES = 20       # minimum messages to retain after compaction
MAX_MESSAGES = 100      # trigger compaction when stored history reaches this
COMPRESSION_WINDOW = 60 # maximum number of oldest messages to compress per run
```

These are deliberately conservative: compaction only fires when the history is long enough that
there is a meaningful "old" segment to compress, and it always leaves a generous tail of recent
context intact. The invariant after any compaction run is:

```
remaining_messages >= MIN_MESSAGES
compressed_messages <= COMPRESSION_WINDOW
```

All three constants can be overridden at runtime. See *Configuring compaction thresholds* below.

### How the split point is found

Compaction must not split in the middle of a tool-call turn. An `AIMessage` with tool calls is
always followed by one or more `ToolMessage`s that carry the tool outputs the model needs to see
together. Splitting there would corrupt the LLM's context on the next call. Instead,
`compact_history()` always splits at a `HumanMessage` boundary.

The algorithm inside `compact_history()` in `birdie/agent/graph.py`:

1. Collect all indices where a `HumanMessage` appears in `all_messages`.
2. Compute `max_split = min(compression_window, len(all_messages) - min_messages)`. This ensures
   that after removing the first `split_at` messages, at least `min_messages` remain.
3. Walk the `HumanMessage` indices in reverse, finding the largest index that is `> 0` and
   `<= max_split`. This becomes `split_at`.
4. Take `all_messages[:split_at]` as the segment to compress.
5. If fewer than `min_messages // 2` messages fall in this segment, skip compaction - the
   overhead of an extra LLM call is not worth it for a small saving.

### The compaction prompt

The segment is rendered as a readable transcript (each message prefixed by its role), then sent to
the LLM with a structured prompt requesting a JSON object with exactly six fields:

```json
{
  "summary": "2-4 sentence narrative of what happened in this segment",
  "extracted_facts": ["specific fact, decision, or named value"],
  "user_preferences": ["how the user likes things done"],
  "world_facts": ["factual observation about the external world"],
  "tool_results": ["key finding or outcome from a tool call"],
  "open_tasks": ["task mentioned but not yet completed"]
}
```

These six categories capture the information that is most likely to be useful on future turns:
- `summary` gives the model a prose narrative it can include in its reasoning.
- `extracted_facts` preserves named values (ports, file paths, variable names) that the user
  mentioned and would not want to repeat.
- `user_preferences` preserves stylistic preferences the model should carry forward.
- `world_facts` preserves observations about the environment.
- `tool_results` preserves non-obvious outcomes (for example, the output of a `strace` run).
- `open_tasks` flags work in progress so the model can pick it up without being reminded.

### Parsing the compaction response

The response parser `_parse_compaction_json()` in `birdie/agent/graph.py` uses a two-level
fallback strategy because LLMs do not always return pure JSON:

1. Try `json.loads()` directly on the full response text.
2. If that fails, use a regex to find the first `{...}` block in the response and try
   `json.loads()` on that substring. This handles the common case where the model wraps the JSON
   in a prose sentence like "Here is the structured summary: { ... }".
3. If both fail, use the raw response text as the `summary` field and empty lists for all other
   fields. This is a last resort: the information is not lost, but it is not structured.

### Storing the result and removing old messages

After a successful compaction:

1. The parsed result is passed to `ltm_store.add(compaction_result)`, which appends a new
   `LTMEntry` to the user's JSON file at `~/.birdie/ltm/<user_id>.json` (see *Long-term memory
   store* below).
2. A `RemoveMessage` is created for every message in the compressed segment that has a non-`None`
   `id`. `RemoveMessage` is a LangGraph primitive: when the node returns these in its state delta,
   the checkpointer deletes the corresponding rows from the SQLite store permanently.

```python
remove_msgs = [RemoveMessage(id=m.id) for m in old_msgs if m.id is not None]
```

3. The node returns `{"messages": compaction_removes + repair_msgs + [response]}`. LangGraph
   processes the `RemoveMessage` entries first (deleting rows from the checkpoint) and then appends
   the new messages. Both operations happen in the same checkpoint write.

### Configuring compaction thresholds

The three thresholds (`min_messages`, `max_messages`, `compression_window`) can be set in the
JSON provider config file passed to `DynamicAgent.from_config()`:

```json
{
  "vendor": "anthropic",
  "model": "claude-sonnet-4-6",
  "min_messages": 10,
  "max_messages": 50,
  "compression_window": 30
}
```

They are extracted from the config dict in `DynamicAgent.from_config()` before the remaining
config is forwarded to the vendor SDK (see `birdie/agent/run.py`), so they never pollute the LLM
provider:

```python
_AGENT_FIELDS = {"min_messages", "max_messages", "compression_window"}
min_messages = int(config_dict.get("min_messages") or MIN_MESSAGES)
max_messages = int(config_dict.get("max_messages") or MAX_MESSAGES)
compression_window = int(config_dict.get("compression_window") or COMPRESSION_WINDOW)
provider_config_clean = {k: v for k, v in config_dict.items() if k not in _AGENT_FIELDS}
```

The values are also wired through `DynamicAgent.__init__()`, `create_agent_graph()`, and
`compact_history()` so all compaction paths - automatic and manual - honour the same settings.

| Field | Default | Effect |
|---|---|---|
| `min_messages` | `20` | Minimum messages to retain in the checkpoint after any compaction run |
| `max_messages` | `100` | Trigger automatic compaction when stored history reaches this count |
| `compression_window` | `60` | Maximum number of oldest messages to compress in a single run |

A low `max_messages` (e.g. 30) is useful for agents that need very tight context budgets.
A high `min_messages` (close to `max_messages`) reduces how aggressively old context is pruned.

### Automatic vs manual compaction

**Automatic compaction** is triggered inside `call_model()` whenever
`len(all_messages) >= MAX_MESSAGES`. The user sees no interruption; the compaction happens
silently before the LLM is called for the current turn. From the user's perspective, the
conversation simply continues; behind the scenes the checkpoint has shrunk.

**Manual compaction** is triggered by the `/compact` slash command in `birdie/cli.py`. The CLI
calls `DynamicAgent.compact_session(thread_id, user_id)`, which:

1. Loads the current checkpoint with `app.aget_state(run_config)`.
2. Calls `compact_history(all_messages, provider, ltm_store=ltm_store, force=True)`. The
   `force=True` flag bypasses the `MAX_MESSAGES` threshold so compaction runs regardless of
   history length.
3. Writes the `RemoveMessage` list back to the checkpoint with
   `app.aupdate_state(run_config, {"messages": removes})`.
4. Returns `(n_removed, summary_text)` to the CLI, which displays the result to the user.

The `/compact` command is implemented in the `_compact()` method of the CLI in `birdie/cli.py`
and dispatched from the main input loop.

---

## System prompt

Each `call_model()` invocation in `birdie/agent/graph.py` assembles the system prompt fresh from
five tiers. This is done by `_build_system_prompt()`. The tiers are evaluated in order and
concatenated; absent tiers are omitted without leaving a gap.

### Tier 0 - custom instructions

If `.birdie/system_prompt.md` exists in the current working directory, its contents are prepended
before all skill context. The file is re-read on every turn, so changes take effect immediately
without restarting the agent.

```bash
cat > .birdie/system_prompt.md << 'EOF'
You are a senior Python engineer working on this codebase.
Always prefer readability over cleverness.
EOF
```

This is the recommended way to give the agent project-specific instructions. The file is
intentionally project-local (`.birdie/`, which should be in `.gitignore`) so different projects
can have different personas without touching global config.

### Tier 1 - skill catalog (always present)

A compact bullet list of every skill currently allowed for the session:

```
You have access to the following skills:

- **Shell**: Execute arbitrary shell commands on the local machine.
- **ssh**: Establish and manage SSH connections...  triggers: ssh, remote server, ...
```

This tier is generated from the in-memory `SkillRegistry` (`core/registry.py`) filtered through
the session's `SkillPolicy` (`core/policy.py`). Skills that are disabled for the current session
are not listed here, so the model does not attempt to call tools it cannot use.

### Tier 2a - always_inject skill bodies

Skills with `always_inject: true` in their SKILL.MD frontmatter have their full prose body
appended on every turn, regardless of what the user said. This is useful for planning or meta
skills whose instructions must always be present in the model's context (for example, a skill
that defines how the agent should break down complex tasks).

### Tier 2b - freetext skill body (on trigger only)

When the most recent `HumanMessage` contains any of a knowledge skill's trigger keywords (a
case-insensitive substring match against the `triggers:` list in the SKILL.MD frontmatter), the
skill's full Markdown body is appended for that turn only. On turns where the keyword is not
present, the body is omitted to save context tokens.

This design keeps the context window lean: large reference documents are only loaded when the
model is likely to need them.

### Tier 3 - long-term memory (manual and semantic, merged)

Two sources are merged into a single `--- Long-term memory ---` block at the end of the system
prompt:

1. **Manual entries**: strings passed via `config["configurable"]["long_term_memory"]`. These come
   from the user's `memory.json` file, populated by the `/remember` command in the CLI. They are
   plain strings, user-authored and unstructured.
2. **Semantic entries**: the top-5 most relevant `LTMEntry` objects retrieved from the `LTMStore`
   via cosine similarity on the current user message (see *Long-term memory store* below). These
   are structured compaction results and are formatted by `ltm_store.format_for_prompt()`.

Example output when both sources are present:

```
--- Long-term memory ---
- prefers concise answers without bullet points   ← manual /remember entry
- The user discussed debugging an async Python service that hangs on startup.
  Facts: asyncio event loop; used uvicorn; port 8000
  Preferences: wants root cause, not workarounds             ← compaction entry
```

Both sources are always included when available. If neither is set (no `/remember` entries, no
compaction results, or no `user_id`), the tier is omitted entirely.

---

## Long-term memory store

The LTM store is a per-user, persistent store for structured semantic memory created by the
compaction pipeline. Unlike the manual `memory.json` store (which is a flat list of user-written
strings), the LTM store contains rich structured objects with embedding vectors for semantic
retrieval.

The implementation spans two files in `birdie/core/`:

- `retrieval.py` - the embedding function and cosine similarity primitive
- `ltm.py` - the `LTMEntry` dataclass and `LTMStore` class

### File location

```
~/.birdie/ltm/<user_id>.json
```

Each user has exactly one LTM file. All compaction entries for all sessions of that user are
accumulated in this single file. Sessions are not separated at the file level; the retrieval step
finds entries relevant to the current conversation regardless of which session created them.

### The LTMEntry data model (`birdie/core/ltm.py`)

Each entry corresponds to one compaction run. The fields map directly to the six categories in
the compaction prompt:

```python
@dataclass
class LTMEntry:
    id: str               # 8-char UUID prefix for display
    user_id: str
    summary: str          # 2-4 sentence narrative
    extracted_facts: List[str]
    user_preferences: List[str]
    world_facts: List[str]
    tool_results: List[str]
    open_tasks: List[str]
    embedding: List[float]  # 512-dim unit vector for retrieval
    created_at: str         # ISO 8601 UTC timestamp
```

The `embedding` field is computed at write time by passing the entry's `summary` through the
`embed()` function from `core/retrieval.py`. It is stored in the JSON file so retrieval does not
require re-embedding on every query.

### LTMStore persistence (`birdie/core/ltm.py`)

`LTMStore` loads lazily: no disk I/O happens until the first method call. This means creating an
`LTMStore` instance for a user who has no history yet is cheap.

Writes are atomic: the new JSON is written to `<user_id>.json.tmp` and then renamed over the real
file with `os.replace()`. This prevents partial writes from corrupting the store if the process is
killed mid-write.

```python
store = LTMStore(user_id="alice")  # no I/O yet
store.add(compaction_result)       # loads (if not loaded), appends, saves atomically
results = store.query("async python service")  # returns up to k=5 LTMEntry objects
```

The on-disk format is a JSON object with a `user_id` field and an `entries` list:

```json
{
  "user_id": "alice",
  "entries": [
    {
      "id": "a3f1c2b0",
      "user_id": "alice",
      "summary": "The user debugged an async Python service...",
      "extracted_facts": ["uvicorn", "port 8000", "asyncio event loop"],
      "user_preferences": ["wants root cause, not workarounds"],
      "world_facts": [],
      "tool_results": ["strace showed hung syscall at accept4()"],
      "open_tasks": [],
      "embedding": [0.012, -0.034, ...],
      "created_at": "2026-05-18T09:12:00+00:00"
    }
  ]
}
```

### Embedding and retrieval (`birdie/core/retrieval.py`)

The public API from this module is three names:

```python
from birdie.core.retrieval import EMBED_DIM, embed, cosine_similarity
```

**`embed(text: str) -> List[float]`** converts a string to a 512-dimensional unit vector using a
hash-trick bag-of-n-grams projection. No model downloads, no external dependencies, fully
deterministic. The algorithm:

1. Lowercase and split `text` into tokens.
2. Generate unigrams and bigrams:
   `tokens + [(a + " " + b) for a, b in zip(tokens, tokens[1:])]`.
   Bigrams capture short phrases ("open tasks", "async python") that a pure unigram approach
   misses because they can appear as single buckets in the embedding.
3. For each n-gram, compute `SHA-256(n-gram)`. Use the first 3 bytes as a bucket index
   `% EMBED_DIM` (512), and byte 4's lowest bit as the sign (+1 or -1).
4. Accumulate signed increments into a 512-dimensional float vector.
5. L2-normalise the vector so that a plain dot product equals cosine similarity.

The hash-trick approach is a standard technique from large-scale machine learning. Its accuracy
is lower than a trained sentence embedding model, but it is entirely self-contained: there is
no model download, no GPU requirement, and no version pinning concern.

**`cosine_similarity(a, b) -> float`** is a plain dot product of two already-normalised vectors.
Because both vectors are unit-length, the dot product equals cosine similarity and lies in
`[-1.0, 1.0]`. A score near 1.0 means the texts are semantically similar; near 0.0 means
unrelated; near -1.0 is rare for natural-language queries (it would require antipodal n-gram
distributions).

```python
vec_a = embed("the user debugged an async python service")
vec_b = embed("async python asyncio service")
score = cosine_similarity(vec_a, vec_b)  # → closer to 1.0
```

### Per-turn semantic retrieval

On every `call_model()` invocation in `birdie/agent/graph.py`:

1. The `user_id` is read from `config["configurable"]["user_id"]`.
2. The agent graph maintains a per-graph `_ltm_cache: dict[str, LTMStore]` so the same
   `LTMStore` instance is reused across turns (avoiding repeated disk reads for the same user).
3. The text of the most recent `HumanMessage` is extracted and passed to
   `ltm_store.query(user_text, k=5)`.
4. `LTMStore.query()` embeds the query text, then scores every stored `LTMEntry` by cosine
   similarity against the entry's pre-computed embedding, and returns the top `k` entries.
5. The top-5 entries are formatted with `ltm_store.format_for_prompt()` and injected into Tier 3
   of the system prompt alongside the manual `/remember` entries.

If `user_id` is empty, or no `ltm_factory` was provided to `create_agent_graph()`, this entire
step is skipped.

### LTM in DynamicAgent (`birdie/agent/run.py`)

`DynamicAgent.from_config()` automatically creates a default LTM factory:

```python
def ltm_store_factory(uid: str) -> LTMStore:
    return LTMStore(uid)   # uses ~/.birdie/ltm/<uid>.json
```

This factory is passed into `create_agent_graph()` and called the first time a given `user_id` is
seen. The returned `LTMStore` is cached in `_ltm_cache` for the lifetime of the graph.

You can override the factory to use a different storage directory or a custom implementation:

```python
from birdie.core.ltm import LTMStore
from pathlib import Path

custom_factory = lambda uid: LTMStore(uid, storage_dir=Path("/var/lib/birdie/ltm"))
agent = DynamicAgent.from_config(config, ltm_store_factory=custom_factory)
```

Pass `ltm_store_factory=None` (or construct `DynamicAgent` directly without one) to disable the
LTM store entirely. When disabled, compaction still fires and removes messages from the checkpoint,
but the structured summaries are not saved anywhere.

---

## System prompt

Each `call_model()` invocation in `birdie/agent/graph.py` assembles the system prompt fresh from
five tiers. This is done by `_build_system_prompt()`. The tiers are evaluated in order and
concatenated; absent tiers are omitted without leaving a gap.

The prompt is rebuilt on every turn. This means enabling a skill, updating `.birdie/system_prompt.md`,
or adding a `/remember` entry all take effect on the very next message without restarting the agent.

---

## Providers

Birdie wraps each vendor SDK behind a common `LLMProvider` interface defined in
`birdie/core/llm_provider.py`:

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

Returned `AIMessage` objects carry `usage_metadata` with `input_tokens`, `output_tokens`, and
`total_tokens`. The CLI status bar reads these to display live context and spend counters.

The provider abstraction is also what allows Birdie to use a different (typically cheaper) model
for the compaction step than for the main agent loop. The compaction call in `compact_history()`
uses the same `provider` object passed in by the caller, but nothing prevents the caller from
constructing a separate, smaller provider and passing it specifically for compaction.

### LangChainProvider

For callers with an existing `BaseChatModel`:

```python
from langchain_openai import ChatOpenAI
agent = DynamicAgent(ChatOpenAI(model="gpt-4o"), skills_dir="birdie/skills")
```

Internally, `LangChainProvider` uses `bind_tools` to attach tool schemas and `ainvoke`/`astream`
from the standard `Runnable` interface. The native providers (for example `MistralProvider` and
`AnthropicProvider`) call their vendor SDKs directly and convert to/from LangChain message types
manually. The native path is preferred when available because it gives Birdie more control over
error handling and token counting.

---

## Access control

Access control is implemented in `birdie/core/policy.py` (skills) and
`birdie/core/agent_registry.py` (sub-agents).

`SkillPolicy` tracks which skills each session may access. All skills are disabled by default;
skills with `enabled_by_default: true` in their SKILL.MD frontmatter seed the initial allow-set
when a session is first created. The policy is consulted on every `call_model()` and
`execute_tools()` invocation so changes take effect immediately on the next turn.

`AgentRegistry` provides the equivalent for sub-agents: session-scoped enable/disable sets,
identical semantics to `SkillPolicy`.

In the CLI, the **session ID** is used as the policy key. Each session has fully independent
skill and agent grants, persisted to the session JSON file and restored when the session is
resumed. This means you can leave a session with `Shell` enabled, resume it later, and the
skill is still enabled - no need to re-grant it.

The separation between skills (what capabilities the model can invoke) and the system prompt
(what the model knows about those capabilities) is deliberate. Disabling a skill removes it from
both the tool list and the skill catalog in Tier 1, so the model does not attempt to call a tool
that would be rejected.

---

## Memory and sessions

Birdie maintains three separate persistent stores per user. Understanding why they are separate
makes the code easier to navigate.

### Short-term memory - LangGraph's checkpointer

All message persistence is delegated to LangGraph's SQLite checkpointer:

```
~/.birdie/sessions/<user_id>/checkpoints.db
```

Before the first node of each turn, the checkpointer loads the full prior history for the current
`thread_id`. After each node, the checkpointer saves the state delta. The application only passes
the new `HumanMessage`; LangGraph handles loading history, appending the new message, and saving
the result.

Each session is a LangGraph **thread**, identified by the session ID. Two different session IDs
produce completely independent conversation histories in the same database. The thread concept is
built into LangGraph's checkpointer protocol: any checkpointer that implements `get()` and `put()`
methods can be swapped in, including remote databases.

The context window trim happens inside `call_model()`, not at the storage layer:

```python
all_messages = list(state["messages"])              # full history from checkpointer
human_indices = [i for i, m in enumerate(all_messages) if isinstance(m, HumanMessage)]
context_msgs = all_messages[human_indices[0]:] if human_indices else all_messages
```

The checkpointer stores everything; the trim is a read-time operation that happens inside the
node function. When compaction fires, old messages are removed from the checkpointer via
`RemoveMessage` entries (see *Conversation compaction* above), so the checkpoint itself shrinks
over time and does not grow without bound.

### Manual long-term memory - `/remember` and `memory.json`

Notes added via `/remember` are written to:

```
~/.birdie/sessions/<user_id>/memory.json
```

This store is **user-scoped**, not session-scoped. A fact added during one session is present in
all future sessions for that user, regardless of which session ID is active. The file is a flat
list of timestamped entries:

```json
{
  "user_id": "alice",
  "entries": [
    {"id": "a3f1c2b0", "timestamp": "2026-04-29T09:12:00+00:00", "content": "prefers concise answers"}
  ]
}
```

At the start of each turn the CLI reads `memory.json` and forwards the contents as
`long_term_memory` through `config["configurable"]`. These are injected as the manual part of
Tier 3 of the system prompt. Long-term memory is never written into the checkpoint and never
mixed with message history; it is injected fresh on every turn.

### Automatic long-term memory - compaction and `ltm.json`

Memory created by the compaction pipeline is stored at:

```
~/.birdie/ltm/<user_id>.json
```

Unlike `memory.json`, which is a flat list of user-authored strings, this store contains
structured `LTMEntry` objects with rich typed fields and a pre-computed embedding vector for
semantic retrieval. The compaction pipeline writes here automatically whenever it processes a
segment of conversation history; the user does not manage this file directly.

The two stores serve different purposes and are never merged on disk. At prompt-injection time
both are read and combined into a single Tier 3 block. The distinction matters because:

- `memory.json` entries are permanent until the user deletes them - they represent explicit
  user intent.
- `ltm.json` entries are added automatically by the compaction pipeline and accumulate over time.
  They represent what the model observed, not what the user chose to record.

### Session files - lightweight metadata

Session JSON files store what neither the checkpointer nor the memory file can represent: skill
and agent grants and administrative metadata.

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

There is no `messages` field - history lives in `checkpoints.db`. There is no `memory` field -
facts live in `memory.json` and `ltm.json`. Each file has exactly one responsibility.

### File layout

```
~/.birdie/
  sessions/
    alice/
      checkpoints.db        ← LangGraph SqliteSaver (all sessions as LG threads)
      memory.json           ← user-authored long-term memory (/remember)
      2026-04-29_1.json     ← session metadata (skill/agent grants, turn count)
      2026-04-29_2.json
    bob/
      checkpoints.db
      memory.json
      2026-04-29_1.json
  ltm/
    alice.json              ← structured LTM from compaction pipeline
    bob.json
```

The `ltm/` directory is a sibling of `sessions/`, not inside it, because LTM is cross-session
by design: a compaction entry created during session `2026-04-29_1` is eligible for retrieval
during session `2026-04-29_2` or any later session.

### The session ID as a shared key

The session ID (`2026-04-29_1`) plays three roles simultaneously:

| Role | Where used | Effect |
|---|---|---|
| LangGraph `thread_id` | `SqliteSaver` checkpointer | Loads and saves this session's message history |
| Policy key | `SkillPolicy`, `AgentRegistry` | Determines which skills and agents are active |
| Filename | `<session_id>.json` | Links to the metadata JSON on disk |

Switching sessions changes all three at once by changing one string. Session IDs use the format
`YYYY-MM-DD_N` where `N` increments from 1 for each new session on that calendar day.

---

## DynamicAgent API

`DynamicAgent` is the main public API for embedding Birdie in other applications. It is defined
in `birdie/agent/run.py` and provides a clean async interface over the LangGraph state machine.

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

# Run to completion - user_id enables LTM retrieval and compaction storage
result = await agent.invoke("list my files", thread_id="session-1", user_id="alice")

# Stream node-level updates
async for update in agent.astream("list my files", thread_id="session-1", user_id="alice"):
    node_name = list(update.keys())[0]     # "agent" or "tools"
    new_messages = update[node_name]["messages"]

# Force-compact a session's history
n_removed, summary = await agent.compact_session("session-1", user_id="alice")
print(f"Compacted {n_removed} messages. Summary: {summary}")

# Skill and agent management
agent.enable_skill("session-1", "Shell")
agent.disable_skill("session-1", "Weather")
agent.enable_agent("session-1", "Summarizer")
```

### `user_id` parameter

When `user_id` is passed to `invoke()` or `astream()`, it is stored in
`config["configurable"]["user_id"]`. The agent graph uses it to:

- Look up (or create) the user's `LTMStore` in the per-graph cache.
- Query the store for semantically relevant entries before each LLM call.
- Pass the store to `compact_history()` so compaction results are saved to
  `~/.birdie/ltm/<user_id>.json`.

If `user_id` is omitted, LTM retrieval and compaction storage are silently disabled; the agent
works exactly as it did before these features were added.

### `compact_session(thread_id, user_id)` method

Force-compacts the stored history for a session without waiting for the automatic threshold to be
reached. The implementation in `birdie/agent/run.py`:

1. Loads the checkpoint with `app.aget_state({"configurable": {"thread_id": thread_id}})`.
2. Calls `compact_history(all_messages, provider, ltm_store=ltm_store, force=True)`.
3. If any `RemoveMessage` entries were returned, writes them back with
   `app.aupdate_state(run_config, {"messages": removes})`.
4. Returns `(len(removes), summary_text)`.

Returns `(0, "")` when there is nothing to compact (history is shorter than `MIN_MESSAGES // 2`).

### Custom LTM factory

```python
from birdie.core.ltm import LTMStore
from pathlib import Path

agent = DynamicAgent.from_config(
    config,
    ltm_store_factory=lambda uid: LTMStore(uid, storage_dir=Path("/tmp/ltm")),
)
```

Pass `ltm_store_factory=None` to disable the LTM store entirely.

### Durable history across restarts

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as cp:
    agent = DynamicAgent.from_config(config, checkpointer=cp)
    result = await agent.invoke("hello", thread_id="my-session", user_id="alice")
```

When no checkpointer is provided, `DynamicAgent.from_config()` creates an
`AsyncSqliteSaver` pointing at `~/.birdie/sessions/<user_id>/checkpoints.db` automatically. The
explicit form above is useful when you want the database at a custom path, or when you want to
share a single checkpointer across multiple `DynamicAgent` instances.
