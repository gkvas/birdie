# birdie-agent Examples

Runnable scripts that show how to use `birdie-agent` programmatically, from a
one-liner hello world to SQLite-persisted sessions and custom skills built from
scratch.

---

## Prerequisites

```bash
# Install birdie-agent (from source in this repo)
pip install -e .

# For the MCP example (10_mcp_tool.py) also install the optional extra
pip install -e ".[mcp]"
```

### Provider credentials

Every example that makes LLM calls reads its provider configuration from
environment variables.  There are two ways to configure a provider:

**Option A - individual variables** (one per vendor):

| Vendor | Environment variables |
|--------|-----------------------|
| Anthropic | `LLM_VENDOR=anthropic` `LLM_MODEL=claude-sonnet-4-6` `ANTHROPIC_API_KEY=sk-ant-…` |
| OpenAI | `LLM_VENDOR=openai` `LLM_MODEL=gpt-4o` `OPENAI_API_KEY=sk-…` |
| Azure OpenAI | `LLM_VENDOR=azure` `LLM_MODEL=<deployment-name>` `AZURE_OPENAI_API_KEY=…` `AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/` |
| Mistral | `LLM_VENDOR=mistral` `LLM_MODEL=mistral-large-latest` `MISTRAL_API_KEY=…` |
| Gemini | `LLM_VENDOR=gemini` `LLM_MODEL=gemini-2.0-flash` `GEMINI_API_KEY=AIza…` |
| Ollama (local) | `LLM_VENDOR=ollama` `LLM_MODEL=llama3` (no key - server must be running) |

Example (Anthropic):

```bash
export LLM_VENDOR=anthropic
export LLM_MODEL=claude-sonnet-4-6
export ANTHROPIC_API_KEY=sk-ant-...
```

Example (Azure OpenAI):

```bash
export LLM_VENDOR=azure
export LLM_MODEL=my-gpt4o-deployment   # your Azure deployment name
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
# Optional: override API version (default: 2024-02-01)
# export AZURE_OPENAI_API_VERSION=2024-05-01-preview
```

**Option B - `LLM_PROVIDER_CONFIG`** (single JSON blob, overrides all other variables):

```bash
export LLM_PROVIDER_CONFIG='{"vendor":"anthropic","model":"claude-sonnet-4-6","api_key":"sk-ant-..."}'
```

Or point to a config file (useful for storing credentials outside shell history):

```bash
export LLM_PROVIDER_CONFIG="$(cat ~/.birdie/anthropic.json)"
```

`LLM_PROVIDER_CONFIG` accepts the same fields as a `--config` file:
`vendor`, `model`, `api_key`, `base_url`, `temperature`, `max_tokens`, `api_version`.
It takes precedence over `LLM_VENDOR`, `LLM_MODEL`, and all vendor-specific key
variables when set.

---

## Example index

| File | API keys needed | What it demonstrates |
|------|----------------|----------------------|
| [01_hello_world.py](01_hello_world.py) | Yes | Simplest possible `DynamicAgent` invocation |
| [02_inspect_skills.py](02_inspect_skills.py) | **No** | Registry and policy API - no LLM calls |
| [03_web_search.py](03_web_search.py) | Yes | DuckDuckGo skill (no extra API key) |
| [04_shell_commands.py](04_shell_commands.py) | Yes | Shell skill for local command execution |
| [05_multi_turn.py](05_multi_turn.py) | Yes | Thread-based conversation continuity |
| [06_streaming.py](06_streaming.py) | Yes | Real-time `astream()` output |
| [07_long_term_memory.py](07_long_term_memory.py) | Yes | `long_term_memory` parameter injection |
| [08_sqlite_persistence.py](08_sqlite_persistence.py) | Yes | Durable sessions with `AsyncSqliteSaver` |
| [09_custom_skill.py](09_custom_skill.py) | Yes | Write a `SKILL.MD` at runtime and run it |
| [10_mcp_tool.py](10_mcp_tool.py) | Yes + `[mcp]` | MCP-backed skill via subprocess server |

---

## Running an example

```bash
# From the repo root
python examples/01_hello_world.py
python examples/02_inspect_skills.py   # no API key required
python examples/03_web_search.py
# ... and so on
```

---

## Key concepts covered

**Agent construction**
- `DynamicAgent.from_config()` - build from env vars or a config dict
- `DynamicAgent(llm, skills_dir=...)` - pass any LangChain `BaseChatModel`

**Skill access control**
- `agent.enable_skill_for_user(session_id, skill_name)` - grant a skill for a session
- `agent.disable_skill_for_user(session_id, skill_name)` - block a skill
- `agent.enable_skills_for_session(session_id, [skill_names])` - fixed skill set
- `agent.registry` / `agent.policy` - inspect loaded skills and current grants

**Invocation**
- `await agent.invoke(message, thread_id=...)` - run to completion, return final state
- `async for update in agent.astream(message, thread_id=...)` - stream node updates

**Memory**
- `thread_id` - identifies the session; the checkpointer loads prior history automatically
- `long_term_memory=[...]` - strings injected into the system prompt every turn
- `AsyncSqliteSaver` - persist history to SQLite across process restarts

**Extensibility**
- Custom `SKILL.MD` with `bash:` entrypoints - no Python code required in the skill
- `python:` entrypoints - call any importable Python function as a tool
- MCP-backed skills - connect to external tool servers via stdio or SSE
