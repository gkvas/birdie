# Skills

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
│  Fixed execution primitives: bash, http, python, ... │
│  Not declared in SKILL.MD - part of the framework.   │
└──────────────────────────────────────────────────────┘
```

Entrypoints answer *how* to run something. Tool skills answer *what* can be run. Knowledge skills answer *when and why* - and delegate the actual execution back down to a tool skill.

---

## SKILL.MD format

Every SKILL.MD file is split into two distinct parts.

**Frontmatter** is the YAML block between the opening and closing `---` delimiters. It is parsed at startup into typed Python fields on the `Skill` model. Frontmatter fields are structural metadata consumed by application code - the registry, the policy engine, the adapter, and the system prompt builder. Nothing in the frontmatter is ever sent verbatim to the LLM.

**Body** is everything after the closing `---`. It is stored as a raw Markdown string in `Skill.body`. The body is not parsed or processed at load time. It may be appended verbatim to the system prompt at turn time, but only under specific conditions.

```
birdie/skills/ssh/SKILL.MD
─────────────────────────────────────────────────────────
---                          ← frontmatter start
name: ssh                    ← parsed into Skill.name        (used by code)
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

---

## Layer 1 - Entrypoints

Entrypoints are the fixed execution mechanisms built into `core/entrypoints.py`. They are not declared in SKILL.MD - they are the substrate that tool skills are built on.

| Scheme | Format | Behaviour |
|---|---|---|
| `bash:` | `bash:{command template}` | Shell command via `subprocess`. `{arg}` placeholders substituted from tool-call arguments. Non-zero exit raises `RuntimeError`. |
| `http:get` | `http:get https://host/path` | HTTP GET; kwargs become query parameters. |
| `http:post` | `http:post https://host/path` | HTTP POST; kwargs become the JSON body. |
| `python:` | `python:module.path.function` | Imports the module and calls the function with kwargs. |
| `grpc:` | `grpc:package.Service/Method` | Stub - wire up a real gRPC channel. |
| `container:` | `container:image_name` | Stub - wire up Docker/Podman. |

`bash:` is the workhorse for local skills. Arguments are injected via `str.format()`, so `bash:cat {path}` called with `path="/etc/hosts"` becomes `cat /etc/hosts`.

> MCP tools do not use an entrypoint scheme. They are declared via `mcp_server` in the frontmatter and loaded as native LangChain tools by `MCPClientManager`. See [mcp.md](mcp.md).

---

## Layer 2 - Tool skills

Tool skills expose callable tools to the LLM. Each tool declares an entrypoint that the framework executes when the LLM calls it.

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

### Frontmatter fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique skill identifier |
| `version` | no | Semver string (default `1.0.0`) |
| `description` | yes | One-line summary sent to the LLM in the skill catalog every turn |
| `tags` | no | Propagated to all tools at registration; used for tag-based lookup |
| `enabled_by_default` | no | If `true`, enabled for all sessions without an explicit grant (default `false`) |
| `always_inject` | no | If `true`, the prose body is appended to the system prompt on every turn |

### Tool block fields

| Field | Required | Description |
|---|---|---|
| `description` | yes | Shown to the LLM as the tool's purpose |
| `entrypoint` | yes | `scheme:target` - see Layer 1 |
| `schema` | yes | JSON Schema object describing the tool's arguments |

### The `always_inject` exception

A structured skill can carry prose alongside its tools by setting `always_inject: true`. The parser stores the prose that appears *before* the `## Tools` section in `Skill.body`, and that prose is appended to the system prompt on **every turn** - useful for skills whose instructions must always be present (e.g. a planning skill that tells the agent how to reason step-by-step).

---

## Layer 3 - Knowledge skills

Knowledge skills carry no tools. Their SKILL.MD is frontmatter plus free-form Markdown prose - no `## Tools` section.

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

### Frontmatter fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique skill identifier |
| `description` | yes | One-line summary included in the skill catalog - always sent to the LLM |
| `triggers` | yes | Keyword phrases; any case-insensitive substring match in the user's message injects the body |

### How trigger injection works

At startup, every enabled knowledge skill's body sits in memory. On each `call_model` invocation, the agent checks whether any trigger phrase appears as a substring of the most recent `HumanMessage` (case-insensitive). If none match, the body is not sent. If any match, the full `Skill.body` is appended to the system prompt for that turn only.

This keeps the baseline system prompt small while making full knowledge available on-demand. A session with 10 knowledge skills enabled pays the prose cost only for the skills relevant to each turn.

The LLM is then responsible for connecting the knowledge to the tools: when the ssh body is injected, the LLM reads it and uses the available tool skills (e.g. Shell's `run_bash`) to construct the appropriate commands.

> Ensure the Shell skill (or another execution skill) is enabled when using knowledge skills that require command execution.

---

## Skill directories

Birdie loads skills from two locations on every startup:

1. **Bundled skills** - `birdie/skills/` shipped inside the package. Always present after `pip install birdie-agent`.
2. **User skills** - `~/.birdie/skills/` on your home directory, if it exists. Drop a subdirectory with a `SKILL.MD` there and it is picked up automatically on next start.

To use a completely different directory instead of the bundled one, pass `--skills-dir PATH`. The user skills directory `~/.birdie/skills/` is always also loaded on top of whichever primary directory is used.

## Built-in skills

All built-in skills are **disabled by default**. Enable them per session:

```
/skill enable Shell
/skill enable DuckDuckGo
```

| Skill | Description |
|---|---|
| `Shell` | Run arbitrary shell commands |
| `Filesystem` | Read and write local files |
| `ssh` | Connect to remote hosts and run commands (knowledge skill) |
| `ToDo` | Step-by-step planning and progress tracking |
| `Weather` | Weather lookup via external API |
| `DuckDuckGo` | Web search with no API key required |
| `mcp_demo` | Demo MCP server (echo and reverse_string) |

---

## Eager vs lazy loading

**Eager - happens once at startup:**

`discover_skills_from_directory` scans every subdirectory, finds `SKILL.MD` files, and parses each into a fully-populated `Skill` object. After startup, no SKILL.MD file is ever read again. All turn-time decisions are made from in-memory `Skill` objects.

**Lazy - happens on every agent turn:**

1. **Tool schema and execution wiring.** `StructuredTool` wrappers are created fresh each turn from the policy-resolved allowed set. A skill can be enabled or disabled between turns.
2. **Trigger matching and body injection.** The prose in `Skill.body` is never sent automatically. It is injected only when a trigger keyword matches the current user message.
3. **MCP tool discovery.** `MCPClientManager.get_tools()` connects to the MCP server on the first call and caches the result for the process lifetime.
