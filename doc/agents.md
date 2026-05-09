# Sub-agents

Sub-agents are self-contained AI agents defined by **AGENT.MD** files. When enabled, a sub-agent appears to the calling LLM as a regular tool. Calling it spins up an ephemeral `DynamicAgent`, runs it to completion with a rendered prompt, and returns its final reply as the tool result.

Sub-agents differ from skills in a few key ways:

| | Skills | Sub-agents |
|---|---|---|
| Defined by | `SKILL.MD` | `AGENT.MD` |
| Executes | A single shell/HTTP/Python call | A full LLM agent loop (may call tools, loop, reason) |
| Output | Tool result string | Final AI message from the inner agent |
| Use case | Atomic operations | Multi-step reasoning, delegation, specialised workflows |

---

## AGENT.MD format

An AGENT.MD file has four sections: frontmatter, `## Input`, `## Output`, and `## Prompt`.

```markdown
---
name: Summarizer
version: 1.0.0
description: Summarize a piece of text into concise bullet points.
enabled_by_default: false
allowed_skills: []
recursion_limit: 25
max_tool_repetitions: 3
---

## Input

### text
type: string
description: The text to summarize
required: true

### max_points
type: integer
description: Maximum number of bullet points (default 5)
required: false

## Output

### summary
type: string
description: Bullet-point summary of the text

## Prompt

Summarize the following text concisely. Return at most {{ max_points }} bullet points.

{{ text }}
```

### Frontmatter fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique agent identifier; becomes the tool name seen by the calling LLM |
| `version` | no | Semver string (default `1.0.0`) |
| `description` | yes | One-line summary - the calling LLM uses this to decide when to invoke the agent |
| `enabled_by_default` | no | If `true`, enabled for all sessions without an explicit grant (default `false`) |
| `vendor` | no | Override the LLM vendor for this agent (default: inherits from the calling agent) |
| `model` | no | Override the LLM model for this agent |
| `allowed_skills` | no | List of skill names the sub-agent may use (default `[]`) |
| `recursion_limit` | no | Maximum LangGraph steps for the inner agent loop (default `25`) |
| `max_tool_repetitions` | no | Block a tool call if it appears this many times consecutively with identical args (default `3`) |

### Input parameters

Each `### ParameterName` block under `## Input` becomes a typed argument on the tool:

| Field | Description |
|---|---|
| `type` | `string` \| `integer` \| `number` \| `boolean` \| `array` \| `object` |
| `description` | Shown to the calling LLM; used to select the right value |
| `required` | `true` (default) or `false` |

### Output parameters

`## Output` documents what the agent returns. These are informational only - the actual return value is always the agent's final AI message text.

### Prompt template

`## Prompt` is the message sent to the sub-agent. Use `{{ parameter_name }}` placeholders for input substitution:

```
Analyze the following C source file for security vulnerabilities.

File: {{ filename }}

Source:
{{ code }}
```

> Do not use `## ` headings inside the `## Prompt` section body - the parser uses `## ` to detect section boundaries. Use `**bold text**` instead.

---

## Agent directories

Birdie loads agents from two locations on every startup:

1. **Bundled agents** - `birdie/agents/` shipped inside the package.
2. **User agents** - `~/.birdie/agents/` on your home directory, if it exists. Drop a subdirectory with an `AGENT.MD` file there and it is picked up automatically on next start.

To use a completely different directory instead of the bundled one, pass `--agents-dir PATH`. The user agents directory `~/.birdie/agents/` is always also loaded on top.

## Bundled agents

| Agent | Description |
|---|---|
| `Summarizer` | Summarize text into concise bullet points (disabled by default) |

---

## Enabling agents

All agents are disabled by default. Enable them per session via the CLI:

```
/agent list
/agent enable Summarizer
/agent disable Summarizer
```

These settings are persisted to the session file and restored on resume.

---

## Runtime controls

### `recursion_limit`

The maximum number of LangGraph steps the inner agent may take before the run is aborted. Each agent node invocation and each tool node invocation counts as one step. The default is 25 steps, which is sufficient for agents with a few tool calls. Tool-heavy agents that need to call many tools in sequence should increase this:

```yaml
recursion_limit: 100
```

### `max_tool_repetitions`

Blocks any tool call that appears more than N times consecutively with identical parameters. When a call is blocked, an error `ToolMessage` is injected so the LLM can see the failure and recover rather than looping forever.

```yaml
max_tool_repetitions: 5
```

The default is 3. The guard applies both to the main agent and to all sub-agents (each with their own configured limit).

---

## Writing a custom agent

### 1. Create the directory and file

```bash
mkdir -p ~/.birdie/agents/my_analyst
cat > ~/.birdie/agents/my_analyst/AGENT.MD << 'EOF'
---
name: MyAnalyst
version: 1.0.0
description: Analyse a Python file and suggest improvements.
enabled_by_default: false
allowed_skills: [Shell]
recursion_limit: 50
max_tool_repetitions: 5
---

## Input

### code
type: string
description: Full content of the Python source file
required: true

### filename
type: string
description: Name of the Python file (e.g. "main.py")
required: true

## Output

### findings
type: array
description: List of suggested improvements

## Prompt

You are an expert Python code reviewer. Analyse the following file for issues,
anti-patterns, and improvement opportunities. Use the Shell skill to run any
static analysis tools (flake8, mypy, bandit) available on the system.

File: {{ filename }}

```python
{{ code }}
```

Return a numbered list of findings, each with severity (low/medium/high),
location, and a concrete suggestion.
EOF
```

### 2. Restart Birdie (agents are loaded at startup)

```
birdie --config ~/.birdie/mistral.json
```

### 3. Enable and use the agent

```
/agent enable MyAnalyst
analyse this file for me
```

The calling LLM will invoke `MyAnalyst` with the file contents and filename as arguments, and display the findings.

---

## Vendor and model overrides

By default a sub-agent uses the same LLM as the calling agent. To use a different model for a specific sub-agent (e.g. a cheaper model for summarisation, a stronger one for security analysis):

```yaml
vendor: anthropic
model: claude-haiku-4-5-20251001
```

These fields override the calling agent's provider for that specific sub-agent only.

---

## Sub-agent output in the CLI

By default, sub-agent transcripts are hidden (`/agent output off`). The calling agent's final answer is shown via the normal tool result display.

To see the sub-agent's work:

```
/agent output short    # transcript truncated to 1000 chars per block
/agent output full     # full untruncated transcript
/agent output off      # silent (default)
```

See [cli.md](cli.md) for the full transcript format description.
