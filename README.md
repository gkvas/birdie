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

A LangGraph-based agent that discovers capabilities at runtime from **SKILL.MD** and **AGENT.MD** files. Skills, tools, and sub-agents are all declared in plain Markdown - no code changes required to add new capabilities.

Birdie is a minimal yet fully functional implementation. The design goal is simplicity and transparency: the codebase is intended to be readable, hackable, and easy to extend.

> **Security notice:** Birdie has no guardrails against local tool misuse. Skills such as `Shell` and `Filesystem` can read, write, and execute anything the running user is permitted to do. Only enable skills you trust and run Birdie under an account with appropriate restrictions.

---

<img src="doc/assets/demo.gif" alt="Birdie CLI demo" width="800">

---

## Installation

**From PyPI (recommended)**

```bash
pip install birdie-agent
```

**From source**

```bash
git clone https://github.com/gkvas/birdie.git
cd birdie
pip install -e .
```

**Optional extras**

```bash
pip install "birdie-agent[mcp]"   # MCP server support
pip install -e ".[dev,mcp]"       # development + MCP (from source)
```

---

## Quick start

Configure your LLM provider and run `birdie`.

**Environment variables**

```bash
# Anthropic
export LLM_VENDOR=anthropic
export LLM_MODEL=claude-sonnet-4-6
export ANTHROPIC_API_KEY=your-key-here
birdie

# OpenAI
export LLM_VENDOR=openai
export LLM_MODEL=gpt-4o
export OPENAI_API_KEY=your-key-here
birdie

# Mistral
export LLM_VENDOR=mistral
export LLM_MODEL=mistral-large-latest
export MISTRAL_API_KEY=your-key-here
birdie

#Azure OpenAI
export  LLM_VENDOR=azure
export LLM_MODEL=your-deployment-name
export AZURE_OPENAI_API_KEY=your-key-here
export AZURE_OPENAI_ENDPOINT=your-base-url
birdie
```

**JSON config file**

```bash
birdie --config ~/.birdie/provider.json
```

```json
{
  "vendor": "anthropic",
  "model": "claude-sonnet-4-6",
  "api_key": "sk-ant-..."
}
```

See [doc/cli.md](doc/cli.md) for all supported vendors, config fields, and environment variable options.

---

## Built-in skills

Skills extend Birdie and are defined in **SKILL.MD** files. All skills are **disabled by default**. Enable them for the current session:

```
/skill enable Shell
/skill enable DuckDuckGo
```

Birdie ships with the following default skills:

| Skill | Description |
|---|---|
| `Shell` | Run arbitrary shell commands |
| `Filesystem` | Read and write local files |
| `ssh` | Connect to remote hosts and run commands |
| `ToDo` | Step-by-step planning and progress tracking |
| `Weather` | Weather lookup via external API |
| `DuckDuckGo` | Web search - no API key required |
| `mcp_demo` | Demo MCP server (echo and reverse_string) |

See [doc/skills.md](doc/skills.md) for the full SKILL.MD format and how to write skills.

---

## Sub-agents

Sub-agents are AI agents defined by **AGENT.MD** files. When enabled, a sub-agent appears to the calling LLM as a regular tool. Invoking it spins up an ephemeral agent, runs it to completion, and returns the result.

All agents are **disabled by default**. Enable them for the current session:

```
/agent enable Summarizer
```

Drop an `AGENT.MD` in `~/.birdie/agents/<name>/` to add custom sub-agents without reinstalling.

See [doc/agents.md](doc/agents.md) for the full AGENT.MD format and how to write custom agents.

---

## Key commands

| Command | Description |
|---|---|
| `/skill list` | List all skills and their status |
| `/skill enable <name>` | Enable a skill for this session |
| `/agent list` | List all sub-agents and their status |
| `/agent enable <name>` | Enable a sub-agent for this session |
| `/agent output short\|full\|off` | Control sub-agent transcript verbosity |
| `/tool output short\|full\|off` | Control tool result verbosity |
| `/remember <text>` | Save a note to long-term memory |
| `/session new` | Start a new session |
| `/session list` | List all sessions |
| `/help` | Show all commands |

---

## Documentation

| Document | Contents |
|---|---|
| [doc/cli.md](doc/cli.md) | CLI flags, provider config reference, all slash commands, key bindings |
| [doc/skills.md](doc/skills.md) | SKILL.MD format, entrypoints, tool and knowledge skills, skill loading |
| [doc/agents.md](doc/agents.md) | AGENT.MD format, sub-agent system, runtime controls, custom agents |
| [doc/mcp.md](doc/mcp.md) | MCP integration, declaring MCP servers, writing MCP servers |
| [doc/architecture.md](doc/architecture.md) | Project layout, agent loop, system prompt, providers, memory and sessions |

---

## Running tests

```bash
pip install -e ".[dev,mcp]"
pytest
```
