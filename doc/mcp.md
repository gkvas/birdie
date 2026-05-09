# MCP integration

[Model Context Protocol](https://modelcontextprotocol.io) (MCP) is an open standard that lets a server expose a set of tools over a well-defined wire protocol. Instead of writing a Python function and wiring it into an entrypoint, you run a separate process that speaks MCP - the agent connects to it, discovers the available tools, and calls them exactly as it would call any other tool.

Birdie integrates MCP through [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters), which converts MCP tool definitions into native LangChain `BaseTool` objects. These are merged with the skill tools before each LLM call and each tool node invocation, so the model sees them alongside any `bash:` or `python:` tools without any special handling.

---

## Declaring an MCP server in SKILL.MD

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

### `mcp_server` fields

| Field | Required | Description |
|---|---|---|
| `transport` | yes | `stdio` or `sse` |
| `command` | stdio only | Executable to launch (e.g. `python`, `node`) |
| `args` | stdio only | List of arguments passed to the command |
| `env` | no | Extra environment variables for the subprocess |
| `cwd` | no | Working directory for the subprocess |
| `url` | sse only | URL of the SSE endpoint |
| `headers` | sse only | HTTP headers to send with the connection |

---

## Writing an MCP server

An MCP server can be written in any language that has an MCP SDK. The Python SDK makes it compact using `FastMCP`:

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

The function's name, docstring, and type annotations become the tool name, description, and argument schema automatically.

---

## The demo server

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

The matching SKILL.MD:

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

To try it:

```
/skill enable mcp_demo
reverse "hello world"
```

The agent will call `reverse_string` via MCP and return `dlrow olleh`.

---

## Installation

MCP support is an optional dependency:

```bash
pip install "birdie-agent[mcp]"
# or from source:
pip install -e ".[mcp]"
```

This adds `mcp` (the official Python SDK) and `langchain-mcp-adapters`. If `mcp_server` is declared in a SKILL.MD but the extra is not installed, `MCPClientManager.get_tools()` raises an `ImportError` with a clear message on first use.

---

## How it works end to end

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

The MCP client opens a fresh session for each tool invocation. This keeps the client stateless and avoids managing long-lived connections.

MCP tools do not go through the entrypoint resolver (`entrypoints.py`). They are handled entirely by `MCPClientManager` (`core/mcp_client.py`) and merged directly into the graph's tool pools where they are used.
