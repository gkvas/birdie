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
