---
name: Shell
version: 1.0.0
description: Execute arbitrary shell commands on the local machine.
tags: [shell, local, system]
enabled_by_default: false
---

## Tools

### run_bash
description: Run a shell command and return its output. Prefer targeted reads over full-file dumps: use `grep -n "pattern" file` to scan, `head -n 80` or `sed -n '50,120p'` to read specific ranges, `wc -l` to check length first. Use `cat` only for files under ~200 lines. Both stdout and stderr are captured; non-zero exit raises an error.
entrypoint: bash:{command}
schema:
  type: object
  properties:
    command:
      type: string
      description: The shell command to execute
  required: [command]
