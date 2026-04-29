---
name: Shell
version: 1.0.0
description: Execute arbitrary shell commands on the local machine.
tags: [shell, local, system]
enabled_by_default: false
---

# Skill: Shell

## Tools

### run_bash
description: Run a shell command and return its output.
entrypoint: bash:{command}
schema:
  type: object
  properties:
    command:
      type: string
      description: The shell command to execute
  required: [command]

## Reading files for analysis

Prefer targeted reads over full-file dumps so output stays within context:

- `grep -n "class\|def " file.py` — scan structure and line numbers
- `head -n 80 file.py` — read the top of a file
- `sed -n '50,120p' file.py` — read a specific line range
- `wc -l file.py` — check file length before deciding how to read it

Use `cat` only for files you know are small (under ~200 lines). For larger
files, read the relevant sections rather than the whole file.

## Permissions
- local_shell
