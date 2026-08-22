---
name: Shell
version: 1.2.0
description: Execute arbitrary shell commands on the local machine.
tags: [shell, local, system]
enabled_by_default: false
---

## Tools

### run_bash
description: Run a shell command and return its output. The session is persistent — working directory, exported variables and functions survive to the next run_bash call, so `source env.sh` or `export PATH=...` once and it sticks. Prefer targeted reads over full-file dumps: use `grep -n "pattern" file` to scan, `head -n 80` or `sed -n '50,120p'` to read specific ranges, `wc -l` to check length first. Use `cat` only for files under ~200 lines. Both stdout and stderr are captured; non-zero exit raises an error. Commands are killed after `timeout_s` seconds (default 120) — set it per call for slow commands (builds, test suites). A timeout kills all of the shell's child processes, background jobs included; start long-lived daemons (servers, emulators) with `setsid <cmd> >/dev/null 2>&1 &` so they detach and survive.
entrypoint: bash:{command}
schema:
  type: object
  properties:
    command:
      type: string
      description: The shell command to execute
  required: [command]
