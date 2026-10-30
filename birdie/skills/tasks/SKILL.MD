---
name: Tasks
version: 1.0.0
description: Start, monitor, and stop detached background tasks (CI checks, builds, servers) without blocking the agent.
tags: [tasks, background, monitoring, async]
enabled_by_default: false
---

## When to use this skill

`run_bash` blocks until the command finishes, and when the timeout hits, the
shell session kills the command's whole process tree. A long-running monitor -
polling `gh pr checks` until CI finishes, a build, a test suite - therefore
either blocks the agent for the entire timeout or has to be interrupted
manually. The tools here start the command **detached** (its own session,
output appended to a log file), so it keeps running after the tool call
returns, survives birdie restarts, and can be inspected at any time.

Typical workflow:

1. `task_start("...", name="pr-85-ci")` - returns a task id immediately.
2. Do other work; check `task_list()` or `task_status(id)` whenever you like.
3. `task_wait(id, timeout=120)` - a bounded wait that returns the final status,
   or reports "still running" and lets you call again - the task is never
   killed by the wait.
4. `task_stop(id)` - kill a task that is no longer needed.

Tasks are stored under `~/.birdie/tasks/<id>/` (`meta.json`, `output.log`,
`exitcode`), so they can be monitored across sessions. Task ids may be given
as a unique prefix.

Example - watch a pull request's CI until all checks complete, then print them:

```
task_start("until gh pr checks 85 --json status -q '[.[] | select(.status != \"COMPLETED\")] | length' | grep -qx 0; do sleep 30; done; gh pr checks 85", name="pr-85-ci")
```

then later: `task_status("pr-85-ci")` or `task_wait("pr-85-ci", timeout=300)`.

## Tools

### task_start
description: Start a shell command as a detached background task and return immediately with its task id. The task keeps running after the call returns (and after birdie exits) - unlike run_bash, whose timeout kills the command's process tree. stdout+stderr are appended to ~/.birdie/tasks/<id>/output.log. Use for long-running monitors and jobs: CI check polling, builds, test suites, servers.
entrypoint: python:birdie.skills.tasks.tools.task_start
schema:
  type: object
  properties:
    command:
      type: string
      description: The shell command to run in the background
    name:
      type: string
      description: Optional short label shown in task_list and task_status
    cwd:
      type: string
      description: Optional working directory for the command (default - current directory)
  required: [command]

### task_list
description: List background tasks, newest first, with status, exit code, elapsed time and command. Optionally filter by status (running, done, failed, stopped).
entrypoint: python:birdie.skills.tasks.tools.task_list
schema:
  type: object
  properties:
    status:
      type: string
      description: Optional status filter - one of running, done, failed, stopped
    limit:
      type: integer
      description: Maximum number of tasks to show (default 50)
      default: 50
  required: []

### task_status
description: Show one task's status, exit code, elapsed time, command, pid and the last 15 lines of its output. Accepts an exact task id or a unique prefix.
entrypoint: python:birdie.skills.tasks.tools.task_status
schema:
  type: object
  properties:
    task_id:
      type: string
      description: Exact task id, or a unique prefix of one
  required: [task_id]

### task_output
description: Return a task's captured output (stdout+stderr) - the last `tail` lines, or all of it (capped) when tail<=0.
entrypoint: python:birdie.skills.tasks.tools.task_output
schema:
  type: object
  properties:
    task_id:
      type: string
      description: Exact task id, or a unique prefix of one
    tail:
      type: integer
      description: Number of trailing lines to return (default 100; <=0 means all, capped at 2000)
      default: 100
  required: [task_id]

### task_wait
description: Wait up to `timeout` seconds for a task to finish and return its final status plus the last 30 output lines. Never kills the task: if the timeout expires the task keeps running in the background and the call returns "still running" - call again later or do other work in the meantime. This is the non-blocking replacement for a run_bash polling loop.
entrypoint: python:birdie.skills.tasks.tools.task_wait
schema:
  type: object
  properties:
    task_id:
      type: string
      description: Exact task id, or a unique prefix of one
    timeout:
      type: number
      description: Seconds to wait before returning (default 60, capped at 300)
      default: 60
  required: [task_id]

### task_stop
description: Stop a running background task (SIGTERM to its process group, then SIGKILL after a short grace period). No-op with a message if the task already finished.
entrypoint: python:birdie.skills.tasks.tools.task_stop
schema:
  type: object
  properties:
    task_id:
      type: string
      description: Exact task id, or a unique prefix of one
  required: [task_id]
