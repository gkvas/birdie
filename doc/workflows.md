# Workflows

A workflow is a YAML file that combines **deterministic steps** (shell
commands, direct tool calls) with **LLM steps** (prompts and sub-agent calls).
The control flow is fixed and owned by the runner. The model fills in
individual steps but never decides what runs next, so a workflow behaves the
same way every time and its cost is bounded.

```yaml
name: fix-tests
description: Run the tests, let the agent fix failures, re-run (max 3 rounds)
inputs:
  command: pytest -q
budget:
  max_llm_calls: 10
steps:
  - id: tests
    retry:
      max: 4
      until: steps.test.ok
    steps:
      - id: test
        run: "{{ inputs.command }}"
        on_error: continue
      - id: fix
        if: "not steps.test.ok and not attempt.last"
        prompt: |
          `{{ inputs.command }}` fails:
          {{ steps.test.stdout | tail(60) }}
          Fix the code. Do not weaken or delete tests.
  - id: diff
    run: git diff --stat
  - echo: "Tests pass after {{ steps.tests.attempts }} run(s)"
```

---

## Running workflows

Workflows are discovered from these directories, first match by name wins:

1. `--workflows-dir PATH` (if given)
2. `~/.birdie/workflows/*.yaml`
3. the bundled `birdie/workflows/` (currently `fix-tests`)

Any `.yaml` file can also be run by path.

| Command | Description |
|---|---|
| `/workflow list` | List workflows with their inputs; broken files are shown with their error |
| `/workflow show <name>` | Show description, inputs, budget and the step tree |
| `/workflow run <name\|file.yaml> [key=value ...]` | Run in the current session |
| `/workflow runs [N]` | List recent runs and their status |
| `/workflow resume <run-id>` | Resume a failed or interrupted run (a unique id prefix is enough) |

Press **Ctrl+C** to interrupt a running workflow. The run is saved as
`interrupted` and can be resumed.

### Headless

```bash
birdie --workflow fix-tests --input command="python -m pytest -q"
birdie --workflow ./release-check.yaml --input version=1.2.0 > summary.txt
birdie --workflow-resume fix-tests-20261101-0930-ab12
```

Progress is written to **stderr**. The output of the last executed step goes
to **stdout**, so it can be piped. Exit codes: `0` completed, `1` failed, `2`
invalid workflow or inputs, `130` interrupted. `ask` steps use their
`default:`, and a step without one fails the run with a resume hint.
`--session-id` runs the prompt steps in an existing session.

---

## File format

| Key | Description |
|---|---|
| `name` | Workflow name (defaults to the file name) |
| `description` | Shown by `/workflow list` and `show` |
| `inputs` | Input parameters, see below |
| `budget` | `max_steps` (default 200), `max_llm_calls`, `max_cost_usd` |
| `context` | Default context for `prompt` steps: `session` (default) or `fresh` |
| `steps` | The list of steps |

Unknown keys are rejected so typos fail loudly. The whole file, including
every expression, is validated before anything runs.

### Inputs

```yaml
inputs:
  path: tests/            # shorthand: default value (type inferred)
  who:                    # shorthand: required string
  rounds:
    type: integer         # string | integer | number | boolean | list
    default: 3
    description: Max fix attempts
```

Values given as `key=value` are converted to the declared type. Lists accept
`a,b,c` or `[a, b]`; booleans accept `true/false/yes/no/1/0`.

### Steps

Each step has exactly one **action**:

| Action | Does | Result fields |
|---|---|---|
| `run: <command>` | Shell command (bash), in its own process group. Extra keys: `cwd`, `env`, `timeout` (seconds, default 600). A non-zero exit fails the step. | `ok exit_code stdout stderr output` |
| `tool: <name>` + `with: {…}` | Calls a skill or MCP tool directly with no LLM involved. Skills that declare permissions still ask for approval. | `ok output` |
| `agent: <name>` + `with: {…}` | Calls an AGENT.MD sub-agent. If the agent declares output fields, the parsed JSON is in `data`. | `ok output data` |
| `prompt: <text>` | One agent turn. `context: session` continues the current conversation, `context: fresh` uses a clean thread (with the session's skill grants). | `ok output cost_usd` |
| `ask: <question>` | Asks the user. `default:` is used on empty input and in headless mode. | `ok answer yes output` |
| `echo: <text>` | Prints a message. | `output` |
| `fail: <message>` | Fails the step (combine with `if`). | |
| `steps: [...]` | A group of nested steps, for multi-step `retry` / `foreach`. Its `output` is the last child's. | `ok output` |

Tools and agents a workflow names do **not** need to be enabled in the
session, because the workflow names them explicitly. Unknown tools and agents
are reported before the first step runs.

**Modifiers**, valid on any step:

| Key | Description |
|---|---|
| `id` | Name for referencing the result as `steps.<id>`. Letters, digits and `_`, unique in the file. Steps without an id get `step_<n>` (nested: `step_2_1`). |
| `if: <expr>` | Skip the step when false. A skipped step has `skipped: true`, `ok: true`. |
| `foreach: <expr or list>` + `as: <var>` | Run the step once per item (default variable `item`, plus `loop.index first last length`). A mapping iterates `{key, value}` pairs. The result has `output` (list) and `items` (per-item results). |
| `retry: N` or `retry: {max, until, delay}` | Run up to `max` times (can be a template) until `until` is true. The default `until` is `result.ok`. Inside, `attempt.number`, `attempt.max` and `attempt.last` are available. The result gains `attempts`. |
| `on_error: stop \| continue` | `stop` (default) ends the workflow (or the enclosing group's attempt) when the step fails. `continue` records the failure and moves on. |

---

## Expressions and templates

`if`, `until` and `foreach` take expressions. Strings take `{{ expr }}`
templates. The language is deliberately tiny and safe: it has no function
calls, no attribute access on Python objects and no imports.

| | |
|---|---|
| Literals | `42` `1.5` `'text'` `"text"` `true` `false` `null` `[1, 'a']` |
| Lookups | `steps.test.ok`, `inputs.files[0]`, `steps['test'].stdout` |
| Operators | `or`, `and`, `not`, `== != < <= > >=`, `in`, `not in`, `+ -` |
| Filters | `value \| name(args)` and `\|` binds loosest, e.g. `(x \| length) > 2` |
| Names | `inputs`, `steps`, `env`, `workflow` (`name`, `run_id`), loop / `attempt` variables, `result` (in `retry.until`) |

Filters: `tail(n)` / `head(n)` (lines), `lines`, `length`, `trim`, `lower`,
`upper`, `default(v)` (for `null`/`""`), `int`, `string`, `json`, `from_json`
(also strips a fenced code block), `join(sep)`, `split(sep)`,
`replace(old, new)`, `quote` (shell-escape).

Referencing a name or key that doesn't exist is an error, not a silent
`null`, so typos surface immediately. A template that is exactly one
`{{ expr }}` keeps the value's type (useful for `foreach` and `with:`).
Otherwise values are stringified: `null` becomes empty and lists/objects
become JSON.

> **Shell safety:** templated values are inserted into `run:` commands
> verbatim. Use `{{ value | quote }}` **without** surrounding quotes for
> anything that came from an input, the LLM or a tool:
> `run: grep -r {{ inputs.pattern | quote }} src/`

---

## Budgets

| Key | Limits |
|---|---|
| `max_steps` | Step executions including every retry attempt and loop iteration (default 200) |
| `max_llm_calls` | `prompt` + `agent` step executions |
| `max_cost_usd` | Accumulated cost of `prompt` steps, as reported by the provider. Sub-agent cost is not tracked. |

When a budget is exhausted the run stops as `failed` and stays resumable. The
counters carry over into the resumed run.

---

## Persistence and resuming

Each run is saved after every step to
`~/.birdie/workflows/runs/<run-id>.json` (override with
`BIRDIE_WORKFLOW_RUNS_DIR`). The file holds the inputs, every step result,
usage counters and the status (`running`, `completed`, `failed`,
`interrupted`).

When you resume:

- completed and skipped steps are not executed again, and their stored
  results remain available as `steps.<id>`
- the step that failed is executed again
- a looping step (`foreach` / `retry`) that had not finished starts again from
  its first iteration or attempt
- if the workflow file changed since the run started, a warning is shown and
  the run continues with the new definition
