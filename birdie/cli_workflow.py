"""
CLI integration for workflows: ``/workflow ...`` slash commands and the
headless ``birdie --workflow NAME`` mode.

The heavy lifting lives in :mod:`birdie.core.workflow`; this module only
renders progress through the REPL's existing machinery (prompt steps stream
like normal turns, command / tool output respects ``/tool output``) and
reads answers for ``ask`` steps.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import shlex
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rich.markup import escape

from .core.workflow import (
    AgentWorkflowHost,
    PromptReply,
    RunStore,
    Step,
    WorkflowDef,
    WorkflowError,
    WorkflowRunner,
    default_workflow_dirs,
    discover_workflows,
    load_workflow,
    parse_kv_args,
    resolve_workflow,
)
from .core.workflow_expr import to_text

if TYPE_CHECKING:  # pragma: no cover
    from .cli import BirdieCLI


WORKFLOW_USAGE = (
    "Usage: /workflow list | show <name> | run <name|file.yaml> [key=value ...] | "
    "runs [N] | resume <run-id>"
)

_STATUS_STYLE = {"completed": "green", "failed": "red", "interrupted": "yellow", "running": "cyan"}


class CLIWorkflowHost(AgentWorkflowHost):
    """Runs workflow steps through a :class:`BirdieCLI`."""

    def __init__(self, cli: "BirdieCLI", interactive: bool = True) -> None:
        super().__init__(
            cli.agent, cli.session.id, cli.user_id,
            cli.user_memory.as_strings() or None,
        )
        self.cli = cli
        self.interactive = interactive

    async def prompt(self, text: str, fresh: bool, step_id: str, run_id: str) -> PromptReply:
        thread = self.thread_for(fresh, step_id, run_id)
        before = self.cli.session.total_cost_usd
        reply = await self.cli._stream_turn(text, thread_id=thread)
        return PromptReply(reply, max(0.0, self.cli.session.total_cost_usd - before))

    async def ask(self, question: str, default: Optional[str]) -> Optional[str]:
        if not self.interactive:
            return default
        hint = f" [{default}]" if default else ""
        answer = (await asyncio.to_thread(input, f"? {question}{hint} ")).strip()
        return answer or default or ""

    def on_event(self, kind: str, step: Optional[Step], **info: Any) -> None:
        c = self.cli.console
        if kind == "warning":
            c.print(f"[yellow]⚠ {escape(info.get('message', ''))}[/yellow]")
        elif step is None:
            return
        elif kind == "step_start":
            if step.action != "echo":
                c.print(f"[cyan]▸[/cyan] [bold]{step.id}[/bold] "
                        f"[dim]{step.action}: {escape(step.summary())}[/dim]")
        elif kind == "step_skip":
            c.print(f"[dim]↷ {step.id} skipped (condition false)[/dim]")
        elif kind == "step_resumed":
            c.print(f"[dim]↷ {step.id} done in a previous run[/dim]")
        elif kind == "retry":
            c.print(f"[yellow]↻ {step.id} attempt {info['attempt']}/{info['max']}[/yellow]")
        elif kind == "echo":
            for line in to_text(info.get("text")).splitlines() or [""]:
                c.print(f"🐦 {escape(line)}", highlight=False)
        elif kind == "step_end":
            result = info["result"]
            body = _result_body(step, result)
            if body.strip():
                self.cli._render_tool_output(step.id, body)
            if not result.get("ok"):
                c.print(f"[red]✗ {step.id}:[/red] {escape(str(result.get('error') or 'failed'))}")
            elif step.action not in ("echo", "prompt"):
                c.print(f"[green]✓[/green] {step.id} [dim]{result.get('duration_s', 0):.1f}s[/dim]")


def _result_body(step: Step, result: Dict[str, Any]) -> str:
    if step.action == "run":
        out = result.get("stdout") or ""
        err = result.get("stderr") or ""
        return out + (("\n" if out and not out.endswith("\n") else "") + "[stderr]\n" + err if err.strip() else "")
    if step.action in ("tool", "agent") and result.get("ok"):
        return to_text(result.get("output"))
    return ""


# -- commands -----------------------------------------------------------------

def _dirs(cli: "BirdieCLI") -> list:
    return default_workflow_dirs(getattr(cli, "workflows_dir", None))


def _signature(wf: WorkflowDef) -> str:
    parts = []
    for i in wf.inputs:
        parts.append(f"{i.name}" if i.required and i.default is None
                     else f"{i.name}={to_text(i.default)}")
    return " ".join(parts)


def _when(ts: Optional[float]) -> str:
    if not ts:
        return "?"
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _show_list(cli: "BirdieCLI") -> None:
    found, errors = discover_workflows(_dirs(cli))
    if not found and not errors:
        cli.console.print(
            "[dim]No workflows found. Add *.yaml files to ~/.birdie/workflows/.[/dim]")
    for name in sorted(found):
        wf = found[name]
        sig = _signature(wf)
        cli.console.print(
            f"  [bold]{escape(name)}[/bold]{(' [dim]' + escape(sig) + '[/dim]') if sig else ''}"
            f"  - {escape(wf.description.splitlines()[0] if wf.description else '')}")
    for path, err in errors.items():
        cli.console.print(f"  [red]✗ {escape(path)}:[/red] {escape(err)}")


def _show_one(cli: "BirdieCLI", wf: WorkflowDef) -> None:
    c = cli.console
    c.print(f"[bold cyan]{escape(wf.name)}[/bold cyan]  [dim]{escape(wf.path or '')}[/dim]")
    if wf.description:
        c.print(escape(wf.description))
    if wf.inputs:
        c.print("[bold]Inputs[/bold]")
        for i in wf.inputs:
            req = "required" if i.required and i.default is None else f"default {to_text(i.default)!r}"
            c.print(f"  {i.name} [dim]({i.type}, {escape(req)})[/dim] {escape(i.description)}")
    if wf.budget:
        c.print("[bold]Budget[/bold] " + ", ".join(f"{k}={v}" for k, v in wf.budget.items()))
    c.print("[bold]Steps[/bold]")

    def walk(steps: List[Step], depth: int) -> None:
        for s in steps:
            mods = [f"{k}: {s.raw[k]}" for k in ("if", "foreach", "retry", "on_error", "context")
                    if k in s.raw]
            c.print(f"  {'  ' * depth}[bold]{s.id}[/bold] [dim]{s.action}: {escape(s.summary())}"
                    f"{('  ' + escape('; '.join(map(str, mods)))) if mods else ''}[/dim]")
            walk(s.children, depth + 1)
    walk(wf.steps, 0)


def _show_runs(cli: "BirdieCLI", arg: str) -> None:
    try:
        limit = int(arg) if arg.strip() else 10
    except ValueError:
        cli.console.print("[red]Usage: /workflow runs [N][/red]")
        return
    runs = RunStore().list(limit=max(1, limit))
    if not runs:
        cli.console.print("[dim]No workflow runs yet.[/dim]")
        return
    for r in runs:
        status = r.get("status", "?")
        style = _STATUS_STYLE.get(status, "white")
        err = f"  [dim]{escape(str(r['error'])[:80])}[/dim]" if r.get("error") else ""
        cli.console.print(f"  [bold]{r.get('run_id')}[/bold]  [{style}]{status}[/{style}]  "
                          f"{_when(r.get('updated_at'))}{err}")


def _load_for_resume(state: Dict[str, Any], dirs: list) -> WorkflowDef:
    path = state.get("workflow_path")
    if path:
        try:
            return load_workflow(path)
        except OSError:
            pass
    return resolve_workflow(state.get("workflow", ""), dirs)


async def run_workflow(
    cli: "BirdieCLI",
    wf: WorkflowDef,
    inputs: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    interactive: bool = True,
) -> Dict[str, Any]:
    """Run (or resume) *wf* in *cli*'s session, rendering progress."""
    runner = WorkflowRunner(wf, CLIWorkflowHost(cli, interactive=interactive), RunStore())
    verb = "Resuming" if state else "Running"
    if state is None:
        state = runner.new_state(inputs or {})   # validates inputs before anything runs
    cli.console.print(f"[bold cyan]{verb} workflow {escape(wf.name)}[/bold cyan] "
                      f"[dim]run {state['run_id']}[/dim]")
    try:
        state = await runner.run(state=state)
    except asyncio.CancelledError:
        cli.console.print(f"\n[yellow]Workflow interrupted.[/yellow] Resume with "
                          f"[bold]/workflow resume {state['run_id']}[/bold]")
        raise
    _print_summary(cli, state)
    return state


def _print_summary(cli: "BirdieCLI", state: Dict[str, Any]) -> None:
    status = state["status"]
    style = _STATUS_STYLE.get(status, "white")
    u = state.get("usage", {})
    cost = f", ${u['cost_usd']:.4f}" if u.get("cost_usd") else ""
    n_steps, n_llm = u.get("steps", 0), u.get("llm_calls", 0)
    cli.console.print(
        f"[{style}]Workflow {status}[/{style}] [dim]({n_steps} step{'s' * (n_steps != 1)}, "
        f"{n_llm} LLM call{'s' * (n_llm != 1)}{cost})[/dim]")
    if status != "completed":
        if state.get("error"):
            cli.console.print(f"[red]{escape(str(state['error']))}[/red]")
        cli.console.print(f"[dim]Resume with /workflow resume {state['run_id']} "
                          f"(or birdie --workflow-resume {state['run_id']})[/dim]")


async def handle_workflow_command(cli: "BirdieCLI", arg: str) -> None:
    """Dispatch ``/workflow <sub> ...``."""
    try:
        tokens = shlex.split(arg)
    except ValueError as exc:
        cli.console.print(f"[red]{escape(str(exc))}[/red]")
        return
    sub = tokens[0].lower() if tokens else "list"
    rest = tokens[1:]
    try:
        if sub in ("list", "ls"):
            _show_list(cli)
        elif sub == "show" and rest:
            _show_one(cli, resolve_workflow(rest[0], _dirs(cli)))
        elif sub == "run" and rest:
            wf = resolve_workflow(rest[0], _dirs(cli))
            await run_workflow(cli, wf, parse_kv_args(rest[1:]))
        elif sub == "runs":
            _show_runs(cli, rest[0] if rest else "")
        elif sub == "resume" and rest:
            state = RunStore().load(rest[0])
            if state.get("status") == "completed":
                cli.console.print(f"[dim]Run {state['run_id']} already completed.[/dim]")
                return
            await run_workflow(cli, _load_for_resume(state, _dirs(cli)), state=state)
        else:
            cli.console.print(f"[red]{WORKFLOW_USAGE}[/red]")
    except WorkflowError as exc:
        cli.console.print(f"[red]Workflow error:[/red] {escape(str(exc))}")


async def run_workflow_headless(
    cli: "BirdieCLI",
    name: Optional[str],
    input_args: List[str],
    resume_id: Optional[str] = None,
) -> int:
    """``birdie --workflow NAME [--input k=v ...]`` - returns the exit code.

    Progress goes to the CLI console (stderr in headless mode); the final
    step's output is printed to stdout so it can be piped.
    """
    try:
        if resume_id:
            state = RunStore().load(resume_id)
            if state.get("status") == "completed":
                print(to_text(state.get("output")))
                return 0
            state = await run_workflow(cli, _load_for_resume(state, _dirs(cli)),
                                       state=state, interactive=False)
        else:
            wf = resolve_workflow(name or "", _dirs(cli))
            state = await run_workflow(cli, wf, parse_kv_args(input_args), interactive=False)
    except WorkflowError as exc:
        cli.console.print(f"[red]Workflow error:[/red] {escape(str(exc))}")
        return 2
    if state["status"] == "completed":
        output = to_text(state.get("output"))
        if output:
            print(output)
        return 0
    return 1
