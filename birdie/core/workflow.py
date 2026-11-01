"""
Workflows: YAML files that combine deterministic steps (shell commands,
direct tool calls) with LLM steps (prompts, sub-agent calls).

Control flow is decided by the runner, never by the model.  A workflow is a
list of steps; each step has exactly one *action* key:

=========  ================================================================
``run``    shell command; result: ``ok exit_code stdout stderr output``
``tool``   call a skill / MCP tool with ``with:`` arguments
``agent``  call an AGENT.MD sub-agent with ``with:`` inputs; ``data`` holds
           the parsed JSON when the agent declares output fields
``prompt`` one agent turn; ``context: session`` (default) continues the
           current conversation, ``fresh`` uses a clean thread
``ask``    ask the user; result: ``answer yes``; ``default:`` is used when
           running non-interactively
``echo``   print a message
``fail``   stop the workflow with a message (combine with ``if``)
``steps``  a group of nested steps (for multi-step retry / foreach)
=========  ================================================================

Modifiers usable on any step: ``id``, ``if`` (expression), ``foreach``
(list expression) with ``as`` (loop variable, default ``item``), ``retry``
(``max`` attempts, ``until`` expression - default ``result.ok`` - and
``delay`` seconds), ``on_error`` (``stop`` default, or ``continue``).

Expressions and ``{{ }}`` templates use the tiny language in
:mod:`birdie.core.workflow_expr` with the names ``inputs``, ``steps``,
``env``, ``workflow``, the loop variables (``item``/``as`` and ``loop``:
``index first last length``), inside a retried step ``attempt``
(``number max last``) and - inside ``retry.until`` - ``result``.

Every run is persisted after each step under ``~/.birdie/workflows/runs/``
(override: ``BIRDIE_WORKFLOW_RUNS_DIR``) so a failed or interrupted run can
be resumed: completed steps are skipped, a looping step (``foreach`` /
``retry``) that did not finish is restarted from its first iteration.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import secrets
import shutil
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from .workflow_expr import ExprError, check_template, compile_expr, render, truthy

log = logging.getLogger(__name__)

ACTIONS = ("run", "tool", "agent", "prompt", "ask", "echo", "fail", "steps")
LLM_ACTIONS = ("prompt", "agent")
_COMMON_KEYS = {"id", "if", "foreach", "as", "retry", "on_error"}
_ACTION_KEYS = {
    "run": {"cwd", "timeout", "env"},
    "tool": {"with"},
    "agent": {"with"},
    "prompt": {"context"},
    "ask": {"default"},
    "echo": set(),
    "fail": set(),
    "steps": set(),
}
_TOP_KEYS = {"name", "description", "version", "inputs", "budget", "context", "steps"}
_INPUT_TYPES = ("string", "integer", "number", "boolean", "list")
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_IDS = {"inputs", "steps", "env", "workflow", "result", "loop", "attempt"}

DEFAULT_MAX_STEPS = 200
DEFAULT_RUN_TIMEOUT = 600.0
MAX_CAPTURE_CHARS = 200_000


class WorkflowError(ValueError):
    """Invalid workflow definition."""


class WorkflowAbort(Exception):
    """Stops the whole run (budget exhausted, input missing, ...)."""


class StepFailed(Exception):
    """A step failed with ``on_error: stop``; propagates to the enclosing group."""

    def __init__(self, step_id: str, error: str) -> None:
        super().__init__(f"step {step_id!r} failed: {error}")
        self.step_id = step_id
        self.error = error


# -- definition model ---------------------------------------------------------

@dataclass
class Step:
    id: str
    action: str
    value: Any
    raw: Dict[str, Any]
    children: List["Step"] = field(default_factory=list)
    descendants: List[str] = field(default_factory=list)

    @property
    def condition(self) -> Optional[str]:
        return self.raw.get("if")

    @property
    def foreach(self) -> Any:
        return self.raw.get("foreach")

    @property
    def loop_var(self) -> str:
        return self.raw.get("as", "item")

    @property
    def retry(self) -> Optional[Dict[str, Any]]:
        return self.raw.get("retry")

    @property
    def on_error(self) -> str:
        return self.raw.get("on_error", "stop")

    @property
    def loops(self) -> bool:
        return self.foreach is not None or self.retry is not None

    def summary(self, width: int = 60) -> str:
        text = str(self.value) if self.action != "steps" else f"{len(self.children)} steps"
        text = " ".join(text.split())
        return text if len(text) <= width else text[: width - 1] + "…"


@dataclass
class InputDef:
    name: str
    type: str = "string"
    default: Any = None
    required: bool = True
    description: str = ""


@dataclass
class WorkflowDef:
    name: str
    description: str
    inputs: List[InputDef]
    steps: List[Step]
    budget: Dict[str, Any]
    context: str = "session"
    path: Optional[str] = None
    source_hash: str = ""

    def all_steps(self) -> List[Step]:
        out: List[Step] = []

        def walk(steps: List[Step]) -> None:
            for s in steps:
                out.append(s)
                walk(s.children)
        walk(self.steps)
        return out

    def find(self, step_id: str) -> Optional[Step]:
        return next((s for s in self.all_steps() if s.id == step_id), None)


# -- parsing / validation -----------------------------------------------------

def _check_expr(where: str, src: Any) -> None:
    if not isinstance(src, str):
        raise WorkflowError(f"{where}: expected an expression string, got {type(src).__name__}")
    try:
        compile_expr(src)
    except ExprError as exc:
        raise WorkflowError(f"{where}: {exc}") from None


def _check_tpl(where: str, value: Any) -> None:
    try:
        check_template(value)
    except ExprError as exc:
        raise WorkflowError(f"{where}: {exc}") from None


def _parse_inputs(raw: Any) -> List[InputDef]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise WorkflowError("'inputs' must be a mapping of input name -> spec")
    inputs = []
    for name, spec in raw.items():
        if not isinstance(name, str) or not _ID_RE.match(name):
            raise WorkflowError(f"invalid input name {name!r} (use letters, digits, _)")
        if isinstance(spec, dict):
            unknown = set(spec) - {"type", "default", "required", "description"}
            if unknown:
                raise WorkflowError(f"input {name!r}: unknown keys {sorted(unknown)}")
            has_default = "default" in spec
            default = spec.get("default")
            typ = spec.get("type") or _infer_type(default)
            required = spec.get("required", not has_default)
            desc = str(spec.get("description", ""))
        else:
            # Shorthand: ``name: value`` (default) or ``name:`` (required).
            default, has_default = spec, spec is not None
            typ, required, desc = _infer_type(spec), spec is None, ""
        if typ not in _INPUT_TYPES:
            raise WorkflowError(f"input {name!r}: unknown type {typ!r} (use {', '.join(_INPUT_TYPES)})")
        if has_default and default is not None:
            default = coerce_input(InputDef(name, typ), default)
        inputs.append(InputDef(name, typ, default, bool(required), desc))
    return inputs


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "list"
    return "string"


def coerce_input(spec: InputDef, value: Any) -> Any:
    """Convert *value* (often a CLI string) to the input's declared type."""
    try:
        if spec.type == "string":
            return value if isinstance(value, str) else json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        if spec.type == "boolean":
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("1", "true", "yes", "y", "on"):
                return True
            if text in ("0", "false", "no", "n", "off", ""):
                return False
            raise ValueError(value)
        if spec.type == "integer":
            if isinstance(value, bool):
                raise ValueError(value)
            return int(value) if not isinstance(value, str) else int(value.strip())
        if spec.type == "number":
            if isinstance(value, bool):
                raise ValueError(value)
            return float(value)
        if spec.type == "list":
            if isinstance(value, list):
                return value
            text = str(value).strip()
            if text.startswith("["):
                parsed = yaml.safe_load(text)
                if isinstance(parsed, list):
                    return parsed
            return [p.strip() for p in text.split(",") if p.strip()]
    except (ValueError, TypeError, yaml.YAMLError):
        pass
    raise WorkflowError(f"input {spec.name!r}: cannot convert {value!r} to {spec.type}")


def _parse_steps(raw: Any, prefix: str, seen: set) -> List[Step]:
    if not isinstance(raw, list) or not raw:
        raise WorkflowError(f"{prefix or 'workflow'}: 'steps' must be a non-empty list")
    steps: List[Step] = []
    for idx, item in enumerate(raw, start=1):
        where = f"step {prefix}{idx}"
        if not isinstance(item, dict):
            raise WorkflowError(f"{where}: must be a mapping, got {type(item).__name__}")
        actions = [a for a in ACTIONS if a in item]
        if len(actions) != 1:
            raise WorkflowError(
                f"{where}: needs exactly one of {', '.join(ACTIONS)} (found {actions or 'none'})"
            )
        action = actions[0]
        unknown = set(item) - _COMMON_KEYS - _ACTION_KEYS[action] - {action}
        if unknown:
            raise WorkflowError(f"{where} ({action}): unknown keys {sorted(unknown)}")

        sid = item.get("id")
        if sid is None:
            sid = f"step_{prefix.replace('.', '_')}{idx}"
        if not isinstance(sid, str) or not _ID_RE.match(sid):
            raise WorkflowError(f"{where}: invalid id {sid!r} (use letters, digits, _)")
        if sid in seen:
            raise WorkflowError(f"{where}: duplicate step id {sid!r}")
        seen.add(sid)
        where = f"step {sid!r}"

        value = item[action]
        if action == "steps":
            children = _parse_steps(value, f"{prefix}{idx}.", seen)
        else:
            children = []
            if action in ("run", "prompt", "ask", "echo", "fail"):
                if isinstance(value, bool):
                    raise WorkflowError(
                        f"{where}: {action!r} must be a string, got the boolean {value} "
                        "(quote it: YAML reads yes/no/on/off/true/false unquoted as booleans)")
                if isinstance(value, (int, float)):
                    value = item[action] = str(value)
                if not isinstance(value, str) or (action != "echo" and not value.strip()):
                    raise WorkflowError(f"{where}: {action!r} must be a non-empty string")
                _check_tpl(where, value)
            elif action in ("tool", "agent"):
                if not isinstance(value, str) or not value.strip():
                    raise WorkflowError(f"{where}: {action!r} must name a {action}")
                args = item.get("with", {})
                if not isinstance(args, dict):
                    raise WorkflowError(f"{where}: 'with' must be a mapping")
                _check_tpl(where, args)

        if "if" in item:
            _check_expr(f"{where} if", item["if"])
        if "foreach" in item:
            fe = item["foreach"]
            if isinstance(fe, str):
                _check_tpl(f"{where} foreach", fe if "{{" in fe else "{{ " + fe + " }}")
            elif not isinstance(fe, list):
                raise WorkflowError(f"{where}: 'foreach' must be a list or an expression")
            var = item.get("as", "item")
            if not isinstance(var, str) or not _ID_RE.match(var) or var in _RESERVED_IDS:
                raise WorkflowError(f"{where}: invalid loop variable {var!r}")
        elif "as" in item:
            raise WorkflowError(f"{where}: 'as' requires 'foreach'")
        if "retry" in item:
            r = item["retry"]
            if isinstance(r, int) and not isinstance(r, bool):
                r = item["retry"] = {"max": r}
            if not isinstance(r, dict) or set(r) - {"max", "until", "delay"}:
                raise WorkflowError(f"{where}: 'retry' must be a number or a mapping with max/until/delay")
            mx = r.get("max", 3)
            if isinstance(mx, str) and "{{" in mx:
                _check_tpl(f"{where} retry.max", mx)
            elif isinstance(mx, bool) or not isinstance(mx, int) or mx < 1:
                raise WorkflowError(f"{where}: retry.max must be an integer >= 1 or a template")
            if "until" in r:
                _check_expr(f"{where} retry.until", r["until"])
            delay = r.get("delay", 0)
            if isinstance(delay, bool) or not isinstance(delay, (int, float)) or delay < 0:
                raise WorkflowError(f"{where}: retry.delay must be a number >= 0")
        if item.get("on_error", "stop") not in ("stop", "continue"):
            raise WorkflowError(f"{where}: on_error must be 'stop' or 'continue'")
        if action == "prompt" and item.get("context", "session") not in ("session", "fresh"):
            raise WorkflowError(f"{where}: context must be 'session' or 'fresh'")
        if action == "run":
            t = item.get("timeout", DEFAULT_RUN_TIMEOUT)
            if isinstance(t, bool) or not isinstance(t, (int, float)) or t <= 0:
                raise WorkflowError(f"{where}: timeout must be a positive number of seconds")
            if "env" in item and not isinstance(item["env"], dict):
                raise WorkflowError(f"{where}: env must be a mapping")
            _check_tpl(where, {k: item.get(k) for k in ("cwd", "env") if k in item})

        step = Step(sid, action, value, item, children)
        step.descendants = [d for c in children for d in [c.id, *c.descendants]]
        steps.append(step)
    return steps


def parse_workflow(text: str, path: Optional[str] = None) -> WorkflowDef:
    """Parse and validate a workflow YAML document."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise WorkflowError(f"invalid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise WorkflowError("a workflow must be a YAML mapping")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise WorkflowError(f"unknown top-level keys {sorted(unknown)} (allowed: {', '.join(sorted(_TOP_KEYS))})")
    name = data.get("name") or (Path(path).stem if path else None)
    if not name or not isinstance(name, str):
        raise WorkflowError("workflow needs a 'name'")
    budget = data.get("budget") or {}
    if not isinstance(budget, dict) or set(budget) - {"max_steps", "max_llm_calls", "max_cost_usd"}:
        raise WorkflowError("'budget' accepts max_steps, max_llm_calls, max_cost_usd")
    for key, val in budget.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)) or val < 0:
            raise WorkflowError(f"budget.{key} must be a number >= 0")
    context = data.get("context", "session")
    if context not in ("session", "fresh"):
        raise WorkflowError("'context' must be 'session' or 'fresh'")
    return WorkflowDef(
        name=name,
        description=str(data.get("description", "")).strip(),
        inputs=_parse_inputs(data.get("inputs")),
        steps=_parse_steps(data.get("steps"), "", set()),
        budget=budget,
        context=context,
        path=str(path) if path else None,
        source_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


def load_workflow(path: Path | str) -> WorkflowDef:
    p = Path(path).expanduser()
    return parse_workflow(p.read_text(encoding="utf-8"), path=str(p.resolve()))


def bundled_workflows_dir() -> Path:
    return Path(__file__).parent.parent / "workflows"


def default_workflow_dirs(extra: Optional[str] = None) -> List[Path]:
    """Search order, highest priority first: *extra*, ~/.birdie/workflows, bundled."""
    dirs = [Path(extra).expanduser()] if extra else []
    dirs += [Path.home() / ".birdie" / "workflows", bundled_workflows_dir()]
    return dirs


def discover_workflows(
    dirs: List[Path],
) -> Tuple[Dict[str, WorkflowDef], Dict[str, str]]:
    """Load ``*.yaml``/``*.yml`` from *dirs*; first definition of a name wins.

    Returns ``(workflows, errors)`` where *errors* maps file path -> message.
    """
    found: Dict[str, WorkflowDef] = {}
    errors: Dict[str, str] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml"))):
            try:
                wf = load_workflow(f)
            except (OSError, WorkflowError) as exc:
                errors[str(f)] = str(exc)
                continue
            found.setdefault(wf.name, wf)
    return found, errors


def resolve_workflow(name_or_path: str, dirs: List[Path]) -> WorkflowDef:
    """Find a workflow by name, or load it from a file path."""
    p = Path(name_or_path).expanduser()
    if p.suffix in (".yaml", ".yml") or os.sep in name_or_path:
        if not p.is_file():
            raise WorkflowError(f"workflow file not found: {name_or_path}")
        return load_workflow(p)
    found, errors = discover_workflows(dirs)
    if name_or_path in found:
        return found[name_or_path]
    lowered = {k.lower(): v for k, v in found.items()}
    if name_or_path.lower() in lowered:
        return lowered[name_or_path.lower()]
    broken = [f"{f}: {e}" for f, e in errors.items() if Path(f).stem == name_or_path]
    if broken:
        raise WorkflowError(broken[0])
    avail = ", ".join(sorted(found)) or "none"
    raise WorkflowError(f"unknown workflow {name_or_path!r} (available: {avail})")


def resolve_inputs(wf: WorkflowDef, given: Mapping[str, Any]) -> Dict[str, Any]:
    known = {i.name: i for i in wf.inputs}
    unknown = set(given) - set(known)
    if unknown:
        raise WorkflowError(
            f"unknown input(s) {', '.join(sorted(unknown))} for workflow {wf.name!r} "
            f"(accepted: {', '.join(known) or 'none'})"
        )
    out: Dict[str, Any] = {}
    for spec in wf.inputs:
        if spec.name in given:
            out[spec.name] = coerce_input(spec, given[spec.name])
        elif spec.default is not None or not spec.required:
            out[spec.name] = spec.default
        else:
            raise WorkflowError(f"missing required input {spec.name!r}")
    return out


def parse_kv_args(items: List[str]) -> Dict[str, str]:
    """Parse ``key=value`` pairs (CLI form)."""
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise WorkflowError(f"expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v
    return out


# -- run state persistence ----------------------------------------------------

def default_runs_dir() -> Path:
    env = os.environ.get("BIRDIE_WORKFLOW_RUNS_DIR")
    return Path(env).expanduser() if env else Path.home() / ".birdie" / "workflows" / "runs"


class RunStore:
    """JSON files, one per run: ``<runs_dir>/<run_id>.json``."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else default_runs_dir()

    def path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def save(self, state: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = time.time()
        target = self.path(state["run_id"])
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, target)

    def load(self, run_id: str) -> Dict[str, Any]:
        """Load by exact id or unique prefix."""
        exact = self.path(run_id)
        if exact.is_file():
            return json.loads(exact.read_text(encoding="utf-8"))
        matches = sorted(self.root.glob(f"{run_id}*.json")) if self.root.is_dir() else []
        if len(matches) == 1:
            return json.loads(matches[0].read_text(encoding="utf-8"))
        if not matches:
            raise WorkflowError(f"no workflow run {run_id!r}")
        raise WorkflowError(f"run id {run_id!r} is ambiguous ({len(matches)} matches)")

    def list(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.root.is_dir():
            return []
        runs = []
        for f in self.root.glob("*.json"):
            try:
                runs.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        runs.sort(key=lambda r: r.get("updated_at", 0), reverse=True)
        return runs[:limit]


# -- host: the side-effecting operations a runner needs -----------------------

@dataclass
class PromptReply:
    text: str
    cost_usd: float = 0.0


def _cap(text: str) -> str:
    if len(text) <= MAX_CAPTURE_CHARS:
        return text
    return f"[... {len(text) - MAX_CAPTURE_CHARS} chars truncated ...]\n" + text[-MAX_CAPTURE_CHARS:]


class WorkflowHost:
    """Base host: shell steps, non-interactive ``ask``, no LLM.

    Subclasses provide tools, agents and prompts (:class:`AgentWorkflowHost`)
    and presentation (``on_event``) / interactive input (``ask``).
    """

    interactive = False

    async def run_shell(
        self, command: str, cwd: Optional[str], timeout: float, env: Optional[Dict[str, str]],
    ) -> Tuple[Optional[int], str, str, bool]:
        """Run *command* in its own process group.

        Returns ``(exit_code, stdout, stderr, timed_out)``.  On timeout or
        cancellation the whole process group is killed.
        """
        full_env = dict(os.environ)
        if env:
            full_env.update({str(k): str(v) for k, v in env.items()})
        shell = shutil.which("bash")
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=os.path.expanduser(cwd) if cwd else None,
            env=full_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            executable=shell,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            if isinstance(exc, asyncio.CancelledError):
                raise
            out, err = await proc.communicate()
            return None, out.decode(errors="replace"), err.decode(errors="replace"), True
        return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace"), False

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        raise WorkflowAbort("tool steps are not available in this context")

    async def call_agent(self, name: str, args: Dict[str, Any]) -> str:
        raise WorkflowAbort("agent steps are not available in this context")

    async def prompt(self, text: str, fresh: bool, step_id: str, run_id: str) -> PromptReply:
        raise WorkflowAbort("prompt steps are not available in this context")

    async def ask(self, question: str, default: Optional[str]) -> Optional[str]:
        """Return the answer, or ``None`` if no answer can be obtained."""
        return default

    async def check(self, step: Step) -> Optional[str]:
        """Preflight a step (e.g. unknown tool); return an error message or None."""
        return None

    def on_event(self, kind: str, step: Optional[Step], **info: Any) -> None:
        """Presentation hook: step_start, step_end, step_skip, step_resumed, retry,
        echo, warning."""


class AgentWorkflowHost(WorkflowHost):
    """Host backed by a :class:`~birdie.agent.run.DynamicAgent`.

    Prompt steps run on *thread_id* (``context: session``) or on a fresh
    thread that inherits the session's skill / agent grants.
    """

    def __init__(self, agent: Any, thread_id: str, user_id: Optional[str] = None,
                 long_term_memory: Optional[List[str]] = None) -> None:
        self.agent = agent
        self.thread_id = thread_id
        self.user_id = user_id
        self.long_term_memory = long_term_memory

    # tools ------------------------------------------------------------------

    async def check(self, step: Step) -> Optional[str]:
        if step.action == "agent" and self.agent.agent_registry.get_agent(step.value) is None:
            names = ", ".join(a.name for a in self.agent.agent_registry.list_agents()) or "none"
            return f"unknown agent {step.value!r} (available: {names})"
        if step.action == "tool" and self.agent.registry.get_tool(step.value) is None:
            mcp_names = await self._mcp_tool_names()
            if mcp_names is not None and step.value not in mcp_names:
                return f"unknown tool {step.value!r}"
        return None

    async def _mcp_tool_names(self) -> Optional[set]:
        """Names of MCP-provided tools (contacts the MCP servers; cached).

        ``None`` means "unknown" (a server could not be listed): the tool is
        then not flagged at preflight and any error surfaces at call time.
        """
        if not hasattr(self, "_mcp_names"):
            self._mcp_names: Optional[set] = set()
            if self.agent.mcp_manager.has_servers:
                names = {s.name for s in self.agent.registry.list_skills()}
                try:
                    self._mcp_names = {t.name for t in await self.agent.mcp_manager.get_tools(names)}
                except Exception as exc:
                    log.warning("could not list MCP tools for preflight: %s", exc)
                    self._mcp_names = None
        return self._mcp_names

    async def _approve(self, tool_name: str, args: Dict[str, Any]) -> None:
        registry, policy = self.agent.registry, self.agent.policy
        owner = registry.skill_for_tool(tool_name)
        skill = registry.get_skill(owner) if owner else None
        cb = self.agent.tool_approval_callback
        if not skill or not skill.permissions or cb is None:
            return
        if policy.has_permission_grant(self.thread_id, skill.name):
            return
        decision = cb(skill.name, list(skill.permissions), tool_name, args)
        if inspect.isawaitable(decision):
            decision = await decision
        if decision == "always":
            policy.grant_permissions(self.thread_id, skill.name)
        elif decision != "allow":
            raise RuntimeError("the user denied permission to run this tool")

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        from .adapter import skilltool_to_langchain_tool

        skill_tool = self.agent.registry.get_tool(name)
        if skill_tool is not None:
            await self._approve(name, args)
            return await skilltool_to_langchain_tool(skill_tool).ainvoke(args)
        if self.agent.mcp_manager.has_servers:
            names = {s.name for s in self.agent.registry.list_skills()}
            for tool in await self.agent.mcp_manager.get_tools(names):
                if tool.name == name:
                    return await tool.ainvoke(args)
        raise RuntimeError(f"unknown tool {name!r}")

    async def call_agent(self, name: str, args: Dict[str, Any]) -> str:
        tools = self.agent.agent_registry.get_tools({name})
        if not tools:
            raise RuntimeError(f"unknown agent {name!r}")
        return await tools[0].ainvoke(args)

    # prompts ----------------------------------------------------------------

    def thread_for(self, fresh: bool, step_id: str, run_id: str) -> str:
        if not fresh:
            return self.thread_id
        thread = f"{self.thread_id}~wf-{run_id}-{step_id}-{secrets.token_hex(2)}"
        self.agent.enable_skills_for_session(
            thread, sorted(self.agent.policy.get_allowed_skills(self.thread_id)))
        self.agent.enable_agents_for_session(
            thread, sorted(self.agent.agent_registry.get_allowed_agents(self.thread_id)))
        return thread

    async def prompt(self, text: str, fresh: bool, step_id: str, run_id: str) -> PromptReply:
        from langchain_core.messages import AIMessage

        thread = self.thread_for(fresh, step_id, run_id)
        reply, cost = "", 0.0
        async for update in self.agent.astream(
            text, thread_id=thread, user_id=self.user_id,
            long_term_memory=self.long_term_memory or None,
        ):
            for _node, data in update.items():
                for msg in (data or {}).get("messages", []):
                    if not isinstance(msg, AIMessage):
                        continue
                    rm_cost = (getattr(msg, "response_metadata", None) or {}).get("cost")
                    if isinstance(rm_cost, dict) and rm_cost.get("amount"):
                        cost += float(rm_cost["amount"])
                    if msg.content and not getattr(msg, "tool_calls", None):
                        reply = message_text(msg.content)
        return PromptReply(reply, cost)


def message_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


# -- runner -------------------------------------------------------------------

def new_run_id(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "workflow"
    return f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


class WorkflowRunner:
    """Executes a :class:`WorkflowDef` against a :class:`WorkflowHost`."""

    def __init__(self, wf: WorkflowDef, host: WorkflowHost,
                 store: Optional[RunStore] = None) -> None:
        self.wf = wf
        self.host = host
        self.store = store
        self.state: Dict[str, Any] = {}

    # public API -------------------------------------------------------------

    async def preflight(self) -> List[str]:
        problems = []
        for s in self.wf.all_steps():
            msg = await self.host.check(s)
            if msg:
                problems.append(f"step {s.id!r}: {msg}")
        return problems

    def new_state(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "run_id": new_run_id(self.wf.name),
            "workflow": self.wf.name,
            "workflow_path": self.wf.path,
            "workflow_hash": self.wf.source_hash,
            "inputs": resolve_inputs(self.wf, inputs),
            "status": "running",
            "error": None,
            "results": {},
            "done": [],
            "usage": {"steps": 0, "llm_calls": 0, "cost_usd": 0.0},
            "output": None,
            "started_at": time.time(),
        }

    async def run(self, inputs: Optional[Mapping[str, Any]] = None,
                  state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start a new run, or resume *state* (from :meth:`RunStore.load`)."""
        if state is None:
            state = self.new_state(inputs or {})
        else:
            if state.get("workflow_hash") and state["workflow_hash"] != self.wf.source_hash:
                self.host.on_event("warning", None,
                                   message="the workflow file changed since this run started")
            state["status"] = "running"
            state["error"] = None
        self.state = state
        problems = await self.preflight()
        if problems:
            state["status"], state["error"] = "failed", "; ".join(problems)
            self._save()
            return state
        self._save()
        try:
            for step in self.wf.steps:
                result = await self._run_step(step, self._scope())
                if not result.get("skipped"):
                    state["output"] = result.get("output")
            state["status"] = "completed"
        except StepFailed as exc:
            state["status"], state["error"] = "failed", str(exc)
        except WorkflowAbort as exc:
            state["status"], state["error"] = "failed", str(exc)
        except asyncio.CancelledError:
            state["status"], state["error"] = "interrupted", "interrupted by user"
            self._save()
            raise
        except Exception as exc:  # unexpected runner bug: keep the run resumable
            state["status"], state["error"] = "failed", f"{type(exc).__name__}: {exc}"
        self._save()
        return state

    # internals --------------------------------------------------------------

    def _save(self) -> None:
        if self.store is not None:
            self.store.save(self.state)

    def _scope(self, **extra: Any) -> Dict[str, Any]:
        scope = {
            "inputs": self.state["inputs"],
            "steps": self.state["results"],
            "env": dict(os.environ),
            "workflow": {"name": self.wf.name, "run_id": self.state["run_id"]},
        }
        scope.update(extra)
        return scope

    def _record(self, step: Step, result: Dict[str, Any], done: bool = True) -> None:
        """Store *result*; *done* steps are skipped when the run is resumed."""
        self.state["results"][step.id] = result
        if done and step.id not in self.state["done"]:
            self.state["done"].append(step.id)
        elif not done:
            self._forget([step.id])
        self._save()

    def _forget(self, ids: List[str]) -> None:
        if ids:
            drop = set(ids)
            self.state["done"] = [d for d in self.state["done"] if d not in drop]

    def _charge(self, step: Step) -> None:
        usage, budget = self.state["usage"], self.wf.budget
        max_steps = budget.get("max_steps", DEFAULT_MAX_STEPS)
        if usage["steps"] >= max_steps:
            raise WorkflowAbort(f"budget exhausted: {usage['steps']} step executions (max_steps {max_steps})")
        if step.action in LLM_ACTIONS:
            if "max_llm_calls" in budget and usage["llm_calls"] >= budget["max_llm_calls"]:
                raise WorkflowAbort(
                    f"budget exhausted: {usage['llm_calls']} LLM calls (max_llm_calls {budget['max_llm_calls']})")
            if "max_cost_usd" in budget and usage["cost_usd"] >= budget["max_cost_usd"]:
                raise WorkflowAbort(
                    f"budget exhausted: ${usage['cost_usd']:.4f} spent (max_cost_usd {budget['max_cost_usd']})")
            usage["llm_calls"] += 1
        usage["steps"] += 1

    async def _run_step(self, step: Step, scope: Dict[str, Any]) -> Dict[str, Any]:
        if step.id in self.state["done"] and step.id in self.state["results"]:
            self.host.on_event("step_resumed", step)
            return self.state["results"][step.id]
        if step.condition is not None:
            try:
                go = truthy(step.condition, scope)
            except ExprError as exc:
                return self._finish(step, {"ok": False, "output": None, "error": f"if: {exc}"})
            if not go:
                self.host.on_event("step_skip", step)
                result = {"ok": True, "skipped": True, "output": None}
                self._record(step, result)
                return result

        if step.foreach is not None:
            result = await self._run_foreach(step, scope)
        else:
            result = await self._run_with_retry(step, scope)
        return self._finish(step, result)

    def _finish(self, step: Step, result: Dict[str, Any]) -> Dict[str, Any]:
        stop = not result.get("ok") and step.on_error == "stop"
        # A step that stopped the run is not "done": resuming re-executes it.
        self._record(step, result, done=not stop)
        if stop:
            raise StepFailed(step.id, result.get("error") or "failed")
        return result

    async def _run_foreach(self, step: Step, scope: Dict[str, Any]) -> Dict[str, Any]:
        spec = step.foreach
        try:
            items = render(spec if not isinstance(spec, str) or "{{" in spec else "{{ " + spec + " }}", scope)
        except ExprError as exc:
            return {"ok": False, "output": None, "error": f"foreach: {exc}"}
        if isinstance(items, dict):
            items = [{"key": k, "value": v} for k, v in items.items()]
        if not isinstance(items, list):
            return {"ok": False, "output": None,
                    "error": f"foreach must evaluate to a list, got {type(items).__name__}"}
        results: List[Dict[str, Any]] = []
        for i, item in enumerate(items):
            loop = {"index": i, "first": i == 0, "last": i == len(items) - 1, "length": len(items)}
            res = await self._run_with_retry(step, {**scope, step.loop_var: item, "loop": loop})
            results.append(res)
            if not res.get("ok") and step.on_error == "stop":
                return {"ok": False, "output": [r.get("output") for r in results],
                        "items": results, "error": f"iteration {i}: {res.get('error') or 'failed'}"}
        return {"ok": all(r.get("ok") for r in results),
                "output": [r.get("output") for r in results], "items": results}

    async def _run_with_retry(self, step: Step, scope: Dict[str, Any]) -> Dict[str, Any]:
        retry = step.retry
        attempts = retry.get("max", 3) if retry else 1
        if isinstance(attempts, str):
            try:
                attempts = int(render(attempts, scope))
            except (ExprError, ValueError, TypeError) as exc:
                return {"ok": False, "output": None, "error": f"retry.max: {exc}"}
            if attempts < 1:
                return {"ok": False, "output": None, "error": f"retry.max must be >= 1, got {attempts}"}
        until = retry.get("until", "result.ok") if retry else None
        delay = float(retry.get("delay", 0)) if retry else 0.0
        result: Dict[str, Any] = {}
        for attempt in range(1, attempts + 1):
            if retry is not None:
                scope = {**scope, "attempt": {"number": attempt, "max": attempts,
                                              "last": attempt == attempts}}
            # Each attempt / iteration of a looping step starts its children
            # afresh; a plain group keeps completed children (resume skips them).
            if step.loops:
                self._forget(step.descendants)
            if attempt > 1:
                self.host.on_event("retry", step, attempt=attempt, max=attempts)
                if delay:
                    await asyncio.sleep(delay)
            result = await self._attempt(step, scope)
            if retry is None:
                return result
            result["attempts"] = attempt
            try:
                satisfied = truthy(until, {**scope, "result": result})
            except ExprError as exc:
                return {**result, "ok": False, "error": f"retry.until: {exc}"}
            if satisfied:
                return {**result, "ok": True} if until != "result.ok" else result
        return {**result, "ok": False,
                "error": f"gave up after {attempts} attempts"
                         + (f" ({result.get('error')})" if result.get("error") else
                            f" (until: {until})")}

    async def _attempt(self, step: Step, scope: Dict[str, Any]) -> Dict[str, Any]:
        self._charge(step)
        self.host.on_event("step_start", step)
        started = time.monotonic()
        try:
            result = await self._execute(step, scope)
        except (WorkflowAbort, asyncio.CancelledError):
            raise
        except ExprError as exc:
            result = {"ok": False, "output": None, "error": f"template: {exc}"}
        except Exception as exc:
            result = {"ok": False, "output": None, "error": f"{type(exc).__name__}: {exc}"}
        result["duration_s"] = round(time.monotonic() - started, 3)
        self.host.on_event("step_end", step, result=result)
        return result

    async def _execute(self, step: Step, scope: Dict[str, Any]) -> Dict[str, Any]:
        a = step.action
        if a == "run":
            command = render(step.value, scope)
            cwd = render(step.raw.get("cwd"), scope)
            env = render(step.raw.get("env"), scope)
            code, out, err, timed_out = await self.host.run_shell(
                str(command), cwd, float(step.raw.get("timeout", DEFAULT_RUN_TIMEOUT)), env)
            res = {"ok": code == 0 and not timed_out, "exit_code": code,
                   "stdout": _cap(out), "stderr": _cap(err), "output": _cap(out.strip())}
            if timed_out:
                res["error"] = f"timed out after {step.raw.get('timeout', DEFAULT_RUN_TIMEOUT)}s"
            elif code != 0:
                tail = (err.strip() or out.strip()).splitlines()[-1:] or [""]
                res["error"] = f"exit code {code}" + (f": {tail[0][:200]}" if tail[0] else "")
            return res
        if a == "tool":
            value = await self.host.call_tool(step.value, render(step.raw.get("with", {}), scope))
            return {"ok": True, "output": value if isinstance(value, (str, int, float, bool, list, dict, type(None)))
                    else str(value)}
        if a == "agent":
            text = await self.host.call_agent(step.value, render(step.raw.get("with", {}), scope))
            text = message_text(text)
            data = None
            try:
                parsed = json.loads(text)
                if isinstance(parsed, (dict, list)):
                    data = parsed
            except (ValueError, TypeError):
                pass
            return {"ok": True, "output": text, "data": data}
        if a == "prompt":
            text = render(step.value, scope)
            fresh = step.raw.get("context", self.wf.context) == "fresh"
            reply = await self.host.prompt(str(text), fresh, step.id, self.state["run_id"])
            self.state["usage"]["cost_usd"] += reply.cost_usd
            return {"ok": True, "output": reply.text, "cost_usd": reply.cost_usd}
        if a == "ask":
            question = str(render(step.value, scope))
            default = render(step.raw.get("default"), scope)
            answer = await self.host.ask(question, None if default is None else str(default))
            if answer is None:
                raise WorkflowAbort(
                    f"step {step.id!r} needs user input ({question!r}); give it a 'default' or "
                    f"resume the run interactively with /workflow resume {self.state['run_id']}")
            answer = answer.strip()
            return {"ok": True, "output": answer, "answer": answer,
                    "yes": answer.lower() in ("y", "yes", "true", "1", "ok")}
        if a == "echo":
            text = render(step.value, scope)
            self.host.on_event("echo", step, text=text)
            return {"ok": True, "output": text}
        if a == "fail":
            return {"ok": False, "output": None, "error": str(render(step.value, scope))}
        if a == "steps":
            last: Dict[str, Any] = {"ok": True, "output": None}
            try:
                for child in step.children:
                    res = await self._run_step(child, scope)
                    if not res.get("skipped"):
                        last = res
            except StepFailed as exc:
                return {"ok": False, "output": None, "error": str(exc)}
            return {"ok": True, "output": last.get("output")}
        raise WorkflowError(f"unknown action {a!r}")  # pragma: no cover
