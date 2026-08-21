"""
Interactive REPL for Birdie.

Presents a prompt-toolkit prompt with:

- Streaming output rendered via Rich (tool calls with 🐦 prefix, AI text
  indented as plain text, tool output indented with ⎿).
- A bottom status bar showing vendor, model, session ID, and token counters.
- Slash commands for session management, skill control, and navigation.
- Ctrl+C to quit; Ctrl+J to insert a newline for multi-line input.

Entry point: ``birdie`` console script → :func:`main`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import PathCompleter, CompleteEvent
from prompt_toolkit.document import Document as _PTDocument
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from rich.console import Console

from .agent.graph import MAX_TOOL_OUTPUT_CAP
from .agent.run import DynamicAgent
from .core.errors import BirdieRateLimitError
from .core.models import Skill
from .core.session import Session, SessionManager, UserMemory


PROMPT_STYLE = Style.from_dict({
    "prompt": "ansicyan bold",
    "ctrlc-hint": "#888888",
})

HELP_TEXT = """
[bold cyan]Birdie CLI - available slash commands[/bold cyan]

  [yellow]/help[/yellow]                         Show this help
  [yellow]/quit[/yellow]  [yellow]/exit[/yellow]                 Exit the session
  [yellow]/new[/yellow]                          Start a fresh conversation (new thread)
  [yellow]/cd <path>[/yellow]                    Change working directory (default: home)
  [yellow]/remember <text>[/yellow]              Save a note to long-term memory
  [yellow]/compact[/yellow]                      Force-compact conversation history into LTM now
  [yellow]/cost[/yellow]                         Show token usage and estimated cost for this session
  [yellow]/history \[N][/yellow]                  Show the last N messages of this session (default 10)
  [yellow]/info[/yellow]                         Show session info (user, session, provider)

  [bold]Tool commands[/bold]
  [yellow]/tool list[/yellow]                    List all available tools
  [yellow]/tool output full[/yellow]             Show complete tool output
  [yellow]/tool output short[/yellow]            Show first 1000 characters + remaining count (default)
  [yellow]/tool output off[/yellow]              Show only line count, no content
  [yellow]/tool cap <N>[/yellow]                 Cap tool output stored in history at N characters
  [yellow]/tool cap off[/yellow]                 Remove the cap (store full tool output)

  [bold]Skill commands[/bold]
  [yellow]/skill list[/yellow]                   List all loaded skills with status
  [yellow]/skill enable <name>[/yellow]          Enable a skill (persists to session)
  [yellow]/skill disable <name>[/yellow]         Disable a skill (persists to session)
  [yellow]/skill reload[/yellow]                 Re-discover skills from disk (picks up edits)
  [yellow]/skill new <name>[/yellow]             Scaffold ~/.birdie/skills/<name>/SKILL.MD

  [bold]Agent commands[/bold]
  [yellow]/agent list[/yellow]                   List all loaded agents with status
  [yellow]/agent enable <name>[/yellow]          Enable an agent (persists to session)
  [yellow]/agent disable <name>[/yellow]         Disable an agent (persists to session)
  [yellow]/agent output full[/yellow]            Show complete sub-agent transcript
  [yellow]/agent output short[/yellow]           Show sub-agent transcript truncated to 1000 chars
  [yellow]/agent output off[/yellow]             Hide sub-agent transcript (default)

  [bold]Logging commands[/bold]
  [yellow]/log llm on[/yellow]                  Enable LLM request/response logging to ~/.birdie/llm.log
  [yellow]/log llm off[/yellow]                 Disable LLM logging
  [yellow]/log http on[/yellow]                 Enable raw HTTP traffic logging to ~/.birdie/http.log
  [yellow]/log http off[/yellow]                Disable HTTP logging

  [bold]Session commands[/bold]
  [yellow]/session new[/yellow]                  Create a new session and switch to it
  [yellow]/session switch <id>[/yellow]          Switch to an existing session
  [yellow]/session delete <id>[/yellow]          Delete a session (creates new if current)
  [yellow]/session list[/yellow]                 List all sessions for this user
  [yellow]/session info[/yellow]                 Show detailed session metadata
"""

_TOOL_OUTPUT_MODES = ("full", "short", "off")
_AGENT_OUTPUT_MODES = ("full", "short", "off")

# The status bar re-renders on every keystroke, so git state is cached
# and refreshed at most once per TTL (see BirdieCLI._git_segment).
_GIT_TTL_SECONDS = 3.0


def _format_git_segment(porcelain: str) -> str:
    """Format ``git status --porcelain=v2 --branch`` output for the toolbar.

    Returns e.g. ``⎇ main``, ``⎇ main*`` (dirty), ``⎇ main* ↑2↓1``
    (ahead/behind upstream), ``⎇ (3f2c1ab)`` (detached HEAD), or ``""``
    when the output carries no branch information.
    """
    branch = ""
    oid = ""
    ahead = behind = 0
    dirty = False
    for line in porcelain.splitlines():
        if line.startswith("# branch.head "):
            branch = line[len("# branch.head "):]
        elif line.startswith("# branch.oid "):
            oid = line[len("# branch.oid "):]
        elif line.startswith("# branch.ab "):
            for part in line[len("# branch.ab "):].split():
                if part.startswith("+"):
                    ahead = int(part[1:])
                elif part.startswith("-"):
                    behind = int(part[1:])
        elif not line.startswith("#"):
            # Any non-header line is a changed/untracked/unmerged entry.
            dirty = True
    if branch == "(detached)":
        branch = f"({oid[:7]})" if oid and oid != "(initial)" else "(detached)"
    if not branch:
        return ""
    segment = f"⎇ {branch}"
    if dirty:
        segment += "*"
    arrows = (f"↑{ahead}" if ahead else "") + (f"↓{behind}" if behind else "")
    if arrows:
        segment += f" {arrows}"
    return segment


def _read_git_status(cwd: str) -> str:
    """Return the git toolbar segment for *cwd*, or ``""`` outside a repo.

    Never raises: a missing git binary, a non-repo directory, or a slow
    filesystem (1 s timeout) all degrade to an empty segment.
    """
    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "status",
             "--porcelain=v2", "--branch"],
            cwd=cwd, capture_output=True, text=True, timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return _format_git_segment(proc.stdout)


class BirdieCLI:
    def __init__(
        self,
        agent: DynamicAgent,
        session_manager: SessionManager,
        session: Session,
        user_id: str,
        user_memory: UserMemory,
        console: Optional[Console] = None,
    ) -> None:
        self.agent = agent
        self.session_manager = session_manager
        self.session = session
        self.user_id = user_id
        self.user_memory = user_memory
        self.console = console or Console()

        self._total_in: int = 0
        self._total_out: int = 0
        self._last_context: int = 0
        self._context_window: int = 0
        self._tool_output_mode: str = "short"
        self._agent_output_mode: str = "off"
        self._tool_output_cap: int = agent._tool_output_cap
        self._llm_log_handler: Optional[logging.FileHandler] = None
        self._orig_async_send = None
        self._orig_sync_send = None
        self._active_status = None  # rich Status while a turn is streaming

        # Route permission approvals for permissioned skills through the CLI.
        agent.tool_approval_callback = self._approve_tool

        # Apply stored skill grants for the initial session
        self._apply_session_policy(session)

        self._ctrl_c_warned: bool = False
        # State for /cd Tab cycling
        self._cd_cycle: dict = {"completions": [], "index": -1, "path": None}
        # TTL cache for the toolbar's git segment, keyed on cwd
        self._git_cache: dict = {"cwd": None, "at": 0.0, "text": ""}

        kb = KeyBindings()

        @kb.add("c-c")
        def _ctrl_c(event):
            buf = event.current_buffer
            if buf.text:
                buf.reset()
                self._ctrl_c_warned = False
                self._cd_cycle["path"] = None
            elif self._ctrl_c_warned:
                event.app.exit(result=None, exception=SystemExit(0))
            else:
                self._ctrl_c_warned = True
                event.app.invalidate()

        @kb.add("c-j")
        def _newline(event):
            """Ctrl+J inserts a newline for multi-line input."""
            event.current_buffer.insert_text("\n")

        _cd_completer = PathCompleter(only_directories=True, expanduser=True)

        @kb.add("tab")
        def _tab(event):
            """Cycle through directory completions for /cd; do nothing otherwise."""
            buf = event.current_buffer
            text = buf.document.text_before_cursor
            _CD_PREFIX = "/cd "
            if not text.lower().startswith(_CD_PREFIX):
                return
            path_part = text[len(_CD_PREFIX):]
            if path_part != self._cd_cycle["path"]:
                # Path changed - ask PathCompleter for fresh matches
                doc = _PTDocument(path_part, len(path_part))
                raw = list(_cd_completer.get_completions(
                    doc, CompleteEvent(completion_requested=True)
                ))
                if not raw:
                    return
                # Convert each Completion to the full path that would follow "/cd "
                # PathCompleter yields: text=suffix, start_position=0 (append at cursor)
                # or start_position<0 (replace last N chars).  The full path is:
                # path_part[:len(path_part)+start_position] + completion.text + "/"
                full_paths = [
                    path_part[: len(path_part) + c.start_position] + c.text + "/"
                    for c in raw
                ]
                self._cd_cycle["completions"] = full_paths
                self._cd_cycle["index"] = 0
                self._cd_cycle["path"] = path_part
            else:
                # Same prefix - advance to next match
                n = len(self._cd_cycle["completions"])
                if n == 0:
                    return
                self._cd_cycle["index"] = (self._cd_cycle["index"] + 1) % n
            selected = self._cd_cycle["completions"][self._cd_cycle["index"]]
            buf.delete_before_cursor(len(path_part))
            buf.insert_text(selected)
            # Track the inserted text so the next Tab advances rather than resets
            self._cd_cycle["path"] = selected

        history_path = Path.home() / ".birdie_history"
        self.session_prompt: PromptSession = PromptSession(
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            style=PROMPT_STYLE,
            key_bindings=kb,
            multiline=False,
            bottom_toolbar=self._get_toolbar,
        )

    # -- policy helpers -------------------------------------------------------

    def _apply_session_policy(self, session: Session) -> None:
        """Apply stored skill and agent grants from session to the policy."""
        for skill in session.enabled_skills:
            self.agent.enable_skill(session.id, skill)
        for skill in session.disabled_skills:
            self.agent.disable_skill(session.id, skill)
        for agent in session.enabled_agents:
            self.agent.enable_agent(session.id, agent)
        for agent in session.disabled_agents:
            self.agent.disable_agent(session.id, agent)
        for skill in session.approved_skills:
            self.agent.policy.grant_permissions(session.id, skill)

    # -- tool permission approval ---------------------------------------------

    async def _approve_tool(
        self, skill_name: str, permissions: list, tool_name: str, args: dict,
    ) -> str:
        """Interactive gate for tools of skills that declare permissions."""
        status = self._active_status
        if status is not None:
            status.stop()
        try:
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            self.console.print(
                f"[yellow]Skill [bold]{skill_name}[/bold] requests permissions:"
                f"[/yellow] {', '.join(permissions)}\n"
                f"[yellow]Tool call:[/yellow] {tool_name}({args_str})"
            )
            answer = (await asyncio.to_thread(
                input, "Allow? [y]es once / [a]lways this session / [N]o: "
            )).strip().lower()
        finally:
            if status is not None:
                status.start()

        if answer in ("a", "always"):
            if skill_name not in self.session.approved_skills:
                self.session.approved_skills.append(skill_name)
                self.session_manager.save(self.session)
            return "always"
        if answer in ("y", "yes"):
            return "allow"
        return "deny"

    async def _approve_acp_permission(self, request: dict) -> str:
        """Interactive gate for ACP session/request_permission requests."""
        always_opt = next(
            (o for o in request.get("options", []) if o.get("kind") == "allow_always"),
            None,
        )
        always_key = (always_opt or {}).get("name") or request.get("title") or ""
        if always_key and always_key in self.session.approved_acp_tools:
            return "allow_always"

        status = self._active_status
        if status is not None:
            status.stop()
        try:
            title = request.get("title") or "unknown tool"
            raw = request.get("raw_input")
            self.console.print(
                f"[yellow]ACP agent requests permission:[/yellow] [bold]{title}[/bold]"
                + (f"\n[dim]{raw}[/dim]" if raw else "")
            )
            answer = (await asyncio.to_thread(
                input, "Allow? [y]es once / [a]lways this session / [N]o: "
            )).strip().lower()
        finally:
            if status is not None:
                status.start()

        if answer in ("a", "always"):
            if always_key and always_key not in self.session.approved_acp_tools:
                self.session.approved_acp_tools.append(always_key)
                self.session_manager.save(self.session)
            return "allow_always"
        if answer in ("y", "yes"):
            return "allow"
        return "deny"

    def _get_prompt(self):
        if self._ctrl_c_warned:
            return [
                ("class:prompt", "you> "),
                ("class:ctrlc-hint", "Press Ctrl+C again to exit, or type new instructions to continue"),
            ]
        return [("class:prompt", "you> ")]

    def _pre_run(self):
        from prompt_toolkit.application import get_app
        app = get_app()
        buf = app.current_buffer

        def _on_changed(_):
            if self._ctrl_c_warned and buf.text:
                self._ctrl_c_warned = False
                app.invalidate()

        buf.on_text_changed += _on_changed

    async def _switch_session(self, session: Session) -> None:
        """Replace the active session, apply its policy, and refresh the display."""
        self.session = session
        self._apply_session_policy(session)
        self.console.clear()
        self._print_welcome()
        if session.turns > 0:
            await self._print_history(session.id)

    # -- history rendering ----------------------------------------------------

    async def _print_history(self, thread_id: str, limit: int = 10) -> None:
        """Replay the last *limit* checkpointed messages of a session."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        try:
            snapshot = await self.agent.app.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
            messages = list((snapshot.values or {}).get("messages", []))
        except Exception as exc:
            self.console.print(f"[dim]Could not load history: {exc}[/dim]")
            return

        if not messages:
            self.console.print("[dim]No prior history in this session.[/dim]")
            return

        recent = messages[-limit:]
        if len(messages) > len(recent):
            self.console.print(
                f"[dim]… showing the last {len(recent)} of "
                f"{len(messages)} messages …[/dim]"
            )

        def _text(content) -> str:
            if isinstance(content, list):
                return "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            return str(content)

        for msg in recent:
            if isinstance(msg, HumanMessage):
                self.console.print(
                    f"[bold cyan]you>[/bold cyan] {_text(msg.content)}",
                    highlight=False,
                )
            elif isinstance(msg, AIMessage):
                for tc in getattr(msg, "tool_calls", []):
                    args_str = ", ".join(
                        f"{k}={v!r}" for k, v in tc["args"].items()
                    )
                    self.console.print(
                        f"[dim]🐦 {tc['name']}({args_str})[/dim]",
                        highlight=False,
                    )
                text = _text(msg.content)
                if text:
                    self.console.print(f"🐦 {text}", highlight=False)
            elif isinstance(msg, ToolMessage):
                n = len(_text(msg.content).splitlines() or [""])
                self.console.print(
                    f"[dim]   ⎿ {msg.name or 'tool'}: "
                    f"{n} line{'s' if n != 1 else ''}[/dim]"
                )
        self.console.print()

    # -- status toolbar -------------------------------------------------------

    def _git_segment(self) -> str:
        """Return the cached git status segment for the current directory."""
        cwd = str(Path.cwd())
        now = time.monotonic()
        cache = self._git_cache
        if cache["cwd"] != cwd or now - cache["at"] >= _GIT_TTL_SECONDS:
            cache.update(cwd=cwd, at=now, text=_read_git_status(cwd))
        return cache["text"]

    def _get_toolbar(self) -> HTML:
        """Render the bottom status bar for prompt_toolkit.

        Left side: provider, cwd, session, and token info.  Right side:
        git status, padded flush-right when the terminal is wide enough,
        otherwise appended as a regular segment.
        """
        vendor = self.agent.provider.vendor_name
        model  = self.agent.provider.model_name
        ctx    = f"{self._last_context:,}" if self._last_context else "-"
        if self._context_window and self._last_context:
            ctx = f"{ctx}/{self._context_window:,}"
        spent  = f"↑{self._total_in:,}  ↓{self._total_out:,}"
        try:
            cwd = Path.cwd().relative_to(Path.home())
            cwd_str = f"~/{cwd}"
        except ValueError:
            cwd_str = str(Path.cwd())
        rest = (
            f" · {model}"
            f"   │   {cwd_str}"
            f"   │   session: {self.session.id}"
            f"   │   ctx: {ctx} tok"
            f"   │   spent: {spent} tok"
        )
        left_html = f" <b>{vendor}</b>{rest}"
        git = self._git_segment()
        if not git:
            return HTML(left_html)
        # Branch names may contain characters with meaning in PT's HTML.
        git_html = (
            git.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        try:
            from prompt_toolkit.application import get_app
            width = get_app().output.get_size().columns
        except Exception:
            width = 0
        left_len = len(f" {vendor}{rest}")
        pad = width - left_len - len(git) - 1
        if pad >= 3:
            return HTML(left_html + " " * pad + git_html + " ")
        return HTML(f"{left_html}   │   {git_html}")

    # -- display helpers ------------------------------------------------------

    def _print_welcome(self) -> None:
        """Print the startup banner with loaded skills and provider info."""
        from importlib.metadata import version, PackageNotFoundError
        try:
            v = version("birdie-agent")
        except PackageNotFoundError:
            v = "dev"
        skill_count = len(self.agent.registry.list_skills())
        agent_count = len(self.agent.agent_registry.list_agents())
        vendor = type(self.agent.provider).__name__.replace("Provider", "").lower()
        self.console.print(
            f"[bold green]Birdie[/bold green] [dim]v{v}[/dim]  "
            f"vendor: [cyan]{vendor}[/cyan]  "
            f"user: [cyan]{self.user_id}[/cyan]  "
            f"session: [cyan]{self.session.id}[/cyan]  "
            f"skills: [yellow]{skill_count}[/yellow]  "
            f"agents: [yellow]{agent_count}[/yellow]"
        )
        self.console.print("[dim]Type /help for commands, /quit to exit.[/dim]")

    def _show_help(self) -> None:
        """Print the slash-command reference."""
        self.console.print(HELP_TEXT)

    def _show_skills(self) -> None:
        """List all loaded skills with their enabled/disabled status."""
        skills: list[Skill] = self.agent.registry.list_skills()
        if not skills:
            self.console.print("[dim]No skills loaded.[/dim]")
            return
        allowed = self.agent.policy.get_allowed_skills(self.session.id)
        for skill in skills:
            status = "[green]enabled[/green]" if skill.name in allowed else "[red]disabled[/red]"
            self.console.print(
                f"  [bold]{skill.name}[/bold] v{skill.version}  {status}  - {skill.description}"
            )

    def _show_agents(self) -> None:
        """List all loaded agents with their enabled/disabled status."""
        from .core.models import AgentDef
        agents: list[AgentDef] = self.agent.agent_registry.list_agents()
        if not agents:
            self.console.print("[dim]No agents loaded.[/dim]")
            return
        allowed = self.agent.agent_registry.get_allowed_agents(self.session.id)
        for agent_def in agents:
            status = "[green]enabled[/green]" if agent_def.name in allowed else "[red]disabled[/red]"
            self.console.print(
                f"  [bold]{agent_def.name}[/bold] v{agent_def.version}  {status}  - {agent_def.description}"
            )

    def _show_tools(self) -> None:
        """List all callable tools available in the current session."""
        allowed = self.agent.policy.get_allowed_skills(self.session.id)
        tools = [
            t for t in self.agent.registry.list_tools()
            if self.agent.registry.is_tool_allowed(t.name, allowed)
        ]
        if not tools:
            self.console.print("[dim]No tools available.[/dim]")
            return
        for tool in tools:
            self.console.print(f"  [bold cyan]{tool.name}[/bold cyan]  - {tool.description}")

    def _show_cost(self) -> None:
        """Print token usage and an estimated cost for the current session."""
        from .core.pricing import estimate_cost

        model = self.agent.provider.model_name
        s_in = self.session.total_input_tokens
        s_out = self.session.total_output_tokens
        if self.session.total_cost_usd:
            self.console.print(
                f"  [dim]model:[/dim]    {model}\n"
                f"  [dim]session:[/dim]  ↑{s_in:,} in  ↓{s_out:,} out tokens\n"
                f"  [dim]cost:[/dim]     ${self.session.total_cost_usd:.4f} (reported by agent)"
            )
            return
        cost = estimate_cost(model, s_in, s_out)
        cost_str = f"${cost:.4f}" if cost is not None else "unknown (no pricing data)"
        self.console.print(
            f"  [dim]model:[/dim]    {model}\n"
            f"  [dim]session:[/dim]  ↑{s_in:,} in  ↓{s_out:,} out tokens\n"
            f"  [dim]cost:[/dim]     ~{cost_str}"
        )
        if cost is not None:
            self.console.print(
                "  [dim]Estimate uses list prices without cache discounts - "
                "treat as an upper bound.[/dim]"
            )

    def _show_info(self) -> None:
        """Print current user, session, and provider info."""
        vendor = type(self.agent.provider).__name__
        n = len(self.user_memory.entries)
        has_ltm = f"yes ({n} entries)" if n else "no"
        self.console.print(
            f"  [dim]user:[/dim]     {self.user_id}\n"
            f"  [dim]session:[/dim]  {self.session.id}\n"
            f"  [dim]turns:[/dim]    {self.session.turns}\n"
            f"  [dim]memory:[/dim]   {has_ltm}\n"
            f"  [dim]provider:[/dim] {vendor}"
        )

    def _render_tool_output(self, name: str, content: str) -> None:
        """Render tool output according to the current _tool_output_mode."""
        lines = content.splitlines() or [""]
        n = len(lines)

        if self._tool_output_mode == "off":
            self.console.print(f"[dim]   {n} line{'s' if n != 1 else ''}[/dim]")
            self.console.print()
            return

        if self._tool_output_mode == "short":
            limit = 1000
            display = content[:limit]
            remaining = len(content) - limit
            display_lines = display.splitlines() or [""]
        else:  # "full"
            display_lines = lines
            remaining = 0

        for line in display_lines:
            self.console.print(f"[dim]   {line}[/dim]", highlight=False)
        if remaining > 0:
            self.console.print(
                f"[dim]   ... {remaining} more character{'s' if remaining != 1 else ''}[/dim]"
            )
        self.console.print()

    def _make_acp_tool_renderer(self, status):
        """Build a callback that renders ACP tool_call session updates live.

        ACP providers execute tools inside the agent subprocess, so the
        graph's tools node never runs.  This callback mirrors the native
        rendering: a bold header when a call starts, dim output when it
        completes (respecting the /tool output mode).
        """
        from rich.markup import escape

        titles: dict[str, str] = {}
        announced_input: set[str] = set()
        finished: set[str] = set()

        def _render(event: dict) -> None:
            tool_id = event.get("id", "")
            if event["event"] == "tool_call":
                title = event.get("title") or event.get("kind") or "tool"
                titles[tool_id] = title
                self.console.print(
                    f"🐦 [bold]{escape(title)}[/bold]", highlight=False
                )
                status.update("[dim]running tools…[/dim]")

            # The call arguments often arrive only in a later update
            # (e.g. claude-agent-acp sends tool_call with empty rawInput).
            raw = event.get("raw_input")
            if isinstance(raw, dict) and raw and tool_id not in announced_input:
                announced_input.add(tool_id)
                cmd = raw.get("command")
                if cmd:
                    line = f"$ {cmd}"
                    desc = raw.get("description")
                    if desc:
                        line += f"  ({desc})"
                else:
                    line = ", ".join(f"{k}={v!r}" for k, v in raw.items())
                self.console.print(
                    f"   [dim]{escape(line)}[/dim]", highlight=False
                )

            if event.get("status") in ("completed", "failed") and tool_id not in finished:
                finished.add(tool_id)
                if event.get("status") == "failed":
                    self.console.print("[red]   ✗ failed[/red]")
                output = event.get("output") or ""
                if output:
                    self._render_tool_output(titles.get(tool_id, "tool"), output)
                status.update("[dim]thinking…[/dim]")

        return _render

    # -- logging --------------------------------------------------------------

    def _handle_log(self, arg: str) -> None:
        """Handle /log sub-commands."""
        parts = arg.strip().split()
        subcmd = parts[0].lower() if parts else ""
        subarg = parts[1].lower() if len(parts) > 1 else ""

        if subcmd == "llm":
            if subarg == "on":
                self._llm_log_on()
            elif subarg == "off":
                self._llm_log_off()
            else:
                self.console.print("[red]Usage: /log llm on|off[/red]")
        elif subcmd == "http":
            if subarg == "on":
                self._http_log_on()
            elif subarg == "off":
                self._http_log_off()
            else:
                self.console.print("[red]Usage: /log http on|off[/red]")
        else:
            self.console.print("[red]Usage: /log llm|http on|off[/red]")

    def _llm_log_on(self) -> None:
        if self._llm_log_handler:
            self.console.print("[dim]LLM logging is already on.[/dim]")
            return
        log_path = Path.home() / ".birdie" / "llm.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(log_path))
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        logger = logging.getLogger("birdie.core.llm_provider")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        self._llm_log_handler = handler
        self.console.print(f"[dim]LLM logging on → [bold]{log_path}[/bold][/dim]")

    def _llm_log_off(self) -> None:
        if not self._llm_log_handler:
            self.console.print("[dim]LLM logging is already off.[/dim]")
            return
        logger = logging.getLogger("birdie.core.llm_provider")
        logger.removeHandler(self._llm_log_handler)
        self._llm_log_handler.close()
        self._llm_log_handler = None
        self.console.print("[dim]LLM logging off.[/dim]")

    def _http_log_on(self) -> None:
        if self._orig_async_send is not None:
            self.console.print("[dim]HTTP logging is already on.[/dim]")
            return
        log_path = Path.home() / ".birdie" / "http.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = str(log_path)

        def _write(label: str, headline: str, body: str) -> None:
            ts = time.strftime("%H:%M:%S")
            try:
                body = json.dumps(json.loads(body), indent=2)
            except Exception:
                pass
            with open(log_file, "a") as f:
                f.write(f"\n{ts} {label} {headline}\n{body}\n")

        orig_async = httpx.AsyncClient.send
        orig_sync = httpx.Client.send
        self._orig_async_send = orig_async
        self._orig_sync_send = orig_sync

        async def _async_send(client_self, request, **kwargs):
            req_body = request.content.decode("utf-8", errors="replace")
            _write(">>", f"{request.method} {request.url}", req_body)
            response = await orig_async(client_self, request, **kwargs)
            if not kwargs.get("stream", False):
                _write("<<", str(response.status_code),
                       response.content.decode("utf-8", errors="replace"))
            else:
                _write("<<", str(response.status_code), "(streaming)")
            return response

        def _sync_send(client_self, request, **kwargs):
            req_body = request.content.decode("utf-8", errors="replace")
            _write(">>", f"{request.method} {request.url}", req_body)
            response = orig_sync(client_self, request, **kwargs)
            if not kwargs.get("stream", False):
                _write("<<", str(response.status_code),
                       response.content.decode("utf-8", errors="replace"))
            else:
                _write("<<", str(response.status_code), "(streaming)")
            return response

        httpx.AsyncClient.send = _async_send
        httpx.Client.send = _sync_send
        self.console.print(f"[dim]HTTP logging on → [bold]{log_path}[/bold][/dim]")

    def _http_log_off(self) -> None:
        if self._orig_async_send is None:
            self.console.print("[dim]HTTP logging is already off.[/dim]")
            return
        httpx.AsyncClient.send = self._orig_async_send
        httpx.Client.send = self._orig_sync_send
        self._orig_async_send = None
        self._orig_sync_send = None
        self.console.print("[dim]HTTP logging off.[/dim]")

    # -- slash command handler ------------------------------------------------

    def _handle_tool(self, arg: str) -> None:
        """Handle /tool sub-commands."""
        parts = arg.strip().split(maxsplit=1)
        subcmd = parts[0].lower() if parts else ""
        subarg = parts[1].lower() if len(parts) > 1 else ""

        if subcmd == "list":
            self._show_tools()

        elif subcmd == "output":
            if subarg not in _TOOL_OUTPUT_MODES:
                self.console.print(
                    "[red]Usage: /tool output full|short|off[/red]"
                )
            else:
                self._tool_output_mode = subarg
                self.console.print(f"[dim]Tool output mode: [bold]{subarg}[/bold][/dim]")

        elif subcmd == "cap":
            if not subarg:
                cap_display = f"{self._tool_output_cap:,}" if self._tool_output_cap else "off"
                self.console.print(f"[dim]Tool output cap: [bold]{cap_display}[/bold] characters[/dim]")
            elif subarg == "off":
                self._tool_output_cap = 0
                self.console.print("[dim]Tool output cap: [bold]off[/bold][/dim]")
            else:
                try:
                    self._tool_output_cap = int(subarg)
                    self.console.print(f"[dim]Tool output cap: [bold]{self._tool_output_cap:,}[/bold] characters[/dim]")
                except ValueError:
                    self.console.print("[red]Usage: /tool cap <N> | off[/red]")

        else:
            self.console.print(
                "[red]Usage: /tool list | output full|short|off | cap <N>|off[/red]"
            )

    def _resolve_skill_name(self, name: str) -> Optional[str]:
        """Return the exact skill name if found, else None. Prints a suggestion on miss."""
        import difflib
        known = [s.name for s in self.agent.registry.list_skills()]
        if name in known:
            return name
        matches = difflib.get_close_matches(name, known, n=1, cutoff=0.5)
        if matches:
            self.console.print(
                f"[red]Skill [bold]{name}[/bold] not found.[/red] "
                f"Did you mean [bold]{matches[0]}[/bold]?"
            )
        else:
            self.console.print(
                f"[red]Skill [bold]{name}[/bold] not found.[/red] "
                f"Use [bold]/skill list[/bold] to see available skills."
            )
        return None

    def _handle_skill(self, arg: str) -> None:
        """Handle /skill sub-commands."""
        parts = arg.strip().split(maxsplit=1)
        subcmd = parts[0].lower() if parts else ""
        subarg = parts[1] if len(parts) > 1 else ""

        if subcmd == "list":
            self._show_skills()

        elif subcmd == "enable":
            if not subarg:
                self.console.print("[red]Usage: /skill enable <SkillName>[/red]")
            else:
                resolved = self._resolve_skill_name(subarg)
                if resolved:
                    self.agent.enable_skill(self.session.id, resolved)
                    if resolved not in self.session.enabled_skills:
                        self.session.enabled_skills.append(resolved)
                    self.session.disabled_skills = [
                        s for s in self.session.disabled_skills if s != resolved
                    ]
                    self.session_manager.save(self.session)
                    self.console.print(f"[green]Enabled[/green] {resolved}")

        elif subcmd == "disable":
            if not subarg:
                self.console.print("[red]Usage: /skill disable <SkillName>[/red]")
            else:
                resolved = self._resolve_skill_name(subarg)
                if resolved:
                    self.agent.disable_skill(self.session.id, resolved)
                    if resolved not in self.session.disabled_skills:
                        self.session.disabled_skills.append(resolved)
                    self.session.enabled_skills = [
                        s for s in self.session.enabled_skills if s != resolved
                    ]
                    self.session_manager.save(self.session)
                    self.console.print(f"[red]Disabled[/red] {resolved}")

        elif subcmd == "reload":
            try:
                n = self.agent.reload_skills()
            except Exception as exc:
                self.console.print(f"[red]Reload failed:[/red] {exc}")
            else:
                self.console.print(f"[dim]Reloaded [bold]{n}[/bold] skills from disk.[/dim]")

        elif subcmd == "new":
            if not subarg:
                self.console.print("[red]Usage: /skill new <SkillName>[/red]")
            else:
                from .core.loader import scaffold_skill
                target_dir = Path.home() / ".birdie" / "skills"
                try:
                    path = scaffold_skill(str(target_dir), subarg)
                except FileExistsError as exc:
                    self.console.print(f"[red]{exc}[/red]")
                else:
                    self.console.print(
                        f"[green]Created[/green] {path}\n"
                        f"[dim]Edit it, then run /skill reload and "
                        f"/skill enable {subarg}.[/dim]"
                    )

        else:
            self.console.print(
                "[red]Usage: /skill list | enable <name> | disable <name> "
                "| reload | new <name>[/red]"
            )

    def _resolve_agent_name(self, name: str) -> Optional[str]:
        """Return the exact agent name if found, else None. Prints a suggestion on miss."""
        import difflib
        known = [a.name for a in self.agent.agent_registry.list_agents()]
        if name in known:
            return name
        matches = difflib.get_close_matches(name, known, n=1, cutoff=0.5)
        if matches:
            self.console.print(
                f"[red]Agent [bold]{name}[/bold] not found.[/red] "
                f"Did you mean [bold]{matches[0]}[/bold]?"
            )
        else:
            self.console.print(
                f"[red]Agent [bold]{name}[/bold] not found.[/red] "
                f"Use [bold]/agent list[/bold] to see available agents."
            )
        return None

    def _handle_agent(self, arg: str) -> None:
        """Handle /agent sub-commands."""
        parts = arg.strip().split(maxsplit=1)
        subcmd = parts[0].lower() if parts else ""
        subarg = parts[1] if len(parts) > 1 else ""

        if subcmd == "list":
            self._show_agents()

        elif subcmd == "enable":
            if not subarg:
                self.console.print("[red]Usage: /agent enable <AgentName>[/red]")
            else:
                resolved = self._resolve_agent_name(subarg)
                if resolved:
                    self.agent.enable_agent(self.session.id, resolved)
                    if resolved not in self.session.enabled_agents:
                        self.session.enabled_agents.append(resolved)
                    self.session.disabled_agents = [
                        a for a in self.session.disabled_agents if a != resolved
                    ]
                    self.session_manager.save(self.session)
                    self.console.print(f"[green]Enabled[/green] {resolved}")

        elif subcmd == "disable":
            if not subarg:
                self.console.print("[red]Usage: /agent disable <AgentName>[/red]")
            else:
                resolved = self._resolve_agent_name(subarg)
                if resolved:
                    self.agent.disable_agent(self.session.id, resolved)
                    if resolved not in self.session.disabled_agents:
                        self.session.disabled_agents.append(resolved)
                    self.session.enabled_agents = [
                        a for a in self.session.enabled_agents if a != resolved
                    ]
                    self.session_manager.save(self.session)
                    self.console.print(f"[red]Disabled[/red] {resolved}")

        elif subcmd == "output":
            if subarg.lower() not in _AGENT_OUTPUT_MODES:
                self.console.print(
                    "[red]Usage: /agent output full|short|off[/red]"
                )
            else:
                self._agent_output_mode = subarg.lower()
                self.agent.agent_output_mode = subarg.lower()
                self.console.print(f"[dim]Agent output mode: [bold]{subarg.lower()}[/bold][/dim]")

        else:
            self.console.print(
                "[red]Usage: /agent list | enable <name> | disable <name> | output full|short|off[/red]"
            )

    async def _handle_session(self, arg: str) -> None:
        """Handle /session sub-commands."""
        parts = arg.strip().split(maxsplit=1)
        subcmd = parts[0].lower() if parts else ""
        subarg = parts[1] if len(parts) > 1 else ""

        if subcmd == "new":
            new_session = self.session_manager.create(self.user_id)
            await self._switch_session(new_session)

        elif subcmd == "switch":
            if not subarg:
                self.console.print("[red]Usage: /session switch <session_id>[/red]")
                return
            try:
                loaded = self.session_manager.load(self.user_id, subarg)
                await self._switch_session(loaded)
            except FileNotFoundError as exc:
                self.console.print(f"[red]{exc}[/red]")

        elif subcmd == "delete":
            if not subarg:
                self.console.print("[red]Usage: /session delete <session_id>[/red]")
                return
            try:
                is_current = subarg == self.session.id
                self.session_manager.delete(self.user_id, subarg)
                # Also drop the conversation history stored under this
                # thread_id, otherwise it lingers in checkpoints.db forever.
                checkpointer = getattr(self.agent.app, "checkpointer", None)
                if checkpointer is not None and hasattr(checkpointer, "adelete_thread"):
                    try:
                        await checkpointer.adelete_thread(subarg)
                    except Exception as exc:
                        self.console.print(
                            f"[dim]Could not delete checkpoint history: {exc}[/dim]"
                        )
                self.console.print(f"[dim]Deleted session {subarg}[/dim]")
                if is_current:
                    new_session = self.session_manager.create(self.user_id)
                    await self._switch_session(new_session)
            except FileNotFoundError as exc:
                self.console.print(f"[red]{exc}[/red]")

        elif subcmd == "list":
            sessions = self.session_manager.list_sessions(self.user_id)
            if not sessions:
                self.console.print("[dim]No sessions found.[/dim]")
                return
            for sid in sessions:
                marker = "  [green]← current[/green]" if sid == self.session.id else ""
                self.console.print(f"  {sid}{marker}")

        elif subcmd == "info":
            s = self.session
            n = len(self.user_memory.entries)
            has_ltm = f"yes ({n} entries, user-scoped)" if n else "no"
            self.console.print(
                f"  [dim]session:[/dim]  {s.id}\n"
                f"  [dim]user:[/dim]     {s.user_id}\n"
                f"  [dim]created:[/dim]  {s.created_at}\n"
                f"  [dim]updated:[/dim]  {s.updated_at}\n"
                f"  [dim]turns:[/dim]    {s.turns}\n"
                f"  [dim]skills:[/dim]   {', '.join(s.enabled_skills) or 'defaults'}\n"
                f"  [dim]agents:[/dim]   {', '.join(s.enabled_agents) or 'none'}\n"
                f"  [dim]memory:[/dim]   {has_ltm}"
            )

        else:
            self.console.print(
                "[red]Usage: /session new | switch <id> | delete <id> | list | info[/red]"
            )

    def _handle_cd(self, arg: str) -> None:
        target = Path(arg.strip()).expanduser() if arg.strip() else Path.home()
        try:
            os.chdir(target)
            try:
                display = f"~/{Path.cwd().relative_to(Path.home())}"
            except ValueError:
                display = str(Path.cwd())
            self.console.print(f"[dim]{display}[/dim]")
        except FileNotFoundError:
            self.console.print(f"[red]No such directory:[/red] {target}")
        except NotADirectoryError:
            self.console.print(f"[red]Not a directory:[/red] {target}")
        except PermissionError:
            self.console.print(f"[red]Permission denied:[/red] {target}")

    async def _handle_slash(self, line: str) -> bool:
        """Return True if line was a slash command (handled here), False otherwise."""
        parts = line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit"):
            self.console.print("[dim]Goodbye.[/dim]")
            sys.exit(0)

        elif cmd == "/help":
            self._show_help()

        elif cmd == "/new":
            # Legacy alias: same as /session new
            new_session = self.session_manager.create(self.user_id)
            await self._switch_session(new_session)

        elif cmd == "/log":
            self._handle_log(arg)

        elif cmd == "/tool":
            self._handle_tool(arg)

        elif cmd == "/skill":
            self._handle_skill(arg)

        elif cmd == "/agent":
            self._handle_agent(arg)

        elif cmd == "/remember":
            if not arg:
                self.console.print("[red]Usage: /remember <text>[/red]")
            else:
                self.user_memory.add(arg)
                self.session_manager.save_user_memory(self.user_memory)
                self.console.print("[dim]Remembered.[/dim]")

        elif cmd == "/cost":
            self._show_cost()

        elif cmd == "/history":
            try:
                limit = int(arg) if arg.strip() else 10
            except ValueError:
                self.console.print("[red]Usage: /history [N][/red]")
                return True
            await self._print_history(self.session.id, limit=max(1, limit))

        elif cmd == "/info":
            self._show_info()

        elif cmd == "/session":
            await self._handle_session(arg)

        elif cmd == "/cd":
            self._handle_cd(arg)

        elif cmd == "/clear":
            self.console.clear()

        else:
            self.console.print(f"[red]Unknown command:[/red] {cmd}  (type /help for list)")

        return True

    # -- streaming turn -------------------------------------------------------

    async def _compact(self) -> None:
        """Force-compact the current session's history into LTM."""
        status = self.console.status("[dim]Compacting…[/dim]", spinner="dots")
        status.start()
        try:
            n_removed, summary = await self.agent.compact_session(
                self.session.id, user_id=self.user_id,
            )
        finally:
            status.stop()

        if n_removed == 0:
            self.console.print(
                "[dim]Nothing to compact - history is too short.[/dim]"
            )
        else:
            self.console.print(
                f"[dim]Compacted {n_removed} messages into LTM.[/dim]"
            )
            if summary:
                for line in summary.splitlines():
                    self.console.print(f"[dim]{line}[/dim]", highlight=False)

    async def _stream_turn(self, message: str) -> None:
        """Send *message* to the agent and render the streamed response.

        History is managed by LangGraph's checkpointer (keyed by session ID as
        thread_id).  Long-term memory is read from the user-scoped memory store
        and injected per-turn via config - it is not stored in the checkpoint.

        Args:
            message: The user's input text for this turn.
        """
        from langchain_core.messages import AIMessage, ToolMessage

        ltm = self.user_memory.as_strings()

        printed_any = False
        status = self.console.status("[dim]thinking…[/dim]", spinner="dots")
        status.start()
        self._active_status = status

        provider = self.agent.provider
        has_tool_events = hasattr(provider, "tool_event_callback")
        if has_tool_events:
            provider.tool_event_callback = self._make_acp_tool_renderer(status)
        has_permission_gate = hasattr(provider, "permission_callback")
        if has_permission_gate:
            provider.permission_callback = self._approve_acp_permission

        try:
            async for update in self.agent.astream(
                message,
                thread_id=self.session.id,
                user_id=self.user_id,
                long_term_memory=ltm if ltm else None,
                config={"configurable": {"tool_output_cap": self._tool_output_cap}},
            ):
                for node_name, node_output in update.items():
                    msgs = node_output.get("messages", [])

                    if node_name == "tools":
                        for msg in msgs:
                            if isinstance(msg, ToolMessage):
                                name = msg.name or ""
                                # Sub-agents print their own transcript; skip
                                # _render_tool_output unless mode is "off" (no
                                # transcript shown) or it's a regular skill tool.
                                is_agent = self.agent.agent_registry.get_agent(name) is not None
                                if is_agent and self._agent_output_mode != "off":
                                    pass  # transcript already printed by the tool
                                else:
                                    self._render_tool_output(name, str(msg.content))
                        status.update("[dim]thinking…[/dim]")

                    elif node_name == "agent":
                        for msg in msgs:
                            if isinstance(msg, AIMessage):
                                um = getattr(msg, "usage_metadata", None)
                                if um:
                                    self._last_context = um.get("input_tokens", 0)
                                    self._total_in  += um.get("input_tokens", 0)
                                    self._total_out += um.get("output_tokens", 0)
                                    self.session.total_input_tokens += um.get("input_tokens", 0)
                                    self.session.total_output_tokens += um.get("output_tokens", 0)
                                rm = getattr(msg, "response_metadata", None) or {}
                                if rm.get("context_window"):
                                    self._context_window = rm["context_window"]
                                rm_cost = rm.get("cost")
                                if isinstance(rm_cost, dict) and rm_cost.get("amount"):
                                    self.session.total_cost_usd += float(rm_cost["amount"])
                                for tc in getattr(msg, "tool_calls", []):
                                    args_str = ", ".join(
                                        f"{k}={v!r}" for k, v in tc["args"].items()
                                    )
                                    self.console.print(
                                        f"🐦 [bold]{tc['name']}[/bold]({args_str})"
                                    )
                                if getattr(msg, "tool_calls", None):
                                    status.update("[dim]running tools…[/dim]")
                                elif msg.content:
                                    status.stop()
                                    content = msg.content
                                    if isinstance(content, list):
                                        content = "\n".join(
                                            b.get("text", "") if isinstance(b, dict) else str(b)
                                            for b in content
                                        )
                                    lines = str(content).splitlines()
                                    for i, line in enumerate(lines):
                                        prefix = "🐦 " if i == 0 else "   "
                                        self.console.print(f"{prefix}{line}", highlight=False)
                                    self.console.print()
                                    printed_any = True
        finally:
            if has_tool_events:
                provider.tool_event_callback = None
            if has_permission_gate:
                provider.permission_callback = None
            status.stop()
            self._active_status = None

        if not printed_any:
            self.console.print("[dim](no response)[/dim]")

        # The checkpointer owns the message history.
        # We only need to update the lightweight session metadata.
        self.session.touch()
        self.session_manager.save(self.session)

    # -- main loop ------------------------------------------------------------

    async def run_non_interactive(self, prompt: str) -> None:
        """Run a single non-interactive turn and print the AI response to stdout."""
        from langchain_core.messages import AIMessage, ToolMessage
        
        # Non-interactive mode: clean output to stdout, no interactive elements
        
        ltm = self.user_memory.as_strings()
        
        try:
            # Collect all output parts (AI messages and tool calls)
            output_parts = []
            
            async for update in self.agent.astream(
                prompt,
                thread_id=self.session.id,
                user_id=self.user_id,
                long_term_memory=ltm if ltm else None,
                config={"configurable": {"tool_output_cap": self._tool_output_cap}},
            ):
                for node_name, node_output in update.items():
                    msgs = node_output.get("messages", [])
                    
                    if node_name == "tools":
                        for msg in msgs:
                            if isinstance(msg, ToolMessage):
                                name = msg.name or ""
                                # Check if this is a sub-agent (they print their own transcript)
                                is_agent = self.agent.agent_registry.get_agent(name) is not None
                                if not is_agent:  # Regular skill tool - include in output
                                    content = str(msg.content) if msg.content else ""
                                    if content:
                                        output_parts.append(f"🐦 {name}: {content}")
                    
                    elif node_name == "agent":
                        for msg in msgs:
                            if isinstance(msg, AIMessage):
                                # Update token usage tracking
                                um = getattr(msg, "usage_metadata", None)
                                if um:
                                    self._last_context = um.get("input_tokens", 0)
                                    self._total_in += um.get("input_tokens", 0)
                                    self._total_out += um.get("output_tokens", 0)
                                    self.session.total_input_tokens += um.get("input_tokens", 0)
                                    self.session.total_output_tokens += um.get("output_tokens", 0)
                                
                                rm = getattr(msg, "response_metadata", None) or {}
                                if rm.get("context_window"):
                                    self._context_window = rm["context_window"]
                                rm_cost = rm.get("cost")
                                if isinstance(rm_cost, dict) and rm_cost.get("amount"):
                                    self.session.total_cost_usd += float(rm_cost["amount"])
                                
                                # Handle tool calls
                                for tc in getattr(msg, "tool_calls", []):
                                    args_str = ", ".join(
                                        f"{k}={v!r}" for k, v in tc["args"].items()
                                    )
                                    output_parts.append(f"🐦 {tc['name']}({args_str})")
                                
                                # Collect AI content
                                content = msg.content
                                if isinstance(content, list):
                                    content = "\n".join(
                                        b.get("text", "") if isinstance(b, dict) else str(b)
                                        for b in content
                                    )
                                if content:
                                    output_parts.append(str(content))
            
            # Print the collected output to stdout
            if output_parts:
                print("\n".join(output_parts))
            else:
                print("(no response)")
                
        except Exception as exc:
            # Print errors to stderr and exit with non-zero code
            print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
            sys.exit(1)
        finally:
            # Update session metadata
            self.session.touch()
            self.session_manager.save(self.session)

    async def run(self) -> None:
        """Start the interactive REPL and block until the user quits."""
        self._print_welcome()
        if self.session.turns > 0:
            # Resumed session: replay the tail of the conversation.
            await self._print_history(self.session.id)

        while True:
            try:
                user_input = await self.session_prompt.prompt_async(
                    self._get_prompt,
                    pre_run=self._pre_run,
                )
                self._ctrl_c_warned = False
            except SystemExit:
                self.console.print("[dim]Goodbye.[/dim]")
                return
            except EOFError:
                self.console.print("[dim]Goodbye.[/dim]")
                return
            except KeyboardInterrupt:
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                if user_input.strip().lower() == "/compact":
                    await self._compact()
                else:
                    await self._handle_slash(user_input)
                continue

            loop = asyncio.get_running_loop()
            task = loop.create_task(self._stream_turn(user_input))
            loop.add_signal_handler(signal.SIGINT, task.cancel)
            try:
                await task
            except asyncio.CancelledError:
                self.console.print("\n[dim]Interrupted.[/dim]")
            except BirdieRateLimitError:
                self.console.print(
                    "[yellow]Rate limit reached - please wait a moment and try again.[/yellow]"
                )
            except Exception as exc:
                self.console.print(
                    f"[red bold]Error:[/red bold] {type(exc).__name__}: {exc}"
                )
                self.console.print_exception(show_locals=False)
            finally:
                loop.remove_signal_handler(signal.SIGINT)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Birdie interactive CLI")
    parser.add_argument(
        "--user",
        metavar="USER_ID",
        default=None,
        help="User identity - organises sessions under ~/.birdie/sessions/<user>/ "
             "(default: system username)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Resume an existing session by ID (e.g. 2026-04-28_1)",
    )
    parser.add_argument(
        "--skills-dir",
        default=None,
        help="Override the built-in skills directory (default: bundled birdie/skills). "
             "Additional skills are always loaded from ~/.birdie/skills/ if it exists.",
    )
    parser.add_argument(
        "--agents-dir",
        default=None,
        help="Override the built-in agents directory (default: bundled birdie/agents). "
             "Additional agents are always loaded from ~/.birdie/agents/ if it exists.",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        default=None,
        help="Path to a JSON provider config file (overrides LLM_VENDOR / LLM_MODEL env vars)",
    )
    parser.add_argument(
        "-p", "--prompt",
        metavar="PROMPT",
        default=None,
        help="Run non-interactively with the given prompt and print the AI response to stdout",
    )
    args = parser.parse_args()

    user_id = (
        args.user
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or "default"
    )
    skills_dir = args.skills_dir or os.path.join(os.path.dirname(__file__), "skills")
    agents_dir = args.agents_dir or os.path.join(os.path.dirname(__file__), "agents")
    provider_config = Path(args.config) if args.config else None

    asyncio.run(_async_main(args.session_id, user_id, skills_dir, agents_dir, provider_config, args.prompt))


_PROVIDER_HELP = """
[bold red]No LLM vendor configured.[/bold red]

Birdie needs to know which LLM provider to use. Configure it with environment
variables or a JSON config file.

[bold]Option 1 - environment variables[/bold]

  [cyan]export LLM_VENDOR=openai[/cyan]          # or: anthropic, mistral, gemini, azure, ollama
  [cyan]export LLM_MODEL=gpt-4o[/cyan]           # optional - uses provider default if omitted
  [cyan]export OPENAI_API_KEY=sk-...[/cyan]       # vendor-specific key variable:
                                   #   OPENAI_API_KEY, ANTHROPIC_API_KEY,
                                   #   MISTRAL_API_KEY, GEMINI_API_KEY,
                                   #   AZURE_OPENAI_API_KEY
  [cyan]birdie[/cyan]

[bold]Option 2 - config file[/bold]

  Create a JSON file, e.g. [cyan]~/.birdie/provider.json[/cyan]:

    {
      "vendor": "openai",
      "model": "gpt-4o",
      "api_key": "sk-..."
    }

  Then start Birdie with:

    [cyan]birdie --config ~/.birdie/provider.json[/cyan]

[bold]Supported vendors:[/bold] openai, anthropic, mistral, gemini, azure, ollama
"""


def _abort(console: Console, message: str) -> None:
    console.print(message)
    sys.exit(1)


async def _async_main(
    session_id_arg: Optional[str],
    user_id: str,
    skills_dir: str,
    agents_dir: Optional[str],
    provider_config,
    prompt: Optional[str] = None,
) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    console = Console()

    # Fail fast if no vendor source is configured at all.
    if (
        provider_config is None
        and not os.environ.get("LLM_PROVIDER_CONFIG")
        and not os.environ.get("LLM_VENDOR")
    ):
        _abort(console, _PROVIDER_HELP)

    session_manager = SessionManager()

    if session_id_arg:
        try:
            session = session_manager.load(user_id, session_id_arg)
        except FileNotFoundError:
            print(f"Error: unknown session {session_id_arg!r}", file=sys.stderr)
            sys.exit(1)
    else:
        session = session_manager.create(user_id)

    user_memory = session_manager.load_user_memory(user_id)

    db_path = session_manager.db_path(user_id)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # AsyncSqliteSaver requires an async context manager to own the connection
    # lifetime, so the entire REPL runs inside it.
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        try:
            agent = DynamicAgent.from_config(
                provider_config, skills_dir=skills_dir, agents_dir=agents_dir,
                agent_console=console, checkpointer=checkpointer,
            )
        except ValueError as exc:
            _abort(console, f"[bold red]Configuration error:[/bold red] {exc}\n\n{_PROVIDER_HELP}")
        except ImportError as exc:
            _abort(console, f"[bold red]Missing dependency:[/bold red] {exc}")

        cli = BirdieCLI(
            agent,
            session_manager=session_manager,
            session=session,
            user_id=user_id,
            user_memory=user_memory,
            console=console,
        )
        
        if prompt is not None:
            # Non-interactive mode: run single prompt and output to stdout
            await cli.run_non_interactive(prompt)
        else:
            # Interactive mode: start the REPL
            await cli.run()


if __name__ == "__main__":
    main()
