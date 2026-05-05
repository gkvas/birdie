"""
Interactive REPL for Birdie.

Presents a prompt-toolkit prompt with:

- Streaming output rendered via Rich (tool calls in yellow panels, AI text
  in green Markdown panels).
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
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .agent.run import DynamicAgent
from .core.models import Skill
from .core.session import Session, SessionManager, UserMemory


PROMPT_STYLE = Style.from_dict({"prompt": "ansicyan bold"})

HELP_TEXT = """
[bold cyan]Birdie CLI - available slash commands[/bold cyan]

  [yellow]/help[/yellow]                         Show this help
  [yellow]/quit[/yellow]  [yellow]/exit[/yellow]                 Exit the session
  [yellow]/new[/yellow]                          Start a fresh conversation (new thread)
  [yellow]/remember <text>[/yellow]              Save a note to long-term memory
  [yellow]/info[/yellow]                         Show session info (user, session, provider)

  [bold]Tool commands[/bold]
  [yellow]/tool list[/yellow]                    List all available tools
  [yellow]/tool output full[/yellow]             Show complete tool output
  [yellow]/tool output short[/yellow]            Show first 1000 characters + remaining count (default)
  [yellow]/tool output off[/yellow]              Show only line count, no content

  [bold]Skill commands[/bold]
  [yellow]/skill list[/yellow]                   List all loaded skills with status
  [yellow]/skill enable <name>[/yellow]          Enable a skill (persists to session)
  [yellow]/skill disable <name>[/yellow]         Disable a skill (persists to session)

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


class BirdieCLI:
    def __init__(
        self,
        agent: DynamicAgent,
        session_manager: SessionManager,
        session: Session,
        user_id: str,
        user_memory: UserMemory,
    ) -> None:
        self.agent = agent
        self.session_manager = session_manager
        self.session = session
        self.user_id = user_id
        self.user_memory = user_memory
        self.console = Console()

        self._total_in: int = 0
        self._total_out: int = 0
        self._last_context: int = 0
        self._tool_output_mode: str = "short"
        self._llm_log_handler: Optional[logging.FileHandler] = None
        self._orig_async_send = None
        self._orig_sync_send = None

        # Apply stored skill grants for the initial session
        self._apply_session_policy(session)

        kb = KeyBindings()

        @kb.add("c-c")
        def _quit(event):
            """Ctrl+C exits Birdie."""
            event.app.exit(result=None, exception=SystemExit(0))

        @kb.add("c-j")
        def _newline(event):
            """Ctrl+J inserts a newline for multi-line input."""
            event.current_buffer.insert_text("\n")

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
        """Apply stored skill grants from session to the policy."""
        for skill in session.enabled_skills:
            self.agent.enable_skill_for_user(session.id, skill)
        for skill in session.disabled_skills:
            self.agent.disable_skill_for_user(session.id, skill)

    def _switch_session(self, session: Session) -> None:
        """Replace the active session, apply its policy, and refresh the display."""
        self.session = session
        self._apply_session_policy(session)
        self.console.clear()
        self._print_welcome()

    # -- status toolbar -------------------------------------------------------

    def _get_toolbar(self) -> HTML:
        """Render the bottom status bar for prompt_toolkit."""
        vendor = self.agent.provider.vendor_name
        model  = self.agent.provider.model_name
        ctx    = f"{self._last_context:,}" if self._last_context else "-"
        spent  = f"↑{self._total_in:,}  ↓{self._total_out:,}"
        return HTML(
            f" <b>{vendor}</b> · {model}"
            f"   │   session: {self.session.id}"
            f"   │   ctx: {ctx} tok"
            f"   │   spent: {spent} tok"
        )

    # -- display helpers ------------------------------------------------------

    def _print_welcome(self) -> None:
        """Print the startup banner with loaded skills and provider info."""
        from importlib.metadata import version, PackageNotFoundError
        try:
            v = version("birdie-agent")
        except PackageNotFoundError:
            v = "dev"
        skill_count = len(self.agent.registry.list_skills())
        vendor = type(self.agent.provider).__name__.replace("Provider", "").lower()
        self.console.print(Panel(
            f"[bold green]Birdie[/bold green] [dim]v{v}[/dim]  |  vendor: [cyan]{vendor}[/cyan]"
            f"  |  user: [cyan]{self.user_id}[/cyan]"
            f"  |  session: [cyan]{self.session.id}[/cyan]"
            f"  |  skills found: [yellow]{skill_count}[/yellow]\n"
            "Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit.",
            border_style="green",
        ))

    def _show_help(self) -> None:
        """Print the slash-command reference."""
        self.console.print(HELP_TEXT)

    def _show_skills(self) -> None:
        """List all loaded skills with their enabled/disabled status."""
        skills: list[Skill] = self.agent.registry.list_skills()
        if not skills:
            self.console.print("[dim]No skills loaded.[/dim]")
            return
        allowed = self.agent.policy.get_allowed_skills_for_user(self.session.id)
        for skill in skills:
            status = "[green]enabled[/green]" if skill.name in allowed else "[red]disabled[/red]"
            self.console.print(
                f"  [bold]{skill.name}[/bold] v{skill.version}  {status}  - {skill.description}"
            )

    def _show_tools(self) -> None:
        """List all callable tools available in the current session."""
        allowed = self.agent.policy.get_allowed_skills_for_user(self.session.id)
        tools = [
            t for t in self.agent.registry.list_tools()
            if self.agent.registry.is_tool_allowed(t.name, allowed)
        ]
        if not tools:
            self.console.print("[dim]No tools available.[/dim]")
            return
        for tool in tools:
            self.console.print(f"  [bold cyan]{tool.name}[/bold cyan]  - {tool.description}")

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
        lines = content.splitlines()
        n = len(lines)

        if self._tool_output_mode == "off":
            self.console.print(
                f"[dim yellow]tool: [bold]{name}[/bold] - {n} line{'s' if n != 1 else ''}[/dim yellow]"
            )

        elif self._tool_output_mode == "short":
            limit = 1000
            body = content[:limit]
            remaining = len(content) - limit
            if remaining > 0:
                body += f"\n[dim]... {remaining} more character{'s' if remaining != 1 else ''}[/dim]"
            self.console.print(Panel(
                body,
                title=f"[yellow]tool: {name}[/yellow]",
                border_style="yellow",
                expand=False,
            ))

        else:  # "full"
            self.console.print(Panel(
                Text(content, style="white"),
                title=f"[yellow]tool: {name}[/yellow]",
                border_style="yellow",
                expand=False,
            ))

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

        else:
            self.console.print(
                "[red]Usage: /tool list | output full|short|off[/red]"
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
                    self.agent.enable_skill_for_user(self.session.id, resolved)
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
                    self.agent.disable_skill_for_user(self.session.id, resolved)
                    if resolved not in self.session.disabled_skills:
                        self.session.disabled_skills.append(resolved)
                    self.session.enabled_skills = [
                        s for s in self.session.enabled_skills if s != resolved
                    ]
                    self.session_manager.save(self.session)
                    self.console.print(f"[red]Disabled[/red] {resolved}")

        else:
            self.console.print(
                "[red]Usage: /skill list | enable <name> | disable <name>[/red]"
            )

    def _handle_session(self, arg: str) -> None:
        """Handle /session sub-commands."""
        parts = arg.strip().split(maxsplit=1)
        subcmd = parts[0].lower() if parts else ""
        subarg = parts[1] if len(parts) > 1 else ""

        if subcmd == "new":
            new_session = self.session_manager.create(self.user_id)
            self._switch_session(new_session)

        elif subcmd == "switch":
            if not subarg:
                self.console.print("[red]Usage: /session switch <session_id>[/red]")
                return
            try:
                loaded = self.session_manager.load(self.user_id, subarg)
                self._switch_session(loaded)
            except FileNotFoundError as exc:
                self.console.print(f"[red]{exc}[/red]")

        elif subcmd == "delete":
            if not subarg:
                self.console.print("[red]Usage: /session delete <session_id>[/red]")
                return
            try:
                is_current = subarg == self.session.id
                self.session_manager.delete(self.user_id, subarg)
                self.console.print(f"[dim]Deleted session {subarg}[/dim]")
                if is_current:
                    new_session = self.session_manager.create(self.user_id)
                    self._switch_session(new_session)
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
                f"  [dim]memory:[/dim]   {has_ltm}"
            )

        else:
            self.console.print(
                "[red]Usage: /session new | switch <id> | delete <id> | list | info[/red]"
            )

    def _handle_slash(self, line: str) -> bool:
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
            self._switch_session(new_session)

        elif cmd == "/log":
            self._handle_log(arg)

        elif cmd == "/tool":
            self._handle_tool(arg)

        elif cmd == "/skill":
            self._handle_skill(arg)

        elif cmd == "/remember":
            if not arg:
                self.console.print("[red]Usage: /remember <text>[/red]")
            else:
                self.user_memory.add(arg)
                self.session_manager.save_user_memory(self.user_memory)
                self.console.print("[dim]Remembered.[/dim]")

        elif cmd == "/info":
            self._show_info()

        elif cmd == "/session":
            self._handle_session(arg)

        elif cmd == "/clear":
            self.console.clear()

        else:
            self.console.print(f"[red]Unknown command:[/red] {cmd}  (type /help for list)")

        return True

    # -- streaming turn -------------------------------------------------------

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

        try:
            async for update in self.agent.astream(
                message,
                thread_id=self.session.id,
                long_term_memory=ltm if ltm else None,
            ):
                for node_name, node_output in update.items():
                    msgs = node_output.get("messages", [])

                    if node_name == "tools":
                        for msg in msgs:
                            if isinstance(msg, ToolMessage):
                                self._render_tool_output(msg.name or "", str(msg.content))
                        status.update("[dim]thinking…[/dim]")

                    elif node_name == "agent":
                        for msg in msgs:
                            if isinstance(msg, AIMessage):
                                um = getattr(msg, "usage_metadata", None)
                                if um:
                                    self._last_context = um.get("input_tokens", 0)
                                    self._total_in  += um.get("input_tokens", 0)
                                    self._total_out += um.get("output_tokens", 0)
                                for tc in getattr(msg, "tool_calls", []):
                                    args_str = ", ".join(
                                        f"{k}={v!r}" for k, v in tc["args"].items()
                                    )
                                    self.console.print(
                                        f"[dim yellow]→ calling [bold]{tc['name']}[/bold]"
                                        f"({args_str})[/dim yellow]"
                                    )
                                if getattr(msg, "tool_calls", None):
                                    status.update("[dim]running tools…[/dim]")
                                elif msg.content:
                                    status.stop()
                                    self.console.print(Panel(
                                        Markdown(msg.content),
                                        title="[green]birdie[/green]",
                                        border_style="green",
                                    ))
                                    printed_any = True
        finally:
            status.stop()

        if not printed_any:
            self.console.print("[dim](no response)[/dim]")

        # The checkpointer owns the message history.
        # We only need to update the lightweight session metadata.
        self.session.touch()
        self.session_manager.save(self.session)

    # -- main loop ------------------------------------------------------------

    async def run(self) -> None:
        """Start the interactive REPL and block until the user quits."""
        self._print_welcome()

        while True:
            try:
                user_input = await self.session_prompt.prompt_async(
                    [("class:prompt", "you> ")],
                )
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
                self._handle_slash(user_input)
                continue

            try:
                await self._stream_turn(user_input)
            except Exception as exc:
                self.console.print(
                    f"[red bold]Error:[/red bold] {type(exc).__name__}: {exc}"
                )
                self.console.print_exception(show_locals=False)


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
        "--config",
        metavar="FILE",
        default=None,
        help="Path to a JSON provider config file (overrides LLM_VENDOR / LLM_MODEL env vars)",
    )
    args = parser.parse_args()

    user_id = (
        args.user
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or "default"
    )
    skills_dir = args.skills_dir or os.path.join(os.path.dirname(__file__), "skills")
    provider_config = Path(args.config) if args.config else None

    asyncio.run(_async_main(args.session_id, user_id, skills_dir, provider_config))


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
    provider_config,
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
                provider_config, skills_dir=skills_dir, checkpointer=checkpointer
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
        )
        await cli.run()


if __name__ == "__main__":
    main()
