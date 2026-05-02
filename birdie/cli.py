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
import os
import sys
from pathlib import Path
from typing import Optional

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

  [yellow]/help[/yellow]                    Show this help
  [yellow]/quit[/yellow]  [yellow]/exit[/yellow]            Exit the session
  [yellow]/new[/yellow]                     Start a fresh conversation (new thread)
  [yellow]/skills[/yellow]                  List all loaded skills
  [yellow]/tools[/yellow]                   List all available tools
  [yellow]/enable <skill>[/yellow]          Enable a skill (persists to session)
  [yellow]/disable <skill>[/yellow]         Disable a skill (persists to session)
  [yellow]/remember <text>[/yellow]         Save a note to long-term memory
  [yellow]/info[/yellow]                    Show session info (user, session, provider)

  [bold]Session commands[/bold]
  [yellow]/session new[/yellow]             Create a new session and switch to it
  [yellow]/session switch <id>[/yellow]     Switch to an existing session
  [yellow]/session delete <id>[/yellow]     Delete a session (creates new if current)
  [yellow]/session list[/yellow]            List all sessions for this user
  [yellow]/session info[/yellow]            Show detailed session metadata
"""


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
        skills = self.agent.registry.list_skills()
        skill_names = ", ".join(s.name for s in skills) if skills else "none"
        vendor = type(self.agent.provider).__name__.replace("Provider", "").lower()
        self.console.print(Panel(
            f"[bold green]Birdie[/bold green]  |  vendor: [cyan]{vendor}[/cyan]"
            f"  |  user: [cyan]{self.user_id}[/cyan]"
            f"  |  session: [cyan]{self.session.id}[/cyan]"
            f"  |  skills: [yellow]{skill_names}[/yellow]\n"
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

    # -- slash command handler ------------------------------------------------

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

        elif cmd == "/skills":
            self._show_skills()

        elif cmd == "/tools":
            self._show_tools()

        elif cmd == "/enable":
            if not arg:
                self.console.print("[red]Usage: /enable <SkillName>[/red]")
            else:
                self.agent.enable_skill_for_user(self.session.id, arg)
                if arg not in self.session.enabled_skills:
                    self.session.enabled_skills.append(arg)
                self.session.disabled_skills = [
                    s for s in self.session.disabled_skills if s != arg
                ]
                self.session_manager.save(self.session)
                self.console.print(f"[green]Enabled[/green] {arg}")

        elif cmd == "/disable":
            if not arg:
                self.console.print("[red]Usage: /disable <SkillName>[/red]")
            else:
                self.agent.disable_skill_for_user(self.session.id, arg)
                if arg not in self.session.disabled_skills:
                    self.session.disabled_skills.append(arg)
                self.session.enabled_skills = [
                    s for s in self.session.enabled_skills if s != arg
                ]
                self.session_manager.save(self.session)
                self.console.print(f"[red]Disabled[/red] {arg}")

        elif cmd == "/remember":
            if not arg:
                self.console.print("[red]Usage: /remember <text>[/red]")
            else:
                self.user_memory.add(arg)
                self.session_manager.save_user_memory(self.user_memory)
                self.console.print(f"[dim]Remembered.[/dim]")

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
                                self.console.print(Panel(
                                    Text(str(msg.content), style="white"),
                                    title=f"[yellow]tool: {msg.name}[/yellow]",
                                    border_style="yellow",
                                    expand=False,
                                ))
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
    parser.add_argument("--skills-dir", default=None, help="Override skills directory")
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


async def _async_main(
    session_id_arg: Optional[str],
    user_id: str,
    skills_dir: str,
    provider_config,
) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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
        agent = DynamicAgent.from_config(
            provider_config, skills_dir=skills_dir, checkpointer=checkpointer
        )
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
