"""
Interactive REPL for Birdie.

Presents a prompt-toolkit prompt with:

- Streaming output rendered via Rich (tool calls in yellow panels, AI text
  in green Markdown panels).
- A bottom status bar showing vendor, model, current context tokens, and
  cumulative spend.
- Slash commands for skill management, user switching, and session control.
- Ctrl+C to quit; Ctrl+J to insert a newline for multi-line input.

Entry point: ``birdie`` console script → :func:`main`.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .agent.run import DynamicAgent
from .core.models import Skill


PROMPT_STYLE = Style.from_dict({"prompt": "ansicyan bold"})

HELP_TEXT = """
[bold cyan]Birdie CLI — available slash commands[/bold cyan]

  [yellow]/help[/yellow]              Show this help
  [yellow]/quit[/yellow]  [yellow]/exit[/yellow]      Exit the session
  [yellow]/new[/yellow]               Start a fresh conversation (new thread)
  [yellow]/skills[/yellow]            List all loaded skills
  [yellow]/tools[/yellow]             List all available tools
  [yellow]/enable <skill>[/yellow]    Enable a skill for the current user
  [yellow]/disable <skill>[/yellow]   Disable a skill for the current user
  [yellow]/user <id>[/yellow]         Switch user identity
  [yellow]/info[/yellow]              Show session info (user, thread, provider)
"""


class BirdieCLI:
    def __init__(self, agent: DynamicAgent, user_id: Optional[str] = None) -> None:
        self.agent = agent
        self.user_id = user_id
        self.thread_id = str(uuid.uuid4())
        self.console = Console()

        self._total_in: int = 0
        self._total_out: int = 0
        self._last_context: int = 0  # input tokens from the most recent request

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
        self.session: PromptSession = PromptSession(
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            style=PROMPT_STYLE,
            key_bindings=kb,
            multiline=False,  # Enter submits; Ctrl+J adds a newline
            bottom_toolbar=self._get_toolbar,
        )

    # -- status toolbar -----------------------------------------------------

    def _get_toolbar(self) -> HTML:
        """Render the bottom status bar for prompt_toolkit."""
        vendor = self.agent.provider.vendor_name
        model  = self.agent.provider.model_name
        ctx    = f"{self._last_context:,}" if self._last_context else "—"
        spent  = f"↑{self._total_in:,}  ↓{self._total_out:,}"
        return HTML(
            f" <b>{vendor}</b> · {model}"
            f"   │   ctx: {ctx} tok"
            f"   │   spent: {spent} tok"
        )

    # -- display helpers ----------------------------------------------------

    def _print_welcome(self) -> None:
        """Print the startup banner with loaded skills and provider info."""
        skills = self.agent.registry.list_skills()
        skill_names = ", ".join(s.name for s in skills) if skills else "none"
        vendor = type(self.agent.provider).__name__.replace("Provider", "").lower()
        self.console.print(Panel(
            f"[bold green]Birdie[/bold green]  |  vendor: [cyan]{vendor}[/cyan]  |  skills: [yellow]{skill_names}[/yellow]\n"
            "Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit.",
            border_style="green",
        ))

    def _show_help(self) -> None:
        """Print the slash-command reference."""
        self.console.print(HELP_TEXT)

    def _show_skills(self) -> None:
        """List all loaded skills with their enabled/disabled status for the current user."""
        skills: list[Skill] = self.agent.registry.list_skills()
        if not skills:
            self.console.print("[dim]No skills loaded.[/dim]")
            return
        for skill in skills:
            enabled = skill.name in self.agent.policy.get_allowed_skills_for_user(self.user_id)
            status = "[green]enabled[/green]" if enabled else "[red]disabled[/red]"
            self.console.print(f"  [bold]{skill.name}[/bold] v{skill.version}  {status}  — {skill.description}")

    def _show_tools(self) -> None:
        """List all callable tools available to the current user."""
        allowed = self.agent.policy.get_allowed_skills_for_user(self.user_id)
        tools = [
            t for t in self.agent.registry.list_tools()
            if self.agent.registry.is_tool_allowed(t.name, allowed)
        ]
        if not tools:
            self.console.print("[dim]No tools available.[/dim]")
            return
        for tool in tools:
            self.console.print(f"  [bold cyan]{tool.name}[/bold cyan]  — {tool.description}")

    def _show_info(self) -> None:
        """Print current user ID, thread ID, and provider class name."""
        vendor = type(self.agent.provider).__name__
        self.console.print(
            f"  [dim]user:[/dim]    {self.user_id or '[anonymous]'}\n"
            f"  [dim]thread:[/dim]  {self.thread_id}\n"
            f"  [dim]provider:[/dim] {vendor}"
        )

    # -- slash command handler ----------------------------------------------

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
            self.thread_id = str(uuid.uuid4())
            self.console.print(f"[dim]Started new conversation thread {self.thread_id}[/dim]")

        elif cmd == "/skills":
            self._show_skills()

        elif cmd == "/tools":
            self._show_tools()

        elif cmd == "/enable":
            if not arg:
                self.console.print("[red]Usage: /enable <SkillName>[/red]")
            else:
                uid = self.user_id or "_cli_"
                self.agent.enable_skill_for_user(uid, arg)
                if self.user_id is None:
                    self.user_id = uid
                self.console.print(f"[green]Enabled[/green] {arg} for user {self.user_id}")

        elif cmd == "/disable":
            if not arg:
                self.console.print("[red]Usage: /disable <SkillName>[/red]")
            else:
                uid = self.user_id or "_cli_"
                self.agent.disable_skill_for_user(uid, arg)
                if self.user_id is None:
                    self.user_id = uid
                self.console.print(f"[red]Disabled[/red] {arg} for user {self.user_id}")

        elif cmd == "/user":
            if not arg:
                self.console.print(f"Current user: {self.user_id or '[anonymous]'}")
            else:
                self.user_id = arg
                self.console.print(f"[dim]Switched to user [bold]{arg}[/bold][/dim]")

        elif cmd == "/info":
            self._show_info()

        elif cmd == "/clear":
            self.console.clear()

        else:
            self.console.print(f"[red]Unknown command:[/red] {cmd}  (type /help for list)")

        return True

    # -- streaming turn -----------------------------------------------------

    async def _stream_turn(self, message: str) -> None:
        """Send *message* to the agent and render the streamed response.

        A spinner runs continuously between LLM calls and tool executions.
        Its label reflects the current phase ("thinking…" or "running tools…").
        Intermediate output (tool intentions, tool results) is printed above
        the spinner while it is still active; the spinner stops just before
        the final text response is rendered.

        Accumulates ``usage_metadata`` from each AIMessage into the running
        token counters used by the status toolbar.

        Args:
            message: The user's input text for this turn.
        """
        from langchain_core.messages import AIMessage, ToolMessage

        printed_any = False
        status = self.console.status("[dim]thinking…[/dim]", spinner="dots")
        status.start()

        try:
            async for update in self.agent.astream(message, self.thread_id, self.user_id):
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
                                    args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                                    self.console.print(
                                        f"[dim yellow]→ calling [bold]{tc['name']}[/bold]({args_str})[/dim yellow]"
                                    )
                                if getattr(msg, "tool_calls", None):
                                    status.update("[dim]running tools…[/dim]")
                                elif msg.content:
                                    status.stop()
                                    self.console.print(
                                        Panel(
                                            Markdown(msg.content),
                                            title="[green]birdie[/green]",
                                            border_style="green",
                                        )
                                    )
                                    printed_any = True
        finally:
            status.stop()

        if not printed_any:
            self.console.print("[dim](no response)[/dim]")

    # -- main loop ----------------------------------------------------------

    async def run(self) -> None:
        """Start the interactive REPL and block until the user quits."""
        self._print_welcome()

        while True:
            try:
                user_input = await self.session.prompt_async(
                    [("class:prompt", "you> ")],
                )
            except SystemExit:
                self.console.print("[dim]Goodbye.[/dim]")
                return
            except EOFError:
                self.console.print("[dim]Goodbye.[/dim]")
                return
            except KeyboardInterrupt:
                continue  # Ctrl+C mid-typing (before our binding fires) — ignore

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_slash(user_input)
                continue

            try:
                await self._stream_turn(user_input)
            except Exception as exc:
                self.console.print(f"[red bold]Error:[/red bold] {exc}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Birdie interactive CLI")
    parser.add_argument("--user", metavar="USER_ID", default=None, help="Set user identity")
    parser.add_argument("--skills-dir", default=None, help="Override skills directory")
    args = parser.parse_args()

    skills_dir = args.skills_dir or os.path.join(os.path.dirname(__file__), "skills")

    agent = DynamicAgent.from_config(skills_dir=skills_dir)
    cli = BirdieCLI(agent, user_id=args.user)

    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
