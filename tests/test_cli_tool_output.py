"""Tests for the CLI's tool-output renderer."""

from unittest.mock import MagicMock

from rich.console import Console

from birdie.cli import BirdieCLI


def _make_cli(mode: str = "full"):
    agent = MagicMock()
    agent._tool_output_cap = 0
    session = MagicMock()
    session.id = "2026-08-22_1"
    session.turns = 0
    session.enabled_skills = []
    session.disabled_skills = []
    session.enabled_agents = []
    session.disabled_agents = []
    session.approved_skills = []
    console = Console(record=True, width=100)
    cli = BirdieCLI(
        agent,
        session_manager=MagicMock(),
        session=session,
        user_id="alice",
        user_memory=MagicMock(),
        console=console,
    )
    cli._tool_output_mode = mode
    return cli, console


def test_render_tool_output_does_not_eat_bracketed_labels():
    """Tool output is data: Rich must not interpret "[stderr]" (or
    any other bracketed text) as markup and drop it from the terminal."""
    cli, console = _make_cli()
    cli._render_tool_output(
        "run_bash", "emulator ready\n[stderr]\nnever booted"
    )
    text = console.export_text()
    assert "[stderr]" in text
    assert "never booted" in text


def test_render_tool_output_short_mode_truncates():
    cli, console = _make_cli("short")
    cli._render_tool_output("run_bash", "x" * 1500)
    text = console.export_text()
    assert "500 more characters" in text
