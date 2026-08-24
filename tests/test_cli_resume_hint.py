"""Tests for the resume hint printed when the session exits."""

import argparse

from unittest.mock import MagicMock

from rich.console import Console

from birdie.cli import BirdieCLI, _resume_flags


def _make_cli(turns=3, resume_flags=None):
    agent = MagicMock()
    agent._tool_output_cap = 0

    session = MagicMock()
    session.id = "2026-08-22_2"
    session.turns = turns
    session.enabled_skills = []
    session.disabled_skills = []
    session.enabled_agents = []
    session.disabled_agents = []
    session.approved_skills = []

    console = Console(record=True, width=120)
    cli = BirdieCLI(
        agent,
        session_manager=MagicMock(),
        session=session,
        user_id="alice",
        user_memory=MagicMock(),
        console=console,
        resume_flags=resume_flags,
    )
    return cli, console


def test_goodbye_shows_resume_command():
    cli, console = _make_cli()
    cli._print_goodbye()
    out = console.export_text()
    assert "Goodbye." in out
    assert "birdie --session-id 2026-08-22_2" in out


def test_goodbye_repeats_startup_flags():
    cli, console = _make_cli(
        resume_flags=["--user", "alice", "--config", "/tmp/prov.json"]
    )
    cli._print_goodbye()
    out = console.export_text()
    assert (
        "birdie --session-id 2026-08-22_2 --user alice "
        "--config /tmp/prov.json" in out
    )


def test_resume_hint_shown_for_untouched_session():
    """A session quit right after startup is still resumable."""
    cli, console = _make_cli(turns=0)
    cli._print_goodbye()
    out = console.export_text()
    assert "Goodbye." in out
    assert "birdie --session-id 2026-08-22_2" in out


def test_resume_flags_only_echo_explicit_options():
    args = argparse.Namespace(
        user=None, skills_dir=None, agents_dir=None, config=None, debug=False
    )
    assert _resume_flags(args) == []

    args = argparse.Namespace(
        user="bob with space",
        skills_dir=None,
        agents_dir="/opt/agents",
        config=None,
        debug=True,
    )
    assert _resume_flags(args) == [
        "--user", "'bob with space'",
        "--agents-dir", "/opt/agents",
        "--debug",
    ]
