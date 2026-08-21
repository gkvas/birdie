"""Tests for the CLI's interactive ACP permission gate."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from birdie.cli import BirdieCLI


def _make_cli(approved_acp_tools=None):
    from rich.console import Console

    agent = MagicMock()
    agent._tool_output_cap = 0

    session = MagicMock()
    session.id = "2026-08-21_1"
    session.turns = 0
    session.enabled_skills = []
    session.disabled_skills = []
    session.enabled_agents = []
    session.disabled_agents = []
    session.approved_skills = []
    session.approved_acp_tools = approved_acp_tools or []

    session_manager = MagicMock()
    console = Console(record=True, width=100)
    cli = BirdieCLI(
        agent,
        session_manager=session_manager,
        session=session,
        user_id="alice",
        user_memory=MagicMock(),
        console=console,
    )
    return cli, session_manager


def _request(title="Bash: ls"):
    return {
        "title": title,
        "kind": "execute",
        "raw_input": {"command": "ls"},
        "options": [
            {"id": "allow_always", "name": "Always allow Bash", "kind": "allow_always"},
            {"id": "allow", "name": "Allow", "kind": "allow_once"},
            {"id": "reject", "name": "Reject", "kind": "reject_once"},
        ],
    }


@pytest.mark.asyncio
async def test_cached_always_approval_skips_prompt():
    cli, _ = _make_cli(approved_acp_tools=["Always allow Bash"])
    with patch("asyncio.to_thread", AsyncMock(side_effect=AssertionError("prompted"))):
        assert await cli._approve_acp_permission(_request()) == "allow_always"


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,expected", [
    ("y", "allow"), ("yes", "allow"),
    ("n", "deny"), ("", "deny"), ("junk", "deny"),
])
async def test_prompt_answers(answer, expected):
    cli, _ = _make_cli()
    with patch("asyncio.to_thread", AsyncMock(return_value=answer)):
        assert await cli._approve_acp_permission(_request()) == expected


@pytest.mark.asyncio
async def test_always_answer_persists_approval():
    cli, session_manager = _make_cli()
    with patch("asyncio.to_thread", AsyncMock(return_value="a")):
        assert await cli._approve_acp_permission(_request()) == "allow_always"
    assert cli.session.approved_acp_tools == ["Always allow Bash"]
    session_manager.save.assert_called_once_with(cli.session)
    # a second request for the same tool no longer prompts
    with patch("asyncio.to_thread", AsyncMock(side_effect=AssertionError("prompted"))):
        assert await cli._approve_acp_permission(_request()) == "allow_always"
