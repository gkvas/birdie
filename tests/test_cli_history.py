"""Tests for the CLI history replay (/history and session resume)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from birdie.cli import BirdieCLI


def _make_cli(messages):
    """Build a BirdieCLI with a stubbed agent whose checkpointer holds *messages*."""
    from rich.console import Console

    agent = MagicMock()
    agent._tool_output_cap = 0
    snapshot = MagicMock()
    snapshot.values = {"messages": messages}
    agent.app.aget_state = AsyncMock(return_value=snapshot)

    session = MagicMock()
    session.id = "2026-07-27_1"
    session.turns = 3
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
    return cli, console


@pytest.mark.asyncio
async def test_history_renders_recent_messages():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    msgs = [
        HumanMessage(content="list the files"),
        AIMessage(content="", tool_calls=[
            {"name": "run_bash", "args": {"command": "ls"}, "id": "tc1"},
        ]),
        ToolMessage(content="a.py\nb.py", tool_call_id="tc1", name="run_bash"),
        AIMessage(content="Two files: a.py and b.py"),
    ]
    cli, console = _make_cli(msgs)
    await cli._print_history("2026-07-27_1")
    out = console.export_text()
    assert "list the files" in out
    assert "run_bash" in out
    assert "2 lines" in out
    assert "Two files" in out


@pytest.mark.asyncio
async def test_history_limits_and_reports_truncation():
    from langchain_core.messages import HumanMessage

    msgs = [HumanMessage(content=f"msg {i}") for i in range(20)]
    cli, console = _make_cli(msgs)
    await cli._print_history("2026-07-27_1", limit=5)
    out = console.export_text()
    assert "msg 19" in out
    assert "msg 14" not in out
    assert "last 5 of 20" in out


@pytest.mark.asyncio
async def test_history_empty_session():
    cli, console = _make_cli([])
    await cli._print_history("2026-07-27_1")
    assert "No prior history" in console.export_text()


class TestToolbar:
    def test_toolbar_renders_with_real_session(self, tmp_path):
        """Regression: the status bar must render with an actual Session object.

        A change once referenced session.model, which does not exist on the
        Session dataclass - the AttributeError fired on the first prompt
        render and crashed the CLI at startup.
        """
        from rich.console import Console
        from birdie.core.session import SessionManager

        mgr = SessionManager(sessions_root=tmp_path)
        session = mgr.create("alice")

        agent = MagicMock()
        agent._tool_output_cap = 0
        agent.provider.vendor_name = "anthropic"
        agent.provider.model_name = "claude-sonnet-4-6"

        cli = BirdieCLI(
            agent,
            session_manager=mgr,
            session=session,
            user_id="alice",
            user_memory=MagicMock(),
            console=Console(record=True, width=120),
        )
        toolbar = cli._get_toolbar()
        assert "claude-sonnet-4-6" in toolbar.value
        assert session.id in toolbar.value
