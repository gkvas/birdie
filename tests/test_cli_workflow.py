"""Tests for the /workflow slash commands and headless --workflow mode."""

import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

from birdie.cli import BirdieCLI


@pytest.fixture(autouse=True)
def _runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BIRDIE_WORKFLOW_RUNS_DIR", str(tmp_path / "runs"))
    # Keep the user's real ~/.birdie/workflows out of discovery.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _make_cli(tmp_path, replies=None):
    from rich.console import Console

    agent = MagicMock()
    agent._tool_output_cap = 0
    agent.registry.get_tool.return_value = None
    agent.agent_registry.get_agent.return_value = None
    agent.mcp_manager.has_servers = False
    agent.policy.get_allowed_skills.return_value = set()
    agent.agent_registry.get_allowed_agents.return_value = set()

    session = MagicMock()
    session.id = "2026-11-01_1"
    session.turns = 0
    session.total_cost_usd = 0.0
    for attr in ("enabled_skills", "disabled_skills", "enabled_agents",
                 "disabled_agents", "approved_skills"):
        setattr(session, attr, [])

    memory = MagicMock()
    memory.as_strings.return_value = []
    console = Console(record=True, width=120)
    wf_dir = tmp_path / "wfs"
    wf_dir.mkdir()
    cli = BirdieCLI(agent, session_manager=MagicMock(), session=session,
                    user_id="alice", user_memory=memory, console=console,
                    workflows_dir=str(wf_dir))
    replies = list(replies or [])
    calls = []

    async def fake_turn(message, thread_id=None):
        calls.append((message, thread_id))
        session.total_cost_usd += 0.5
        return replies.pop(0) if replies else "ok"

    cli._stream_turn = fake_turn
    return cli, console, wf_dir, calls


def _write(wf_dir, name, body):
    (wf_dir / f"{name}.yaml").write_text(textwrap.dedent(body))


@pytest.mark.asyncio
async def test_list_and_show(tmp_path):
    cli, console, wf_dir, _ = _make_cli(tmp_path)
    _write(wf_dir, "greet", """
        name: greet
        description: Say hello
        inputs: {who: world}
        steps:
          - id: hi
            echo: "hello {{ inputs.who }}"
            if: "inputs.who != ''"
    """)
    (wf_dir / "broken.yaml").write_text("name: broken\nsteps: [{zzz: 1}]")
    await cli._handle_slash("/workflow list")
    await cli._handle_slash("/workflow show greet")
    out = console.export_text()
    assert "greet who=world  - Say hello" in out
    assert "fix-tests" in out  # bundled workflow
    assert "broken.yaml" in out and "exactly one of" in out
    assert "who (string, default 'world')" in out
    assert "hi echo: hello {{ inputs.who }}  if: inputs.who != ''" in out


@pytest.mark.asyncio
async def test_run_renders_steps_and_prompt_uses_session(tmp_path):
    cli, console, wf_dir, calls = _make_cli(tmp_path, replies=["fixed it"])
    _write(wf_dir, "demo", """
        name: demo
        inputs: {n: 1}
        steps:
          - id: count
            run: echo "n={{ inputs.n }}"
          - id: ask_llm
            prompt: "value is {{ steps.count.output }}"
          - id: fresh
            prompt: again
            context: fresh
          - echo: "llm said {{ steps.ask_llm.output }}"
    """)
    await cli._handle_slash("/workflow run demo n=5")
    out = console.export_text()
    assert calls[0] == ("value is n=5", "2026-11-01_1")
    assert calls[1][0] == "again" and calls[1][1].startswith("2026-11-01_1~wf-demo-")
    assert "▸ count run: echo \"n={{ inputs.n }}\"" in out
    assert "n=5" in out and "✓ count" in out
    assert "🐦 llm said fixed it" in out
    assert "Workflow completed (4 steps, 2 LLM calls, $1.0000)" in out


@pytest.mark.asyncio
async def test_failed_run_then_resume(tmp_path):
    cli, console, wf_dir, _ = _make_cli(tmp_path)
    gate = tmp_path / "gate"
    _write(wf_dir, "gated", f"""
        name: gated
        steps:
          - id: first
            run: echo once >> {tmp_path / 'log'}
          - id: check
            run: test -f {gate}
    """)
    await cli._handle_slash("/workflow run gated")
    out = console.export_text()
    assert "✗ check:" in out and "Workflow failed" in out
    run_id = out.split("Resume with /workflow resume ")[1].split()[0]

    gate.write_text("")
    await cli._handle_slash("/workflow runs")
    await cli._handle_slash(f"/workflow resume {run_id}")
    out = console.export_text()
    assert f"{run_id}  failed" in out
    assert "↷ first done in a previous run" in out
    assert "Workflow completed" in out
    assert (tmp_path / "log").read_text() == "once\n"
    await cli._handle_slash(f"/workflow resume {run_id}")
    assert "already completed" in console.export_text()


@pytest.mark.asyncio
async def test_errors_are_reported(tmp_path):
    cli, console, wf_dir, _ = _make_cli(tmp_path)
    _write(wf_dir, "needs", "name: needs\ninputs: {who: }\nsteps: [{echo: hi}]")
    await cli._handle_slash("/workflow run needs")
    await cli._handle_slash("/workflow run nope")
    await cli._handle_slash("/workflow run needs bogus")
    await cli._handle_slash("/workflow frobnicate")
    await cli._handle_slash("/workflow resume zzz")
    out = console.export_text()
    assert "missing required input 'who'" in out
    assert "unknown workflow 'nope'" in out
    assert "expected key=value" in out
    assert "Usage: /workflow" in out
    assert "no workflow run 'zzz'" in out


@pytest.mark.asyncio
async def test_ask_step_reads_input_interactively(tmp_path, monkeypatch):
    cli, console, wf_dir, _ = _make_cli(tmp_path)
    _write(wf_dir, "confirm", """
        name: confirm
        steps:
          - id: q
            ask: Ship it?
            default: "no"
          - echo: "yes={{ steps.q.yes }}"
    """)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    await cli._handle_slash("/workflow run confirm")
    assert "yes=true" in console.export_text()


@pytest.mark.asyncio
async def test_headless_exit_codes_and_stdout(tmp_path, capsys):
    from birdie.cli_workflow import run_workflow_headless

    from rich.console import Console

    cli, _, wf_dir, _ = _make_cli(tmp_path)
    # Headless mode renders progress on stderr (see _async_main).
    cli.console = console = Console(record=True, width=120, stderr=True)
    _write(wf_dir, "ok", """
        name: ok
        inputs: {x: a}
        steps:
          - id: q
            ask: Sure?
            default: "yes"
          - echo: "result {{ inputs.x }} {{ steps.q.answer }}"
    """)
    _write(wf_dir, "asks", "name: asks\nsteps: [{ask: 'Really?'}]")
    assert await run_workflow_headless(cli, "ok", ["x=b"]) == 0
    assert capsys.readouterr().out == "result b yes\n"
    assert await run_workflow_headless(cli, "asks", []) == 1
    assert "needs user input" in console.export_text()
    assert await run_workflow_headless(cli, "missing", []) == 2


def test_input_requires_workflow_flag(monkeypatch, capsys):
    from birdie import cli as cli_mod

    monkeypatch.setattr("sys.argv", ["birdie", "--input", "a=1"])
    with pytest.raises(SystemExit) as exc:
        cli_mod.main()
    assert exc.value.code == 2
    assert "--input requires --workflow" in capsys.readouterr().err

    monkeypatch.setattr("sys.argv", ["birdie", "-p", "hi", "--workflow", "x"])
    with pytest.raises(SystemExit):
        cli_mod.main()
    assert "mutually exclusive" in capsys.readouterr().err
