"""Tests for workflow parsing, the runner, persistence and resumption."""

import asyncio
import json
import textwrap

import pytest
from langchain_core.messages import AIMessage

from birdie.core.workflow import (
    AgentWorkflowHost,
    PromptReply,
    RunStore,
    WorkflowError,
    WorkflowHost,
    WorkflowRunner,
    bundled_workflows_dir,
    discover_workflows,
    load_workflow,
    parse_kv_args,
    parse_workflow,
    resolve_inputs,
    resolve_workflow,
)


def wf(text: str):
    return parse_workflow(textwrap.dedent(text))


class FakeHost(WorkflowHost):
    """Records events; scripted prompt / agent / tool / ask replies."""

    def __init__(self, prompts=None, answers=None, interactive=False):
        self.events = []
        self.prompts = list(prompts or [])
        self.prompt_calls = []
        self.answers = list(answers or [])
        self.interactive = interactive
        self.tool_calls = []

    async def prompt(self, text, fresh, step_id, run_id):
        self.prompt_calls.append((text, fresh, step_id))
        reply = self.prompts.pop(0) if self.prompts else "ok"
        if isinstance(reply, Exception):
            raise reply
        return PromptReply(reply, cost_usd=0.25)

    async def call_tool(self, name, args):
        self.tool_calls.append((name, args))
        if name == "boom":
            raise RuntimeError("tool exploded")
        return f"{name}:{json.dumps(args, sort_keys=True)}"

    async def call_agent(self, name, args):
        return json.dumps({"summary": f"summary of {args['text']}"})

    async def ask(self, question, default):
        if self.answers:
            return self.answers.pop(0)
        return default

    def on_event(self, kind, step, **info):
        self.events.append((kind, step.id if step else None))


def run(workflow, host=None, inputs=None, store=None, state=None):
    runner = WorkflowRunner(workflow, host or FakeHost(), store)
    return asyncio.run(runner.run(inputs or {}, state=state))


# -- parsing -------------------------------------------------------------------

@pytest.mark.parametrize("doc,msg", [
    ("- run: x", "must be a YAML mapping"),
    ("name: x\nsteps: []", "non-empty list"),
    ("name: x\nbogus: 1\nsteps: [{echo: hi}]", "unknown top-level keys"),
    ("name: x\nsteps: [{run: a, prompt: b}]", "exactly one of"),
    ("name: x\nsteps: [{id: a}]", "exactly one of"),
    ("name: x\nsteps: [{run: a, with: {}}]", "unknown keys"),
    ("name: x\nsteps: [{id: a, echo: x}, {id: a, echo: y}]", "duplicate step id"),
    ("name: x\nsteps: [{id: 'bad-id', echo: x}]", "invalid id"),
    ("name: x\nsteps: [{echo: x, if: 'a ==' }]", "if"),
    ("name: x\nsteps: [{echo: '{{ 1 + }}'}]", "unexpected"),
    ("name: x\nsteps: [{echo: x, on_error: maybe}]", "on_error"),
    ("name: x\nsteps: [{echo: x, retry: {max: 0}}]", "retry.max"),
    ("name: x\nsteps: [{echo: x, retry: {tries: 2}}]", "retry"),
    ("name: x\nsteps: [{echo: x, as: y}]", "'as' requires 'foreach'"),
    ("name: x\nsteps: [{echo: x, foreach: [1], as: steps}]", "invalid loop variable"),
    ("name: x\nsteps: [{prompt: x, context: global}]", "context"),
    ("name: x\nsteps: [{run: x, timeout: -1}]", "timeout"),
    ("name: x\nsteps: [{tool: t, with: [1]}]", "'with' must be a mapping"),
    ("name: x\ninputs: {n: {type: float}}\nsteps: [{echo: x}]", "unknown type"),
    ("name: x\nbudget: {max_tokens: 1}\nsteps: [{echo: x}]", "budget"),
    ("name: x\nsteps: [{steps: []}]", "non-empty list"),
    ("name: x\nsteps: [{run: '   '}]", "non-empty string"),
    ("name: x\nsteps: [{echo: no}]", "quote it"),
])
def test_parse_errors(doc, msg):
    with pytest.raises(WorkflowError, match=msg):
        parse_workflow(doc)


def test_parse_assigns_ids_and_descendants():
    w = wf("""
        name: demo
        steps:
          - run: a
          - id: grp
            steps:
              - echo: x
              - id: inner
                steps: [{echo: y}]
    """)
    assert [s.id for s in w.all_steps()] == ["step_1", "grp", "step_2_1", "inner", "step_2_2_1"]
    assert w.find("grp").descendants == ["step_2_1", "inner", "step_2_2_1"]


def test_name_defaults_to_file_stem(tmp_path):
    f = tmp_path / "my-flow.yaml"
    f.write_text("steps: [{echo: hi}]")
    assert load_workflow(f).name == "my-flow"


def test_inputs_shorthand_and_coercion():
    w = wf("""
        name: x
        inputs:
          path: tests/
          n: 3
          flag: false
          ratio: {type: number, default: 0.5}
          tags: {type: list}
          who:
        steps: [{echo: hi}]
    """)
    specs = {i.name: i for i in w.inputs}
    assert specs["n"].type == "integer" and not specs["n"].required
    assert specs["who"].required and specs["who"].type == "string"
    assert specs["tags"].required
    got = resolve_inputs(w, {"who": "me", "n": "7", "flag": "yes", "tags": "a, b", "ratio": "2"})
    assert got == {"path": "tests/", "n": 7, "flag": True, "ratio": 2.0,
                   "tags": ["a", "b"], "who": "me"}
    assert resolve_inputs(w, {"who": "x", "tags": "[1, 2]"})["tags"] == [1, 2]
    with pytest.raises(WorkflowError, match="missing required input 'who'"):
        resolve_inputs(w, {"tags": "a"})
    with pytest.raises(WorkflowError, match="unknown input"):
        resolve_inputs(w, {"who": "x", "tags": "a", "zzz": 1})
    with pytest.raises(WorkflowError, match="cannot convert"):
        resolve_inputs(w, {"who": "x", "tags": "a", "n": "seven"})


def test_parse_kv_args():
    assert parse_kv_args(["a=1", "b=x=y", "c="]) == {"a": "1", "b": "x=y", "c": ""}
    with pytest.raises(WorkflowError):
        parse_kv_args(["novalue"])


def test_bundled_workflows_are_valid():
    found, errors = discover_workflows([bundled_workflows_dir()])
    assert not errors
    assert "fix-tests" in found


def test_discovery_priority_and_errors(tmp_path):
    hi, lo = tmp_path / "hi", tmp_path / "lo"
    hi.mkdir(), lo.mkdir()
    (hi / "a.yaml").write_text("name: a\ndescription: high\nsteps: [{echo: x}]")
    (lo / "a.yml").write_text("name: a\ndescription: low\nsteps: [{echo: x}]")
    (lo / "broken.yaml").write_text("name: broken\nsteps: [{nope: 1}]")
    found, errors = discover_workflows([hi, lo])
    assert found["a"].description == "high"
    assert list(errors) == [str(lo / "broken.yaml")]
    assert resolve_workflow("A", [hi, lo]).name == "a"
    with pytest.raises(WorkflowError, match="exactly one of"):
        resolve_workflow("broken", [hi, lo])
    with pytest.raises(WorkflowError, match="unknown workflow 'zzz'"):
        resolve_workflow("zzz", [hi, lo])
    assert resolve_workflow(str(lo / "a.yml"), []).description == "low"


# -- runner ----------------------------------------------------------------------

def test_run_steps_and_outputs():
    w = wf("""
        name: x
        inputs: {who: world}
        steps:
          - id: hello
            run: echo "hello {{ inputs.who | quote }}"; echo oops >&2
          - id: shout
            echo: "{{ steps.hello.output | upper }}"
    """)
    state = run(w)
    assert state["status"] == "completed"
    hello = state["results"]["hello"]
    assert hello["ok"] and hello["exit_code"] == 0
    assert hello["stdout"] == "hello world\n" and hello["stderr"] == "oops\n"
    assert state["output"] == "HELLO WORLD"


def test_failing_command_stops_run():
    w = wf("""
        name: x
        steps:
          - id: bad
            run: echo partial; exit 3
          - id: never
            echo: never
    """)
    state = run(w)
    assert state["status"] == "failed"
    assert "step 'bad' failed: exit code 3" in state["error"]
    assert "never" not in state["results"]
    assert "bad" not in state["done"]  # failed step is re-run on resume


def test_on_error_continue_and_conditions():
    w = wf("""
        name: x
        steps:
          - id: bad
            run: exit 2
            on_error: continue
          - id: handle
            if: "not steps.bad.ok"
            echo: "handled {{ steps.bad.exit_code }}"
          - id: skipped
            if: steps.bad.ok
            fail: should not run
    """)
    host = FakeHost()
    state = run(w, host)
    assert state["status"] == "completed"
    assert state["results"]["handle"]["output"] == "handled 2"
    assert state["results"]["skipped"] == {"ok": True, "skipped": True, "output": None}
    assert ("step_skip", "skipped") in host.events
    assert state["output"] == "handled 2"  # skipped steps don't override output


def test_fail_step():
    w = wf("""
        name: x
        steps:
          - fail: "stop here: {{ 1 + 1 }}"
    """)
    state = run(w)
    assert state["status"] == "failed" and "stop here: 2" in state["error"]


def test_bad_expression_at_runtime_fails_step():
    w = wf("""
        name: x
        steps:
          - echo: "{{ steps.nothing.output }}"
    """)
    state = run(w)
    assert state["status"] == "failed" and "has no key 'nothing'" in state["error"]


def test_foreach_with_loop_vars():
    w = wf("""
        name: x
        inputs: {files: {type: list, default: [a, b, c]}}
        steps:
          - id: each
            foreach: inputs.files
            as: f
            echo: "{{ loop.index }}/{{ loop.length }}={{ f }}{{ '!' if false else '' }}"
    """.replace("{{ '!' if false else '' }}", ""))
    state = run(w)
    assert state["results"]["each"]["output"] == ["0/3=a", "1/3=b", "2/3=c"]
    assert state["usage"]["steps"] == 3


def test_foreach_over_group_and_dict():
    w = wf("""
        name: x
        steps:
          - id: each
            foreach: "{{ {'a': 1} if false else [1, 2] }}"
            steps:
              - id: double
                run: echo $(( {{ item }} * 2 ))
    """.replace("{{ {'a': 1} if false else [1, 2] }}", "[1, 2]"))
    state = run(w)
    assert state["results"]["each"]["output"] == ["2", "4"]


def test_foreach_requires_list():
    w = wf("""
        name: x
        steps:
          - foreach: "'abc'"
            echo: x
    """)
    state = run(w)
    assert state["status"] == "failed" and "must evaluate to a list" in state["error"]


def test_foreach_stops_on_failed_iteration():
    w = wf("""
        name: x
        steps:
          - id: each
            foreach: [1, 2, 3]
            run: test {{ item }} -ne 2
    """)
    state = run(w)
    assert state["status"] == "failed" and "iteration 1" in state["error"]
    assert state["usage"]["steps"] == 2


def test_retry_single_step_until_ok(tmp_path):
    counter = tmp_path / "n"
    w = wf(f"""
        name: x
        steps:
          - id: flaky
            retry: 3
            run: n=$(cat {counter} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {counter}; test $n -ge 3
    """)
    host = FakeHost()
    state = run(w, host)
    assert state["status"] == "completed"
    assert state["results"]["flaky"]["attempts"] == 3
    assert [e for e in host.events if e[0] == "retry"] == [("retry", "flaky")] * 2


def test_retry_exhausted_fails():
    w = wf("""
        name: x
        steps:
          - id: never
            retry: {max: 2}
            run: exit 1
    """)
    state = run(w)
    assert state["status"] == "failed"
    assert "gave up after 2 attempts" in state["error"]


def test_retry_group_fix_loop_with_attempt_vars(tmp_path):
    """The canonical pattern: test -> prompt to fix -> re-test, bounded."""
    marker = tmp_path / "fixed"
    w = wf(f"""
        name: x
        inputs: {{rounds: 2}}
        steps:
          - id: tests
            retry:
              max: "{{{{ inputs.rounds + 1 }}}}"
              until: steps.test.ok
            steps:
              - id: test
                run: test -f {marker}
                on_error: continue
              - id: fix
                if: "not steps.test.ok and not attempt.last"
                prompt: "fix it (attempt {{{{ attempt.number }}}}/{{{{ attempt.max }}}})"
    """)

    class FixingHost(FakeHost):
        async def prompt(self, text, fresh, step_id, run_id):
            self.prompt_calls.append(text)
            if len(self.prompt_calls) == 2:
                marker.write_text("x")
            return PromptReply("done", 0.0)

    host = FixingHost()
    state = run(w, host)
    assert state["status"] == "completed"
    assert host.prompt_calls == ["fix it (attempt 1/3)", "fix it (attempt 2/3)"]
    assert state["results"]["tests"]["attempts"] == 3
    assert state["results"]["fix"]["skipped"]  # last attempt passed; no fix needed


def test_retry_group_never_fixed_gives_up():
    w = wf("""
        name: x
        steps:
          - id: tests
            retry: {max: 2, until: steps.test.ok}
            steps:
              - id: test
                run: exit 1
                on_error: continue
              - if: "not attempt.last"
                prompt: fix
    """)
    host = FakeHost()
    state = run(w, host)
    assert state["status"] == "failed"
    assert len(host.prompt_calls) == 1


def test_group_child_failure_fails_group():
    w = wf("""
        name: x
        steps:
          - id: grp
            on_error: continue
            steps:
              - run: exit 4
              - id: after
                echo: never
          - echo: "grp ok={{ steps.grp.ok }}"
    """)
    state = run(w)
    assert state["status"] == "completed"
    assert "exit code 4" in state["results"]["grp"]["error"]
    assert "after" not in state["results"]
    assert state["output"] == "grp ok=false"


def test_prompt_tool_agent_ask_steps():
    w = wf("""
        name: x
        context: fresh
        steps:
          - id: p1
            prompt: "summarise {{ inputs.topic }}"
          - id: p2
            prompt: follow up
            context: session
          - id: t
            tool: read_file
            with: {path: "{{ inputs.topic }}", n: 3}
          - id: a
            agent: Summarizer
            with: {text: "{{ steps.p1.output }}"}
          - id: q
            ask: Proceed?
            default: "no"
          - echo: "{{ steps.a.data.summary }} / {{ steps.q.yes }}"
        inputs: {topic: README}
    """)
    host = FakeHost(prompts=["S1", "S2"], answers=["Yes"])
    state = run(w, host)
    assert state["status"] == "completed", state["error"]
    assert host.prompt_calls == [("summarise README", True, "p1"), ("follow up", False, "p2")]
    assert host.tool_calls == [("read_file", {"path": "README", "n": 3})]
    assert state["results"]["t"]["output"] == 'read_file:{"n": 3, "path": "README"}'
    assert state["results"]["a"]["data"] == {"summary": "summary of S1"}
    assert state["output"] == "summary of S1 / true"
    assert state["usage"] == {"steps": 6, "llm_calls": 3, "cost_usd": 0.5}


def test_tool_error_fails_step():
    w = wf("""
        name: x
        steps:
          - tool: boom
    """)
    state = run(w)
    assert state["status"] == "failed" and "RuntimeError: tool exploded" in state["error"]


def test_ask_without_answer_aborts_with_resume_hint():
    w = wf("""
        name: x
        steps:
          - id: q
            ask: Continue?
    """)
    state = run(w, FakeHost())
    assert state["status"] == "failed"
    assert "needs user input" in state["error"] and "/workflow resume" in state["error"]


def test_budget_max_steps_stops_loops():
    w = wf("""
        name: x
        budget: {max_steps: 3}
        steps:
          - retry: 10
            run: exit 1
    """)
    state = run(w)
    assert state["status"] == "failed"
    assert "max_steps 3" in state["error"]
    assert state["usage"]["steps"] == 3


def test_budget_llm_calls_and_cost():
    w = wf("""
        name: x
        budget: {max_llm_calls: 2}
        steps:
          - prompt: a
          - prompt: b
          - prompt: c
    """)
    state = run(w)
    assert state["status"] == "failed" and "max_llm_calls 2" in state["error"]
    w = wf("""
        name: x
        budget: {max_cost_usd: 0.5}
        steps:
          - prompt: a
          - prompt: b
          - prompt: c
    """)
    state = run(w)  # each fake prompt costs 0.25
    assert state["status"] == "failed" and "max_cost_usd" in state["error"]
    assert state["usage"]["llm_calls"] == 2


def test_run_timeout_kills_process_group(tmp_path):
    marker = tmp_path / "late"
    w = wf(f"""
        name: x
        steps:
          - id: slow
            run: (sleep 2; touch {marker}) & sleep 5
            timeout: 0.3
    """)
    state = run(w)
    assert state["status"] == "failed" and "timed out" in state["error"]
    assert state["results"]["slow"]["exit_code"] is None
    import time
    time.sleep(2.3)
    assert not marker.exists()  # the backgrounded child was killed too


def test_run_cwd_and_env(tmp_path):
    w = wf(f"""
        name: x
        steps:
          - run: pwd; echo "$GREETING"
            cwd: "{tmp_path}"
            env: {{GREETING: "hi {{{{ inputs.n }}}}"}}
        inputs: {{n: 1}}
    """)
    state = run(w)
    assert state["output"].splitlines() == [str(tmp_path.resolve()), "hi 1"]


def test_preflight_errors_stop_before_running():
    class CheckingHost(FakeHost):
        async def check(self, step):
            return "unknown agent" if step.action == "agent" else None

    w = wf("""
        name: x
        steps:
          - id: side_effect
            run: echo hi
          - id: a
            agent: Nope
    """)
    state = run(w, CheckingHost())
    assert state["status"] == "failed" and "step 'a': unknown agent" in state["error"]
    assert state["results"] == {}


# -- persistence & resume ---------------------------------------------------------

def test_state_persisted_and_resume_skips_done_steps(tmp_path):
    store = RunStore(tmp_path / "runs")
    count, gate = tmp_path / "count", tmp_path / "gate"
    w = wf(f"""
        name: resumable
        steps:
          - id: once
            run: echo x >> {count}
          - id: gated
            run: test -f {gate}
          - echo: done
    """)
    state = run(w, store=store)
    assert state["status"] == "failed"
    saved = store.load(state["run_id"][:-2])  # unique prefix works
    assert saved["status"] == "failed" and saved["done"] == ["once"]

    gate.write_text("open")
    host = FakeHost()
    resumed = run(w, host, store=store, state=saved)
    assert resumed["status"] == "completed"
    assert resumed["run_id"] == state["run_id"]
    assert count.read_text() == "x\n"  # 'once' did not run again
    assert ("step_resumed", "once") in host.events
    assert store.list()[0]["status"] == "completed"


def test_resume_inside_plain_group_skips_done_children(tmp_path):
    store = RunStore(tmp_path / "runs")
    log, gate = tmp_path / "log", tmp_path / "gate"
    w = wf(f"""
        name: grouped
        steps:
          - id: grp
            steps:
              - id: once
                run: echo once >> {log}
              - id: check
                run: test -f {gate}
    """)
    first = run(w, store=store)
    assert first["status"] == "failed"
    gate.write_text("")
    second = run(w, store=store, state=store.load(first["run_id"]))
    assert second["status"] == "completed"
    assert log.read_text() == "once\n"


def test_resume_restarts_unfinished_loop_from_scratch(tmp_path):
    store = RunStore(tmp_path / "runs")
    log, gate = tmp_path / "log", tmp_path / "gate"
    w = wf(f"""
        name: loopy
        steps:
          - id: each
            foreach: [1, 2]
            steps:
              - id: mark
                run: echo {{{{ item }}}} >> {log}
              - id: check
                run: test {{{{ item }}}} -eq 1 -o -f {gate}
    """)
    first = run(w, store=store)
    assert first["status"] == "failed"
    assert log.read_text() == "1\n2\n"
    gate.write_text("")
    second = run(w, store=store, state=store.load(first["run_id"]))
    assert second["status"] == "completed"
    assert log.read_text() == "1\n2\n1\n2\n"


def test_resume_warns_when_workflow_changed(tmp_path):
    store = RunStore(tmp_path / "runs")
    w1 = wf("name: x\nsteps: [{fail: nope}]")
    state = run(w1, store=store)
    w2 = wf("name: x\nsteps: [{echo: fixed}]")
    host = FakeHost()
    out = run(w2, host, store=store, state=store.load(state["run_id"]))
    assert out["status"] == "completed"
    assert ("warning", None) in host.events


def test_cancel_marks_run_interrupted(tmp_path):
    store = RunStore(tmp_path / "runs")
    w = wf("""
        name: x
        steps:
          - id: first
            echo: hi
          - id: slow
            run: sleep 10
    """)

    async def go():
        runner = WorkflowRunner(w, FakeHost(), store)
        task = asyncio.create_task(runner.run({}))
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return runner.state["run_id"]

    run_id = asyncio.run(go())
    saved = store.load(run_id)
    assert saved["status"] == "interrupted"
    assert saved["done"] == ["first"]


def test_store_errors(tmp_path):
    store = RunStore(tmp_path)
    with pytest.raises(WorkflowError, match="no workflow run"):
        store.load("nothing")
    for rid in ("abc-1", "abc-2"):
        store.save({"run_id": rid})
    with pytest.raises(WorkflowError, match="ambiguous"):
        store.load("abc")
    assert {r["run_id"] for r in store.list()} == {"abc-1", "abc-2"}


# -- AgentWorkflowHost with a real DynamicAgent -----------------------------------

SKILL_MD = """---
name: EchoSkill
version: 1.0.0
description: echo
---

## Tools

### echo
description: Echo a message
entrypoint: python:tests.test_integration.echo_tool
schema:
  type: object
  properties:
    message:
      type: string
  required: [message]

## Permissions

- network
"""


class _ReplyProvider:
    def __init__(self):
        self.seen = []

    def supports_tools(self):
        return True

    async def achat(self, messages, tools=None, system_prompt=None, **kw):
        self.seen.append([m.content for m in messages])
        return AIMessage(content=f"reply {len(self.seen)}",
                         response_metadata={"cost": {"amount": 0.01}})


def _agent(tmp_path):
    from birdie.agent.run import DynamicAgent
    from birdie.core.llm_provider import LLMProvider

    LLMProvider.register(_ReplyProvider)
    d = tmp_path / "skills" / "echo"
    d.mkdir(parents=True)
    (d / "SKILL.MD").write_text(SKILL_MD)
    provider = _ReplyProvider()
    agent = DynamicAgent(provider, skills_dir=str(tmp_path / "skills"),
                         agents_dir=str(tmp_path / "skills"))
    return agent, provider


def test_agent_host_prompt_session_vs_fresh(tmp_path):
    agent, provider = _agent(tmp_path)
    w = wf("""
        name: x
        steps:
          - id: a
            prompt: first
          - id: b
            prompt: second
          - id: c
            prompt: isolated
            context: fresh
    """)
    host = AgentWorkflowHost(agent, thread_id="sess-1")
    state = run(w, host)
    assert state["status"] == "completed", state["error"]
    assert [state["results"][k]["output"] for k in "abc"] == ["reply 1", "reply 2", "reply 3"]
    # (the first human message of a thread carries a project-instructions prefix)
    seen = [" | ".join(map(str, msgs)) for msgs in provider.seen]
    assert "first" in seen[1] and "second" in seen[1]  # session context carried over
    assert "isolated" in seen[2] and "first" not in seen[2]  # fresh thread
    assert state["usage"]["cost_usd"] == pytest.approx(0.03)


def test_agent_host_tool_step_respects_permissions(tmp_path):
    agent, _ = _agent(tmp_path)
    decisions = []

    def approve(skill, perms, tool, args):
        decisions.append((skill, tool))
        return "deny" if len(decisions) == 1 else "always"

    agent.tool_approval_callback = approve
    w = wf("""
        name: x
        steps:
          - id: t1
            tool: echo
            with: {message: hi}
            on_error: continue
          - id: t2
            tool: echo
            with: {message: hi}
          - id: t3
            tool: echo
            with: {message: again}
    """)
    state = run(w, AgentWorkflowHost(agent, thread_id="s"))
    assert "denied" in state["results"]["t1"]["error"]
    assert state["results"]["t2"]["output"] == "Echo: hi"
    assert state["results"]["t3"]["output"] == "Echo: again"
    assert decisions == [("EchoSkill", "echo"), ("EchoSkill", "echo")]  # 'always' sticks


def test_agent_host_preflight_unknown_tool_and_agent(tmp_path):
    agent, _ = _agent(tmp_path)
    w = wf("""
        name: x
        steps:
          - tool: nope
          - agent: Nobody
    """)
    state = run(w, AgentWorkflowHost(agent, thread_id="s"))
    assert state["status"] == "failed"
    assert "unknown tool 'nope'" in state["error"] and "unknown agent 'Nobody'" in state["error"]
