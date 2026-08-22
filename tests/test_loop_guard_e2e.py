"""End-to-end: a model that loops on one tool call gets the turn ended."""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from birdie.agent.graph import create_agent_graph
from birdie.core.loader import load_skill_from_markdown
from birdie.core.policy import SkillPolicy
from birdie.core.registry import SkillRegistry


class LoopingProvider:
    """Fake LLMProvider that always asks for the same tool call."""

    def __init__(self):
        self.calls = 0

    def supports_tools(self):
        return True

    async def achat(self, **_):
        self.calls += 1
        return AIMessage(content="", tool_calls=[
            {"name": "todo_create_plan", "args": {"steps": ["a"]},
             "id": f"c{self.calls}"}])

    def __getattr__(self, name):
        # Anything else the graph asks of the provider is irrelevant here.
        return lambda *a, **k: None


def test_looping_model_is_stopped(tmp_path):
    registry = SkillRegistry()
    registry.register_skill(load_skill_from_markdown("birdie/skills/todo/SKILL.MD"))
    policy = SkillPolicy()
    policy.enable_skill("t", "ToDo")
    provider = LoopingProvider()
    graph = create_agent_graph(provider, registry, policy).compile()
    cfg = {"configurable": {"thread_id": "t", "user_id": "u",
                            "max_tool_repetitions": 3}}
    out = asyncio.run(graph.ainvoke(
        {"messages": [HumanMessage(content="plan")]}, cfg))
    last = out["messages"][-1]
    assert isinstance(last, AIMessage) and "Stopped" in last.content
    # The tool really ran (skill wiring is right), it was the model looping.
    from langchain_core.messages import ToolMessage
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].content.startswith("Plan (1 step)")
    # 3 allowed reps + warning + one more -> guard twice -> stop.
    assert provider.calls <= 6


class MixedBatchProvider:
    """First turn: two tool calls, one of which fails. Second turn: done."""

    def __init__(self):
        self.calls = 0

    def supports_tools(self):
        return True

    async def achat(self, **_):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[
                {"name": "todo_complete_step",
                 "args": {"step_number": 1, "summary": "ok"}, "id": "good"},
                {"name": "run_bash",
                 "args": {"command": "ls /definitely/missing"}, "id": "bad"},
            ])
        return AIMessage(content="done")

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_failing_call_does_not_clobber_sibling_results():
    from langchain_core.messages import ToolMessage
    registry = SkillRegistry()
    for skill in ("todo", "shell"):
        registry.register_skill(
            load_skill_from_markdown(f"birdie/skills/{skill}/SKILL.MD"))
    policy = SkillPolicy()
    policy.enable_skill("t", "ToDo")
    policy.enable_skill("t", "Shell")
    graph = create_agent_graph(MixedBatchProvider(), registry, policy).compile()
    cfg = {"configurable": {"thread_id": "t", "user_id": "u"}}
    out = asyncio.run(graph.ainvoke(
        {"messages": [HumanMessage(content="go")]}, cfg))
    by_id = {m.tool_call_id: str(m.content)
             for m in out["messages"] if isinstance(m, ToolMessage)}
    assert by_id["good"].startswith("[x] Step 1 done")
    assert by_id["bad"].startswith("Error:")
    assert "No such file" in by_id["bad"]


class CreativeLoopProvider:
    """Never repeats exactly (varies a sleep suffix), so the exact-match guard
    never fires; only the per-turn budget can stop it."""

    def __init__(self):
        self.calls = 0

    def supports_tools(self):
        return True

    async def achat(self, **_):
        self.calls += 1
        return AIMessage(content="", tool_calls=[
            {"name": "run_bash",
             "args": {"command": f"true && sleep 0.0{self.calls}"},
             "id": f"c{self.calls}"}])

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_tool_call_budget_stops_creative_loop():
    registry = SkillRegistry()
    registry.register_skill(load_skill_from_markdown("birdie/skills/shell/SKILL.MD"))
    policy = SkillPolicy()
    policy.enable_skill("t", "Shell")
    provider = CreativeLoopProvider()
    graph = create_agent_graph(provider, registry, policy).compile()
    cfg = {"configurable": {"thread_id": "t", "user_id": "u",
                            "max_tool_calls_per_turn": 5}}
    out = asyncio.run(graph.ainvoke(
        {"messages": [HumanMessage(content="go")]}, cfg))
    last = out["messages"][-1]
    assert isinstance(last, AIMessage) and "more than 5 tool calls" in last.content
    assert provider.calls == 6  # 5 allowed, the 6th is cut off


def test_tool_calls_this_turn_resets_on_human_message():
    from birdie.agent.graph import _tool_calls_this_turn
    msgs = [HumanMessage(content="a"), AIMessage(content="", tool_calls=[
        {"name": "x", "args": {}, "id": "1"}, {"name": "y", "args": {}, "id": "2"}]),
        HumanMessage(content="b"), AIMessage(content="", tool_calls=[
            {"name": "x", "args": {}, "id": "3"}])]
    assert _tool_calls_this_turn(msgs) == 1
