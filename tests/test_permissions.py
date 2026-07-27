"""Tests for skill permission enforcement (human-in-the-loop tool gate)."""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from birdie.agent.run import DynamicAgent
from birdie.core.llm_provider import LLMProvider
from birdie.core.policy import SkillPolicy

SKILL_MD = """---
name: PermSkill
version: 1.0.0
description: A skill that declares permissions
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
- filesystem
"""


def _write_perm_skill(tmp_path):
    d = tmp_path / "skills" / "permskill"
    d.mkdir(parents=True)
    (d / "SKILL.MD").write_text(SKILL_MD)
    return str(tmp_path / "skills")


class _ToolCallingProvider:
    """Calls echo on the first turn-cycle, then finishes."""

    def __init__(self):
        self.calls = 0

    def supports_tools(self):
        return True

    async def achat(self, messages, tools=None, system_prompt=None, **kw):
        self.calls += 1
        if self.calls % 2 == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "echo", "args": {"message": "hi"},
                             "id": f"tc{self.calls}"}],
            )
        return AIMessage(content="done")


LLMProvider.register(_ToolCallingProvider)


def _tool_messages(result):
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


@pytest.mark.asyncio
async def test_deny_blocks_execution(tmp_path):
    skills_dir = _write_perm_skill(tmp_path)
    decisions = []

    async def cb(skill_name, permissions, tool_name, args):
        decisions.append((skill_name, tuple(permissions), tool_name))
        return "deny"

    agent = DynamicAgent(_ToolCallingProvider(), skills_dir=skills_dir,
                         skills_enabled=["PermSkill"])
    agent.tool_approval_callback = cb
    result = await agent.invoke("run it", thread_id="t1")

    assert decisions == [("PermSkill", ("network", "filesystem"), "echo")]
    tm = _tool_messages(result)[0]
    assert "denied" in tm.content
    assert "Echo: hi" not in tm.content


@pytest.mark.asyncio
async def test_allow_executes_and_prompts_again(tmp_path):
    skills_dir = _write_perm_skill(tmp_path)
    decisions = []

    def cb(skill_name, permissions, tool_name, args):  # sync callback works too
        decisions.append(tool_name)
        return "allow"

    agent = DynamicAgent(_ToolCallingProvider(), skills_dir=skills_dir,
                         skills_enabled=["PermSkill"])
    agent.tool_approval_callback = cb
    r1 = await agent.invoke("run it", thread_id="t2")
    r2 = await agent.invoke("run it again", thread_id="t2")

    assert "Echo: hi" in _tool_messages(r1)[0].content
    assert len(decisions) == 2  # "allow" is per-call, so it prompts again


@pytest.mark.asyncio
async def test_always_grants_for_session(tmp_path):
    skills_dir = _write_perm_skill(tmp_path)
    decisions = []

    async def cb(skill_name, permissions, tool_name, args):
        decisions.append(tool_name)
        return "always"

    agent = DynamicAgent(_ToolCallingProvider(), skills_dir=skills_dir,
                         skills_enabled=["PermSkill"])
    agent.tool_approval_callback = cb
    r1 = await agent.invoke("run it", thread_id="t3")
    r2 = await agent.invoke("run it again", thread_id="t3")

    assert "Echo: hi" in _tool_messages(r1)[0].content
    assert "Echo: hi" in _tool_messages(r2)[0].content
    assert len(decisions) == 1  # second call served by the standing grant
    assert agent.policy.has_permission_grant("t3", "PermSkill")


@pytest.mark.asyncio
async def test_no_callback_allows_everything(tmp_path):
    skills_dir = _write_perm_skill(tmp_path)
    agent = DynamicAgent(_ToolCallingProvider(), skills_dir=skills_dir,
                         skills_enabled=["PermSkill"])
    result = await agent.invoke("run it", thread_id="t4")
    assert "Echo: hi" in _tool_messages(result)[0].content


def test_policy_grant_roundtrip():
    p = SkillPolicy()
    assert not p.has_permission_grant("s1", "X")
    p.grant_permissions("s1", "X")
    assert p.has_permission_grant("s1", "X")
    assert not p.has_permission_grant("s2", "X")


def test_session_approved_skills_roundtrip(tmp_path):
    from birdie.core.session import SessionManager
    mgr = SessionManager(sessions_root=tmp_path)
    session = mgr.create("alice")
    session.approved_skills.append("PermSkill")
    mgr.save(session)
    loaded = mgr.load("alice", session.id)
    assert loaded.approved_skills == ["PermSkill"]
