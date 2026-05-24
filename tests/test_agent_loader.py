"""Tests for AGENT.MD parsing and agent discovery."""

import os
import tempfile
from pathlib import Path

import pytest

from birdie.core.agent_loader import (
    parse_agent_markdown,
    load_agent_from_markdown,
    discover_agents_from_directory,
)
from birdie.core.models import AgentDef, AgentParam


MINIMAL_AGENTS_MD = """\
---
name: TestAgent
description: A minimal test agent
---

## Prompt

Hello {{ name }}
"""

FULL_AGENTS_MD = """\
---
name: Summarizer
version: 1.2.0
description: Summarizes text into bullet points
vendor: anthropic
model: claude-haiku-4-5-20251001
allowed_skills:
  - search
  - calculator
---

## Input

### text
type: string
description: The text to summarize
required: true

### max_points
type: integer
description: Maximum number of bullet points
required: false

## Output

### summary
type: string
description: Concise summary paragraph

## Prompt

Summarize the following text. Use at most {{ max_points }} bullet points.

{{ text }}
"""


class TestParseAgentMarkdown:
    def test_minimal_parses(self):
        agent = parse_agent_markdown(MINIMAL_AGENTS_MD)
        assert agent.name == "TestAgent"
        assert agent.description == "A minimal test agent"
        assert agent.prompt == "Hello {{ name }}"

    def test_minimal_defaults(self):
        agent = parse_agent_markdown(MINIMAL_AGENTS_MD)
        assert agent.version == "1.0.0"
        assert agent.vendor is None
        assert agent.model is None
        assert agent.allowed_skills == []
        assert agent.input_params == []
        assert agent.output_params == []

    def test_full_frontmatter(self):
        agent = parse_agent_markdown(FULL_AGENTS_MD)
        assert agent.name == "Summarizer"
        assert agent.version == "1.2.0"
        assert agent.vendor == "anthropic"
        assert agent.model == "claude-haiku-4-5-20251001"
        assert agent.allowed_skills == ["search", "calculator"]

    def test_input_params(self):
        agent = parse_agent_markdown(FULL_AGENTS_MD)
        assert len(agent.input_params) == 2

        text_param = agent.input_params[0]
        assert text_param.name == "text"
        assert text_param.type == "string"
        assert text_param.description == "The text to summarize"
        assert text_param.required is True

        max_param = agent.input_params[1]
        assert max_param.name == "max_points"
        assert max_param.type == "integer"
        assert max_param.required is False

    def test_output_params(self):
        agent = parse_agent_markdown(FULL_AGENTS_MD)
        assert len(agent.output_params) == 1
        assert agent.output_params[0].name == "summary"
        assert agent.output_params[0].type == "string"

    def test_prompt_extracted(self):
        agent = parse_agent_markdown(FULL_AGENTS_MD)
        assert "{{ max_points }}" in agent.prompt
        assert "{{ text }}" in agent.prompt

    def test_missing_frontmatter_raises(self):
        with pytest.raises(ValueError, match="No YAML frontmatter"):
            parse_agent_markdown("## Prompt\nHello")

    def test_missing_prompt_raises(self):
        content = "---\nname: X\ndescription: y\n---\n\n## Input\n"
        with pytest.raises(ValueError, match="No ## Prompt section"):
            parse_agent_markdown(content)

    def test_no_input_section(self):
        agent = parse_agent_markdown(MINIMAL_AGENTS_MD)
        assert agent.input_params == []

    def test_param_default_required_true(self):
        content = """\
---
name: A
description: b
---

## Input

### thing
type: string
description: a thing

## Prompt

do {{ thing }}
"""
        agent = parse_agent_markdown(content)
        assert agent.input_params[0].required is True


class TestLoadAgentFromMarkdown:
    def test_loads_from_file(self, tmp_path):
        f = tmp_path / "AGENT.MD"
        f.write_text(MINIMAL_AGENTS_MD)
        agent = load_agent_from_markdown(str(f))
        assert agent.name == "TestAgent"


class TestDiscoverAgentsFromDirectory:
    def test_discovers_in_subdirs(self, tmp_path):
        (tmp_path / "my_agent").mkdir()
        (tmp_path / "my_agent" / "AGENT.MD").write_text(MINIMAL_AGENTS_MD)

        agents = discover_agents_from_directory(str(tmp_path))
        assert len(agents) == 1
        assert agents[0].name == "TestAgent"

    def test_ignores_non_dirs(self, tmp_path):
        (tmp_path / "AGENT.MD").write_text(MINIMAL_AGENTS_MD)
        agents = discover_agents_from_directory(str(tmp_path))
        assert agents == []

    def test_skips_broken_files(self, tmp_path):
        (tmp_path / "bad_agent").mkdir()
        (tmp_path / "bad_agent" / "AGENT.MD").write_text("not yaml at all ### broken")

        agents = discover_agents_from_directory(str(tmp_path))
        assert agents == []

    def test_discovers_multiple(self, tmp_path):
        for i in range(3):
            d = tmp_path / f"agent_{i}"
            d.mkdir()
            content = MINIMAL_AGENTS_MD.replace("TestAgent", f"Agent{i}")
            (d / "AGENT.MD").write_text(content)

        agents = discover_agents_from_directory(str(tmp_path))
        assert len(agents) == 3

    def test_empty_directory(self, tmp_path):
        agents = discover_agents_from_directory(str(tmp_path))
        assert agents == []
