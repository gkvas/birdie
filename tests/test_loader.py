"""
Unit tests for the SKILL.MD loader.
"""

import pytest
import tempfile
import os
from birdie.core.loader import parse_skill_markdown, load_skill_from_markdown
from birdie.core.models import Skill


def test_parse_skill_markdown():
    """Test parsing a SKILL.MD file."""
    content = """---
name: TestSkill
version: 1.0.0
description: A test skill
tags: [test, example]
---

# Skill: TestSkill

## Tools

### test_tool
description: A test tool
entrypoint: bash:echo {param}
schema:
  type: object
  properties:
    param:
      type: string
  required: [param]

## Permissions
- local_shell
"""
    
    skill = parse_skill_markdown(content)
    assert skill.name == "TestSkill"
    assert skill.version == "1.0.0"
    assert skill.description == "A test skill"
    assert len(skill.tools) == 1
    assert skill.tools[0].name == "test_tool"
    assert skill.tools[0].description == "A test tool"
    assert skill.tools[0].entrypoint == "bash:echo {param}"
    assert "local_shell" in skill.permissions


def test_load_skill_from_markdown():
    """Test loading a skill from a markdown file."""
    content = """---
name: FileSkill
version: 1.0.0
description: A file skill
tags: [file, test]
---

# Skill: FileSkill

## Tools

### read_file
description: Read a file
entrypoint: bash:cat {path}
schema:
  type: object
  properties:
    path:
      type: string
  required: [path]
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    try:
        skill = load_skill_from_markdown(temp_path)
        assert skill.name == "FileSkill"
        assert len(skill.tools) == 1
        assert skill.tools[0].name == "read_file"
    finally:
        os.unlink(temp_path)