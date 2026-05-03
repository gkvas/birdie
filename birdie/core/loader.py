"""
SKILL.MD parser and skill discovery.

Parses the two SKILL.MD formats supported by Birdie:

- **Structured skills** - YAML frontmatter + ``## Tools`` section.  Each
  ``### tool_name`` block becomes a callable ``SkillTool`` with an entrypoint
  and JSON Schema.

- **Freetext skills** - YAML frontmatter only (no ``## Tools``).  The entire
  Markdown body is stored in ``Skill.body`` and injected into the system
  prompt when the skill's trigger keywords fire.

Skills with ``always_inject: true`` may carry both tools *and* a prose body
(the prose before ``## Tools`` is stored and injected every turn).
"""

import re
import sys
import yaml
from pathlib import Path
from typing import List
from .models import Skill, SkillTool, MCPServerConfig


def parse_skill_markdown(content: str) -> Skill:
    """
    Parse a SKILL.MD file into a Skill object.

    Two formats are supported:

    Structured skills
        Frontmatter + ``## Tools`` section with ``### tool_name`` / ``entrypoint``
        / ``schema`` blocks.  Each tool becomes a callable ``SkillTool``.

    Freetext skills
        Frontmatter only (no ``## Tools`` section).  The full Markdown body is
        stored in ``skill.body`` and injected into the system prompt when the
        skill is triggered.  ``skill.tools`` is empty.
    """
    # -- 1. Frontmatter -------------------------------------------------------
    fm_match = re.search(r'^---(.*?)---', content, re.DOTALL)
    if not fm_match:
        raise ValueError("No YAML frontmatter found in SKILL.MD")
    frontmatter = yaml.safe_load(fm_match.group(1))

    # Body = everything after the closing ---
    body_start = fm_match.end()
    full_body = content[body_start:].strip() or None

    always_inject = bool(frontmatter.get('always_inject', False))

    # -- 2. Tools section (structured skills only) ----------------------------
    tools_match = re.search(r'## Tools(.*?)(?=\n## |\Z)', content, re.DOTALL)

    tools: List[SkillTool] = []
    if tools_match:
        tools_body = tools_match.group(1)
        for block in re.finditer(r'\n### (.*?)\n(.*?)(?=\n### |\Z)', tools_body, re.DOTALL):
            tool_name = block.group(1).strip()
            tool_content = block.group(2)

            desc_match = re.search(r'description:\s*(.*?)(?=\n\w|$)', tool_content)
            description = desc_match.group(1).strip() if desc_match else ""

            ep_match = re.search(r'entrypoint:\s*(.*?)(?=\n\w|$)', tool_content)
            entrypoint = ep_match.group(1).strip() if ep_match else ""

            schema_match = re.search(r'schema:(.*?)(?=\n### |\n## |\Z)', tool_content, re.DOTALL)
            schema = yaml.safe_load(schema_match.group(1)) if schema_match else {}

            tools.append(SkillTool(
                name=tool_name,
                description=description,
                entrypoint=entrypoint,
                schema=schema or {},
                tags=frontmatter.get('tool_tags', []),
            ))

    # -- 3. Permissions section -----------------------------------------------
    perms_match = re.search(r'## Permissions(.*?)(?=\n## |\Z)', content, re.DOTALL)
    permissions: List[str] = []
    if perms_match:
        for line in perms_match.group(1).split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('- '):
                line = line[2:]
            permissions.append(line)

    # Determine which prose to store in body:
    # - Freetext skills (no tools): full body
    # - always_inject structured skills: prose before the ## Tools section
    # - Other structured skills: no body stored
    if not tools:
        body = full_body
    elif always_inject:
        tools_pos = content.find('\n## Tools', body_start)
        if tools_pos >= 0:
            body = content[body_start:tools_pos].strip() or None
        else:
            body = full_body
    else:
        body = None

    mcp_server: MCPServerConfig | None = None
    if 'mcp_server' in frontmatter:
        mcp_server = MCPServerConfig(**frontmatter['mcp_server'])

    return Skill(
        name=frontmatter['name'],
        version=str(frontmatter.get('version', '1.0.0')),
        description=frontmatter['description'],
        tools=tools,
        tags=frontmatter.get('tags', []),
        triggers=frontmatter.get('triggers', []),
        enabled_by_default=frontmatter.get('enabled_by_default', True),
        always_inject=always_inject,
        permissions=permissions,
        body=body,
        mcp_server=mcp_server,
    )


def load_skill_from_markdown(path: str) -> Skill:
    """Parse a single SKILL.MD file from disk.

    Args:
        path: Absolute or relative path to the SKILL.MD file.

    Returns:
        A fully populated ``Skill`` object.

    Raises:
        ValueError: If the file contains no YAML frontmatter.
    """
    with open(path, "r") as f:
        content = f.read()
    skill = parse_skill_markdown(content)
    if skill.mcp_server and skill.mcp_server.args:
        skill_dir = Path(path).resolve().parent
        resolved = [
            str(skill_dir / arg) if not arg.startswith("-") and not Path(arg).is_absolute() else arg
            for arg in skill.mcp_server.args
        ]
        skill = skill.model_copy(update={"mcp_server": skill.mcp_server.model_copy(update={"args": resolved})})
    return skill


def discover_skills_from_directory(directory: str) -> List[Skill]:
    """Scan a directory tree for SKILL.MD files and return the parsed skills.

    Each immediate subdirectory of *directory* is checked for a ``SKILL.MD``
    file.  Parse errors are printed and skipped so a single bad skill does not
    prevent the rest from loading.

    Args:
        directory: Path to the skills root (e.g. ``birdie/skills``).

    Returns:
        List of successfully parsed ``Skill`` objects.
    """
    # Add the skills root to sys.path so that python: entrypoints in user skill
    # directories (e.g. python:tools.search_repos) can import their local modules.
    skills_root = str(Path(directory).resolve())
    if skills_root not in sys.path:
        sys.path.insert(0, skills_root)

    skills: List[Skill] = []
    for skill_dir in Path(directory).iterdir():
        if skill_dir.is_dir():
            skill_md = skill_dir / "SKILL.MD"
            if skill_md.exists():
                try:
                    skills.append(load_skill_from_markdown(str(skill_md)))
                except Exception as e:
                    print(f"Error loading skill from {skill_md}: {e}")
    return skills
