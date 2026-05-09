"""
AGENTS.MD parser and agent discovery.

An AGENTS.MD file defines a sub-agent with structured input/output and a
prompt template.  Each agent surfaces to the calling LLM as a regular async
tool whose arguments are the declared input parameters.

File format
-----------
::

    ---
    name: Summarizer
    version: 1.0.0
    description: Summarizes text into bullet points
    enabled_by_default: false
    vendor: anthropic
    model: claude-haiku-4-5-20251001
    allowed_skills: []
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

    ### points
    type: array
    description: Bullet point list

    ## Prompt

    Summarize the following text.  Use at most {{ max_points }} bullet points.

    {{ text }}

    Return JSON: {"summary": "...", "points": ["..."]}
"""

import re
import yaml
from pathlib import Path
from typing import List
from .models import AgentDef, AgentParam


def parse_agent_markdown(content: str) -> AgentDef:
    """Parse an AGENTS.MD file into an AgentDef object."""
    fm_match = re.search(r'^---(.*?)---', content, re.DOTALL)
    if not fm_match:
        raise ValueError("No YAML frontmatter found in AGENTS.MD")
    frontmatter = yaml.safe_load(fm_match.group(1))

    input_params = _parse_params_section(content, "Input")
    output_params = _parse_params_section(content, "Output")

    prompt_match = re.search(r'## Prompt\s*\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
    if not prompt_match:
        raise ValueError("No ## Prompt section found in AGENTS.MD")
    prompt = prompt_match.group(1).strip()

    return AgentDef(
        name=frontmatter['name'],
        version=str(frontmatter.get('version', '1.0.0')),
        description=frontmatter['description'],
        enabled_by_default=frontmatter.get('enabled_by_default', False),
        vendor=frontmatter.get('vendor'),
        model=frontmatter.get('model'),
        allowed_skills=frontmatter.get('allowed_skills', []),
        input_params=input_params,
        output_params=output_params,
        prompt=prompt,
    )


def _parse_params_section(content: str, section: str) -> List[AgentParam]:
    """Extract AgentParam objects from a named ## section."""
    section_match = re.search(
        rf'## {section}\s*\n(.*?)(?=\n## |\Z)', content, re.DOTALL
    )
    if not section_match:
        return []

    # Prepend \n so the first ### block is also matched by \n### pattern.
    body = "\n" + section_match.group(1)
    params = []
    for block in re.finditer(r'\n### (\w+)\n(.*?)(?=\n### |\Z)', body, re.DOTALL):
        name = block.group(1).strip()
        body = block.group(2)

        type_match = re.search(r'type:\s*(\S+)', body)
        desc_match = re.search(r'description:\s*(.*?)(?=\n\w|$)', body)
        req_match = re.search(r'required:\s*(\S+)', body)

        params.append(AgentParam(
            name=name,
            type=type_match.group(1) if type_match else "string",
            description=desc_match.group(1).strip() if desc_match else "",
            required=(req_match.group(1).lower() == "true") if req_match else True,
        ))
    return params


def load_agent_from_markdown(path: str) -> AgentDef:
    """Parse a single AGENTS.MD file from disk."""
    with open(path, "r") as f:
        content = f.read()
    return parse_agent_markdown(content)


def discover_agents_from_directory(directory: str) -> List[AgentDef]:
    """Scan a directory tree for AGENTS.MD files and return parsed AgentDefs.

    Each immediate subdirectory of *directory* is checked for an ``AGENTS.MD``
    file.  Parse errors are printed and skipped.
    """
    agents: List[AgentDef] = []
    for agent_dir in Path(directory).iterdir():
        if agent_dir.is_dir():
            agents_md = agent_dir / "AGENTS.MD"
            if agents_md.exists():
                try:
                    agents.append(load_agent_from_markdown(str(agents_md)))
                except Exception as e:
                    print(f"Error loading agent from {agents_md}: {e}")
    return agents
