"""
Core data models for the dynamic skill system.
"""

import warnings
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal

# Pydantic v2 reserves `model_` prefixed names; `schema` shadows the legacy
# .schema() class-method.  We silence the warning because the SKILL.MD spec
# mandates this field name and all callers access it as an instance attribute.
warnings.filterwarnings(
    "ignore",
    message="Field name \"schema\" in \"SkillTool\" shadows",
    category=UserWarning,
)


class MCPServerConfig(BaseModel):
    """Connection config for a single MCP server declared in a SKILL.MD file."""
    transport: Literal["stdio", "sse"] = "stdio"
    # stdio fields
    command: Optional[str] = None
    args: List[str] = []
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    # sse / http fields
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


class SkillTool(BaseModel):
    """
    A single tool within a skill that can be executed.
    """
    name: str
    description: str
    entrypoint: str
    schema: Dict[str, Any]
    tags: List[str] = []
    

class AgentParam(BaseModel):
    """A single input or output parameter declared in an AGENT.MD file."""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


class AgentDef(BaseModel):
    """A sub-agent defined by an AGENT.MD file.

    Each AgentDef surfaces as an async tool to the calling agent.  At
    invocation time the prompt template is rendered with the input params,
    then an ephemeral DynamicAgent is run and its final reply is returned.
    """
    name: str
    version: str = "1.0.0"
    description: str
    enabled_by_default: bool = False
    vendor: Optional[str] = None
    model: Optional[str] = None
    allowed_skills: List[str] = []
    recursion_limit: int = 25
    max_tool_repetitions: int = 3
    input_params: List[AgentParam] = []
    output_params: List[AgentParam] = []
    prompt: str


class Skill(BaseModel):
    """
    A self-contained capability bundle.

    Structured skills define explicit tools (entrypoint + schema).
    Freetext skills carry instructional prose in `body` that is injected into
    the system prompt when the skill is triggered; their `tools` list is empty.
    """
    name: str
    version: str
    description: str
    tools: List[SkillTool] = []
    tags: List[str] = []
    triggers: List[str] = []
    enabled_by_default: bool = False
    always_inject: bool = False   # inject body into system prompt every turn
    permissions: List[str] = []
    body: Optional[str] = None  # prose body injected into system prompt
    mcp_server: Optional[MCPServerConfig] = None  # set for MCP-backed skills