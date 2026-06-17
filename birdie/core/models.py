"""
Core data models for the dynamic skill system.
"""

import warnings
from pydantic import BaseModel, model_validator
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
    """Connection config for a single MCP server declared in a SKILL.MD file.

    Three transports are supported:

    - ``stdio``: the server is launched as a subprocess (``command`` + ``args``).
    - ``sse``: connect to a long-lived Server-Sent Events endpoint (``url``).
    - ``streamable_http``: connect to a Streamable HTTP endpoint (``url``).  The
      friendly alias ``http`` is accepted and normalized to ``streamable_http``.
    """
    transport: Literal["stdio", "sse", "streamable_http"] = "stdio"
    # stdio fields
    command: Optional[str] = None
    args: List[str] = []
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    # sse / streamable_http fields
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    # connection timeout in seconds (sse / streamable_http only)
    timeout: Optional[float] = None
    # read timeout for the SSE event stream in seconds (sse / streamable_http only)
    sse_read_timeout: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_transport(cls, data: Any) -> Any:
        """Accept ``http`` as a friendly alias for ``streamable_http``."""
        if isinstance(data, dict) and data.get("transport") == "http":
            data = {**data, "transport": "streamable_http"}
        return data

    @model_validator(mode="after")
    def _check_required_fields(self) -> "MCPServerConfig":
        """Ensure the fields required by the chosen transport are present."""
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("MCP stdio transport requires 'command'")
        else:
            if not self.url:
                raise ValueError(
                    f"MCP {self.transport} transport requires 'url'"
                )
        return self


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
    vendor: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
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
    Freetext skills carry instructional prose in `body` that is progressively
    loaded into the system prompt on LLM request via the get_skill tool.
    """
    name: str
    version: str
    description: str
    tools: List[SkillTool] = []
    tags: List[str] = []
    triggers: List[str] = []  # deprecated: kept for backward compat, no longer used
    always_inject: bool = False   # inject body into system prompt every turn
    permissions: List[str] = []
    body: Optional[str] = None  # prose body loaded on demand via get_skill
    location: Optional[str] = None  # identifier used to load this skill; defaults to name
    mcp_server: Optional[MCPServerConfig] = None  # set for MCP-backed skills