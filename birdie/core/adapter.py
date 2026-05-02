"""
Adapter layer: converts SkillTool objects into LangChain StructuredTools.

SkillTools are declarative (name, entrypoint, JSON schema).  This module
bridges them to the executable LangChain tool interface expected by ToolNode.
"""

from langchain_core.tools import StructuredTool
from typing import Any
from .models import SkillTool
from .entrypoints import resolve_entrypoint


def skilltool_to_langchain_tool(skill_tool: SkillTool) -> StructuredTool:
    """Wrap a SkillTool as a LangChain StructuredTool backed by its entrypoint.

    A new wrapper is created on every call so the active entrypoint resolver
    is always fresh - important when skills are enabled/disabled between turns.

    Args:
        skill_tool: The declarative skill tool to wrap.

    Returns:
        An executable LangChain StructuredTool with the same name, description,
        and argument schema as the source SkillTool.
    """
    resolver = resolve_entrypoint(skill_tool.entrypoint)

    def _wrapped(**kwargs: Any) -> Any:
        if "required" in skill_tool.schema:
            for field in skill_tool.schema["required"]:
                if field not in kwargs:
                    raise ValueError(f"Missing required field: {field}")
        return resolver(skill_tool.entrypoint, **kwargs)

    return StructuredTool.from_function(
        func=_wrapped,
        name=skill_tool.name,
        description=skill_tool.description,
        args_schema=create_args_schema(skill_tool.schema),
    )


def create_args_schema(schema: dict) -> type:
    """Build a Pydantic model class from a JSON Schema object.

    Used to give LangChain's StructuredTool a typed argument schema that it
    can validate and surface to the LLM as the function parameter spec.

    Args:
        schema: A JSON Schema ``object`` dict with optional ``properties`` and
            ``required`` keys.  Unknown types fall back to ``Any``.

    Returns:
        A dynamically created Pydantic BaseModel subclass whose fields match
        the schema properties, with required fields marked as mandatory.
    """
    from pydantic import BaseModel, create_model
    from typing import Optional, Any as AnyType

    if "properties" not in schema:
        return create_model("EmptyArgs", __base__=BaseModel)

    _TYPE_MAP = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    fields = {}
    required = schema.get("required", [])
    for field_name, field_schema in schema["properties"].items():
        python_type = _TYPE_MAP.get(field_schema.get("type", ""), AnyType)
        if field_name in required:
            fields[field_name] = (python_type, ...)
        else:
            fields[field_name] = (Optional[python_type], None)

    return create_model("ToolArgs", **fields)
