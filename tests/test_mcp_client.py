"""
Unit tests for MCP server config and adapter conversion (all transports).
"""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from birdie.core.models import MCPServerConfig
from birdie.core.mcp_client import _to_adapter_config


def test_stdio_config():
    cfg = MCPServerConfig(
        transport="stdio", command="python", args=["server.py"]
    )
    entry = _to_adapter_config("demo", cfg)
    assert entry == {
        "transport": "stdio",
        "command": "python",
        "args": ["server.py"],
    }


def test_stdio_requires_command():
    with pytest.raises(ValidationError):
        MCPServerConfig(transport="stdio")


def test_sse_config():
    cfg = MCPServerConfig(transport="sse", url="http://localhost:8080/sse")
    entry = _to_adapter_config("remote", cfg)
    assert entry == {"transport": "sse", "url": "http://localhost:8080/sse"}


def test_sse_timeouts_stay_floats():
    cfg = MCPServerConfig(
        transport="sse",
        url="http://localhost:8080/sse",
        timeout=5,
        sse_read_timeout=30,
    )
    entry = _to_adapter_config("remote", cfg)
    assert entry["timeout"] == 5
    assert entry["sse_read_timeout"] == 30


def test_streamable_http_config():
    cfg = MCPServerConfig(
        transport="streamable_http",
        url="http://localhost:8080/mcp",
        headers={"Authorization": "Bearer x"},
    )
    entry = _to_adapter_config("remote", cfg)
    assert entry == {
        "transport": "streamable_http",
        "url": "http://localhost:8080/mcp",
        "headers": {"Authorization": "Bearer x"},
    }


def test_streamable_http_timeouts_become_timedelta():
    cfg = MCPServerConfig(
        transport="streamable_http",
        url="http://localhost:8080/mcp",
        timeout=5,
        sse_read_timeout=30,
    )
    entry = _to_adapter_config("remote", cfg)
    assert entry["timeout"] == timedelta(seconds=5)
    assert entry["sse_read_timeout"] == timedelta(seconds=30)


def test_http_alias_normalizes_to_streamable_http():
    cfg = MCPServerConfig(transport="http", url="http://localhost:8080/mcp")
    assert cfg.transport == "streamable_http"


def test_remote_transport_requires_url():
    with pytest.raises(ValidationError):
        MCPServerConfig(transport="sse")
    with pytest.raises(ValidationError):
        MCPServerConfig(transport="streamable_http")
