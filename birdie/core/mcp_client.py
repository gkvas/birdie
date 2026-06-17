"""
MCP client manager.

Wraps ``langchain-mcp-adapters`` ``MultiServerMCPClient`` so the rest of the
codebase only needs to call ``get_tools()`` - connection details stay here.

Each call to ``get_tools()`` opens a fresh MCP session, which matches the
library's design (stateless per-call sessions).  Normalized tool definitions
(schemas only, no callables) are cached after the first fetch so ``call_model``
can be re-entered cheaply.
"""

import logging
from datetime import timedelta
from typing import List, Optional

from langchain_core.tools import BaseTool

from .models import MCPServerConfig

log = logging.getLogger(__name__)


def _to_adapter_config(name: str, cfg: MCPServerConfig) -> dict:
    """Convert MCPServerConfig to the dict format expected by MultiServerMCPClient.

    Handles all three transports: ``stdio`` (subprocess), ``sse`` (Server-Sent
    Events endpoint), and ``streamable_http`` (Streamable HTTP endpoint).  The
    ``sse`` and ``streamable_http`` transports share the same URL/header/timeout
    fields; the only wire difference is the transport string and that the
    Streamable HTTP client expects ``timedelta`` timeouts where SSE takes floats.
    """
    if cfg.transport == "stdio":
        entry: dict = {
            "transport": "stdio",
            "command": cfg.command,
            "args": cfg.args,
        }
        if cfg.env is not None:
            entry["env"] = cfg.env
        if cfg.cwd is not None:
            entry["cwd"] = cfg.cwd
    else:
        # sse and streamable_http both connect to a URL.
        entry = {
            "transport": cfg.transport,
            "url": cfg.url,
        }
        if cfg.headers is not None:
            entry["headers"] = cfg.headers
        # The Streamable HTTP adapter expects timedelta timeouts; SSE takes floats.
        wrap = (
            (lambda s: timedelta(seconds=s))
            if cfg.transport == "streamable_http"
            else (lambda s: s)
        )
        if cfg.timeout is not None:
            entry["timeout"] = wrap(cfg.timeout)
        if cfg.sse_read_timeout is not None:
            entry["sse_read_timeout"] = wrap(cfg.sse_read_timeout)
    return entry


class MCPClientManager:
    """
    Manages MCP server registrations and provides their tools as LangChain
    BaseTool objects.

    Usage
    -----
    Register servers (sync, at startup)::

        manager = MCPClientManager()
        manager.register_server("my_server", MCPServerConfig(
            transport="stdio", command="python", args=["server.py"]
        ))

    Then retrieve tools (async, at tool-call time)::

        tools = await manager.get_tools()

    ``get_tools()`` opens a new MCP session on each call - this is the
    recommended pattern from ``langchain-mcp-adapters`` (each tool invocation
    is self-contained).  Normalized definitions (name/description/schema) are
    cached for the lifetime of the process since schemas do not change at runtime.
    """

    def __init__(self) -> None:
        self._configs: dict[str, dict] = {}
        self._cached_tools: Optional[List[BaseTool]] = None
        self._cache_key: Optional[frozenset] = None

    def register_server(self, name: str, config: MCPServerConfig) -> None:
        """Register an MCP server under *name*.  Call before any ``get_tools()``."""
        if self._cached_tools is not None:
            log.warning(
                "MCPClientManager: registering server '%s' after tools were already "
                "cached - cache cleared", name
            )
            self._cached_tools = None
            self._cache_key = None
        self._configs[name] = _to_adapter_config(name, config)
        log.debug("Registered MCP server '%s' (%s)", name, config.transport)

    @property
    def has_servers(self) -> bool:
        return bool(self._configs)

    async def get_tools(self, allowed: set = None) -> List[BaseTool]:
        """Return LangChain tools from enabled MCP servers.

        Only servers whose skill name appears in *allowed* are contacted.
        Pass ``None`` to include all registered servers.

        Results are cached per allowed-set so repeated calls within the same
        turn are free.  The cache is invalidated when a server is registered.
        """
        configs = (
            {k: v for k, v in self._configs.items() if k in allowed}
            if allowed is not None else self._configs
        )

        if not configs:
            return []

        cache_key = frozenset(configs)
        if self._cached_tools is not None and cache_key == self._cache_key:
            return self._cached_tools

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:
            raise ImportError(
                "MCP support requires the 'mcp' optional extra: "
                "pip install 'birdie[mcp]'"
            ) from exc

        client = MultiServerMCPClient(configs)
        tools = await client.get_tools()
        self._cached_tools = tools
        self._cache_key = cache_key
        log.debug(
            "MCP: loaded %d tool(s) from %d server(s)",
            len(tools), len(configs),
        )
        return tools
