"""
Vendor-agnostic LLM provider layer.

Each provider converts between the internal normalized format
(LangChain BaseMessage objects + NormalizedToolDef dicts) and the
vendor's wire format.  The agent calls achat() / stream_chat() and
never touches a vendor SDK directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional, Union, TYPE_CHECKING

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalized types
# ---------------------------------------------------------------------------

class NormalizedToolDef(TypedDict):
    """Vendor-agnostic tool definition derived from a SKILL.MD SkillTool."""
    name: str
    description: str
    parameters: dict   # JSON Schema object
    entrypoint: NotRequired[str]  # present for SkillTool-derived defs; used by ACPProvider MCP bridge


@dataclass
class ModelInfo:
    """Normalized metadata about a single model."""
    id: str
    context_window: int = 0
    supports_tools: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = False
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ProviderConfig - validated, JSON-serialisable provider configuration
# ---------------------------------------------------------------------------

class ProviderConfig(BaseModel):
    """
    Fully validated, JSON-round-trippable configuration for any LLMProvider.

    ``extra="allow"`` lets vendor-specific fields pass through transparently
    (e.g. ``default_max_tokens`` for Anthropic, ``seed`` for OpenAI).

    Accepted sources
    ----------------
    - Python dict::

        ProviderConfig(vendor="anthropic", model="claude-sonnet-4-6")

    - JSON string (``model_validate_json``)::

        ProviderConfig.model_validate_json('{"vendor":"openai","model":"gpt-4o"}')

    - JSON file::

        ProviderConfig.from_file("provider.json")

    Full JSON schema
    ----------------
    .. code-block:: json

        {
          "vendor":      "openai",
          "model":       "gpt-4o",
          "api_key":     "sk-...",
          "base_url":    null,
          "temperature": 0.0,
          "max_tokens":  null
        }

    Any extra keys are forwarded as ``**kwargs`` to the provider constructor.
    """

    model_config = ConfigDict(extra="allow")

    vendor: str = Field(
        default="openai",
        description=(
            "LLM vendor identifier.  "
            "Supported: openai | azure | anthropic | mistral | gemini | ollama | langchain | acp"
        ),
    )
    model: Optional[str] = Field(
        default=None,
        description="Model identifier (vendor-specific).  Falls back to provider default.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key.  Falls back to the vendor environment variable "
            "(OPENAI_API_KEY, AZURE_OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY, GEMINI_API_KEY)."
        ),
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Override the vendor API endpoint (proxy, local server, …).",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature applied to every completion by default.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens per completion.  None means vendor default.",
    )
    min_messages_auto: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Auto-compaction floor: minimum messages to retain after compaction.  "
            "None uses the built-in default (20)."
        ),
    )
    min_messages_forced: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Forced-compaction floor (/compact): minimum messages to retain.  "
            "None uses the built-in default (4)."
        ),
    )
    compression_window_size: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Maximum number of oldest messages to compress per compaction run.  "
            "None uses the built-in default (60)."
        ),
    )

    @classmethod
    def from_json(cls, json_str: str) -> "ProviderConfig":
        """Parse a JSON string into a ProviderConfig."""
        return cls.model_validate_json(json_str)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ProviderConfig":
        """Load a ProviderConfig from a JSON file."""
        return cls.model_validate_json(Path(path).read_text())

    def to_json(self, **kwargs: Any) -> str:
        """Serialise to a JSON string (``api_key`` excluded by default)."""
        data = self.model_dump(exclude_none=True, **kwargs)
        return json.dumps(data)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """
    Unified interface to any LLM vendor.

    The agent must call achat() for all completions; stream_chat() /
    astream_chat() for token-level streaming.  The provider handles all
    vendor-specific serialization internally.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> BaseMessage:
        """Synchronous single-turn completion."""

    @abstractmethod
    async def achat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> BaseMessage:
        """Async single-turn completion."""

    @abstractmethod
    def stream_chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        """Synchronous streaming; yields one chunk per token/delta."""

    @abstractmethod
    async def astream_chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessage]:
        """Async streaming; yields one chunk per token/delta."""

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Return available models for this vendor."""

    # -- capability flags (subclasses override as needed) -------------------

    def supports_tools(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        return False

    def supports_json_mode(self) -> bool:
        return False

    # -- identity (subclasses may override) ---------------------------------

    @property
    def vendor_name(self) -> str:
        """Human-readable vendor identifier derived from the class name."""
        return type(self).__name__.replace("Provider", "").lower()

    @property
    def model_name(self) -> str:
        """Active model identifier."""
        return getattr(self, "_model", "unknown")

    # -- traffic logging helpers --------------------------------------------

    def _log_request(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None,
        system_prompt: str | None = None,
    ) -> None:
        if not log.isEnabledFor(logging.DEBUG):
            return
        tool_names = [t["name"] for t in (tools or [])]
        last_human = next(
            (str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)),
            "",
        )
        log.debug(
            "REQUEST  model=%s  messages=%d  tools=%s\n  last_user: %s",
            self.model_name, len(messages), tool_names, last_human[:2000],
        )

    def _log_response(self, msg: BaseMessage) -> None:
        if not log.isEnabledFor(logging.DEBUG):
            return
        tc = [tc["name"] for tc in getattr(msg, "tool_calls", [])]
        log.debug(
            "RESPONSE  model=%s  tool_calls=%s\n  content: %s",
            self.model_name, tc, str(msg.content)[:2000],
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible message conversion helpers
# ---------------------------------------------------------------------------

# Tool results larger than this are truncated before being sent to the LLM.
# Large shell output (e.g. `cat` on a big file) can easily exceed provider
# context limits and trigger cryptic API errors.
_MAX_TOOL_CONTENT_CHARS = 32_000


def _lc_to_openai_messages(
    messages: list[BaseMessage],
    system_prompt: str | None = None,
) -> list[dict]:
    """Convert LangChain messages to the OpenAI chat format."""
    result: list[dict] = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
    for msg in messages:
        if isinstance(msg, SystemMessage):
            if not system_prompt:  # avoid duplicate system messages
                result.append({"role": "system", "content": str(msg.content)})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                # Omit content when tool_calls are present - Mistral (and OpenAI)
                # reject content="" alongside tool_calls in history messages.
                m: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            else:
                m = {"role": "assistant", "content": msg.content or ""}
            result.append(m)
        elif isinstance(msg, ToolMessage):
            content = str(msg.content)
            if len(content) > _MAX_TOOL_CONTENT_CHARS:
                dropped = len(content) - _MAX_TOOL_CONTENT_CHARS
                content = (
                    content[:_MAX_TOOL_CONTENT_CHARS]
                    + f"\n[...{dropped} characters truncated]"
                )
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": content,
            }
            if msg.name:
                tool_msg["name"] = msg.name
            result.append(tool_msg)
    return result


def _openai_msg_to_lc(raw: Any, usage: Any = None) -> AIMessage:
    """
    Convert an OpenAI response message object (or dict) to a LangChain
    AIMessage, preserving tool_calls when present.

    ``usage`` accepts an OpenAI-style usage object with ``prompt_tokens`` and
    ``completion_tokens`` attributes (or a plain dict with the same keys).
    """
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()

    tool_calls = []
    for tc in raw.get("tool_calls") or []:
        fn = tc.get("function", tc)  # handle both nested and flat shapes
        args = fn.get("arguments", "{}")
        tool_calls.append({
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "args": json.loads(args) if isinstance(args, str) else args,
            "type": "tool_call",
        })

    usage_metadata = None
    if usage is not None:
        if hasattr(usage, "prompt_tokens"):
            inp  = usage.prompt_tokens or 0
            out  = usage.completion_tokens or 0
        else:
            inp  = (usage.get("prompt_tokens") or 0)
            out  = (usage.get("completion_tokens") or 0)
        usage_metadata = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

    return AIMessage(
        content=raw.get("content") or "",
        tool_calls=tool_calls,
        usage_metadata=usage_metadata,
    )


def _tools_to_openai_functions(tools: list[NormalizedToolDef]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# OpenAI-compatible base provider
# ---------------------------------------------------------------------------

class _OpenAICompatibleProvider(LLMProvider):
    """
    Shared implementation for any OpenAI-compatible HTTP endpoint.

    Subclasses set _default_base_url and _env_key_name, or supply them
    directly via __init__.
    """

    _default_base_url: str | None = None
    _env_key_name: str = "OPENAI_API_KEY"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError("pip install openai") from e

        resolved_key = api_key or os.environ.get(self._env_key_name, "")
        resolved_url = base_url or self._default_base_url

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = openai.OpenAI(api_key=resolved_key, base_url=resolved_url)
        self._async_client = openai.AsyncOpenAI(api_key=resolved_key, base_url=resolved_url)
        self._extra = kwargs

    # -- capabilities -------------------------------------------------------

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True

    # -- core ---------------------------------------------------------------

    def _build_kwargs(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> dict:
        kw: dict[str, Any] = {
            "model": self._model,
            "messages": _lc_to_openai_messages(messages, system_prompt),
            "temperature": temperature if temperature is not None else self._temperature,
        }
        resolved_max = max_tokens if max_tokens is not None else self._max_tokens
        if resolved_max:
            kw["max_tokens"] = resolved_max
        if tools:
            kw["tools"] = _tools_to_openai_functions(tools)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        return kw

    def chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> BaseMessage:
        kw = self._build_kwargs(messages, tools, system_prompt, temperature, max_tokens, json_mode)
        self._log_request(messages, tools, system_prompt)
        response = self._client.chat.completions.create(**kw)
        result = _openai_msg_to_lc(response.choices[0].message, response.usage)
        self._log_response(result)
        return result

    async def achat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> BaseMessage:
        kw = self._build_kwargs(messages, tools, system_prompt, temperature, max_tokens, json_mode)
        self._log_request(messages, tools, system_prompt)
        response = await self._async_client.chat.completions.create(**kw)
        result = _openai_msg_to_lc(response.choices[0].message, response.usage)
        self._log_response(result)
        return result

    def stream_chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        kw = self._build_kwargs(messages, tools, system_prompt, kwargs.pop("temperature", 0.0), None, False)
        with self._client.chat.completions.create(**kw, stream=True) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield AIMessageChunk(content=delta.content)

    async def astream_chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessage]:
        kw = self._build_kwargs(messages, tools, system_prompt, kwargs.pop("temperature", 0.0), None, False)
        async with await self._async_client.chat.completions.create(**kw, stream=True) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield AIMessageChunk(content=delta.content)

    def list_models(self) -> list[ModelInfo]:
        try:
            models = self._client.models.list()
            return [ModelInfo(id=m.id) for m in models.data]
        except Exception as e:
            log.warning("list_models failed: %s", e)
            return []


# ---------------------------------------------------------------------------
# Concrete OpenAI-compatible providers
# ---------------------------------------------------------------------------

class OpenAIProvider(_OpenAICompatibleProvider):
    """OpenAI (GPT series)."""

    _env_key_name = "OPENAI_API_KEY"

    def __init__(self, model: str = "gpt-4o", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)

    def list_models(self) -> list[ModelInfo]:
        _KNOWN: dict[str, dict] = {
            "gpt-4o":            {"context_window": 128_000, "supports_tools": True, "supports_json_mode": True, "supports_streaming": True},
            "gpt-4o-mini":       {"context_window": 128_000, "supports_tools": True, "supports_json_mode": True, "supports_streaming": True},
            "gpt-4-turbo":       {"context_window": 128_000, "supports_tools": True, "supports_json_mode": True, "supports_streaming": True},
            "gpt-3.5-turbo":     {"context_window":  16_385, "supports_tools": True, "supports_json_mode": True, "supports_streaming": True},
        }
        return [ModelInfo(id=k, **v) for k, v in _KNOWN.items()]


class GeminiProvider(_OpenAICompatibleProvider):
    """Google Gemini via the OpenAI-compatible endpoint.

    Uses Google's drop-in OpenAI-compatible API so no additional SDK is
    required beyond the ``openai`` package already pulled in by langchain-openai.

    Install: no extra dependency - set GEMINI_API_KEY and go.
    """

    _default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    _env_key_name = "GEMINI_API_KEY"

    def __init__(self, model: str = "gemini-2.0-flash", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)

    def list_models(self) -> list[ModelInfo]:
        _KNOWN: dict[str, dict] = {
            "gemini-2.5-pro":       {"context_window": 1_048_576, "supports_tools": True, "supports_json_mode": True, "supports_streaming": True},
            "gemini-2.0-flash":     {"context_window": 1_048_576, "supports_tools": True, "supports_json_mode": True, "supports_streaming": True},
            "gemini-2.0-flash-lite":{"context_window": 1_048_576, "supports_tools": True, "supports_json_mode": True, "supports_streaming": True},
            "gemini-1.5-pro":       {"context_window": 2_097_152, "supports_tools": True, "supports_json_mode": True, "supports_streaming": True},
            "gemini-1.5-flash":     {"context_window": 1_048_576, "supports_tools": True, "supports_json_mode": True, "supports_streaming": True},
        }
        return [ModelInfo(id=k, **v) for k, v in _KNOWN.items()]


class OllamaProvider(_OpenAICompatibleProvider):
    """Local Ollama server (OpenAI-compatible endpoint)."""

    _default_base_url = "http://localhost:11434/v1"
    _env_key_name = "OLLAMA_API_KEY"  # Ollama typically ignores this

    def __init__(self, model: str = "llama3", **kwargs: Any) -> None:
        super().__init__(model=model, api_key="ollama", **kwargs)

    def supports_json_mode(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Mistral provider (native mistralai SDK)
# ---------------------------------------------------------------------------

class MistralProvider(LLMProvider):
    """
    Mistral AI via the official mistralai SDK.

    Install: pip install mistralai
    """

    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        try:
            from mistralai.client import Mistral  # mistralai>=1.0 (namespace pkg in v2)
        except ImportError as e:
            raise ImportError("pip install 'mistralai>=1.0'") from e

        key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        # The Mistral SDK default read timeout is ~5 s - far too short for
        # large payloads.  120 s matches the OpenAI SDK default.
        self._client = Mistral(api_key=key, timeout_ms=int(timeout * 1000))
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True

    def _build_kwargs(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> dict:
        kw: dict[str, Any] = {
            "model": self._model,
            "messages": _lc_to_openai_messages(messages, system_prompt),
            "temperature": temperature if temperature is not None else self._temperature,
        }
        resolved_max = max_tokens if max_tokens is not None else self._max_tokens
        if resolved_max:
            kw["max_tokens"] = resolved_max
        if tools:
            kw["tools"] = _tools_to_openai_functions(tools)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        return kw

    def chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> BaseMessage:
        kw = self._build_kwargs(messages, tools, system_prompt, temperature, max_tokens, json_mode)
        self._log_request(messages, tools, system_prompt)
        response = self._client.chat.complete(**kw)
        result = _openai_msg_to_lc(response.choices[0].message, response.usage)
        self._log_response(result)
        return result

    async def achat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> BaseMessage:
        kw = self._build_kwargs(messages, tools, system_prompt, temperature, max_tokens, json_mode)
        self._log_request(messages, tools, system_prompt)
        response = await self._client.chat.complete_async(**kw)
        result = _openai_msg_to_lc(response.choices[0].message, response.usage)
        self._log_response(result)
        return result

    def stream_chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        kw = self._build_kwargs(messages, tools, system_prompt, kwargs.pop("temperature", 0.0), None, False)
        for chunk in self._client.chat.stream(**kw):
            delta = chunk.data.choices[0].delta if chunk.data.choices else None
            if delta and delta.content:
                yield AIMessageChunk(content=delta.content)

    async def astream_chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessage]:
        kw = self._build_kwargs(messages, tools, system_prompt, kwargs.pop("temperature", 0.0), None, False)
        async for chunk in await self._client.chat.stream_async(**kw):
            delta = chunk.data.choices[0].delta if chunk.data.choices else None
            if delta and delta.content:
                yield AIMessageChunk(content=delta.content)

    def list_models(self) -> list[ModelInfo]:
        _KNOWN = {
            "mistral-large-latest": {"context_window": 128_000, "supports_tools": True},
            "mistral-small-latest": {"context_window": 32_000,  "supports_tools": True},
            "codestral-latest":     {"context_window": 32_000,  "supports_tools": False},
            "open-mistral-nemo":    {"context_window": 128_000, "supports_tools": True},
        }
        return [
            ModelInfo(id=k, supports_streaming=True, supports_json_mode=True, **v)
            for k, v in _KNOWN.items()
        ]


# ---------------------------------------------------------------------------
# Anthropic provider (native anthropic SDK)
# ---------------------------------------------------------------------------

def _lc_to_anthropic_messages(messages: list[BaseMessage]) -> list[dict]:
    """
    Convert LangChain messages to Anthropic's format.

    Key differences from OpenAI:
    - SystemMessage is handled as a top-level field (extracted by caller).
    - ToolMessages must be batched into a single "user" message as content blocks.
    - AIMessages with tool_calls emit "tool_use" content blocks.
    """
    result: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, SystemMessage):
            i += 1  # caller extracts this separately
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": str(msg.content)})
            i += 1
        elif isinstance(msg, AIMessage):
            content: list[dict] = []
            if msg.content:
                content.append({"type": "text", "text": str(msg.content)})
            for tc in msg.tool_calls or []:
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["args"],
                })
            result.append({"role": "assistant", "content": content or str(msg.content)})
            i += 1
        elif isinstance(msg, ToolMessage):
            # Batch consecutive ToolMessages into one user message
            blocks: list[dict] = []
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                tm = messages[i]
                blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tm.tool_call_id,
                    "content": str(tm.content),
                })
                i += 1
            result.append({"role": "user", "content": blocks})
        else:
            i += 1
    return result


def _anthropic_response_to_lc(response: Any) -> AIMessage:
    """Convert an Anthropic Message object to a LangChain AIMessage."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "args": block.input,
                "type": "tool_call",
            })

    usage_metadata = None
    if hasattr(response, "usage") and response.usage:
        inp = getattr(response.usage, "input_tokens", 0) or 0
        out = getattr(response.usage, "output_tokens", 0) or 0
        usage_metadata = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

    return AIMessage(
        content=" ".join(text_parts),
        tool_calls=tool_calls,
        usage_metadata=usage_metadata,
    )


def _tools_to_anthropic(tools: list[NormalizedToolDef]) -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],  # Anthropic uses input_schema
        }
        for t in tools
    ]


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude via the official anthropic SDK.

    Install: pip install anthropic
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic") from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = _anthropic.Anthropic(api_key=key)
        self._async_client = _anthropic.AsyncAnthropic(api_key=key)
        self._model = model
        self._temperature = temperature
        # Anthropic requires max_tokens; use 4096 as a safe default
        self._max_tokens = max_tokens or 4096

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        # Anthropic doesn't have a JSON mode flag; callers instruct via system prompt
        return False

    def _build_kwargs(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict:
        resolved_system = system_prompt
        if not resolved_system:
            system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
            if system_msgs:
                resolved_system = str(system_msgs[-1].content)

        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        kw: dict[str, Any] = {
            "model": self._model,
            "messages": _lc_to_anthropic_messages(non_system),
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        if resolved_system:
            kw["system"] = resolved_system
        if tools:
            kw["tools"] = _tools_to_anthropic(tools)
        return kw

    def chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> BaseMessage:
        kw = self._build_kwargs(messages, tools, system_prompt, temperature, max_tokens)
        self._log_request(messages, tools, system_prompt)
        response = self._client.messages.create(**kw)
        result = _anthropic_response_to_lc(response)
        self._log_response(result)
        return result

    async def achat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> BaseMessage:
        kw = self._build_kwargs(messages, tools, system_prompt, temperature, max_tokens)
        self._log_request(messages, tools, system_prompt)
        response = await self._async_client.messages.create(**kw)
        result = _anthropic_response_to_lc(response)
        self._log_response(result)
        return result

    def stream_chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        kw = self._build_kwargs(messages, tools, system_prompt, kwargs.pop("temperature", 0.0), None)
        with self._client.messages.stream(**kw) as stream:
            for text in stream.text_stream:
                yield AIMessageChunk(content=text)

    async def astream_chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessage]:
        kw = self._build_kwargs(messages, tools, system_prompt, kwargs.pop("temperature", 0.0), None)
        async with self._async_client.messages.stream(**kw) as stream:
            async for text in stream.text_stream:
                yield AIMessageChunk(content=text)

    def list_models(self) -> list[ModelInfo]:
        _KNOWN = {
            "claude-opus-4-7":          {"context_window": 200_000},
            "claude-sonnet-4-6":        {"context_window": 200_000},
            "claude-haiku-4-5-20251001":{"context_window": 200_000},
        }
        return [
            ModelInfo(
                id=k,
                supports_tools=True,
                supports_streaming=True,
                supports_json_mode=False,
                **v,
            )
            for k, v in _KNOWN.items()
        ]


# ---------------------------------------------------------------------------
# LangChain adapter (wraps any BaseChatModel)
# ---------------------------------------------------------------------------

class LangChainProvider(LLMProvider):
    """
    Wraps any LangChain BaseChatModel as an LLMProvider.

    This preserves backward compatibility: callers that already have a
    ChatOpenAI / ChatAnthropic / etc. instance can wrap it here without
    touching the rest of the agent.
    """

    def __init__(self, llm: "BaseChatModel") -> None:
        self._llm = llm

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return False

    def _with_tools(self, tools: list[NormalizedToolDef] | None):
        if not tools:
            return self._llm
        lc_tools = [_normalized_tool_to_lc_schema(t) for t in tools]
        return self._llm.bind_tools(lc_tools)

    def _inject_system(
        self, messages: list[BaseMessage], system_prompt: str | None
    ) -> list[BaseMessage]:
        if not system_prompt:
            return messages
        if messages and isinstance(messages[0], SystemMessage):
            return messages
        return [SystemMessage(content=system_prompt), *messages]

    def chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> BaseMessage:
        msgs = self._inject_system(messages, system_prompt)
        self._log_request(messages, tools, system_prompt)
        result = self._with_tools(tools).invoke(msgs)
        self._log_response(result)
        return result

    async def achat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> BaseMessage:
        msgs = self._inject_system(messages, system_prompt)
        self._log_request(messages, tools, system_prompt)
        result = await self._with_tools(tools).ainvoke(msgs)
        self._log_response(result)
        return result

    def stream_chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        msgs = self._inject_system(messages, system_prompt)
        yield from self._with_tools(tools).stream(msgs)

    async def astream_chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessage]:
        msgs = self._inject_system(messages, system_prompt)
        async for chunk in self._with_tools(tools).astream(msgs):
            yield chunk

    def list_models(self) -> list[ModelInfo]:
        model_name = getattr(self._llm, "model_name", None) or getattr(self._llm, "model", "unknown")
        return [ModelInfo(id=str(model_name), supports_tools=True, supports_streaming=True)]


class AzureOpenAIProvider(LangChainProvider):
    """Azure OpenAI Service using AzureChatOpenAI from langchain-openai.

    Requires three Azure-specific values beyond the base OpenAI config:

    - ``base_url`` / ``AZURE_OPENAI_ENDPOINT`` - e.g.
      ``https://<resource>.openai.azure.com/``
    - ``model`` - the *deployment name* chosen in Azure Portal, not the
      canonical model name (e.g. ``my-gpt4o`` not ``gpt-4o``)
    - ``api_version`` - Azure API version string, e.g. ``2024-02-01``
      (pass as an extra JSON field; forwarded via ``ProviderConfig``'s
      ``extra="allow"``)

    Install: no extra dependency - ``langchain-openai`` is already present.
    Set ``AZURE_OPENAI_API_KEY`` (or ``api_key`` in the config).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        api_version: str = "2024-02-01",
        **kwargs: Any,
    ) -> None:
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:
            raise ImportError("pip install langchain-openai") from e

        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        resolved_endpoint = base_url or os.environ.get("AZURE_OPENAI_ENDPOINT", "")

        llm_kw: dict[str, Any] = {
            "azure_deployment": model,
            "azure_endpoint": resolved_endpoint,
            "api_version": api_version,
            "temperature": temperature,
        }
        if resolved_key:
            llm_kw["api_key"] = resolved_key
        if max_tokens:
            llm_kw["max_tokens"] = max_tokens

        super().__init__(llm=AzureChatOpenAI(**llm_kw))


class ACPProvider(LLMProvider):
    """Agent Client Protocol (ACP) provider over stdio (JSON-RPC 2.0).

    Spawns an ACP-compatible agent binary as a child process and communicates
    via stdin/stdout using newline-delimited JSON-RPC 2.0 messages.  No
    network server is required - the binary is started on demand.

    The protocol follows three mandatory phases per turn:
      1. initialize  - negotiate protocol version
      2. session/new - create a session, receive sessionId
      3. session/prompt - send user message, receive streaming updates + final response

    The provider also handles incoming requests from the agent:
      - session/request_permission -> auto-allow
      - fs/read_text_file / fs/write_text_file -> perform local I/O
      - terminal/create -> run the command and return output

    - ``command`` - binary to spawn, e.g. ``"claude-agent-acp"`` (maps from
      the ``model`` config field).  Accepts a string or a list for extra args.
    - ``cwd`` - working directory reported to the agent (defaults to os.getcwd()).

    Example config::

        {"vendor": "acp", "model": "claude-agent-acp"}

    The binary must be on PATH (e.g. ``npm install -g @agentclientprotocol/claude-agent-acp``).
    """

    _PROTOCOL_VERSION = 1  # integer, as specified by the ACP protocol

    def __init__(
        self,
        command: str | list[str] = "claude-agent-acp",
        cwd: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._command = [command] if isinstance(command, str) else list(command)
        self._cwd = cwd or os.getcwd()

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return False

    # -- MCP bridge ---------------------------------------------------------

    def _mcp_server_entry(self, tools: list[NormalizedToolDef]) -> dict | None:
        """Build the mcpServers list entry that exposes Birdie skill tools and agents.

        The ``tools`` list is partitioned into two groups:

        * **Skill tools** – entries that carry an ``entrypoint`` key
          (SkillTool-derived).  Passed to the subprocess via
          ``BIRDIE_TOOLS_JSON``.
        * **Agent tools** – entries that carry a ``_agent_def`` key
          (produced by :func:`agentdef_to_normalized_def`).  Passed via
          ``BIRDIE_AGENTS_JSON`` so the MCP server can reconstruct and run
          each agent in-process.

        Returns None when there is nothing to expose.
        """
        bridgeable_tools = [t for t in tools if "entrypoint" in t]
        bridgeable_agents = [t for t in tools if "_agent_def" in t]
        if not bridgeable_tools and not bridgeable_agents:
            return None
        tools_json = json.dumps(bridgeable_tools)
        agents_json = json.dumps(bridgeable_agents)
        return {
            "name": "birdie",
            "command": sys.executable,
            "args": ["-m", "birdie.core.acp_mcp_server"],
            "env": [
                {"name": "BIRDIE_TOOLS_JSON", "value": tools_json},
                {"name": "BIRDIE_AGENTS_JSON", "value": agents_json},
            ],
        }

    # -- in-process MCP SSE server (async paths) ----------------------------

    # -- message helpers ----------------------------------------------------

    def _build_conversation_prompt(
        self,
        messages: list[BaseMessage],
        system_prompt: str | None,
    ) -> str:
        """Format the full conversation history as a text prompt for the ACP subprocess."""
        effective_system = system_prompt
        dialogue: list[str] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                if not effective_system:
                    effective_system = str(msg.content)
            elif isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                dialogue.append(f"Human: {content}")
            elif isinstance(msg, AIMessage):
                if isinstance(msg.content, str):
                    text = msg.content
                elif isinstance(msg.content, list):
                    text = " ".join(
                        block.get("text", "")
                        for block in msg.content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                else:
                    text = str(msg.content)
                tool_calls = getattr(msg, "tool_calls", [])
                if text:
                    dialogue.append(f"Assistant: {text}")
                elif tool_calls:
                    tc_strs = [f"[called {tc['name']}({tc['args']})]" for tc in tool_calls]
                    dialogue.append(f"Assistant: {' '.join(tc_strs)}")
            elif isinstance(msg, ToolMessage):
                name = getattr(msg, "name", "tool")
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                dialogue.append(f"[{name} result: {content}]")

        result = "\n\n".join(dialogue)
        if effective_system:
            return f"{effective_system}\n\n{result}"
        return result

    def _prompt_blocks(self, text: str) -> list:
        """Build the ACP content blocks array for session/prompt."""
        return [{"type": "text", "text": text}]

    def _extract_chunk_text(self, params: dict) -> str | None:
        """Extract text from an agent_message_chunk session/update notification."""
        update = params.get("update", {})
        if update.get("sessionUpdate") != "agent_message_chunk":
            return None
        content = update.get("content", {})
        if isinstance(content, dict) and content.get("type") == "text":
            return content.get("text") or None
        return None

    # -- sync low-level -----------------------------------------------------

    def _sync_send(self, stdin: Any, msg: dict) -> None:
        stdin.write((json.dumps(msg) + "\n").encode())
        stdin.flush()

    def _sync_recv(self, stdout: Any) -> dict:
        line = stdout.readline()
        if not line:
            raise EOFError("ACP subprocess closed stdout unexpectedly")
        return json.loads(line.decode())

    def _sync_handle_agent_request(self, stdin: Any, msg: dict, mcp_mode: bool = False) -> None:
        """Respond to an incoming JSON-RPC request from the agent (sync)."""
        method = msg.get("method", "")
        req_id = msg["id"]
        params = msg.get("params", {})

        if mcp_mode and method in ("terminal/create", "fs/read_text_file", "fs/write_text_file"):
            self._sync_send(stdin, {"jsonrpc": "2.0", "id": req_id,
                                    "error": {"code": -32601, "message": f"Use MCP tools instead of {method}"}})
            return

        if method == "session/request_permission":
            self._sync_send(stdin, {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"outcome": {"outcome": "selected", "optionId": "allow"}},
            })

        elif method == "fs/read_text_file":
            try:
                content = Path(params["path"]).read_text()
                self._sync_send(stdin, {"jsonrpc": "2.0", "id": req_id, "result": {"content": content}})
            except Exception as exc:
                self._sync_send(stdin, {"jsonrpc": "2.0", "id": req_id,
                                        "error": {"code": -32000, "message": str(exc)}})

        elif method == "fs/write_text_file":
            try:
                Path(params["path"]).write_text(params.get("content", ""))
                self._sync_send(stdin, {"jsonrpc": "2.0", "id": req_id, "result": {}})
            except Exception as exc:
                self._sync_send(stdin, {"jsonrpc": "2.0", "id": req_id,
                                        "error": {"code": -32000, "message": str(exc)}})

        elif method == "terminal/create":
            try:
                proc = subprocess.run(
                    params.get("command", ""), shell=True, capture_output=True,
                    text=True, cwd=params.get("cwd", self._cwd), timeout=60,
                )
                output = proc.stdout + proc.stderr
                self._sync_send(stdin, {"jsonrpc": "2.0", "id": req_id,
                                        "result": {"terminalId": "term_001", "output": output}})
            except Exception as exc:
                self._sync_send(stdin, {"jsonrpc": "2.0", "id": req_id,
                                        "error": {"code": -32000, "message": str(exc)}})

        else:
            self._sync_send(stdin, {"jsonrpc": "2.0", "id": req_id,
                                    "error": {"code": -32601, "message": f"Method not found: {method}"}})

    def chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        mcp_entry = self._mcp_server_entry(tools) if tools else None
        mcp_mode = mcp_entry is not None
        mcp_servers = [mcp_entry] if mcp_entry else []

        proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            # Phase 1: initialize
            self._sync_send(proc.stdin, {
                "jsonrpc": "2.0", "id": 0, "method": "initialize",
                "params": {
                    "protocolVersion": self._PROTOCOL_VERSION,
                    "clientInfo": {"name": "birdie", "version": "0.2.5"},
                    "clientCapabilities": {},
                },
            })
            self._sync_recv(proc.stdout)  # consume initialize response

            # Phase 2: session/new
            session_params: dict = {"cwd": self._cwd, "mcpServers": mcp_servers}
            if mcp_mode:
                session_params["_meta"] = {"disableBuiltInTools": True}
            self._sync_send(proc.stdin, {
                "jsonrpc": "2.0", "id": 1, "method": "session/new",
                "params": session_params,
            })
            session_resp = self._sync_recv(proc.stdout)
            session_id = session_resp.get("result", {}).get("sessionId", "")

            # Phase 3: session/prompt
            self._sync_send(proc.stdin, {
                "jsonrpc": "2.0", "id": 2, "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": self._prompt_blocks(self._build_conversation_prompt(messages, system_prompt)),
                },
            })

            text_parts: list[str] = []
            while True:
                msg = self._sync_recv(proc.stdout)
                if "id" in msg and msg.get("id") == 2 and "result" in msg:
                    return AIMessage(content="".join(text_parts))
                if "id" in msg and "method" in msg:
                    self._sync_handle_agent_request(proc.stdin, msg, mcp_mode=mcp_mode)
                elif "method" in msg and "id" not in msg:
                    chunk = self._extract_chunk_text(msg.get("params", {}))
                    if chunk:
                        text_parts.append(chunk)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.wait()

    def stream_chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        yield self.chat(messages, tools=tools, system_prompt=system_prompt)

    # -- async low-level ----------------------------------------------------

    async def _async_send(self, stdin: Any, msg: dict) -> None:
        stdin.write((json.dumps(msg) + "\n").encode())
        await stdin.drain()

    async def _async_recv(self, stdout: Any, timeout: float = 300.0) -> dict:
        line = await asyncio.wait_for(stdout.readline(), timeout=timeout)
        if not line:
            raise EOFError("ACP subprocess closed stdout unexpectedly")
        return json.loads(line.decode())

    async def _async_handle_agent_request(self, stdin: Any, msg: dict, mcp_mode: bool = False) -> None:
        """Respond to an incoming JSON-RPC request from the agent (async)."""
        method = msg.get("method", "")
        req_id = msg["id"]
        params = msg.get("params", {})

        if mcp_mode and method in ("terminal/create", "fs/read_text_file", "fs/write_text_file"):
            await self._async_send(stdin, {"jsonrpc": "2.0", "id": req_id,
                                           "error": {"code": -32601, "message": f"Use MCP tools instead of {method}"}})
            return

        if method == "session/request_permission":
            await self._async_send(stdin, {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"outcome": {"outcome": "selected", "optionId": "allow"}},
            })

        elif method == "fs/read_text_file":
            try:
                content = Path(params["path"]).read_text()
                await self._async_send(stdin, {"jsonrpc": "2.0", "id": req_id, "result": {"content": content}})
            except Exception as exc:
                await self._async_send(stdin, {"jsonrpc": "2.0", "id": req_id,
                                               "error": {"code": -32000, "message": str(exc)}})

        elif method == "fs/write_text_file":
            try:
                Path(params["path"]).write_text(params.get("content", ""))
                await self._async_send(stdin, {"jsonrpc": "2.0", "id": req_id, "result": {}})
            except Exception as exc:
                await self._async_send(stdin, {"jsonrpc": "2.0", "id": req_id,
                                               "error": {"code": -32000, "message": str(exc)}})

        elif method == "terminal/create":
            try:
                p = await asyncio.create_subprocess_shell(
                    params.get("command", ""),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=params.get("cwd", self._cwd),
                )
                out, err = await asyncio.wait_for(p.communicate(), timeout=60)
                output = (out or b"").decode() + (err or b"").decode()
                await self._async_send(stdin, {"jsonrpc": "2.0", "id": req_id,
                                               "result": {"terminalId": "term_001", "output": output}})
            except Exception as exc:
                await self._async_send(stdin, {"jsonrpc": "2.0", "id": req_id,
                                               "error": {"code": -32000, "message": str(exc)}})

        else:
            await self._async_send(stdin, {"jsonrpc": "2.0", "id": req_id,
                                           "error": {"code": -32601, "message": f"Method not found: {method}"}})

    async def _async_initialize_session(
        self, proc: Any, mcp_servers: list[dict] | None = None
    ) -> str:
        """Run Phase 1 + Phase 2; return the sessionId."""
        await self._async_send(proc.stdin, {
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {
                "protocolVersion": self._PROTOCOL_VERSION,
                "clientInfo": {"name": "birdie", "version": "0.2.5"},
                "clientCapabilities": {},
            },
        })
        await self._async_recv(proc.stdout, timeout=30)

        session_params: dict = {"cwd": self._cwd, "mcpServers": mcp_servers or []}
        if mcp_servers:
            session_params["_meta"] = {"disableBuiltInTools": True}
        await self._async_send(proc.stdin, {
            "jsonrpc": "2.0", "id": 1, "method": "session/new",
            "params": session_params,
        })
        session_resp = await self._async_recv(proc.stdout, timeout=30)
        return session_resp.get("result", {}).get("sessionId", "")

    async def achat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        mcp_entry = self._mcp_server_entry(tools or [])
        mcp_mode = mcp_entry is not None
        mcp_servers = [mcp_entry] if mcp_entry else []

        proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            session_id = await self._async_initialize_session(proc, mcp_servers=mcp_servers)

            prompt_text = self._build_conversation_prompt(messages, system_prompt)
            log.debug("REQUEST  acp=%s\n  prompt: %s", self._command[0], prompt_text[:2000])
            await self._async_send(proc.stdin, {
                "jsonrpc": "2.0", "id": 2, "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": self._prompt_blocks(prompt_text),
                },
            })

            text_parts: list[str] = []
            while True:
                msg = await self._async_recv(proc.stdout)
                if "id" in msg and msg.get("id") == 2 and "result" in msg:
                    result = AIMessage(content="".join(text_parts))
                    log.debug("RESPONSE  acp=%s\n  content: %s", self._command[0], result.content[:2000])
                    return result
                if "id" in msg and "method" in msg:
                    await self._async_handle_agent_request(proc.stdin, msg, mcp_mode=mcp_mode)
                elif "method" in msg and "id" not in msg:
                    chunk = self._extract_chunk_text(msg.get("params", {}))
                    if chunk:
                        text_parts.append(chunk)
        finally:
            proc.stdin.close()
            await proc.wait()

    async def astream_chat(
        self,
        messages: list[BaseMessage],
        tools: list[NormalizedToolDef] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessage]:
        mcp_entry = self._mcp_server_entry(tools or [])
        mcp_mode = mcp_entry is not None
        mcp_servers = [mcp_entry] if mcp_entry else []

        proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            session_id = await self._async_initialize_session(proc, mcp_servers=mcp_servers)

            await self._async_send(proc.stdin, {
                "jsonrpc": "2.0", "id": 2, "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": self._prompt_blocks(self._build_conversation_prompt(messages, system_prompt)),
                },
            })

            while True:
                msg = await self._async_recv(proc.stdout)
                if "id" in msg and msg.get("id") == 2 and "result" in msg:
                    break
                if "id" in msg and "method" in msg:
                    method = msg.get("method", "")
                    req_id = msg["id"]
                    params = msg.get("params", {})
                    if mcp_mode and method in ("terminal/create", "fs/read_text_file", "fs/write_text_file"):
                        # MCP mode: reject built-in callbacks so the subprocess uses MCP tools
                        await self._async_send(proc.stdin, {
                            "jsonrpc": "2.0", "id": req_id,
                            "error": {"code": -32601, "message": f"Use MCP tools instead of {method}"},
                        })
                    elif not mcp_mode and method == "terminal/create":
                        # Legacy mode: handle inline and yield visibility chunk
                        cmd = params.get("command", "")
                        cwd_param = params.get("cwd", self._cwd)
                        try:
                            p = await asyncio.create_subprocess_shell(
                                cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                cwd=cwd_param,
                            )
                            out, err = await asyncio.wait_for(p.communicate(), timeout=60)
                            output = (out or b"").decode() + (err or b"").decode()
                            await self._async_send(proc.stdin, {
                                "jsonrpc": "2.0", "id": req_id,
                                "result": {"terminalId": "term_001", "output": output},
                            })
                            yield AIMessageChunk(content=f"run_bash(command={cmd!r})\n{output}\n")
                        except Exception as exc:
                            await self._async_send(proc.stdin, {
                                "jsonrpc": "2.0", "id": req_id,
                                "error": {"code": -32000, "message": str(exc)},
                            })
                    elif not mcp_mode and method == "fs/read_text_file":
                        # Legacy mode: handle inline and yield visibility chunk
                        path_str = params.get("path", "")
                        try:
                            content = Path(path_str).read_text()
                            await self._async_send(proc.stdin, {
                                "jsonrpc": "2.0", "id": req_id, "result": {"content": content},
                            })
                            yield AIMessageChunk(content=f"read_file({path_str!r})\n")
                        except Exception as exc:
                            await self._async_send(proc.stdin, {
                                "jsonrpc": "2.0", "id": req_id,
                                "error": {"code": -32000, "message": str(exc)},
                            })
                    else:
                        await self._async_handle_agent_request(proc.stdin, msg)
                elif "method" in msg and "id" not in msg:
                    chunk = self._extract_chunk_text(msg.get("params", {}))
                    if chunk:
                        yield AIMessageChunk(content=chunk)
        finally:
            proc.stdin.close()
            await proc.wait()

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=self._command[0], supports_streaming=True)]


def _normalized_tool_to_lc_schema(t: NormalizedToolDef):
    """
    Create a schema-only LangChain StructuredTool from a NormalizedToolDef.

    Used by LangChainProvider to bind tool schemas to the LLM.  Execution
    happens separately in the ToolNode via skilltool_to_langchain_tool().
    """
    from langchain_core.tools import StructuredTool
    from birdie.core.adapter import create_args_schema

    def _noop(**kwargs: Any) -> None:
        raise NotImplementedError("This is a schema-only tool; execution is handled by ToolNode")

    return StructuredTool.from_function(
        func=_noop,
        name=t["name"],
        description=t["description"],
        args_schema=create_args_schema(t["parameters"]),
    )


# ---------------------------------------------------------------------------
# Utility: convert SkillTool → NormalizedToolDef
# ---------------------------------------------------------------------------

def skilltool_to_normalized_def(skill_tool: Any) -> NormalizedToolDef:
    """Convert a SkillTool Pydantic model to a NormalizedToolDef dict."""
    d: NormalizedToolDef = {
        "name": skill_tool.name,
        "description": skill_tool.description,
        "parameters": skill_tool.schema,
    }
    if skill_tool.entrypoint:
        d["entrypoint"] = skill_tool.entrypoint
    return d


def lc_tool_to_normalized_def(tool: Any) -> NormalizedToolDef:
    """Convert a LangChain BaseTool (e.g. from MCP) to a NormalizedToolDef dict."""
    args_schema = tool.args_schema
    if args_schema is None:
        schema: dict = {"type": "object", "properties": {}}
    elif isinstance(args_schema, dict):
        # MCP tools provide args_schema as a plain JSON Schema dict
        schema = dict(args_schema)
    else:
        # Pydantic model class (StructuredTool pattern)
        schema = args_schema.model_json_schema()
    schema.pop("title", None)
    schema.pop("$defs", None)
    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": schema,
    }


def agentdef_to_normalized_def(
    agent_def: Any,
    provider_config: dict | None = None,
    skills_dir: str = "skills",
    agents_dir: str | None = None,
) -> NormalizedToolDef:
    """Convert an AgentDef to a NormalizedToolDef for use with ACPProvider.

    The returned dict carries an ``"_agent_def"`` key (the serialised AgentDef)
    plus execution-context keys (``_provider_config``, ``_skills_dir``,
    ``_agents_dir``) so that ``acp_mcp_server.py`` can reconstruct and run the
    agent without any extra state.

    The ``parameters`` field is built from the agent's ``input_params`` so the
    ACP subprocess knows the tool's JSON Schema and can validate arguments.

    Args:
        agent_def: A parsed ``AgentDef`` Pydantic model.
        provider_config: Full provider config dict from the parent agent
            (vendor, api_key, base_url, …).  Forwarded to the ephemeral
            DynamicAgent so it uses the same LLM backend.
        skills_dir: Skills directory forwarded to the ephemeral DynamicAgent.
        agents_dir: Agents directory forwarded to the ephemeral DynamicAgent.

    Returns:
        A ``NormalizedToolDef`` dict that ``_mcp_server_entry`` can serialise
        into ``BIRDIE_AGENTS_JSON``.
    """
    _TYPE_MAP = {
        "string": "string", "integer": "integer", "number": "number",
        "boolean": "boolean", "array": "array", "object": "object",
    }
    properties: dict = {}
    required: list = []
    for p in agent_def.input_params:
        properties[p.name] = {
            "type": _TYPE_MAP.get(p.type, "string"),
            "description": p.description,
        }
        if p.required:
            required.append(p.name)
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    d: NormalizedToolDef = {
        "name": agent_def.name,
        "description": agent_def.description,
        "parameters": schema,
        # Execution-context metadata consumed by acp_mcp_server._invoke_agent()
        "_agent_def": agent_def.model_dump(),
        "_provider_config": provider_config or {},
        "_skills_dir": skills_dir,
        "_agents_dir": agents_dir,
    }
    return d


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def get_llm_provider(
    config: Union[dict, str, Path, "ProviderConfig"],
) -> LLMProvider:
    """
    Instantiate the correct LLMProvider from any JSON-compatible source.

    Accepted input types
    --------------------
    ``dict``
        Plain Python dict - the original interface, still supported::

            get_llm_provider({"vendor": "openai", "model": "gpt-4o"})

    ``str``
        Raw JSON string::

            get_llm_provider('{"vendor":"anthropic","model":"claude-sonnet-4-6"}')

    ``pathlib.Path`` or path-like string ending in ``.json``
        Path to a JSON config file::

            get_llm_provider(Path("provider.json"))

    ``ProviderConfig``
        Pre-validated Pydantic model::

            get_llm_provider(ProviderConfig(vendor="mistral"))

    Config fields
    -------------
    See :class:`ProviderConfig` for the full field reference.
    Extra fields are forwarded as ``**kwargs`` to the provider constructor.

    The ``vendor`` field also reads from the ``LLM_VENDOR`` environment
    variable when omitted from the config.
    """
    # -- normalise to ProviderConfig ----------------------------------------
    if isinstance(config, ProviderConfig):
        cfg = config
    elif isinstance(config, str):
        try:
            cfg = ProviderConfig.model_validate_json(config)
        except Exception:
            # Treat as a file path if JSON parsing fails
            cfg = ProviderConfig.from_file(config)
    elif isinstance(config, Path):
        cfg = ProviderConfig.from_file(config)
    elif isinstance(config, dict):
        # Pull the non-JSON "llm" key out before Pydantic sees the dict,
        # as BaseChatModel instances cannot be validated by Pydantic.
        llm_obj = config.pop("llm", None)
        cfg = ProviderConfig.model_validate(config)
        if llm_obj is not None:
            config["llm"] = llm_obj  # restore caller's dict
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    vendor = cfg.vendor.lower()

    # Extra fields defined in JSON beyond the standard ProviderConfig fields
    extra_kw: dict[str, Any] = cfg.model_extra or {}

    # -- dispatch -----------------------------------------------------------
    if vendor == "openai":
        kw: dict[str, Any] = {"temperature": cfg.temperature}
        if cfg.model:      kw["model"]     = cfg.model
        if cfg.api_key:    kw["api_key"]   = cfg.api_key
        if cfg.base_url:   kw["base_url"]  = cfg.base_url
        if cfg.max_tokens: kw["max_tokens"] = cfg.max_tokens
        return OpenAIProvider(**kw, **extra_kw)

    if vendor == "anthropic":
        kw = {"temperature": cfg.temperature}
        if cfg.model:      kw["model"]     = cfg.model
        if cfg.api_key:    kw["api_key"]   = cfg.api_key
        if cfg.max_tokens: kw["max_tokens"] = cfg.max_tokens
        return AnthropicProvider(**kw, **extra_kw)

    if vendor == "mistral":
        kw = {"temperature": cfg.temperature}
        if cfg.model:      kw["model"]     = cfg.model
        if cfg.api_key:    kw["api_key"]   = cfg.api_key
        if cfg.max_tokens: kw["max_tokens"] = cfg.max_tokens
        return MistralProvider(**kw, **extra_kw)

    if vendor == "azure":
        kw = {"temperature": cfg.temperature}
        if cfg.model:      kw["model"]     = cfg.model
        if cfg.api_key:    kw["api_key"]   = cfg.api_key
        if cfg.base_url:   kw["base_url"]  = cfg.base_url
        if cfg.max_tokens: kw["max_tokens"] = cfg.max_tokens
        return AzureOpenAIProvider(**kw, **extra_kw)

    if vendor == "gemini":
        kw = {"temperature": cfg.temperature}
        if cfg.model:      kw["model"]     = cfg.model
        if cfg.api_key:    kw["api_key"]   = cfg.api_key
        if cfg.base_url:   kw["base_url"]  = cfg.base_url
        if cfg.max_tokens: kw["max_tokens"] = cfg.max_tokens
        return GeminiProvider(**kw, **extra_kw)

    if vendor == "ollama":
        kw = {"temperature": cfg.temperature}
        if cfg.model:    kw["model"]    = cfg.model
        if cfg.base_url: kw["base_url"] = cfg.base_url
        return OllamaProvider(**kw, **extra_kw)

    if vendor == "langchain":
        # "llm" can't be stored in ProviderConfig (not JSON-serialisable);
        # accept it via extra_kw when the caller passes a dict.
        llm_obj = extra_kw.pop("llm", None)
        if llm_obj is None and isinstance(config, dict):
            llm_obj = config.get("llm")
        if llm_obj is None:
            raise ValueError(
                "vendor='langchain' requires an 'llm' key containing a "
                "pre-built BaseChatModel instance"
            )
        return LangChainProvider(llm_obj)

    if vendor == "acp":
        kw: dict[str, Any] = {}
        if cfg.model: kw["command"] = cfg.model
        return ACPProvider(**kw, **extra_kw)

    raise ValueError(
        f"Unknown vendor '{vendor}'. "
        "Supported: openai, azure, anthropic, mistral, gemini, ollama, langchain, acp"
    )


def get_llm_provider_from_json(json_str: str) -> LLMProvider:
    """Parse a JSON string and return the configured provider."""
    return get_llm_provider(json_str)


def get_llm_provider_from_file(path: Union[str, Path]) -> LLMProvider:
    """Load provider configuration from a JSON file."""
    return get_llm_provider(Path(path))
