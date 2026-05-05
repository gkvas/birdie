"""
Tests for the vendor-agnostic LLM provider layer.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import json
import tempfile
from pathlib import Path

from birdie.core.llm_provider import (
    NormalizedToolDef,
    ModelInfo,
    ProviderConfig,
    LangChainProvider,
    LLMProvider,
    _lc_to_openai_messages,
    _openai_msg_to_lc,
    _lc_to_anthropic_messages,
    _anthropic_response_to_lc,
    _tools_to_openai_functions,
    _tools_to_anthropic,
    skilltool_to_normalized_def,
    get_llm_provider,
    get_llm_provider_from_json,
    get_llm_provider_from_file,
)
from birdie.core.models import SkillTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tools() -> list[NormalizedToolDef]:
    return [
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


@pytest.fixture
def sample_messages() -> list:
    return [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="What is the weather in Graz?"),
    ]


# ---------------------------------------------------------------------------
# Message conversion — OpenAI format
# ---------------------------------------------------------------------------

class TestOpenAIMessageConversion:
    def test_human_message(self):
        result = _lc_to_openai_messages([HumanMessage(content="hello")])
        assert result == [{"role": "user", "content": "hello"}]

    def test_system_message_from_list(self):
        result = _lc_to_openai_messages([SystemMessage(content="sys"), HumanMessage(content="hi")])
        assert result[0] == {"role": "system", "content": "sys"}

    def test_system_prompt_param_takes_priority(self):
        result = _lc_to_openai_messages(
            [HumanMessage(content="hi")],
            system_prompt="override",
        )
        assert result[0] == {"role": "system", "content": "override"}
        assert result[1] == {"role": "user", "content": "hi"}

    def test_ai_message_no_tools(self):
        result = _lc_to_openai_messages([AIMessage(content="pong")])
        assert result == [{"role": "assistant", "content": "pong"}]

    def test_ai_message_with_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "get_weather", "args": {"city": "Graz"}, "type": "tool_call"}],
        )
        result = _lc_to_openai_messages([msg])
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "c1"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Graz"}

    def test_tool_message(self):
        msg = ToolMessage(content="sunny", tool_call_id="c1")
        result = _lc_to_openai_messages([msg])
        assert result == [{"role": "tool", "tool_call_id": "c1", "content": "sunny"}]


class TestOpenAIResponseConversion:
    def test_plain_text_response(self):
        raw = {"content": "Hello", "tool_calls": None}
        msg = _openai_msg_to_lc(raw)
        assert isinstance(msg, AIMessage)
        assert msg.content == "Hello"
        assert msg.tool_calls == []

    def test_tool_call_response(self):
        raw = {
            "content": "",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Graz"}'},
                }
            ],
        }
        msg = _openai_msg_to_lc(raw)
        assert msg.tool_calls[0]["name"] == "get_weather"
        assert msg.tool_calls[0]["args"] == {"city": "Graz"}


class TestToolDefConversion:
    def test_openai_tool_format(self, sample_tools):
        result = _tools_to_openai_functions(sample_tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert "parameters" in result[0]["function"]

    def test_anthropic_tool_format(self, sample_tools):
        result = _tools_to_anthropic(sample_tools)
        assert result[0]["name"] == "get_weather"
        assert "input_schema" in result[0]  # Anthropic uses input_schema, not parameters
        assert "parameters" not in result[0]


# ---------------------------------------------------------------------------
# Message conversion — Anthropic format
# ---------------------------------------------------------------------------

class TestAnthropicMessageConversion:
    def test_human_message(self):
        result = _lc_to_anthropic_messages([HumanMessage(content="hi")])
        assert result == [{"role": "user", "content": "hi"}]

    def test_system_message_excluded(self):
        # SystemMessage is handled as a top-level Anthropic field, not a message
        result = _lc_to_anthropic_messages([SystemMessage(content="sys"), HumanMessage(content="hi")])
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_ai_message_with_tool_calls(self):
        msg = AIMessage(
            content="calling tool",
            tool_calls=[{"id": "tc1", "name": "get_weather", "args": {"city": "Graz"}, "type": "tool_call"}],
        )
        result = _lc_to_anthropic_messages([msg])
        content = result[0]["content"]
        assert any(b["type"] == "text" for b in content)
        tool_block = next(b for b in content if b["type"] == "tool_use")
        assert tool_block["id"] == "tc1"
        assert tool_block["name"] == "get_weather"
        assert tool_block["input"] == {"city": "Graz"}

    def test_tool_messages_batched_into_single_user_turn(self):
        msgs = [
            ToolMessage(content="sunny", tool_call_id="tc1"),
            ToolMessage(content="22°C", tool_call_id="tc2"),
        ]
        result = _lc_to_anthropic_messages(msgs)
        # Both ToolMessages must be merged into ONE user turn
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert all(b["type"] == "tool_result" for b in result[0]["content"])


class TestAnthropicResponseConversion:
    def test_text_only_response(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello")]
        msg = _anthropic_response_to_lc(mock_response)
        assert isinstance(msg, AIMessage)
        assert "Hello" in msg.content
        assert msg.tool_calls == []

    def test_tool_use_response(self):
        mock_block = MagicMock()
        mock_block.type = "tool_use"
        mock_block.id = "toolu_01"
        mock_block.name = "get_weather"
        mock_block.input = {"city": "Graz"}

        mock_response = MagicMock()
        mock_response.content = [mock_block]

        msg = _anthropic_response_to_lc(mock_response)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"
        assert msg.tool_calls[0]["args"] == {"city": "Graz"}


# ---------------------------------------------------------------------------
# SkillTool → NormalizedToolDef
# ---------------------------------------------------------------------------

class TestSkillToolNormalization:
    def test_basic_conversion(self):
        tool = SkillTool(
            name="read_file",
            description="Read a file",
            entrypoint="bash:cat {path}",
            schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )
        normalized = skilltool_to_normalized_def(tool)
        assert normalized["name"] == "read_file"
        assert normalized["description"] == "Read a file"
        assert "properties" in normalized["parameters"]


# ---------------------------------------------------------------------------
# LangChainProvider
# ---------------------------------------------------------------------------

class TestLangChainProvider:
    def _make_mock_llm(self, response_content="Hello", tool_calls=None):
        ai_msg = AIMessage(content=response_content, tool_calls=tool_calls or [])
        llm = MagicMock()
        llm.invoke.return_value = ai_msg
        llm.ainvoke = AsyncMock(return_value=ai_msg)
        bound = MagicMock()
        bound.invoke.return_value = ai_msg
        bound.ainvoke = AsyncMock(return_value=ai_msg)
        llm.bind_tools.return_value = bound
        return llm

    def test_chat_no_tools(self, sample_messages):
        llm = self._make_mock_llm("Paris is sunny")
        provider = LangChainProvider(llm)
        result = provider.chat(sample_messages)
        assert isinstance(result, AIMessage)
        assert result.content == "Paris is sunny"
        llm.bind_tools.assert_not_called()

    def test_chat_with_tools_binds_schema(self, sample_messages, sample_tools):
        llm = self._make_mock_llm()
        provider = LangChainProvider(llm)
        provider.chat(sample_messages, tools=sample_tools)
        llm.bind_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_achat_no_tools(self, sample_messages):
        llm = self._make_mock_llm("async response")
        provider = LangChainProvider(llm)
        result = await provider.achat(sample_messages)
        assert result.content == "async response"

    def test_system_prompt_injected(self, sample_tools):
        llm = self._make_mock_llm()
        provider = LangChainProvider(llm)
        msgs = [HumanMessage(content="hi")]
        provider.chat(msgs, system_prompt="Be concise")
        # The injected system message should be the first element passed to invoke
        call_args = llm.invoke.call_args[0][0]
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == "Be concise"

    def test_no_duplicate_system_message(self):
        llm = self._make_mock_llm()
        provider = LangChainProvider(llm)
        msgs = [SystemMessage(content="existing"), HumanMessage(content="hi")]
        provider.chat(msgs, system_prompt="override ignored")
        # Already has SystemMessage → don't prepend another one
        call_args = llm.invoke.call_args[0][0]
        system_msgs = [m for m in call_args if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1

    def test_list_models(self):
        llm = MagicMock()
        llm.model_name = "gpt-4o"
        provider = LangChainProvider(llm)
        models = provider.list_models()
        assert models[0].id == "gpt-4o"

    def test_capability_flags(self):
        provider = LangChainProvider(MagicMock())
        assert provider.supports_tools() is True
        assert provider.supports_streaming() is True


# ---------------------------------------------------------------------------
# AzureOpenAIProvider
# ---------------------------------------------------------------------------

class TestAzureOpenAIProvider:
    def _make_mock_azure_llm(self, response_content="Hello"):
        ai_msg = AIMessage(content=response_content, tool_calls=[])
        llm = MagicMock()
        llm.invoke.return_value = ai_msg
        llm.model_name = "my-gpt4o-deployment"
        bound = MagicMock()
        bound.invoke.return_value = ai_msg
        llm.bind_tools.return_value = bound
        return llm

    @patch("langchain_openai.AzureChatOpenAI")
    def test_init_passes_correct_params(self, MockAzureChatOpenAI):
        from birdie.core.llm_provider import AzureOpenAIProvider
        MockAzureChatOpenAI.return_value = self._make_mock_azure_llm()
        AzureOpenAIProvider(
            model="my-gpt4o-deployment",
            api_key="azure-key",
            base_url="https://my-resource.openai.azure.com/",
            api_version="2024-02-01",
            temperature=0.5,
        )
        MockAzureChatOpenAI.assert_called_once_with(
            azure_deployment="my-gpt4o-deployment",
            azure_endpoint="https://my-resource.openai.azure.com/",
            api_version="2024-02-01",
            temperature=0.5,
            api_key="azure-key",
        )

    @patch("langchain_openai.AzureChatOpenAI")
    def test_tools_use_bind_tools(self, MockAzureChatOpenAI, sample_messages, sample_tools):
        from birdie.core.llm_provider import AzureOpenAIProvider
        mock_llm = self._make_mock_azure_llm()
        MockAzureChatOpenAI.return_value = mock_llm
        provider = AzureOpenAIProvider(
            model="my-gpt4o-deployment",
            base_url="https://my-resource.openai.azure.com/",
        )
        provider.chat(sample_messages, tools=sample_tools)
        mock_llm.bind_tools.assert_called_once()

    @patch("langchain_openai.AzureChatOpenAI")
    def test_env_var_fallback(self, MockAzureChatOpenAI, monkeypatch):
        from birdie.core.llm_provider import AzureOpenAIProvider
        MockAzureChatOpenAI.return_value = self._make_mock_azure_llm()
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env-resource.openai.azure.com/")
        AzureOpenAIProvider(model="my-deployment")
        call_kw = MockAzureChatOpenAI.call_args[1]
        assert call_kw["api_key"] == "env-key"
        assert call_kw["azure_endpoint"] == "https://env-resource.openai.azure.com/"


# ---------------------------------------------------------------------------
# ACPProvider
# ---------------------------------------------------------------------------

class TestACPProvider:

    def _make_proc(self, *response_lines):
        """Return a mock Popen whose stdout yields initialize + session/new + response lines."""
        init_resp = json.dumps({
            "jsonrpc": "2.0", "id": 0,
            "result": {"protocolVersion": 1, "agentInfo": {"name": "test-agent", "version": "1.0.0"}, "agentCapabilities": {}},
        }) + "\n"
        session_resp = json.dumps({
            "jsonrpc": "2.0", "id": 1, "result": {"sessionId": "sess_test123"},
        }) + "\n"
        lines = [init_resp.encode(), session_resp.encode()] + [
            (json.dumps(r) + "\n").encode() for r in response_lines
        ]
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stdout.readline.side_effect = lines
        mock_proc.wait.return_value = 0
        return mock_proc

    def _chunk_notification(self, text):
        return {"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": "sess_test123",
            "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}},
        }}

    def _prompt_result(self):
        return {"jsonrpc": "2.0", "id": 2, "result": {"stopReason": "end_turn"}}

    def test_chat(self, sample_messages):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(self._chunk_notification("Hi there"), self._prompt_result())
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            result = provider.chat(sample_messages)
        assert isinstance(result, AIMessage)
        assert result.content == "Hi there"

    def test_chat_sends_correct_rpc(self, sample_messages):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(self._prompt_result())
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.chat(sample_messages, system_prompt="Be helpful")
        calls = mock_proc.stdin.write.call_args_list
        # calls[0]=initialize, calls[1]=session/new, calls[2]=session/prompt
        assert json.loads(calls[0][0][0].decode())["method"] == "initialize"
        assert json.loads(calls[1][0][0].decode())["method"] == "session/new"
        prompt_msg = json.loads(calls[2][0][0].decode())
        assert prompt_msg["method"] == "session/prompt"
        blocks = prompt_msg["params"]["prompt"]
        text = blocks[0]["text"]
        assert "Be helpful" in text

    def test_chat_sends_last_human_message(self):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(self._prompt_result())
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.chat(
                [HumanMessage(content="Hello"), AIMessage(content="Hi"), HumanMessage(content="Follow up")],
                system_prompt="Be helpful",
            )
        calls = mock_proc.stdin.write.call_args_list
        prompt_msg = json.loads(calls[2][0][0].decode())
        blocks = prompt_msg["params"]["prompt"]
        text = blocks[0]["text"]
        assert "Follow up" in text
        assert "Be helpful" in text

    def test_list_models(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        assert provider.list_models()[0].id == "claude-agent-acp"

    def test_capability_flags(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        assert provider.supports_tools() is False
        assert provider.supports_streaming() is True
        assert provider.supports_json_mode() is False

    def test_extract_chunk_text(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        params = {"update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "chunk"}}}
        assert provider._extract_chunk_text(params) == "chunk"
        assert provider._extract_chunk_text({}) is None
        params2 = {"update": {"sessionUpdate": "tool_call_update"}}
        assert provider._extract_chunk_text(params2) is None

    @patch("birdie.core.llm_provider.ACPProvider.__init__", return_value=None)
    def test_acp_vendor_factory(self, mock_init):
        get_llm_provider({"vendor": "acp", "model": "claude-agent-acp"})
        mock_init.assert_called_once_with(command="claude-agent-acp")


# ---------------------------------------------------------------------------
# get_llm_provider factory
# ---------------------------------------------------------------------------

class TestProviderConfig:
    """ProviderConfig validation and JSON round-tripping."""

    def test_defaults(self):
        cfg = ProviderConfig()
        assert cfg.vendor == "openai"
        assert cfg.model is None
        assert cfg.temperature == 0.0
        assert cfg.max_tokens is None

    def test_from_dict(self):
        cfg = ProviderConfig.model_validate({"vendor": "anthropic", "model": "claude-sonnet-4-6"})
        assert cfg.vendor == "anthropic"
        assert cfg.model == "claude-sonnet-4-6"

    def test_from_json_string(self):
        cfg = ProviderConfig.from_json('{"vendor":"mistral","temperature":0.7}')
        assert cfg.vendor == "mistral"
        assert cfg.temperature == 0.7

    def test_from_file(self, tmp_path):
        p = tmp_path / "provider.json"
        p.write_text('{"vendor":"gemini","model":"gemini-2.0-flash","temperature":0.5}')
        cfg = ProviderConfig.from_file(p)
        assert cfg.vendor == "gemini"
        assert cfg.temperature == 0.5

    def test_to_json_excludes_none(self):
        cfg = ProviderConfig(vendor="openai", model="gpt-4o")
        data = json.loads(cfg.to_json())
        assert "model" in data
        assert "api_key" not in data  # excluded because None

    def test_to_json_roundtrip(self):
        original = ProviderConfig(vendor="mistral", model="mistral-large-latest", temperature=0.3)
        restored = ProviderConfig.from_json(original.to_json())
        assert restored.vendor == original.vendor
        assert restored.model == original.model
        assert restored.temperature == original.temperature

    def test_extra_fields_allowed(self):
        cfg = ProviderConfig.model_validate({"vendor": "openai", "seed": 42})
        assert cfg.model_extra["seed"] == 42

    def test_temperature_bounds(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ProviderConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            ProviderConfig(temperature=2.1)

    def test_max_tokens_positive(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ProviderConfig(max_tokens=0)


class TestGetLLMProvider:
    def test_langchain_vendor(self):
        mock_llm = MagicMock()
        provider = get_llm_provider({"vendor": "langchain", "llm": mock_llm})
        assert isinstance(provider, LangChainProvider)

    def test_langchain_vendor_missing_llm_raises(self):
        with pytest.raises(ValueError, match="'llm' key"):
            get_llm_provider({"vendor": "langchain"})

    def test_unknown_vendor_raises(self):
        with pytest.raises(ValueError, match="Unknown vendor"):
            get_llm_provider({"vendor": "acme_llm"})

    @patch("birdie.core.llm_provider.OpenAIProvider.__init__", return_value=None)
    def test_openai_vendor(self, mock_init):
        get_llm_provider({"vendor": "openai", "model": "gpt-4o", "api_key": "sk-test"})
        mock_init.assert_called_once_with(model="gpt-4o", api_key="sk-test", temperature=0.0)

    @patch("birdie.core.llm_provider.AnthropicProvider.__init__", return_value=None)
    def test_anthropic_vendor(self, mock_init):
        get_llm_provider({"vendor": "anthropic", "model": "claude-sonnet-4-6", "api_key": "sk-ant"})
        mock_init.assert_called_once_with(
            model="claude-sonnet-4-6", api_key="sk-ant", temperature=0.0
        )

    @patch("birdie.core.llm_provider.MistralProvider.__init__", return_value=None)
    def test_mistral_vendor(self, mock_init):
        get_llm_provider({"vendor": "mistral", "model": "mistral-large-latest", "api_key": "sk-m"})
        mock_init.assert_called_once_with(
            model="mistral-large-latest", api_key="sk-m", temperature=0.0
        )

    @patch("birdie.core.llm_provider.GeminiProvider.__init__", return_value=None)
    def test_gemini_vendor(self, mock_init):
        get_llm_provider({"vendor": "gemini", "model": "gemini-2.0-flash", "api_key": "AIza_test"})
        mock_init.assert_called_once_with(model="gemini-2.0-flash", api_key="AIza_test", temperature=0.0)

    @patch("birdie.core.llm_provider.AzureOpenAIProvider.__init__", return_value=None)
    def test_azure_vendor(self, mock_init):
        get_llm_provider({
            "vendor": "azure",
            "model": "my-gpt4o-deployment",
            "api_key": "azure-key",
            "base_url": "https://my-resource.openai.azure.com/",
            "api_version": "2024-02-01",
        })
        mock_init.assert_called_once_with(
            model="my-gpt4o-deployment",
            api_key="azure-key",
            base_url="https://my-resource.openai.azure.com/",
            temperature=0.0,
            api_version="2024-02-01",
        )

    # -- JSON input forms ---------------------------------------------------

    @patch("birdie.core.llm_provider.OpenAIProvider.__init__", return_value=None)
    def test_accepts_json_string(self, mock_init):
        get_llm_provider('{"vendor":"openai","model":"gpt-4o","api_key":"sk-test"}')
        mock_init.assert_called_once_with(model="gpt-4o", api_key="sk-test", temperature=0.0)

    @patch("birdie.core.llm_provider.AnthropicProvider.__init__", return_value=None)
    def test_accepts_json_file(self, mock_init, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text('{"vendor":"anthropic","model":"claude-sonnet-4-6","api_key":"sk-ant"}')
        get_llm_provider(p)
        mock_init.assert_called_once_with(
            model="claude-sonnet-4-6", api_key="sk-ant", temperature=0.0
        )

    @patch("birdie.core.llm_provider.MistralProvider.__init__", return_value=None)
    def test_accepts_provider_config_object(self, mock_init):
        cfg = ProviderConfig(vendor="mistral", model="mistral-large-latest", api_key="sk-m")
        get_llm_provider(cfg)
        mock_init.assert_called_once_with(
            model="mistral-large-latest", api_key="sk-m", temperature=0.0
        )

    @patch("birdie.core.llm_provider.OpenAIProvider.__init__", return_value=None)
    def test_temperature_forwarded(self, mock_init):
        get_llm_provider({"vendor": "openai", "temperature": 0.9})
        _, kw = mock_init.call_args
        assert kw.get("temperature") == 0.9

    @patch("birdie.core.llm_provider.OpenAIProvider.__init__", return_value=None)
    def test_max_tokens_forwarded(self, mock_init):
        get_llm_provider({"vendor": "openai", "max_tokens": 512})
        _, kw = mock_init.call_args
        assert kw.get("max_tokens") == 512

    @patch("birdie.core.llm_provider.OpenAIProvider.__init__", return_value=None)
    def test_extra_fields_forwarded(self, mock_init):
        get_llm_provider({"vendor": "openai", "seed": 42})
        _, kw = mock_init.call_args
        assert kw.get("seed") == 42

    @patch("birdie.core.llm_provider.OpenAIProvider.__init__", return_value=None)
    def test_get_llm_provider_from_json_helper(self, mock_init):
        get_llm_provider_from_json('{"vendor":"openai","model":"gpt-4o"}')
        mock_init.assert_called_once()

    @patch("birdie.core.llm_provider.OpenAIProvider.__init__", return_value=None)
    def test_get_llm_provider_from_file_helper(self, mock_init, tmp_path):
        p = tmp_path / "c.json"
        p.write_text('{"vendor":"openai","model":"gpt-4o"}')
        get_llm_provider_from_file(p)
        mock_init.assert_called_once()


class TestFromConfigEnvVars:
    """DynamicAgent.from_config env-var overrides."""

    @patch("birdie.core.llm_provider.OpenAIProvider.__init__", return_value=None)
    def test_llm_vendor_env_overrides_dict(self, mock_init, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_VENDOR", "openai")
        monkeypatch.delenv("LLM_PROVIDER_CONFIG", raising=False)
        from birdie.agent.run import DynamicAgent
        DynamicAgent.from_config({"vendor": "anthropic"}, skills_dir=str(tmp_path))
        mock_init.assert_called_once()

    @patch("birdie.core.llm_provider.MistralProvider.__init__", return_value=None)
    def test_llm_provider_config_env_overrides_all(self, mock_init, monkeypatch, tmp_path):
        monkeypatch.setenv(
            "LLM_PROVIDER_CONFIG",
            '{"vendor":"mistral","model":"mistral-large-latest"}',
        )
        from birdie.agent.run import DynamicAgent
        DynamicAgent.from_config({"vendor": "openai"}, skills_dir=str(tmp_path))
        mock_init.assert_called_once_with(
            model="mistral-large-latest", temperature=0.0
        )


# ---------------------------------------------------------------------------
# DynamicAgent with LLMProvider (via LangChainProvider wrapper)
# ---------------------------------------------------------------------------

class TestAgentWithProvider:
    @pytest.mark.asyncio
    async def test_agent_uses_provider_achat(self, tmp_path):
        """DynamicAgent routes every LLM call through provider.achat()."""
        import tempfile, os
        from birdie.agent.run import DynamicAgent

        # Create a minimal skill
        skill_dir = tmp_path / "EchoSkill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.MD").write_text("""---
name: EchoSkill
version: 1.0.0
description: Echo skill
tags: []
enabled_by_default: true
---
## Tools
### echo
description: Echo a message
entrypoint: python:tests.test_integration.echo_tool
schema:
  type: object
  properties:
    message:
      type: string
  required: [message]
""")

        call_log = []

        class TrackingProvider(LLMProvider):
            def __init__(self): self.call_count = 0
            def supports_tools(self): return True
            def supports_streaming(self): return False
            def supports_json_mode(self): return False

            def chat(self, messages, tools=None, **kw):
                raise NotImplementedError

            async def achat(self, messages, tools=None, **kw):
                self.call_count += 1
                call_log.append({"tools": tools, "num_messages": len(messages)})
                return AIMessage(content="done")

            def stream_chat(self, messages, **kw):
                raise NotImplementedError

            async def astream_chat(self, messages, **kw):
                raise NotImplementedError

            def list_models(self): return []

        provider = TrackingProvider()
        agent = DynamicAgent(provider, skills_dir=str(tmp_path))
        await agent.invoke("test")

        assert provider.call_count == 1
        # Provider should have received the EchoSkill's tool
        assert call_log[0]["tools"] is not None
        assert call_log[0]["tools"][0]["name"] == "echo"

    @pytest.mark.asyncio
    async def test_from_config_uses_env_var(self, tmp_path, monkeypatch):
        """from_config respects LLM_VENDOR env override."""
        monkeypatch.setenv("LLM_VENDOR", "langchain")

        mock_llm = MagicMock()
        ai_msg = AIMessage(content="ok")
        mock_llm.ainvoke = AsyncMock(return_value=ai_msg)
        mock_llm.invoke.return_value = ai_msg

        from birdie.agent.run import DynamicAgent
        agent = DynamicAgent.from_config(
            {"vendor": "openai", "llm": mock_llm},  # vendor overridden by env
            skills_dir=str(tmp_path),
        )
        assert isinstance(agent.provider, LangChainProvider)
