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
    _anthropic_accepts_temperature,
    _is_temperature_rejection,
    AnthropicProvider,
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

    def test_ai_message_with_tool_calls_keeps_text_content(self):
        msg = AIMessage(
            content="Checking the weather now.",
            tool_calls=[{"id": "c1", "name": "get_weather", "args": {"city": "Graz"}, "type": "tool_call"}],
        )
        result = _lc_to_openai_messages([msg])
        assert result[0]["content"] == "Checking the weather now."
        assert result[0]["tool_calls"][0]["id"] == "c1"

    def test_ai_message_with_tool_calls_omits_empty_content(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "get_weather", "args": {}, "type": "tool_call"}],
        )
        result = _lc_to_openai_messages([msg])
        assert "content" not in result[0]

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
# Anthropic temperature handling
# ---------------------------------------------------------------------------

def _make_anthropic_provider(model: str) -> AnthropicProvider:
    """Build an AnthropicProvider without touching the anthropic SDK."""
    p = AnthropicProvider.__new__(AnthropicProvider)
    p._model = model
    p._temperature = 0.3
    p._max_tokens = 4096
    p._send_temperature = _anthropic_accepts_temperature(model)
    p._prompt_cache = True
    p._client = MagicMock()
    p._async_client = MagicMock()
    return p


def _anthropic_400(message: str) -> Exception:
    exc = Exception(f"Error code: 400 - {message}")
    exc.status_code = 400
    return exc


class TestAnthropicTemperature:
    @pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"])
    def test_older_models_accept_temperature(self, model):
        assert _anthropic_accepts_temperature(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-fable-5",
            "claude-opus-5",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-5",
            "anthropic.claude-opus-5",
        ],
    )
    def test_new_models_reject_temperature(self, model):
        assert _anthropic_accepts_temperature(model) is False

    def test_temperature_sent_for_supported_model(self):
        p = _make_anthropic_provider("claude-sonnet-4-6")
        kw = p._build_kwargs([HumanMessage(content="hi")], None, None, None, None)
        assert kw["temperature"] == 0.3

    def test_temperature_omitted_for_unsupported_model(self):
        p = _make_anthropic_provider("claude-fable-5")
        kw = p._build_kwargs([HumanMessage(content="hi")], None, None, 0.7, None)
        assert "temperature" not in kw

    def test_chat_retries_without_temperature_on_400(self):
        p = _make_anthropic_provider("claude-sonnet-4-6")
        ok = MagicMock()
        ok.content = [MagicMock(type="text", text="Hello")]
        p._client.messages.create.side_effect = [
            _anthropic_400("`temperature` is deprecated for this model."),
            ok,
        ]

        msg = p.chat([HumanMessage(content="hi")])

        assert "Hello" in msg.content
        assert p._client.messages.create.call_count == 2
        assert "temperature" in p._client.messages.create.call_args_list[0].kwargs
        assert "temperature" not in p._client.messages.create.call_args_list[1].kwargs
        # The provider remembers, so later calls skip temperature entirely
        assert p._send_temperature is False

    def test_chat_reraises_unrelated_400(self):
        p = _make_anthropic_provider("claude-sonnet-4-6")
        p._client.messages.create.side_effect = _anthropic_400("max_tokens is required")
        with pytest.raises(Exception, match="max_tokens"):
            p.chat([HumanMessage(content="hi")])
        assert p._client.messages.create.call_count == 1

    def test_rejection_detector_ignores_non_400(self):
        exc = Exception("temperature something")
        exc.status_code = 500
        assert _is_temperature_rejection(exc) is False


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
        assert normalized["entrypoint"] == "bash:cat {path}"


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

    def _make_proc(self, *response_lines, session_result=None):
        """Return a mock Popen whose stdout yields initialize + session/new + response lines."""
        init_resp = json.dumps({
            "jsonrpc": "2.0", "id": 0,
            "result": {"protocolVersion": 1, "agentInfo": {"name": "test-agent", "version": "1.0.0"}, "agentCapabilities": {}},
        }) + "\n"
        session_resp = json.dumps({
            "jsonrpc": "2.0", "id": 1, "result": session_result or {"sessionId": "sess_test123"},
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

    def test_chat_captures_model_name(self, sample_messages):
        from birdie.core.llm_provider import ACPProvider
        session_result = {
            "sessionId": "sess_test123",
            "models": {"availableModels": [], "currentModelId": "claude-fable-5"},
        }
        mock_proc = self._make_proc(self._prompt_result(), session_result=session_result)
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            assert provider.model_name == "unknown"
            provider.chat(sample_messages)
        assert provider.model_name == "claude-fable-5"

    def test_chat_without_model_info_keeps_unknown(self, sample_messages):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(self._prompt_result())
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.chat(sample_messages)
        assert provider.model_name == "unknown"

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

    def test_chat_sends_full_conversation_history(self):
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
        assert "Hello" in text
        assert "Hi" in text

    def test_list_models(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        assert provider.list_models()[0].id == "claude-agent-acp"

    def test_capability_flags(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        assert provider.supports_tools() is True
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

    def _tool_call_notification(self, tool_id="call_1", title="List files", status="pending", raw_input=None):
        update = {
            "sessionUpdate": "tool_call", "toolCallId": tool_id,
            "title": title, "kind": "execute", "status": status,
        }
        if raw_input is not None:
            update["rawInput"] = raw_input
        return {"jsonrpc": "2.0", "method": "session/update",
                "params": {"sessionId": "sess_test123", "update": update}}

    def _tool_update_notification(self, tool_id="call_1", status="completed", output=None):
        update = {"sessionUpdate": "tool_call_update", "toolCallId": tool_id, "status": status}
        if output is not None:
            update["content"] = [{"type": "content", "content": {"type": "text", "text": output}}]
        return {"jsonrpc": "2.0", "method": "session/update",
                "params": {"sessionId": "sess_test123", "update": update}}

    def test_extract_tool_event(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        params = self._tool_call_notification(raw_input={"command": "ls -la"})["params"]
        event = provider._extract_tool_event(params)
        assert event["event"] == "tool_call"
        assert event["id"] == "call_1"
        assert event["title"] == "List files"
        assert event["kind"] == "execute"
        assert event["status"] == "pending"
        assert event["raw_input"] == {"command": "ls -la"}
        assert event["output"] == ""

    def test_extract_tool_event_collects_output(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        params = self._tool_update_notification(output="file1.py\nfile2.py")["params"]
        params["update"]["content"].append({"type": "diff", "path": "/tmp/x.py"})
        event = provider._extract_tool_event(params)
        assert event["event"] == "tool_call_update"
        assert event["status"] == "completed"
        assert event["output"] == "file1.py\nfile2.py\n[diff] /tmp/x.py"

    def test_extract_tool_event_ignores_other_updates(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        chunk_params = self._chunk_notification("hi")["params"]
        assert provider._extract_tool_event(chunk_params) is None
        assert provider._extract_tool_event({}) is None

    def test_chat_fires_tool_event_callback(self, sample_messages):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(
            self._tool_call_notification(raw_input={"command": "ls"}),
            self._tool_update_notification(output="file1.py"),
            self._chunk_notification("Done"),
            self._prompt_result(),
        )
        events = []
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.tool_event_callback = events.append
            result = provider.chat(sample_messages)
        assert result.content == "Done"
        assert [e["event"] for e in events] == ["tool_call", "tool_call_update"]
        assert events[0]["title"] == "List files"
        assert events[1]["output"] == "file1.py"

    def test_chat_callback_exception_does_not_break_turn(self, sample_messages):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(
            self._tool_call_notification(),
            self._chunk_notification("Done"),
            self._prompt_result(),
        )

        def boom(event):
            raise RuntimeError("render failed")

        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.tool_event_callback = boom
            result = provider.chat(sample_messages)
        assert result.content == "Done"

    def _usage_notification(self, used=None, size=None, cost=None):
        update = {"sessionUpdate": "usage_update"}
        if used is not None:
            update["used"] = used
        if size is not None:
            update["size"] = size
        if cost is not None:
            update["cost"] = cost
        return {"jsonrpc": "2.0", "method": "session/update",
                "params": {"sessionId": "sess_test123", "update": update}}

    def test_extract_usage(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        params = self._usage_notification(
            used=53_000, size=200_000, cost={"amount": 0.045, "currency": "USD"},
        )["params"]
        usage = provider._extract_usage(params)
        assert usage == {"used": 53_000, "size": 200_000,
                         "cost": {"amount": 0.045, "currency": "USD"}}
        assert provider._extract_usage(self._chunk_notification("hi")["params"]) is None
        assert provider._extract_usage({}) is None

    def test_chat_attaches_usage_metadata(self, sample_messages):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(
            self._chunk_notification("Done"),
            self._usage_notification(used=41_000, size=200_000),
            self._usage_notification(used=42_000, cost={"amount": 0.05, "currency": "USD"}),
            self._prompt_result(),
        )
        events = []
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.tool_event_callback = events.append
            result = provider.chat(sample_messages)
        assert result.content == "Done"
        assert result.usage_metadata["input_tokens"] == 42_000
        assert result.usage_metadata["output_tokens"] == 0
        assert result.usage_metadata["total_tokens"] == 42_000
        assert result.response_metadata["context_window"] == 200_000
        assert result.response_metadata["cost"] == {"amount": 0.05, "currency": "USD"}
        # usage updates must not be routed to the tool event callback
        assert events == []

    def test_chat_without_usage_has_no_usage_metadata(self, sample_messages):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(self._chunk_notification("Hi"), self._prompt_result())
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            result = provider.chat(sample_messages)
        assert result.usage_metadata is None

    @pytest.mark.asyncio
    async def test_achat_attaches_usage_metadata(self):
        from birdie.core.llm_provider import ACPProvider

        init_resp = json.dumps({
            "jsonrpc": "2.0", "id": 0,
            "result": {"protocolVersion": 1, "agentInfo": {"name": "t", "version": "0"}, "agentCapabilities": {}},
        }).encode() + b"\n"
        session_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"sessionId": "s1"}}).encode() + b"\n"
        chunk_msg = json.dumps(self._chunk_notification("Done")).encode() + b"\n"
        usage_msg = json.dumps(self._usage_notification(
            used=12_345, size=200_000, cost={"amount": 0.01, "currency": "USD"},
        )).encode() + b"\n"
        final = json.dumps(self._prompt_result()).encode() + b"\n"

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()
        mock_stdin.close = MagicMock()

        read_lines = [init_resp, session_resp, chunk_msg, usage_msg, final]
        read_idx = 0

        async def fake_read(n=-1):
            nonlocal read_idx
            if read_idx < len(read_lines):
                line = read_lines[read_idx]
                read_idx += 1
                return line
            return b""

        mock_stdout = AsyncMock()
        mock_stdout.read = fake_read

        mock_proc = AsyncMock()
        mock_proc.stdin = mock_stdin
        mock_proc.stdout = mock_stdout
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            result = await provider.achat([HumanMessage(content="List files")])

        assert result.content == "Done"
        assert result.usage_metadata["input_tokens"] == 12_345
        assert result.response_metadata["context_window"] == 200_000
        assert result.response_metadata["cost"] == {"amount": 0.01, "currency": "USD"}

    def _permission_request(self, req_id=10, options=None, title="Bash: ls"):
        if options is None:
            options = [
                {"kind": "allow_always", "name": "Always allow Bash", "optionId": "allow_always"},
                {"kind": "allow_once", "name": "Allow", "optionId": "allow"},
                {"kind": "reject_once", "name": "Reject", "optionId": "reject"},
            ]
        return {"jsonrpc": "2.0", "id": req_id, "method": "session/request_permission",
                "params": {"sessionId": "sess_test123",
                           "toolCall": {"toolCallId": "call_1", "title": title,
                                        "kind": "execute", "rawInput": {"command": "ls"}},
                           "options": options}}

    def test_normalize_permission_request(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        request = provider._normalize_permission_request(self._permission_request()["params"])
        assert request["title"] == "Bash: ls"
        assert request["kind"] == "execute"
        assert request["raw_input"] == {"command": "ls"}
        assert request["options"][0] == {
            "id": "allow_always", "name": "Always allow Bash", "kind": "allow_always"}

    def test_permission_outcome_mapping(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        params = self._permission_request()["params"]
        assert provider._permission_outcome(params, "allow") == {
            "outcome": {"outcome": "selected", "optionId": "allow"}}
        assert provider._permission_outcome(params, "allow_always")["outcome"]["optionId"] == "allow_always"
        assert provider._permission_outcome(params, "deny")["outcome"]["optionId"] == "reject"
        # no options offered: fall back to the legacy hard-coded ids
        assert provider._permission_outcome({}, "allow")["outcome"]["optionId"] == "allow"
        assert provider._permission_outcome({}, "deny")["outcome"]["optionId"] == "reject"

    def _permission_response(self, mock_proc, idx=3):
        return json.loads(mock_proc.stdin.write.call_args_list[idx][0][0].decode())

    def test_chat_auto_allows_permission_without_callback(self, sample_messages):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(
            self._permission_request(),
            self._chunk_notification("Done"),
            self._prompt_result(),
        )
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            result = provider.chat(sample_messages)
        assert result.content == "Done"
        resp = self._permission_response(mock_proc)
        assert resp["id"] == 10
        assert resp["result"]["outcome"] == {"outcome": "selected", "optionId": "allow"}

    def test_chat_permission_callback_denies(self, sample_messages):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(
            self._permission_request(),
            self._chunk_notification("Done"),
            self._prompt_result(),
        )
        seen = []
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.permission_callback = lambda req: (seen.append(req), "deny")[1]
            result = provider.chat(sample_messages)
        assert result.content == "Done"
        assert seen[0]["title"] == "Bash: ls"
        resp = self._permission_response(mock_proc)
        assert resp["result"]["outcome"]["optionId"] == "reject"

    def test_chat_permission_callback_error_fails_closed(self, sample_messages):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(
            self._permission_request(),
            self._chunk_notification("Done"),
            self._prompt_result(),
        )

        def boom(req):
            raise RuntimeError("gate crashed")

        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.permission_callback = boom
            result = provider.chat(sample_messages)
        assert result.content == "Done"
        resp = self._permission_response(mock_proc)
        assert resp["result"]["outcome"]["optionId"] == "reject"

    @pytest.mark.asyncio
    async def test_achat_awaits_async_permission_callback(self):
        from birdie.core.llm_provider import ACPProvider

        init_resp = json.dumps({
            "jsonrpc": "2.0", "id": 0,
            "result": {"protocolVersion": 1, "agentInfo": {"name": "t", "version": "0"}, "agentCapabilities": {}},
        }).encode() + b"\n"
        session_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"sessionId": "s1"}}).encode() + b"\n"
        perm_msg = json.dumps(self._permission_request()).encode() + b"\n"
        chunk_msg = json.dumps(self._chunk_notification("Done")).encode() + b"\n"
        final = json.dumps(self._prompt_result()).encode() + b"\n"

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()
        mock_stdin.close = MagicMock()

        read_lines = [init_resp, session_resp, perm_msg, chunk_msg, final]
        read_idx = 0

        async def fake_read(n=-1):
            nonlocal read_idx
            if read_idx < len(read_lines):
                line = read_lines[read_idx]
                read_idx += 1
                return line
            return b""

        mock_stdout = AsyncMock()
        mock_stdout.read = fake_read

        mock_proc = AsyncMock()
        mock_proc.stdin = mock_stdin
        mock_proc.stdout = mock_stdout
        mock_proc.wait = AsyncMock(return_value=0)

        async def gate(request):
            assert request["title"] == "Bash: ls"
            return "allow_always"

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.permission_callback = gate
            result = await provider.achat([HumanMessage(content="List files")])

        assert result.content == "Done"
        resp = json.loads(mock_stdin.write.call_args_list[3][0][0].decode())
        assert resp["id"] == 10
        assert resp["result"]["outcome"]["optionId"] == "allow_always"

    @pytest.mark.asyncio
    async def test_achat_fires_tool_event_callback(self):
        from birdie.core.llm_provider import ACPProvider

        init_resp = json.dumps({
            "jsonrpc": "2.0", "id": 0,
            "result": {"protocolVersion": 1, "agentInfo": {"name": "t", "version": "0"}, "agentCapabilities": {}},
        }).encode() + b"\n"
        session_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"sessionId": "s1"}}).encode() + b"\n"
        tool_call = json.dumps(self._tool_call_notification(raw_input={"command": "ls"})).encode() + b"\n"
        tool_update = json.dumps(self._tool_update_notification(output="file1.py")).encode() + b"\n"
        chunk_msg = json.dumps(self._chunk_notification("Done")).encode() + b"\n"
        final = json.dumps(self._prompt_result()).encode() + b"\n"

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()
        mock_stdin.close = MagicMock()

        read_lines = [init_resp, session_resp, tool_call, tool_update, chunk_msg, final]
        read_idx = 0

        async def fake_read(n=-1):
            nonlocal read_idx
            if read_idx < len(read_lines):
                line = read_lines[read_idx]
                read_idx += 1
                return line
            return b""

        mock_stdout = AsyncMock()
        mock_stdout.read = fake_read

        mock_proc = AsyncMock()
        mock_proc.stdin = mock_stdin
        mock_proc.stdout = mock_stdout
        mock_proc.wait = AsyncMock(return_value=0)

        events = []
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.tool_event_callback = events.append
            result = await provider.achat([HumanMessage(content="List files")])

        assert result.content == "Done"
        assert [e["event"] for e in events] == ["tool_call", "tool_call_update"]
        assert events[0]["raw_input"] == {"command": "ls"}
        assert events[1]["output"] == "file1.py"

    def test_chat_includes_tool_messages_in_history(self):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(self._prompt_result())
        messages = [
            HumanMessage(content="List files"),
            AIMessage(content="", tool_calls=[{"name": "run_bash", "args": {"command": "ls"}, "id": "tc1"}]),
            ToolMessage(content="file1.py\nfile2.py", tool_call_id="tc1", name="run_bash"),
            HumanMessage(content="Which is larger?"),
        ]
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.chat(messages, system_prompt="Be helpful")
        calls = mock_proc.stdin.write.call_args_list
        prompt_msg = json.loads(calls[2][0][0].decode())
        text = prompt_msg["params"]["prompt"][0]["text"]
        assert "List files" in text
        assert "Which is larger?" in text
        assert "run_bash" in text
        assert "file1.py" in text

    @pytest.mark.asyncio
    async def test_astream_chat_yields_terminal_call(self):
        from birdie.core.llm_provider import ACPProvider
        from langchain_core.messages import AIMessageChunk

        terminal_request = {
            "jsonrpc": "2.0", "id": 10, "method": "terminal/create",
            "params": {"command": "ls -la", "cwd": "/tmp"},
        }
        terminal_response = {
            "jsonrpc": "2.0", "id": 10,
            "result": {"terminalId": "term_001", "output": "file.txt\n"},
        }

        init_resp = json.dumps({
            "jsonrpc": "2.0", "id": 0,
            "result": {"protocolVersion": 1, "agentInfo": {"name": "t", "version": "0"}, "agentCapabilities": {}},
        }).encode() + b"\n"
        session_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"sessionId": "s1"}}).encode() + b"\n"
        chunk_msg = json.dumps(self._chunk_notification("Here are the files:")).encode() + b"\n"
        term_req = json.dumps(terminal_request).encode() + b"\n"
        final = json.dumps(self._prompt_result()).encode() + b"\n"

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        read_lines = [init_resp, session_resp, term_req, chunk_msg, final]
        read_idx = 0

        async def fake_read(n=-1):
            nonlocal read_idx
            if read_idx < len(read_lines):
                line = read_lines[read_idx]
                read_idx += 1
                return line
            return b""

        mock_stdout = AsyncMock()
        mock_stdout.read = fake_read

        mock_proc = AsyncMock()
        mock_proc.stdin = mock_stdin
        mock_proc.stdout = mock_stdout
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch("asyncio.create_subprocess_shell") as mock_shell:
            mock_shell_proc = AsyncMock()
            mock_shell_proc.communicate = AsyncMock(return_value=(b"file.txt\n", b""))
            mock_shell.return_value = mock_shell_proc

            provider = ACPProvider(command="claude-agent-acp")
            chunks = [c async for c in provider.astream_chat([HumanMessage(content="List files")])]

        content = "".join(c.content for c in chunks)
        assert "run_bash" in content
        assert "ls -la" in content
        assert "Here are the files:" in content

    def test_mcp_server_entry_with_entrypoint_tools(self):
        from birdie.core.llm_provider import ACPProvider
        import sys
        provider = ACPProvider(command="claude-agent-acp")
        tools = [
            {"name": "run_bash", "description": "Run a bash command", "parameters": {}, "entrypoint": "bash:{command}"},
            {"name": "search", "description": "Search the web", "parameters": {}, "entrypoint": "python:birdie.skills.ddg.run"},
        ]
        entry = provider._mcp_server_entry(tools)
        assert entry is not None
        assert entry["name"] == "birdie"
        assert entry["command"] == sys.executable
        assert "-m" in entry["args"]
        assert "birdie.core.acp_mcp_server" in entry["args"]
        env_dict = {e["name"]: e["value"] for e in entry["env"]}
        decoded = json.loads(env_dict["BIRDIE_TOOLS_JSON"])
        assert len(decoded) == 2
        assert decoded[0]["entrypoint"] == "bash:{command}"

    def test_mcp_server_entry_without_entrypoint_tools(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        tools = [
            {"name": "mcp_tool", "description": "An MCP tool", "parameters": {}},
        ]
        entry = provider._mcp_server_entry(tools)
        assert entry is None

    def test_chat_with_tools_sends_mcp_server_entry(self):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(self._prompt_result())
        tools = [{"name": "run_bash", "description": "run", "parameters": {}, "entrypoint": "bash:{command}"}]
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.chat([HumanMessage(content="hi")], tools=tools)
        calls = mock_proc.stdin.write.call_args_list
        session_new = json.loads(calls[1][0][0].decode())
        assert session_new["method"] == "session/new"
        servers = session_new["params"]["mcpServers"]
        assert len(servers) == 1
        assert servers[0]["name"] == "birdie"
        assert session_new["params"]["_meta"]["disableBuiltInTools"] is True

    def test_chat_without_tools_sends_empty_mcp_servers(self):
        from birdie.core.llm_provider import ACPProvider
        mock_proc = self._make_proc(self._prompt_result())
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.chat([HumanMessage(content="hi")])
        calls = mock_proc.stdin.write.call_args_list
        session_new = json.loads(calls[1][0][0].decode())
        assert session_new["params"]["mcpServers"] == []
        assert "_meta" not in session_new["params"]

    def test_chat_mcp_mode_rejects_terminal_create(self):
        from birdie.core.llm_provider import ACPProvider
        terminal_req = {
            "jsonrpc": "2.0", "id": 5, "method": "terminal/create",
            "params": {"command": "ls"},
        }
        mock_proc = self._make_proc(terminal_req, self._prompt_result())
        tools = [{"name": "run_bash", "description": "run", "parameters": {}, "entrypoint": "bash:{command}"}]
        with patch("subprocess.Popen", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            provider.chat([HumanMessage(content="hi")], tools=tools)
        written = [json.loads(c[0][0].decode()) for c in mock_proc.stdin.write.call_args_list]
        responses = [m for m in written if m.get("id") == 5]
        assert len(responses) == 1
        assert "error" in responses[0]
        assert responses[0]["error"]["code"] == -32601

    def _make_async_proc(self, *response_lines, session_result=None):
        """Build a mock async ACP subprocess whose stdout yields the given JSON lines."""
        init_resp = json.dumps({
            "jsonrpc": "2.0", "id": 0,
            "result": {"protocolVersion": 1, "agentInfo": {"name": "t", "version": "0"}, "agentCapabilities": {}},
        }).encode() + b"\n"
        session_resp = json.dumps({
            "jsonrpc": "2.0", "id": 1, "result": session_result or {"sessionId": "s_async"},
        }).encode() + b"\n"
        lines = [init_resp, session_resp] + [(json.dumps(r) + "\n").encode() for r in response_lines]
        idx = 0

        async def read(n=-1):
            nonlocal idx
            if idx < len(lines):
                line = lines[idx]; idx += 1; return line
            return b""

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()
        mock_stdout = AsyncMock()
        mock_stdout.read = read
        mock_proc = AsyncMock()
        mock_proc.stdin = mock_stdin
        mock_proc.stdout = mock_stdout
        mock_proc.wait = AsyncMock(return_value=0)
        return mock_proc

    @pytest.mark.asyncio
    async def test_achat_with_tools_sends_mcp_entry(self):
        import sys
        from birdie.core.llm_provider import ACPProvider
        tools = [{"name": "run_bash", "description": "run", "parameters": {}, "entrypoint": "bash:{command}"}]
        mock_proc = self._make_async_proc(self._prompt_result())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            await provider.achat([HumanMessage(content="hi")], tools=tools)

        written = [json.loads(c[0][0].decode()) for c in mock_proc.stdin.write.call_args_list]
        session_new = next(m for m in written if m.get("method") == "session/new")
        servers = session_new["params"]["mcpServers"]
        assert len(servers) == 1
        entry = servers[0]
        assert entry["name"] == "birdie"
        assert entry["command"] == sys.executable
        assert "-m" in entry["args"]
        assert "birdie.core.acp_mcp_server" in entry["args"]
        assert any(e["name"] == "BIRDIE_TOOLS_JSON" for e in entry["env"])

    @pytest.mark.asyncio
    async def test_achat_captures_model_name(self):
        from birdie.core.llm_provider import ACPProvider
        session_result = {
            "sessionId": "s_async",
            "models": {"availableModels": [], "currentModelId": "claude-fable-5"},
        }
        mock_proc = self._make_async_proc(self._prompt_result(), session_result=session_result)
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            await provider.achat([HumanMessage(content="hi")])
        assert provider.model_name == "claude-fable-5"

    @pytest.mark.asyncio
    async def test_astream_chat_with_tools_rejects_terminal_create(self):
        from birdie.core.llm_provider import ACPProvider
        terminal_req = {"jsonrpc": "2.0", "id": 7, "method": "terminal/create", "params": {"command": "ls"}}
        tools = [{"name": "run_bash", "description": "run", "parameters": {}, "entrypoint": "bash:{command}"}]
        mock_proc = self._make_async_proc(terminal_req, self._prompt_result())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            provider = ACPProvider(command="claude-agent-acp")
            chunks = [c async for c in provider.astream_chat([HumanMessage(content="hi")], tools=tools)]

        written = [json.loads(c[0][0].decode()) for c in mock_proc.stdin.write.call_args_list]
        rejection = next((m for m in written if m.get("id") == 7), None)
        assert rejection is not None
        assert "error" in rejection
        assert rejection["error"]["code"] == -32601

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
        agent = DynamicAgent(provider, skills_dir=str(tmp_path), skills_enabled=["EchoSkill"])
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


# ---------------------------------------------------------------------------
# agentdef_to_normalized_def
# ---------------------------------------------------------------------------

class TestAgentdefToNormalizedDef:
    """Unit tests for the agentdef_to_normalized_def() utility."""

    def _make_agent_def(self, name="Summarizer", input_params=None):
        from birdie.core.models import AgentDef, AgentParam
        params = input_params or [
            AgentParam(name="text", type="string", description="Text to summarize", required=True),
            AgentParam(name="max_points", type="integer", description="Max bullet points", required=False),
        ]
        return AgentDef(
            name=name,
            description="Summarizes text into bullet points",
            prompt="Summarize: {{ text }}",
            input_params=params,
        )

    def test_basic_fields(self):
        from birdie.core.llm_provider import agentdef_to_normalized_def
        agent_def = self._make_agent_def()
        result = agentdef_to_normalized_def(agent_def)

        assert result["name"] == "Summarizer"
        assert result["description"] == "Summarizes text into bullet points"

    def test_parameters_schema_built_from_input_params(self):
        from birdie.core.llm_provider import agentdef_to_normalized_def
        agent_def = self._make_agent_def()
        result = agentdef_to_normalized_def(agent_def)

        schema = result["parameters"]
        assert schema["type"] == "object"
        assert "text" in schema["properties"]
        assert schema["properties"]["text"]["type"] == "string"
        assert "max_points" in schema["properties"]
        assert schema["properties"]["max_points"]["type"] == "integer"
        # Only required params appear in the required list
        assert "text" in schema["required"]
        assert "max_points" not in schema.get("required", [])

    def test_no_required_params_omits_required_key(self):
        from birdie.core.llm_provider import agentdef_to_normalized_def
        from birdie.core.models import AgentDef, AgentParam
        agent_def = AgentDef(
            name="NoReq",
            description="No required params",
            prompt="do it",
            input_params=[AgentParam(name="opt", type="string", description="optional", required=False)],
        )
        result = agentdef_to_normalized_def(agent_def)
        assert "required" not in result["parameters"]

    def test_agent_def_embedded_in_result(self):
        from birdie.core.llm_provider import agentdef_to_normalized_def
        agent_def = self._make_agent_def()
        result = agentdef_to_normalized_def(agent_def)

        assert "_agent_def" in result
        assert result["_agent_def"]["name"] == "Summarizer"
        assert result["_agent_def"]["prompt"] == "Summarize: {{ text }}"

    def test_provider_config_forwarded(self):
        from birdie.core.llm_provider import agentdef_to_normalized_def
        agent_def = self._make_agent_def()
        cfg = {"vendor": "anthropic", "model": "claude-haiku-4-5-20251001", "api_key": "sk-test"}
        result = agentdef_to_normalized_def(agent_def, provider_config=cfg)

        # api_key must never be serialised into the tool def (it lands in the
        # MCP subprocess environment); everything else is forwarded.
        assert result["_provider_config"] == {
            "vendor": "anthropic", "model": "claude-haiku-4-5-20251001",
        }

    def test_skills_dir_and_agents_dir_forwarded(self):
        from birdie.core.llm_provider import agentdef_to_normalized_def
        agent_def = self._make_agent_def()
        result = agentdef_to_normalized_def(
            agent_def, skills_dir="/custom/skills", agents_dir="/custom/agents"
        )
        assert result["_skills_dir"] == "/custom/skills"
        assert result["_agents_dir"] == "/custom/agents"

    def test_defaults_when_no_provider_config(self):
        from birdie.core.llm_provider import agentdef_to_normalized_def
        agent_def = self._make_agent_def()
        result = agentdef_to_normalized_def(agent_def)

        assert result["_provider_config"] == {}
        assert result["_skills_dir"] == "skills"
        assert result["_agents_dir"] is None

    def test_no_entrypoint_key(self):
        """Agent defs must NOT have an entrypoint key (that's for skill tools)."""
        from birdie.core.llm_provider import agentdef_to_normalized_def
        agent_def = self._make_agent_def()
        result = agentdef_to_normalized_def(agent_def)
        assert "entrypoint" not in result

    def test_empty_input_params(self):
        from birdie.core.llm_provider import agentdef_to_normalized_def
        from birdie.core.models import AgentDef
        agent_def = AgentDef(name="Empty", description="No params", prompt="go", input_params=[])
        result = agentdef_to_normalized_def(agent_def)
        assert result["parameters"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# ACPProvider._mcp_server_entry with agent defs
# ---------------------------------------------------------------------------

class TestACPProviderMcpServerEntryWithAgents:
    """Tests for _mcp_server_entry when agent NormalizedToolDefs are present."""

    def _make_agent_normalized(self, name="Summarizer"):
        from birdie.core.llm_provider import agentdef_to_normalized_def
        from birdie.core.models import AgentDef, AgentParam
        agent_def = AgentDef(
            name=name,
            description=f"Agent {name}",
            prompt="do {{ task }}",
            input_params=[AgentParam(name="task", type="string", description="task", required=True)],
        )
        return agentdef_to_normalized_def(agent_def, provider_config={"vendor": "openai"})

    def _make_skill_tool(self, name="search"):
        return {
            "name": name,
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            "entrypoint": "python:birdie.skills.duckduckgo.tools.search",
        }

    def test_agent_only_produces_entry(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        agent_tool = self._make_agent_normalized()
        entry = provider._mcp_server_entry([agent_tool])

        assert entry is not None
        assert entry["name"] == "birdie"
        assert "birdie.core.acp_mcp_server" in entry["args"]

    def test_agent_goes_into_birdie_agents_json(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        agent_tool = self._make_agent_normalized("Summarizer")
        entry = provider._mcp_server_entry([agent_tool])

        env_dict = {e["name"]: e["value"] for e in entry["env"]}
        agents = json.loads(env_dict["BIRDIE_AGENTS_JSON"])
        assert len(agents) == 1
        assert agents[0]["name"] == "Summarizer"
        assert "_agent_def" in agents[0]
        assert agents[0]["_agent_def"]["name"] == "Summarizer"

    def test_agent_not_in_birdie_tools_json(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        agent_tool = self._make_agent_normalized()
        entry = provider._mcp_server_entry([agent_tool])

        env_dict = {e["name"]: e["value"] for e in entry["env"]}
        tools = json.loads(env_dict["BIRDIE_TOOLS_JSON"])
        assert tools == []

    def test_skill_tool_goes_into_birdie_tools_json(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        skill_tool = self._make_skill_tool("search")
        entry = provider._mcp_server_entry([skill_tool])

        env_dict = {e["name"]: e["value"] for e in entry["env"]}
        tools = json.loads(env_dict["BIRDIE_TOOLS_JSON"])
        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    def test_mixed_tools_and_agents_both_partitioned(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        skill_tool = self._make_skill_tool("search")
        agent_tool = self._make_agent_normalized("Summarizer")
        entry = provider._mcp_server_entry([skill_tool, agent_tool])

        assert entry is not None
        env_dict = {e["name"]: e["value"] for e in entry["env"]}

        tools = json.loads(env_dict["BIRDIE_TOOLS_JSON"])
        agents = json.loads(env_dict["BIRDIE_AGENTS_JSON"])

        assert len(tools) == 1 and tools[0]["name"] == "search"
        assert len(agents) == 1 and agents[0]["name"] == "Summarizer"

    def test_multiple_agents_all_included(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        tools = [self._make_agent_normalized(n) for n in ["AgentA", "AgentB", "AgentC"]]
        entry = provider._mcp_server_entry(tools)

        env_dict = {e["name"]: e["value"] for e in entry["env"]}
        agents = json.loads(env_dict["BIRDIE_AGENTS_JSON"])
        names = {a["name"] for a in agents}
        assert names == {"AgentA", "AgentB", "AgentC"}

    def test_no_tools_no_agents_returns_none(self):
        from birdie.core.llm_provider import ACPProvider
        provider = ACPProvider(command="claude-agent-acp")
        # A tool without entrypoint and without _agent_def (e.g. MCP tool) is skipped
        mcp_tool = {
            "name": "mcp_tool",
            "description": "some mcp tool",
            "parameters": {"type": "object", "properties": {}},
        }
        entry = provider._mcp_server_entry([mcp_tool])
        assert entry is None

    def test_provider_config_embedded_in_agent_entry(self):
        from birdie.core.llm_provider import ACPProvider, agentdef_to_normalized_def
        from birdie.core.models import AgentDef
        provider = ACPProvider(command="claude-agent-acp")
        agent_def = AgentDef(name="MyAgent", description="test", prompt="go")
        cfg = {"vendor": "anthropic", "model": "claude-haiku-4-5-20251001"}
        agent_tool = agentdef_to_normalized_def(agent_def, provider_config=cfg)
        entry = provider._mcp_server_entry([agent_tool])

        env_dict = {e["name"]: e["value"] for e in entry["env"]}
        agents = json.loads(env_dict["BIRDIE_AGENTS_JSON"])
        assert agents[0]["_provider_config"] == cfg


# ---------------------------------------------------------------------------
# acp_mcp_server._build_server with agent defs
# ---------------------------------------------------------------------------

class TestAcpMcpServerWithAgents:
    """Unit tests for the MCP server builder with agent definitions."""

    def _make_agent_raw(self, name="Summarizer"):
        from birdie.core.llm_provider import agentdef_to_normalized_def
        from birdie.core.models import AgentDef, AgentParam
        agent_def = AgentDef(
            name=name,
            description=f"Agent {name}",
            prompt="Summarize: {{ text }}",
            input_params=[AgentParam(name="text", type="string", description="text", required=True)],
        )
        return agentdef_to_normalized_def(agent_def)

    async def _list_tools(self, server):
        """Invoke the list_tools request handler directly."""
        import mcp.types as types
        handler = server.request_handlers[types.ListToolsRequest]
        result = await handler(types.ListToolsRequest(method="tools/list", params=None))
        return result.root.tools

    async def _call_tool(self, server, name, arguments):
        """Invoke the call_tool request handler directly."""
        import mcp.types as types
        handler = server.request_handlers[types.CallToolRequest]
        req = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name=name, arguments=arguments),
        )
        return await handler(req)

    @pytest.mark.asyncio
    async def test_list_tools_includes_agents(self):
        from birdie.core.acp_mcp_server import _build_server
        agent_raw = self._make_agent_raw("Summarizer")
        server = _build_server(tool_defs=[], agent_defs=[agent_raw])

        tools = await self._list_tools(server)
        names = [t.name for t in tools]
        assert "Summarizer" in names

    @pytest.mark.asyncio
    async def test_list_tools_includes_both_skill_tools_and_agents(self):
        from birdie.core.acp_mcp_server import _build_server
        skill_tool = {
            "name": "search",
            "description": "Search",
            "parameters": {"type": "object", "properties": {}},
            "entrypoint": "python:birdie.skills.duckduckgo.tools.search",
        }
        agent_raw = self._make_agent_raw("Summarizer")
        server = _build_server(tool_defs=[skill_tool], agent_defs=[agent_raw])

        tools = await self._list_tools(server)
        names = [t.name for t in tools]
        assert "search" in names
        assert "Summarizer" in names

    @pytest.mark.asyncio
    async def test_agent_tool_input_schema_forwarded(self):
        from birdie.core.acp_mcp_server import _build_server
        agent_raw = self._make_agent_raw("Summarizer")
        server = _build_server(tool_defs=[], agent_defs=[agent_raw])

        tools = await self._list_tools(server)
        agent_tool = next(t for t in tools if t.name == "Summarizer")
        assert "text" in agent_tool.inputSchema.get("properties", {})

    @pytest.mark.asyncio
    async def test_call_unknown_tool_returns_error_result(self):
        """The MCP framework wraps ValueError into an isError CallToolResult."""
        from birdie.core.acp_mcp_server import _build_server
        server = _build_server(tool_defs=[], agent_defs=[])
        result = await self._call_tool(server, "nonexistent", {})
        # MCP framework converts the ValueError into an error result
        assert result.root.isError is True
        assert any("Unknown tool" in c.text for c in result.root.content)

    @pytest.mark.asyncio
    async def test_call_agent_tool_invokes_invoke_agent(self):
        """call_tool for an agent entry delegates to _invoke_agent."""
        from birdie.core.acp_mcp_server import _build_server
        from unittest.mock import patch, AsyncMock
        agent_raw = self._make_agent_raw("Summarizer")
        server = _build_server(tool_defs=[], agent_defs=[agent_raw])

        with patch("birdie.core.acp_mcp_server._invoke_agent", new=AsyncMock(return_value="bullet summary")) as mock_invoke:
            result = await self._call_tool(server, "Summarizer", {"text": "hello world"})

        mock_invoke.assert_awaited_once()
        call_args = mock_invoke.call_args
        assert call_args[0][0]["name"] == "Summarizer"
        assert call_args[0][1] == {"text": "hello world"}
        # Result should contain the mocked return value
        assert any("bullet summary" in c.text for c in result.root.content)

    @pytest.mark.asyncio
    async def test_skill_tool_still_uses_entrypoint(self):
        """Skill tools continue to use resolve_entrypoint, not _invoke_agent."""
        from birdie.core.acp_mcp_server import _build_server
        from unittest.mock import patch, AsyncMock
        skill_tool = {
            "name": "search",
            "description": "Search",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            "entrypoint": "python:birdie.skills.duckduckgo.tools.search",
        }
        server = _build_server(tool_defs=[skill_tool], agent_defs=[])

        with patch("birdie.core.acp_mcp_server.resolve_entrypoint") as mock_resolve:
            mock_fn = MagicMock(return_value="search results")
            mock_resolve.return_value = mock_fn
            result = await self._call_tool(server, "search", {"query": "python"})

        mock_resolve.assert_called_once_with("python:birdie.skills.duckduckgo.tools.search")
        mock_fn.assert_called_once_with("python:birdie.skills.duckduckgo.tools.search", query="python")


class TestProviderConfigSecrets:
    def test_to_json_always_excludes_api_key(self):
        from birdie.core.llm_provider import ProviderConfig
        cfg = ProviderConfig(vendor="openai", model="gpt-4o", api_key="sk-secret")
        data = json.loads(cfg.to_json())
        assert "api_key" not in data
        assert data["model"] == "gpt-4o"


class TestMalformedToolArgs:
    def test_malformed_arguments_degrade_to_empty_args(self):
        raw = {
            "content": "",
            "tool_calls": [{
                "id": "tc1", "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": broken'},
            }],
        }
        msg = _openai_msg_to_lc(raw)
        assert msg.tool_calls[0]["name"] == "get_weather"
        assert msg.tool_calls[0]["args"] == {}


class TestAnthropicToolResultTruncation:
    def test_oversized_tool_result_truncated(self):
        from birdie.core.llm_provider import _MAX_TOOL_CONTENT_CHARS
        big = "x" * (_MAX_TOOL_CONTENT_CHARS + 500)
        result = _lc_to_anthropic_messages(
            [ToolMessage(content=big, tool_call_id="tc1")]
        )
        block = result[0]["content"][0]
        assert len(block["content"]) < len(big)
        assert "characters truncated" in block["content"]


class TestAnthropicModelCatalog:
    def test_current_generation_models_listed(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            provider = AnthropicProvider(api_key="test")
        ids = {m.id for m in provider.list_models()}
        assert {"claude-fable-5", "claude-opus-5", "claude-opus-4-8",
                "claude-sonnet-5", "claude-sonnet-4-6"} <= ids
        by_id = {m.id: m for m in provider.list_models()}
        assert by_id["claude-opus-5"].context_window == 1_000_000
        assert by_id["claude-haiku-4-5-20251001"].context_window == 200_000


class TestRetryableErrors:
    def test_transient_5xx_is_retryable(self):
        from birdie.agent.graph import _is_retryable_error

        class _ServerError(Exception):
            status_code = 500

        class _Unavailable(Exception):
            status_code = 503

        assert _is_retryable_error(_ServerError("boom"))
        assert _is_retryable_error(_Unavailable("down"))

    def test_client_errors_not_retryable(self):
        from birdie.agent.graph import _is_retryable_error

        class _BadRequest(Exception):
            status_code = 400

        assert not _is_retryable_error(_BadRequest("bad"))

    def test_connection_error_name_is_retryable(self):
        from birdie.agent.graph import _is_retryable_error

        class APIConnectionError(Exception):
            pass

        assert _is_retryable_error(APIConnectionError("net down"))


class TestAnthropicPromptCaching:
    def test_cache_breakpoints_on_tools_system_and_last_stable_message(self):
        p = _make_anthropic_provider("claude-opus-5")
        tools = [
            {"name": "a", "description": "d", "parameters": {"type": "object", "properties": {}}},
            {"name": "b", "description": "d", "parameters": {"type": "object", "properties": {}}},
        ]
        ephemeral = HumanMessage(
            content="<session_context>volatile</session_context>",
            additional_kwargs={"birdie_ephemeral": True},
        )
        kw = p._build_kwargs(
            [HumanMessage(content="hi"), AIMessage(content="hello"),
             HumanMessage(content="do it"), ephemeral],
            tools, "stable system", None, None,
        )
        # Tools: breakpoint on the last tool only
        assert "cache_control" not in kw["tools"][0]
        assert kw["tools"][-1]["cache_control"] == {"type": "ephemeral"}
        # System: block form with breakpoint
        assert kw["system"][0]["text"] == "stable system"
        assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
        # Messages: breakpoint on the message BEFORE the ephemeral context
        msgs = kw["messages"]
        assert "cache_control" not in str(msgs[-1])  # volatile tail untouched
        stable_last = msgs[-2]
        assert stable_last["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_breakpoint_on_last_message_when_no_ephemeral_tail(self):
        p = _make_anthropic_provider("claude-opus-5")
        kw = p._build_kwargs([HumanMessage(content="hi")], None, None, None, None)
        assert kw["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_prompt_cache_disabled_leaves_plain_shapes(self):
        p = _make_anthropic_provider("claude-opus-5")
        p._prompt_cache = False
        tools = [{"name": "a", "description": "d", "parameters": {"type": "object", "properties": {}}}]
        kw = p._build_kwargs([HumanMessage(content="hi")], tools, "sys", None, None)
        assert kw["system"] == "sys"
        assert "cache_control" not in kw["tools"][0]
        assert kw["messages"][-1]["content"] == "hi"
