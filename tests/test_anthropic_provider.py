from anthropic.types.beta import BetaMessage

from matchuco.messages import (
    Message,
    OpaqueBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from matchuco.providers.anthropic import (
    AnthropicProvider,
    from_anthropic_message,
    to_anthropic_messages,
)

TOOL = ToolSpec(
    name="read_file",
    description="Read a file",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
)


def make_provider(model: str) -> AnthropicProvider:
    # A dummy key: these tests never hit the network.
    from anthropic import AsyncAnthropic

    return AnthropicProvider(model, client=AsyncAnthropic(api_key="test"))


def test_build_request_enables_caching_thinking_and_fallbacks() -> None:
    req = make_provider("claude-opus-5").build_request("sys", [Message.user("hi")], [TOOL])
    assert req["cache_control"] == {"type": "ephemeral"}
    assert req["system"] == "sys"
    assert req["thinking"]["type"] == "adaptive"
    assert req["fallbacks"] == "default"
    assert req["tools"][0]["eager_input_streaming"] is True
    assert req["tools"][0]["input_schema"] == TOOL.input_schema


def test_build_request_model_specific_options() -> None:
    req = make_provider("claude-haiku-4-5").build_request("", [Message.user("hi")], [])
    assert "thinking" not in req
    assert "fallbacks" not in req
    assert "system" not in req
    assert "tools" not in req


def test_thinking_is_replayed_only_to_its_own_provider() -> None:
    raw = {"type": "thinking", "thinking": "hmm", "signature": "sig"}
    history = [
        Message.user("q"),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(text="hmm", provider="anthropic", raw=raw),
                ThinkingBlock(text="other", provider="openai"),
                TextBlock(text="answer"),
            ],
        ),
    ]
    out = to_anthropic_messages(history)
    assert out[1]["content"] == [raw, {"type": "text", "text": "answer"}]


def test_empty_assistant_turn_gets_placeholder() -> None:
    history = [Message(role="assistant", content=[ThinkingBlock(provider="openai")])]
    assert to_anthropic_messages(history)[0]["content"] == [
        {"type": "text", "text": "(no content)"}
    ]


def test_tool_blocks_convert() -> None:
    history = [
        Message(role="assistant", content=[ToolUseBlock(id="t1", name="x", input={"a": 1})]),
        Message(
            role="user", content=[ToolResultBlock(tool_use_id="t1", content="boom", is_error=True)]
        ),
    ]
    out = to_anthropic_messages(history)
    assert out[0]["content"][0] == {"type": "tool_use", "id": "t1", "name": "x", "input": {"a": 1}}
    assert out[1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "boom",
        "is_error": True,
    }


def test_from_anthropic_message() -> None:
    api_msg = BetaMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "plan", "signature": "sig"},
                {"type": "text", "text": "Reading it."},
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}},
            ],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 7,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 20,
            },
        }
    )
    resp = from_anthropic_message(api_msg)
    assert resp.stop_reason == "tool_use"
    thinking, text, tool = resp.message.content
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.raw is not None and thinking.raw["signature"] == "sig"
    assert text == TextBlock(text="Reading it.")
    assert tool == ToolUseBlock(id="t1", name="read_file", input={"path": "a"})
    assert resp.usage.cache_read_tokens == 100
    assert resp.usage.cache_write_tokens == 20


def test_opaque_blocks_roundtrip() -> None:
    raw = {"type": "some_future_block", "data": 1}
    history = [Message(role="assistant", content=[OpaqueBlock(provider="anthropic", raw=raw)])]
    assert to_anthropic_messages(history)[0]["content"] == [raw]
