from typing import Any

from openai.types.chat import ChatCompletionChunk

from matchuco.messages import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from matchuco.providers.base import TextDelta, ToolUseStart
from matchuco.providers.openai_compat import _Accumulator, to_openai_messages


def chunk(
    delta: dict[str, Any], finish: str | None = None, usage: Any = None
) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-test",
            "choices": (
                [{"index": 0, "delta": delta, "finish_reason": finish}] if delta or finish else []
            ),
            "usage": usage,
        }
    )


def test_messages_convert_with_tool_results_first() -> None:
    history = [
        Message.user("hi"),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(provider="anthropic", raw={"type": "thinking"}),
                TextBlock(text="checking"),
                ToolUseBlock(id="t1", name="glob", input={"pattern": "*"}),
            ],
        ),
        Message(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="t1", content="nope", is_error=True),
                TextBlock(text="also this"),
            ],
        ),
    ]
    out = to_openai_messages("sys", history)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "user"]
    assert out[2]["content"] == "checking"
    assert out[2]["tool_calls"][0]["function"] == {"name": "glob", "arguments": '{"pattern": "*"}'}
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": "Error: nope"}


def test_accumulator_reassembles_streamed_tool_calls() -> None:
    acc = _Accumulator()
    events = []
    for c in [
        chunk({"role": "assistant", "content": "Let me "}),
        chunk({"content": "look."}),
        chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "call_a", "function": {"name": "glob", "arguments": ""}}
                ]
            }
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"pat'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'tern": "*.py"}'}}]}),
        chunk({}, finish="tool_calls"),
        chunk(
            {},
            usage={
                "prompt_tokens": 50,
                "completion_tokens": 9,
                "total_tokens": 59,
                "prompt_tokens_details": {"cached_tokens": 30},
            },
        ),
    ]:
        events += acc.feed(c)

    assert events == [TextDelta("Let me "), TextDelta("look."), ToolUseStart("call_a", "glob")]
    resp = acc.response("fallback")
    assert resp.stop_reason == "tool_use"
    assert resp.message.text == "Let me look."
    assert resp.message.tool_uses == [
        ToolUseBlock(id="call_a", name="glob", input={"pattern": "*.py"})
    ]
    assert resp.usage.input_tokens == 20
    assert resp.usage.cache_read_tokens == 30
    assert resp.model == "gpt-test"


def test_invalid_tool_json_is_preserved_for_the_tool_layer() -> None:
    acc = _Accumulator()
    acc.feed(
        chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "x", "function": {"name": "f", "arguments": "{bad"}}
                ]
            }
        )
    )
    acc.feed(chunk({}, finish="stop"))
    resp = acc.response("m")
    # Some servers say "stop" even with tool calls; we still report tool_use.
    assert resp.stop_reason == "tool_use"
    assert resp.message.tool_uses[0].input == {"__invalid_json__": "{bad"}
