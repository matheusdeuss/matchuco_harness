from pydantic import TypeAdapter

from matchuco.messages import (
    Message,
    OpaqueBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


def test_conversation_roundtrips_through_json() -> None:
    conversation = [
        Message.user("list the files"),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(text="I should glob", provider="anthropic", raw={"type": "thinking"}),
                TextBlock(text="Let me look."),
                ToolUseBlock(id="t1", name="glob", input={"pattern": "*.py"}),
                OpaqueBlock(provider="anthropic", raw={"type": "fallback"}),
            ],
        ),
        Message(role="user", content=[ToolResultBlock(tool_use_id="t1", content="a.py")]),
    ]
    adapter = TypeAdapter(list[Message])
    restored = adapter.validate_json(adapter.dump_json(conversation))
    assert restored == conversation
    assert isinstance(restored[1].content[2], ToolUseBlock)


def test_message_helpers() -> None:
    msg = Message(
        role="assistant",
        content=[
            TextBlock(text="a"),
            ToolUseBlock(id="1", name="x", input={}),
            TextBlock(text="b"),
        ],
    )
    assert msg.text == "ab"
    assert [t.id for t in msg.tool_uses] == ["1"]


def test_usage_adds() -> None:
    total = Usage(input_tokens=1, output_tokens=2) + Usage(input_tokens=3, cache_read_tokens=4)
    assert total == Usage(input_tokens=4, output_tokens=2, cache_read_tokens=4)
