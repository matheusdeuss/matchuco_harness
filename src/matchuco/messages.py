"""Provider-neutral conversation types.

Every provider speaks its own wire format (Anthropic content blocks, OpenAI
chat messages with `tool_calls`, ...). The rest of the harness only ever sees
the types in this module; each provider adapter translates to and from them.

They are pydantic models with a `type` discriminator, so a whole conversation
can be dumped to JSON and loaded back losslessly -- that is what session
transcripts (JSONL) are built on later.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    """The model asking the harness to run a tool."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    """The harness answering a ToolUseBlock. Always sent in a `user` message."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


class ThinkingBlock(BaseModel):
    """Model reasoning. `text` is for display; `raw` is replayed verbatim.

    Some providers (Anthropic) require reasoning blocks to be sent back
    unchanged -- signature included -- on the next request, and only to the
    same provider. So we keep the provider's original payload in `raw`.
    """

    type: Literal["thinking"] = "thinking"
    text: str = ""
    provider: str
    raw: dict[str, Any] | None = None


class OpaqueBlock(BaseModel):
    """A provider-specific block the harness does not understand but must keep.

    Example: Anthropic `fallback` blocks. The owning provider replays `raw`;
    every other provider drops it.
    """

    type: Literal["opaque"] = "opaque"
    provider: str
    raw: dict[str, Any]


ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock | OpaqueBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=[TextBlock(text=text)])

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


class ToolSpec(BaseModel):
    """What the model sees about a tool: name, description, JSON Schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


# Normalized reasons a model turn ended.
#   end_turn   -> the model is done and waiting for the user
#   tool_use   -> the model wants tool results before continuing
#   max_tokens -> output was cut off
#   refusal    -> the provider declined the request
#   pause_turn -> the provider paused a long turn; resend to continue
StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "pause_turn"]


class Response(BaseModel):
    message: Message
    stop_reason: StopReason
    usage: Usage
    model: str
