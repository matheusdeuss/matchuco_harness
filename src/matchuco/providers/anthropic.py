"""Anthropic (Claude) provider, using the official `anthropic` SDK.

Notable Claude-specific behaviour handled here:
- Prompt caching: a top-level `cache_control` caches the longest stable prefix
  (tools -> system -> messages), which an agent loop re-sends every turn.
- Thinking blocks are replayed verbatim (signature included) via `raw`.
- Eager input streaming: tool inputs stream as they are generated; the SDK
  raises ValueError if the streamed JSON is unparseable, which we surface as
  a retryable ProviderError. Tool inputs are validated by the tools themselves.
- Refusal fallbacks: for models that support it, the server retries a refused
  request on a fallback model inside the same call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic

from matchuco.messages import (
    ContentBlock,
    Message,
    OpaqueBlock,
    Response,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from matchuco.providers.base import (
    Done,
    ProviderError,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolUseStart,
)

DEFAULT_MODEL = "claude-opus-5"
PROVIDER_NAME = "anthropic"

# Models that accept server-side refusal fallbacks (`fallbacks="default"`).
_FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "max_tokens",
    "refusal": "refusal",
    "pause_turn": "pause_turn",
}


class AnthropicProvider:
    name = PROVIDER_NAME

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 64_000,
        thinking: bool = True,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self._client = client or anthropic.AsyncAnthropic()

    def build_request(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> dict[str, Any]:
        """Translate the neutral conversation into Messages API kwargs."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": to_anthropic_messages(messages),
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [to_anthropic_tool(t) for t in tools]
        # Haiku 4.5 does not support adaptive thinking; everything current does.
        if self.thinking and "haiku" not in self.model:
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if self.model in _FALLBACK_MODELS:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    async def stream(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[StreamEvent]:
        kwargs = self.build_request(system, messages, tools)
        try:
            async with self._client.beta.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield TextDelta(event.text)
                    elif event.type == "thinking":
                        yield ThinkingDelta(event.thinking)
                    elif (
                        event.type == "content_block_start"
                        and event.content_block.type == "tool_use"
                    ):
                        yield ToolUseStart(event.content_block.id, event.content_block.name)
                final = await stream.get_final_message()
        except ValueError as e:
            # Eager input streaming produced JSON the SDK could not parse at all.
            raise ProviderError(f"unparseable tool input from model: {e}", retryable=True) from e
        except (
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
        ) as e:
            raise ProviderError(f"anthropic: {e}", retryable=True) from e
        except anthropic.AuthenticationError as e:
            raise ProviderError("anthropic: authentication failed; set ANTHROPIC_API_KEY") from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"anthropic: {e.status_code} {e.message}") from e
        except TypeError as e:
            # The SDK raises TypeError (not an API error) when no credentials resolve.
            if "authentication" not in str(e):
                raise
            raise ProviderError("anthropic: no credentials found; set ANTHROPIC_API_KEY") from e

        yield Done(from_anthropic_message(final))


def to_anthropic_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "eager_input_streaming": True,
    }


def to_anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        blocks = [b for b in (_to_anthropic_block(b) for b in msg.content) if b is not None]
        if not blocks:
            # Happens when e.g. an assistant turn from another provider only had
            # reasoning we cannot replay. The API rejects empty content.
            blocks = [{"type": "text", "text": "(no content)"}]
        out.append({"role": msg.role, "content": blocks})
    return out


def _to_anthropic_block(block: ContentBlock) -> dict[str, Any] | None:
    match block:
        case TextBlock():
            return {"type": "text", "text": block.text}
        case ToolUseBlock():
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
        case ToolResultBlock():
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
                "is_error": block.is_error,
            }
        case ThinkingBlock() | OpaqueBlock():
            # Only replayable to the provider that produced it.
            if block.provider == PROVIDER_NAME and block.raw is not None:
                return block.raw
            return None
    return None


def from_anthropic_message(msg: Any) -> Response:
    content: list[ContentBlock] = []
    for block in msg.content:
        raw = block.model_dump(mode="json", by_alias=True, exclude_none=True)
        if block.type == "text":
            content.append(TextBlock(text=block.text))
        elif block.type == "tool_use":
            content.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input)))
        elif block.type == "thinking":
            content.append(ThinkingBlock(text=block.thinking, provider=PROVIDER_NAME, raw=raw))
        elif block.type == "redacted_thinking":
            content.append(ThinkingBlock(provider=PROVIDER_NAME, raw=raw))
        else:
            content.append(OpaqueBlock(provider=PROVIDER_NAME, raw=raw))

    u = msg.usage
    usage = Usage(
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", None) or 0,
    )
    return Response(
        message=Message(role="assistant", content=content),
        stop_reason=_STOP_REASONS.get(msg.stop_reason or "end_turn", "end_turn"),
        usage=usage,
        model=msg.model,
    )
