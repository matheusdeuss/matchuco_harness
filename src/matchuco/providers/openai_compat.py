"""OpenAI Chat Completions provider -- also covers any OpenAI-compatible server.

Chat Completions is the lowest common denominator that Ollama, LM Studio,
vLLM, OpenRouter and friends all speak, so one adapter reaches many models.

The interesting part is the shape mismatch with our neutral format:
- Tool calls live in `assistant.tool_calls` with JSON-*string* arguments.
- Tool results are separate messages with `role: "tool"`, not blocks inside a
  user message, and must directly follow the assistant message that asked.
- Streamed tool calls arrive as fragments keyed by `index` that we reassemble.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import openai

from matchuco.messages import (
    ContentBlock,
    Message,
    Response,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from matchuco.providers.base import Done, ProviderError, StreamEvent, TextDelta, ToolUseStart

DEFAULT_MODEL = "gpt-5.4-mini"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_DEFAULT_MODEL = "qwen3-coder"

# Chat Completions does not report a model's context window, so we keep a small
# table of known prefixes. Unknown models get a conservative default; pass
# --context-window to override (local servers are often configured smaller).
_CONTEXT_WINDOWS = (
    ("gpt-4.1", 1_047_576),
    ("gpt-5", 400_000),
    ("gpt-6", 400_000),
    ("o4", 200_000),
    ("o3", 200_000),
)
DEFAULT_CONTEXT_WINDOW = 128_000
OLLAMA_CONTEXT_WINDOW = 32_768


def context_window_for(model: str) -> int:
    return next(
        (size for prefix, size in _CONTEXT_WINDOWS if model.startswith(prefix)),
        DEFAULT_CONTEXT_WINDOW,
    )


_STOP_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAICompatProvider:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        name: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        context_window: int | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.context_window = context_window or context_window_for(model)
        try:
            self._client = client or openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        except openai.OpenAIError as e:  # raised at construction when no API key is set
            raise ProviderError(f"{name}: {e}") from e

    def build_request(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(system, messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = [to_openai_tool(t) for t in tools]
        return kwargs

    async def stream(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[StreamEvent]:
        kwargs = self.build_request(system, messages, tools)
        acc = _Accumulator()
        try:
            chunks = await self._client.chat.completions.create(**kwargs)
            async for chunk in chunks:
                for event in acc.feed(chunk):
                    yield event
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError) as e:
            raise ProviderError(f"{self.name}: {e}", retryable=True) from e
        except openai.AuthenticationError as e:
            raise ProviderError(f"{self.name}: authentication failed; check your API key") from e
        except openai.APIStatusError as e:
            raise ProviderError(f"{self.name}: {e.status_code} {e.message}") from e

        yield Done(acc.response(self.model))


def to_openai_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def to_openai_messages(system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for msg in messages:
        if msg.role == "assistant":
            out.append(_assistant_message(msg))
            continue
        # A neutral user message may carry tool results *and* text. OpenAI
        # needs the tool results first, as individual `tool` messages.
        for b in msg.content:
            if isinstance(b, ToolResultBlock):
                content = f"Error: {b.content}" if b.is_error else b.content
                out.append({"role": "tool", "tool_call_id": b.tool_use_id, "content": content})
        text = msg.text
        if text:
            out.append({"role": "user", "content": text})
    return out


def _assistant_message(msg: Message) -> dict[str, Any]:
    # Thinking/opaque blocks from other providers are dropped: not replayable here.
    out: dict[str, Any] = {"role": "assistant", "content": msg.text or None}
    if msg.tool_uses:
        out["tool_calls"] = [
            {
                "id": t.id,
                "type": "function",
                "function": {"name": t.name, "arguments": json.dumps(t.input)},
            }
            for t in msg.tool_uses
        ]
    return out


@dataclass
class _PartialToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _Accumulator:
    """Reassembles a streamed chat completion into a neutral Response."""

    text: list[str] = field(default_factory=list)
    tool_calls: dict[int, _PartialToolCall] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    model: str | None = None

    def feed(self, chunk: Any) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        self.model = chunk.model or self.model
        if chunk.usage:
            details = getattr(chunk.usage, "prompt_tokens_details", None)
            cached = (getattr(details, "cached_tokens", None) or 0) if details else 0
            self.usage = Usage(
                input_tokens=(chunk.usage.prompt_tokens or 0) - cached,
                output_tokens=chunk.usage.completion_tokens or 0,
                cache_read_tokens=cached,
            )
        for choice in chunk.choices:
            delta = choice.delta
            if delta.content:
                self.text.append(delta.content)
                events.append(TextDelta(delta.content))
            for tc in delta.tool_calls or []:
                partial = self.tool_calls.setdefault(tc.index, _PartialToolCall())
                if tc.id:
                    partial.id = tc.id
                if tc.function and tc.function.name:
                    if not partial.name:
                        events.append(ToolUseStart(partial.id, tc.function.name))
                    partial.name += tc.function.name
                if tc.function and tc.function.arguments:
                    partial.arguments += tc.function.arguments
            if choice.finish_reason:
                self.finish_reason = choice.finish_reason
        return events

    def response(self, fallback_model: str) -> Response:
        content: list[ContentBlock] = []
        if self.text:
            content.append(TextBlock(text="".join(self.text)))
        for i in sorted(self.tool_calls):
            tc = self.tool_calls[i]
            content.append(
                ToolUseBlock(id=tc.id or f"call_{i}", name=tc.name, input=_parse_args(tc.arguments))
            )
        stop = _STOP_REASONS.get(self.finish_reason or "stop", "end_turn")
        # Some compatible servers report "stop" even when they emitted tool calls.
        if self.tool_calls and stop == "end_turn":
            stop = "tool_use"
        return Response(
            message=Message(role="assistant", content=content),
            stop_reason=stop,
            usage=self.usage,
            model=self.model or fallback_model,
        )


def _parse_args(arguments: str) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string; keep bad JSON visible to the tool layer."""
    if not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"__invalid_json__": arguments}
    return parsed if isinstance(parsed, dict) else {"__invalid_json__": arguments}
