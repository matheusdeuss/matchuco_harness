"""The Provider interface every model backend implements.

A provider takes the neutral conversation (system prompt, messages, tool
specs) and streams back events. The last event is always `Done`, carrying the
complete assistant `Response`. Streaming lets the UI print tokens as they
arrive, while the agent loop only needs the final `Done`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from matchuco.messages import Message, Response, ToolSpec


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True)
class ToolUseStart:
    """The model began emitting a tool call (its input is still streaming)."""

    id: str
    name: str


@dataclass(frozen=True)
class Done:
    response: Response


StreamEvent = TextDelta | ThinkingDelta | ToolUseStart | Done


class ProviderError(Exception):
    """A provider call failed. `retryable` tells the caller if trying again may help."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class Provider(Protocol):
    name: str
    model: str

    def stream(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[StreamEvent]:
        """Stream one model turn. The final event is always `Done`."""
        ...


async def complete(
    provider: Provider,
    system: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec] = (),
) -> Response:
    """Run a turn without caring about intermediate events."""
    async for event in provider.stream(system, messages, tools):
        if isinstance(event, Done):
            return event.response
    raise ProviderError(f"{provider.name} stream ended without a Done event")
