"""A scripted provider for deterministic tests -- no network, no API key.

Give it the assistant turns you want the "model" to produce, in order. It
streams each one back and records every request it received, so tests can
assert exactly what the harness sent (system prompt, history, tools).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from matchuco.messages import (
    Message,
    Response,
    StopReason,
    TextBlock,
    ToolSpec,
    Usage,
)
from matchuco.providers.base import (
    Done,
    ProviderError,
    StreamEvent,
    TextDelta,
    ToolUseStart,
)


@dataclass
class FakeRequest:
    system: str
    messages: list[Message]
    tools: list[ToolSpec]


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        script: Sequence[Message | str] = (),
        model: str = "fake-model",
        context_window: int = 200_000,
    ) -> None:
        self.model = model
        self.context_window = context_window
        self._script = [
            Message(role="assistant", content=[TextBlock(text=s)]) if isinstance(s, str) else s
            for s in script
        ]
        self.requests: list[FakeRequest] = []

    async def stream(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[StreamEvent]:
        # Deep-copy so later mutation of the history doesn't rewrite what we recorded.
        self.requests.append(
            FakeRequest(system, [m.model_copy(deep=True) for m in messages], list(tools))
        )
        if not self._script:
            raise ProviderError("FakeProvider script exhausted")
        reply = self._script.pop(0)

        for word in _chunks(reply.text):
            yield TextDelta(word)
        for tool_use in reply.tool_uses:
            yield ToolUseStart(tool_use.id, tool_use.name)

        stop: StopReason = "tool_use" if reply.tool_uses else "end_turn"
        yield Done(
            Response(
                message=reply,
                stop_reason=stop,
                usage=Usage(input_tokens=10, output_tokens=len(reply.text.split())),
                model=self.model,
            )
        )


def _chunks(text: str) -> list[str]:
    """Split text into word-ish chunks that concatenate back to the original."""
    return re.findall(r"\s*\S+|\s+$", text)
