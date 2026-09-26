"""The agentic loop: the part that makes a chatbot into an agent.

One user prompt can take many model turns. The loop is small enough to state
in full:

    send the conversation + tool specs to the model
    if the model asked for tools: run them, append the results, send again
    otherwise: the turn is over

Everything else in this module exists to keep that loop honest -- a step limit
so a confused model cannot spin forever, tool failures fed back as results
instead of exceptions, and an event stream so the UI can show work in progress
without the loop knowing anything about terminals.

The conversation lives on the `Agent`, which is what later phases hook into:
sessions persist `history`, compaction rewrites it, permissions sit between
`ToolStarted` and the actual call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from matchuco.messages import (
    Message,
    Response,
    StopReason,
    ToolResultBlock,
    Usage,
)
from matchuco.providers.base import (
    Done,
    Provider,
    ProviderError,
    TextDelta,
    ThinkingDelta,
    ToolUseStart,
)
from matchuco.tools import Tool, ToolContext, ToolRegistry, default_tools

__all__ = [
    "Agent",
    "AgentEvent",
    "TextDelta",
    "ThinkingDelta",
    "ToolFinished",
    "ToolStarted",
    "TurnEnd",
]

DEFAULT_MAX_STEPS = 50

SYSTEM_PROMPT = """\
You are matchuco, a coding agent running in the user's terminal.

You work by calling tools. Prefer acting over asking: if a question can be \
answered by reading the repository, read it instead of asking the user.

Guidelines:
- Find before you read: use glob and grep to locate code, then read only the \
relevant files. Reading whole trees wastes the context window.
- Read a file before you edit or write it. The tools enforce this.
- Prefer edit over write for existing files; write replaces the entire file.
- Verify your work when there is a cheap way to: run the tests, the linter, or \
the command the user mentioned.
- When a tool fails, read the error and adjust. Do not repeat the same failing \
call.
- Keep replies short and concrete. The user sees the tool calls, so do not \
narrate what you are about to do at length or repeat file contents back.

Workspace root: {root}
"""


@dataclass(frozen=True)
class ToolStarted:
    """A tool call is about to run, with the arguments fully decoded."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolFinished:
    id: str
    name: str
    result: ToolResultBlock

    @property
    def failed(self) -> bool:
        return self.result.is_error


@dataclass(frozen=True)
class TurnEnd:
    """The agent finished the user's request (or hit the step limit)."""

    reason: StopReason | Literal["max_steps"]
    usage: Usage
    steps: int


AgentEvent = TextDelta | ThinkingDelta | ToolStarted | ToolFinished | TurnEnd


class Agent:
    """A provider, a tool set and a conversation, wired into a loop."""

    def __init__(
        self,
        provider: Provider,
        *,
        tools: Sequence[Tool[Any]] | None = None,
        root: Path | None = None,
        system: str = SYSTEM_PROMPT,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self.provider = provider
        self.registry = ToolRegistry(list(default_tools() if tools is None else tools))
        self.context = ToolContext(root=root or Path.cwd())
        self.system = system.format(root=self.context.root)
        self.max_steps = max_steps
        self.history: list[Message] = []
        self.usage = Usage()

    def clear(self) -> None:
        """Forget the conversation. Tool state (files read) goes with it."""
        self.history.clear()
        self.context.files_read.clear()

    async def run(self, prompt: str) -> AsyncIterator[AgentEvent]:
        """Answer one user prompt, streaming events until `TurnEnd`.

        If the turn dies early -- provider error, Ctrl-C, caller stops reading
        -- the history is repaired before the exception escapes, so the next
        prompt still goes out as a valid request.
        """
        self.history.append(Message.user(prompt))
        try:
            async for event in self._loop():
                yield event
        except BaseException:
            self._repair_history()
            raise

    def _repair_history(self) -> None:
        """Drop the trailing half-finished exchange.

        A request is only valid if every `tool_use` the assistant emitted has a
        matching `tool_result`, and the last message is the assistant's. An
        interrupted turn breaks both, so we pop until it holds again -- which,
        for a turn cut short, unwinds the whole prompt.
        """
        while self.history:
            last = self.history[-1]
            if last.role == "assistant" and not last.tool_uses:
                return
            self.history.pop()

    async def _loop(self) -> AsyncIterator[AgentEvent]:
        turn_usage = Usage()
        for step in range(1, self.max_steps + 1):
            response: Response | None = None
            async for event in self.provider.stream(self.system, self.history, self.registry.specs):
                match event:
                    case TextDelta() | ThinkingDelta():
                        yield event
                    case ToolUseStart():
                        pass  # arguments are still streaming; we announce them below
                    case Done(response=done_response):
                        response = done_response
            if response is None:
                raise ProviderError(f"{self.provider.name} stream ended without a Done event")

            self.history.append(response.message)
            turn_usage += response.usage
            self.usage += response.usage

            tool_uses = response.message.tool_uses
            if not tool_uses:
                # `pause_turn` means the provider cut a long turn short; resending
                # the same history lets the model pick up where it left off.
                if response.stop_reason == "pause_turn":
                    continue
                yield TurnEnd(response.stop_reason, turn_usage, step)
                return

            results: list[ToolResultBlock] = []
            for block in tool_uses:
                yield ToolStarted(block.id, block.name, block.input)
                result = await self.registry.execute(block, self.context)
                yield ToolFinished(block.id, block.name, result)
                results.append(result)
            # Every tool_use must be answered in a single user message, in order,
            # or the next request is rejected as malformed.
            self.history.append(Message(role="user", content=list(results)))

        yield TurnEnd("max_steps", turn_usage, self.max_steps)
