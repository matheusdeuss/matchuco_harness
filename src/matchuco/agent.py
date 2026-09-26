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

The conversation lives on the `Agent`, and it is **append-only**: the loop
only ever adds messages to the end. Two things are allowed to replace it
wholesale -- `clear()` and compaction -- and both start a fresh conversation
rather than editing the old one (see `compact`).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from matchuco.context import (
    SystemPrompt,
    block_category,
    build_system_prompt,
    compact_instructions,
    estimate_block,
    estimate_messages,
    estimate_tokens,
    estimate_tools,
)
from matchuco.messages import (
    ContentBlock,
    Message,
    Response,
    StopReason,
    TextBlock,
    ToolResultBlock,
    Usage,
)
from matchuco.permissions import Approver, Mode, PermissionPolicy
from matchuco.providers.base import (
    Done,
    Provider,
    ProviderError,
    TextDelta,
    ThinkingDelta,
    ToolUseStart,
    complete,
)
from matchuco.tools import Tool, ToolContext, ToolRegistry, default_tools

__all__ = [
    "Agent",
    "AgentEvent",
    "Compacted",
    "Compacting",
    "ContextOverflowError",
    "ContextReport",
    "TextDelta",
    "ThinkingDelta",
    "ToolFinished",
    "ToolStarted",
    "TurnEnd",
]

DEFAULT_MAX_STEPS = 50
# Compact when the conversation would fill this fraction of the window. The
# rest is headroom for the model's answer and for the summary request itself.
DEFAULT_COMPACT_THRESHOLD = 0.8
# Compactions allowed within one turn before we give up (see `_loop`).
MAX_COMPACTIONS_PER_TURN = 2

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

# Mode changes are announced inside the next user message rather than by
# editing the system prompt: the system prompt is part of the cached prefix,
# and rewriting it on every mode switch would throw the cache away.
PLAN_MODE_REMINDER = """\
<system-reminder>
Plan mode is active. You must not change anything: only read-only tools \
(glob, grep, read) will run; edits and shell commands are refused. Explore the \
code, then call exit_plan_mode with a concrete plan (what changes, where, and \
how you will verify it). The user will approve or reject it.
</system-reminder>"""

PLAN_MODE_ENDED = """\
<system-reminder>
Plan mode has ended. You may edit files and run commands again, subject to \
the user's permission settings.
</system-reminder>"""

# What to keep when the conversation is squeezed into one message. The last
# sentence matters: the request still carries the tool list (dropping it
# would change the cached prefix), so the model must be told not to use it.
SUMMARY_PROMPT = """\
The conversation is about to run out of context. Summarize the transcript \
inside <summary></summary> tags so the work can continue in a new context \
window without redoing anything or asking the user to repeat themselves. \
Be sure to preserve:
1. The user's requests, decisions, constraints and preferences, stated exactly \
and close to their own words.
2. Exactly where things stand: what is done, which files were changed and how, \
and what was verified (tests run and their results).
3. Problems that came up and how they were resolved; approaches tried or ruled \
out, and why.
4. What is still open or expected to happen next.
5. Specific details that would be hard to reconstruct: file paths, function \
names, commands, error messages, numbers.
Be complete on these even at the cost of length; condense your own reasoning \
to its conclusions. Do not call any tools while writing this summary; respond \
with text only."""

SUMMARY_HEADER = """\
<system-reminder>
This session continues a conversation that ran out of context and was \
compacted. Summary of everything so far:

{summary}

Files you read before the compaction must be read again before you edit them.
</system-reminder>"""

CONTINUE_AFTER_COMPACTION = (
    "Continue the task from where you left off, without asking the user to repeat anything."
)

_SUMMARY_RE = re.compile(r"<summary>(.*?)(?:</summary>|\Z)", re.DOTALL)
_OWN_REMINDERS = (PLAN_MODE_REMINDER, PLAN_MODE_ENDED)


class ContextOverflowError(ProviderError):
    """The conversation cannot be made to fit the context window."""


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
class Compacting:
    """The conversation is being summarized (a model call, so it takes a moment)."""

    auto: bool
    tokens: int


@dataclass(frozen=True)
class Compacted:
    auto: bool
    tokens_before: int
    tokens_after: int
    summary: str


@dataclass(frozen=True)
class TurnEnd:
    """The agent finished the user's request (or hit the step limit)."""

    reason: StopReason | Literal["max_steps"]
    usage: Usage
    steps: int


AgentEvent = (
    TextDelta | ThinkingDelta | ToolStarted | ToolFinished | Compacting | Compacted | TurnEnd
)


@dataclass(frozen=True)
class ContextReport:
    """What is in the context window, by category, for /context."""

    window: int
    threshold: int
    categories: list[tuple[str, int]]
    last_reported: int | None  # what the provider said the last request used

    @property
    def used(self) -> int:
        return sum(tokens for _, tokens in self.categories)


class Agent:
    """A provider, a tool set and a conversation, wired into a loop."""

    def __init__(
        self,
        provider: Provider,
        *,
        tools: Sequence[Tool[Any]] | None = None,
        root: Path | None = None,
        system: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        permissions: PermissionPolicy | None = None,
        approver: Approver | None = None,
        context_window: int | None = None,
        compact_threshold: float = DEFAULT_COMPACT_THRESHOLD,
    ) -> None:
        self.provider = provider
        self.registry = ToolRegistry(list(default_tools() if tools is None else tools))
        self.context = ToolContext(root=root or Path.cwd())
        # system=None: the full prompt with environment and AGENTS.md files,
        # rebuilt on /clear. An explicit string is used as-is (tests, embedding).
        self._custom_system = system
        self.system_prompt = self._build_system_prompt()
        self.max_steps = max_steps
        # Secure by default: without an approver, anything that needs one is denied.
        self.permissions = permissions or PermissionPolicy()
        self.approver = approver
        self.context_window = context_window or provider.context_window
        self.compact_threshold = compact_threshold
        self.history: list[Message] = []
        self.usage = Usage()
        self._announced_mode: Mode = "default"
        # Blocks to put at the start of the next user message (a compaction summary).
        self._carryover: list[ContentBlock] = []
        # (tokens the provider reported for the last request + its output,
        #  len(history) at that moment). None = no trustworthy measurement yet.
        self._baseline: tuple[int, int] | None = None
        self._last_reported: int | None = None

    # --- system prompt ---------------------------------------------------------

    def _build_system_prompt(self) -> SystemPrompt:
        root = self.context.root
        if self._custom_system is not None:
            return SystemPrompt((("system prompt", self._custom_system.format(root=root)),))
        return build_system_prompt(SYSTEM_PROMPT.format(root=root), root)

    @property
    def system(self) -> str:
        return self.system_prompt.text

    # --- conversation state ----------------------------------------------------

    def _mode_reminder(self) -> str | None:
        """The reminder to prepend to the next prompt, if the model's view of the mode is stale."""
        mode, previous = self.permissions.mode, self._announced_mode
        self._announced_mode = mode
        if mode == previous:
            return None
        if mode == "plan":
            return PLAN_MODE_REMINDER
        if previous == "plan":
            return PLAN_MODE_ENDED
        return None

    def clear(self) -> None:
        """Start a new conversation. Instruction files are re-read, tool state is dropped."""
        self.history.clear()
        self.context.files_read.clear()
        self._announced_mode = "default"  # a fresh conversation has heard nothing yet
        self._carryover = []
        self._baseline = None
        self._last_reported = None
        self.system_prompt = self._build_system_prompt()

    # --- measuring ---------------------------------------------------------------

    def context_tokens(self) -> int:
        """Best estimate of what the next request will hold.

        The last provider-reported size is exact; only what was appended since
        is estimated. Without a measurement (new session, just compacted) the
        whole request is estimated.
        """
        pending = sum(estimate_block(b) for b in self._carryover)
        if self._baseline is not None and self._baseline[1] <= len(self.history):
            reported, length = self._baseline
            return reported + estimate_messages(self.history[length:]) + pending
        return (
            estimate_tokens(self.system)
            + estimate_tools(self.registry.specs)
            + estimate_messages(self.history)
            + pending
        )

    @property
    def compact_at(self) -> int:
        return int(self.context_window * self.compact_threshold)

    def context_report(self) -> ContextReport:
        categories: list[tuple[str, int]] = [
            (f"system: {label}", estimate_tokens(text)) for label, text in self.system_prompt.parts
        ]
        categories.append(("tool definitions", estimate_tools(self.registry.specs)))
        totals: dict[str, int] = {}
        for message in self.history:
            for block in message.content:
                name = block_category(block, message.role)
                totals[name] = totals.get(name, 0) + estimate_block(block)
        if self._carryover:
            totals["compaction summary"] = sum(estimate_block(b) for b in self._carryover)
        categories += sorted(totals.items(), key=lambda item: -item[1])
        return ContextReport(self.context_window, self.compact_at, categories, self._last_reported)

    def _record(self, response: Response) -> None:
        """Remember what the provider says this request cost, as the new baseline."""
        u = response.usage
        reported = u.input_tokens + u.cache_read_tokens + u.cache_write_tokens
        if reported:
            self._last_reported = reported
            # The response we just appended is part of the next request too.
            self._baseline = (reported + u.output_tokens, len(self.history))
        else:
            self._baseline = None  # provider gave no usage; fall back to estimating

    # --- the loop ---------------------------------------------------------------

    async def run(self, prompt: str) -> AsyncIterator[AgentEvent]:
        """Answer one user prompt, streaming events until `TurnEnd`.

        If the turn dies early -- provider error, Ctrl-C, caller stops reading
        -- the conversation is restored to how it was before the prompt, so the
        next prompt still goes out as a valid request. (Restoring a snapshot,
        rather than popping messages off the end, also undoes a compaction
        that happened during the failed turn.)
        """
        snapshot = (list(self.history), list(self._carryover), self._announced_mode, self._baseline)
        content: list[ContentBlock] = [*self._carryover]
        if reminder := self._mode_reminder():
            content.append(TextBlock(text=reminder))
        content.append(TextBlock(text=prompt))
        self._carryover = []
        self.history.append(Message(role="user", content=content))

        finished = False
        try:
            async for event in self._loop():
                finished = isinstance(event, TurnEnd)
                yield event
        except BaseException:
            if not finished:
                self.history[:], self._carryover, self._announced_mode, self._baseline = (
                    snapshot[0],
                    snapshot[1],
                    snapshot[2],
                    snapshot[3],
                )
            raise

    async def _loop(self) -> AsyncIterator[AgentEvent]:
        turn_usage = Usage()
        compactions = 0
        for step in range(1, self.max_steps + 1):
            # Check before every request, not just before a new prompt: a long
            # tool loop is exactly how a conversation outgrows its window.
            if self.context_tokens() > self.compact_at:
                compactions += 1
                if compactions > MAX_COMPACTIONS_PER_TURN:
                    raise ContextOverflowError(
                        "the context keeps refilling right after compaction; something in "
                        "the conversation (a huge file or command output?) is too large"
                    )
                async for compaction_event in self._compact(auto=True):
                    yield compaction_event
                if self.context_tokens() > self.compact_at:
                    raise ContextOverflowError(
                        f"even after compaction the conversation needs ~{self.context_tokens()} "
                        f"tokens, over the {self.compact_at} limit"
                    )

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
            self._record(response)
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
            gate = self.permissions.gate(self.context, self.approver)
            for block in tool_uses:
                yield ToolStarted(block.id, block.name, block.input)
                result = await self.registry.execute(block, self.context, gate)
                yield ToolFinished(block.id, block.name, result)
                results.append(result)
            # Every tool_use must be answered in a single user message, in order,
            # or the next request is rejected as malformed.
            self.history.append(Message(role="user", content=list(results)))

        yield TurnEnd("max_steps", turn_usage, self.max_steps)

    # --- compaction ---------------------------------------------------------------

    async def compact(self, focus: str = "") -> Compacted | None:
        """Summarize the conversation now (the /compact command). None if it is empty."""
        result: Compacted | None = None
        async for event in self._compact(auto=False, focus=focus):
            if isinstance(event, Compacted):
                result = event
        return result

    async def _compact(
        self, *, auto: bool, focus: str = ""
    ) -> AsyncIterator[Compacting | Compacted]:
        """Replace the whole conversation with a summary of it.

        This is "simple compaction": no recent turns are kept verbatim. That
        sounds lossy, but it is the shape that stays valid everywhere -- keeping
        a tail would replay reasoning blocks that were produced with the full
        history in front of them, which newer models reject -- and models are
        trained to continue from exactly this kind of summary.

        It only ever runs at a request boundary, when no `tool_use` is waiting
        for its result. The new conversation starts with the summary plus
        whatever was pending: the new prompt, or an instruction to continue
        the interrupted task.
        """
        messages = list(self.history)
        held: list[ContentBlock] = []
        if messages and messages[-1].role == "user" and not _has_tool_results(messages[-1]):
            # A fresh prompt that has not been answered yet: keep it out of the
            # summary and put it back, verbatim, after it.
            held = [b for b in messages.pop().content if not _is_own_reminder(b)]
        if not messages:
            return

        before = self.context_tokens()
        yield Compacting(auto, before)
        summary, usage = await self._summarize(messages, focus)
        self.usage += usage

        # A fresh conversation: re-announce the mode, forget what was read.
        self._announced_mode = "default"
        header: list[ContentBlock] = [TextBlock(text=SUMMARY_HEADER.format(summary=summary))]
        if reminder := self._mode_reminder():
            header.append(TextBlock(text=reminder))
        self.context.files_read.clear()
        self._baseline = None

        if held:
            self.history[:] = [Message(role="user", content=[*header, *held])]
        elif messages[-1].role == "user":
            # Mid-turn (last message = tool results): the loop sends this next.
            self.history[:] = [
                Message(role="user", content=[*header, TextBlock(text=CONTINUE_AFTER_COMPACTION)])
            ]
        else:
            # Between turns: nothing to send yet, so the summary rides along
            # with the user's next prompt instead of becoming a message of its own.
            self.history.clear()
            self._carryover = header

        yield Compacted(auto, before, self.context_tokens(), summary)

    async def _summarize(self, messages: list[Message], focus: str) -> tuple[str, Usage]:
        instructions = SUMMARY_PROMPT
        if extra := compact_instructions(list(self.system_prompt.instructions)):
            instructions += f"\n\nThe project asks you to also keep:\n{extra}"
        if focus:
            instructions += f"\n\nThe user asked this summary to focus on: {focus}"

        # Same system prompt and tools as the conversation, with the request
        # appended at the end: the whole conversation is a cache hit.
        request = list(messages)
        last = request[-1]
        if last.role == "user":
            request[-1] = Message(
                role="user", content=[*last.content, TextBlock(text=instructions)]
            )
        else:
            request.append(Message.user(instructions))

        response = await complete(self.provider, self.system, request, self.registry.specs)
        text = response.message.text
        match = _SUMMARY_RE.search(text)
        summary = (match.group(1) if match else text).strip()
        if not summary:
            raise ProviderError("compaction failed: the model returned an empty summary")
        return summary, response.usage


def _has_tool_results(message: Message) -> bool:
    return any(isinstance(b, ToolResultBlock) for b in message.content)


def _is_own_reminder(block: ContentBlock) -> bool:
    return isinstance(block, TextBlock) and block.text in _OWN_REMINDERS
