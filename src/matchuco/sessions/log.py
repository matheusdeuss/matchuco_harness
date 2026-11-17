"""A session is an append-only JSONL file: `.matchuco/sessions/<id>.jsonl`.

Each line is one typed event. The file is never rewritten. `resume` rebuilds
the in-memory conversation by walking the events; `fork` copies the file to a
new id and leaves the original bytes untouched. `rewind` appends another
event. Rebuilding then ignores everything between the chosen checkpoint and
that rewind, which is how the transcript can undo work without forgetting
that the work happened.

Checkpoint events are written *before* the assistant message that asked for
the edit. The file bytes they point at were captured immediately before the
write, but placing the marker first means a rewind drops the tool call
together with its result. The provider never sees a `tool_use` without the
matching `tool_result`.

Edits that the model requested in one response share a `group`. Rewinding to
any of them restores every file in the group (and every later checkpoint),
because the conversation drops the whole response, not one of the calls.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from matchuco.messages import ContentBlock, Message, ToolResultBlock

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SessionError(Exception):
    """The transcript is missing, unreadable, or the rewind target is not active."""


class UserMessageEvent(BaseModel):
    type: Literal["user_message"] = "user_message"
    message: Message


class ModelMessageEvent(BaseModel):
    type: Literal["model_message"] = "model_message"
    message: Message


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    result: ToolResultBlock
    # Complete read-before-write map after this result, relative path -> mtime.
    files_read: dict[str, float] | None = None
    # Set when the tool was `task` and a child session was opened (phase 6).
    child_session_id: str | None = None


class CompactionEvent(BaseModel):
    """The history was replaced by a summary. Earlier events stay in the file."""

    type: Literal["compaction"] = "compaction"
    summary: str
    auto: bool = False
    messages: list[Message] = Field(default_factory=list)
    carryover: list[ContentBlock] = Field(default_factory=list)


class CheckpointEvent(BaseModel):
    """A file snapshot. The bytes live in `.matchuco/checkpoints/<id>.json`."""

    type: Literal["checkpoint"] = "checkpoint"
    id: str
    path: str
    existed: bool
    # Checkpoints taken for the same assistant response share a group.
    group: str = ""
    # Read-before-write map *before* this response's tools ran.
    files_read: dict[str, float] | None = None


class RewindEvent(BaseModel):
    type: Literal["rewind"] = "rewind"
    checkpoint_id: str
    # Mtimes refreshed from disk after the files were restored. Overrides the
    # checkpoint snapshot so the read-before-write check matches the filesystem.
    files_read: dict[str, float] | None = None


SessionEvent = Annotated[
    UserMessageEvent
    | ModelMessageEvent
    | ToolResultEvent
    | CompactionEvent
    | CheckpointEvent
    | RewindEvent,
    Field(discriminator="type"),
]

_EVENT: TypeAdapter[SessionEvent] = TypeAdapter(SessionEvent)


@dataclass
class SessionState:
    """What `resume` / `fork` / `rewind` put back on the agent."""

    messages: list[Message] = field(default_factory=list)
    carryover: list[ContentBlock] = field(default_factory=list)
    files_read: dict[str, float] = field(default_factory=dict)
    checkpoints: list[CheckpointEvent] = field(default_factory=list)


@dataclass
class _Point:
    """Conversation state just after a checkpoint event was applied."""

    messages: list[Message]
    pending: list[ToolResultBlock]
    carryover: list[ContentBlock]
    files_read: dict[str, float]
    checkpoints: list[CheckpointEvent]


def _check_id(session_id: str) -> None:
    if _ID_RE.fullmatch(session_id) is None:
        raise SessionError(f"invalid session id {session_id!r}")


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [m.model_copy(deep=True) for m in messages]


def _copy_blocks(blocks: list[ContentBlock]) -> list[ContentBlock]:
    return [b.model_copy(deep=True) for b in blocks]


def _has_tool_results(message: Message) -> bool:
    return any(isinstance(block, ToolResultBlock) for block in message.content)


def _without_dangling_tool_calls(messages: list[Message]) -> list[Message]:
    """Drop an assistant turn whose tool results were rewound away.

    The log order already puts the checkpoint before the tool call, so this is
    a safety net: a `tool_use` with no following `tool_result` would make the
    next provider request invalid.
    """
    kept: list[Message] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "assistant" and message.tool_uses:
            nxt = messages[index + 1] if index + 1 < len(messages) else None
            if nxt is None or not _has_tool_results(nxt):
                break
        kept.append(message)
        index += 1
    return kept


def _walk(events: list[SessionEvent]) -> tuple[SessionState, dict[str, _Point]]:
    messages: list[Message] = []
    pending: list[ToolResultBlock] = []
    carryover: list[ContentBlock] = []
    files_read: dict[str, float] = {}
    checkpoints: list[CheckpointEvent] = []
    points: dict[str, _Point] = {}

    def flush() -> None:
        nonlocal pending
        if pending:
            messages.append(Message(role="user", content=list(pending)))
            pending = []

    def remember(snapshot: dict[str, float] | None) -> None:
        nonlocal files_read
        if snapshot is not None:
            files_read = dict(snapshot)

    for event in events:
        match event:
            case UserMessageEvent(message=message):
                flush()
                messages.append(message.model_copy(deep=True))
                carryover = []
            case ModelMessageEvent(message=message):
                flush()
                messages.append(message.model_copy(deep=True))
            case ToolResultEvent(result=result, files_read=seen):
                pending.append(result.model_copy(deep=True))
                remember(seen)
            case CompactionEvent(messages=replaced, carryover=held):
                flush()
                messages = _copy_messages(replaced)
                carryover = _copy_blocks(held)
                files_read = {}
            case CheckpointEvent() as checkpoint:
                remember(checkpoint.files_read)
                checkpoints.append(checkpoint)
                points[checkpoint.id] = _Point(
                    _copy_messages(messages),
                    [item.model_copy(deep=True) for item in pending],
                    _copy_blocks(carryover),
                    dict(files_read),
                    list(checkpoints),
                )
            case RewindEvent(checkpoint_id=checkpoint_id, files_read=seen):
                point = points.get(checkpoint_id)
                if point is None:
                    raise SessionError(
                        f"checkpoint {checkpoint_id!r} is not active; it was already rewound "
                        "or never recorded"
                    )
                messages = _copy_messages(point.messages)
                pending = [item.model_copy(deep=True) for item in point.pending]
                carryover = _copy_blocks(point.carryover)
                checkpoints = list(point.checkpoints)
                files_read = dict(seen) if seen is not None else dict(point.files_read)
                kept_points: dict[str, _Point] = {}
                for item in checkpoints:
                    saved = points.get(item.id)
                    if saved is not None:
                        kept_points[item.id] = saved
                points = kept_points
            case _:
                raise SessionError(f"unknown session event {event!r}")

    flush()
    messages = _without_dangling_tool_calls(messages)
    state = SessionState(messages, carryover, files_read, checkpoints)
    return state, points


def reconstruct(events: list[SessionEvent]) -> SessionState:
    """Rebuild the conversation a provider should see."""
    return _walk(events)[0]


def state_at_checkpoint(events: list[SessionEvent], checkpoint_id: str) -> SessionState:
    """The conversation as of `checkpoint_id`, before the edit it guards."""
    _, points = _walk(events)
    point = points.get(checkpoint_id)
    if point is None:
        raise SessionError(f"checkpoint {checkpoint_id!r} is not in the active session")
    messages = _without_dangling_tool_calls(_copy_messages(point.messages))
    return SessionState(
        messages,
        _copy_blocks(point.carryover),
        dict(point.files_read),
        list(point.checkpoints),
    )


def _group(checkpoint: CheckpointEvent) -> str:
    return checkpoint.group or checkpoint.id


def checkpoints_to_restore(events: list[SessionEvent], checkpoint_id: str) -> list[str]:
    """Checkpoint ids whose files must be restored, oldest first.

    Includes the whole group of the target (one assistant response may have
    edited several files) and every checkpoint that comes after it, since
    those edits are about to leave the reconstructed history.
    """
    state = reconstruct(events)
    index = next((i for i, item in enumerate(state.checkpoints) if item.id == checkpoint_id), None)
    if index is None:
        raise SessionError(f"checkpoint {checkpoint_id!r} is not in the active session")
    start = index
    group = _group(state.checkpoints[index])
    while start > 0 and _group(state.checkpoints[start - 1]) == group:
        start -= 1
    return [item.id for item in state.checkpoints[start:]]


def refresh_files_read(root: Path, files_read: dict[str, float]) -> dict[str, float]:
    """Replace stored mtimes with what the filesystem reports after a restore.

    Paths that no longer exist are dropped: a new file the rewind deleted is
    not something the model has read.
    """
    root = root.resolve()
    refreshed: dict[str, float] = {}
    for relative, mtime in files_read.items():
        path = (root / relative).resolve()
        if path.is_file() and (path == root or root in path.parents):
            refreshed[relative] = path.stat().st_mtime
        elif not path.exists():
            continue
        else:
            refreshed[relative] = mtime
    return refreshed


class SessionLog:
    """One JSONL transcript on disk."""

    def __init__(self, root: Path, session_id: str) -> None:
        _check_id(session_id)
        self.root = root.resolve()
        self.id = session_id
        self.path = self.root / ".matchuco" / "sessions" / f"{session_id}.jsonl"

    @classmethod
    def create(cls, root: Path, session_id: str | None = None) -> SessionLog:
        log = cls(root, session_id or uuid.uuid4().hex[:12])
        log.path.parent.mkdir(parents=True, exist_ok=True)
        log.path.touch(exist_ok=True)
        return log

    @classmethod
    def resume(cls, root: Path, session_id: str) -> tuple[SessionLog, SessionState]:
        log = cls(root, session_id)
        if not log.path.is_file():
            raise SessionError(f"no session {session_id!r} in {log.path.parent}")
        return log, reconstruct(log.read_events())

    @classmethod
    def fork(cls, root: Path, session_id: str) -> tuple[SessionLog, SessionState]:
        """Copy the transcript to a new id. The source file is not opened for write."""
        source = cls(root, session_id)
        if not source.path.is_file():
            raise SessionError(f"no session {session_id!r} in {source.path.parent}")
        forked = cls.create(root)
        forked.path.write_bytes(source.path.read_bytes())
        return forked, reconstruct(forked.read_events())

    def append(self, event: SessionEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = _EVENT.dump_json(event, exclude_none=True).decode("utf-8")
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    def read_events(self) -> list[SessionEvent]:
        if not self.path.is_file():
            return []
        events: list[SessionEvent] = []
        text = self.path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                events.append(_EVENT.validate_json(line))
            except ValidationError as e:
                raise SessionError(f"{self.path}:{number}: {e}") from e
        return events
