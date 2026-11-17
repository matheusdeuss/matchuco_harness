"""Hooks: in-process callbacks around a tool call the permission gate already allowed.

Two events, and only these two:

- **PreToolUse** runs after `permissions.py` has authorized the call. A deny
  rule is never shown to a hook, and a hook cannot allow what that rule
  blocked. The hook may still refuse: the refusal comes back as an error
  `tool_result`, the same way a permission denial does, so the model can
  change course. It is not an exception that kills the loop.
- **PostToolUse** sees the tool's result and may append a note. The note is
  part of that new `tool_result`. The frozen system prompt is not rewritten.

Hooks are ordinary callables (sync or async). Tests register them in process.
`settings.json` can declare the same events — a fixed denial or a fixed note —
and those declarations become callables in this registry. There is no second
config file and no shell.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

from pydantic import BaseModel

from matchuco.config import HooksSettings
from matchuco.messages import ToolResultBlock
from matchuco.tools.base import Tool

__all__ = [
    "HookFailure",
    "HookRegistry",
    "PostToolUse",
    "PreToolUse",
    "hooks_from_settings",
]


@dataclass(frozen=True)
class PreToolUse:
    """A call the permission gate has already allowed."""

    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class PostToolUse:
    """A finished tool call, before any hook note is appended."""

    name: str
    args: dict[str, Any]
    content: str
    is_error: bool


PreHook = Callable[[PreToolUse], str | Awaitable[str | None] | None]
PostHook = Callable[[PostToolUse], str | Awaitable[str | None] | None]


class HookFailure(Exception):
    """A hook returned something unusable or raised. The loop turns this into a result."""


class HookRegistry:
    """PreToolUse and PostToolUse callbacks, matched by tool name (`*` or a glob)."""

    def __init__(self) -> None:
        self._pre: list[tuple[str, PreHook]] = []
        self._post: list[tuple[str, PostHook]] = []

    def on_pre(self, fn: PreHook, *, matcher: str = "*") -> None:
        self._pre.append((matcher, fn))

    def on_post(self, fn: PostHook, *, matcher: str = "*") -> None:
        self._post.append((matcher, fn))

    async def pre_tool_use(self, tool: Tool[Any], args: BaseModel) -> str | None:
        """Return a denial message, or None to let the tool run.

        The first denial wins; later hooks are not called. A hook that raises
        denies the call instead of escaping.
        """
        event = PreToolUse(tool.name, args.model_dump())
        for matcher, fn in self._pre:
            if not fnmatchcase(tool.name, matcher):
                continue
            try:
                message = await _call(fn, event)
            except HookFailure as exc:
                return f"Hook denied: PreToolUse hook failed ({exc})."
            if message:
                return f"Hook denied: {message}."
        return None

    async def post_tool_use(
        self, tool: Tool[Any], args: BaseModel, result: ToolResultBlock
    ) -> str | None:
        """Return notes to append, or None. A broken hook becomes a note, not a crash."""
        event = PostToolUse(tool.name, args.model_dump(), result.content, result.is_error)
        notes: list[str] = []
        for matcher, fn in self._post:
            if not fnmatchcase(tool.name, matcher):
                continue
            try:
                note = await _call(fn, event)
            except HookFailure as exc:
                notes.append(f"PostToolUse hook failed ({exc}).")
                continue
            if note:
                notes.append(note)
        if not notes:
            return None
        return "\n\n".join(notes)


def hooks_from_settings(hooks: HooksSettings) -> HookRegistry:
    """Turn the `hooks` section of settings into callables on one registry."""
    registry = HookRegistry()
    for pre in hooks.PreToolUse:
        registry.on_pre(_deny(pre.deny), matcher=pre.matcher)
    for post in hooks.PostToolUse:
        registry.on_post(_note(post.note), matcher=post.matcher)
    return registry


def _deny(message: str) -> PreHook:
    def hook(call: PreToolUse) -> str:
        del call
        return message

    return hook


def _note(message: str) -> PostHook:
    def hook(call: PostToolUse) -> str:
        del call
        return message

    return hook


async def _call(fn: Callable[[Any], Any], event: Any) -> str | None:
    try:
        result = fn(event)
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:
        raise HookFailure(f"{type(exc).__name__}: {exc}") from exc
    if result is None:
        return None
    if isinstance(result, str):
        text = result.strip()
        return text or None
    raise HookFailure(f"returned {type(result).__name__}, expected str or None")
