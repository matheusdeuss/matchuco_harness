"""The Tool interface, the execution context, and the registry.

A tool is three things: a name and description the model reads, a JSON Schema
for its arguments, and code that runs. Here each tool declares a pydantic model
for its input, so the schema and the runtime validation come from one source.

The registry is what turns a `ToolUseBlock` from the model into a
`ToolResultBlock` for the next request. Crucially, a failing tool is *not* an
exception for the agent loop: bad arguments, a missing file or a non-zero exit
code all come back as a `ToolResultBlock` with `is_error=True`, so the model
can read the message and try something else.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from matchuco.messages import ToolResultBlock, ToolSpec, ToolUseBlock

InputT = TypeVar("InputT", bound=BaseModel)


class ToolError(Exception):
    """A tool failed in a way the model should see and can recover from."""


@dataclass
class ToolContext:
    """State shared by every tool call in a session.

    `root` is the workspace boundary: every path a tool touches is resolved
    against `root` and rejected if it escapes. That is a blunt rule -- Phase 3
    replaces it with real permission modes -- but it keeps a hallucinated
    `../../.ssh/id_rsa` from being a single tool call away.

    `files_read` powers the read-before-write rule: the model may only write to
    a file it has read, and only if the file has not changed since. Without it
    a model that guesses at a file's contents can silently destroy work.
    """

    root: Path = field(default_factory=Path.cwd)
    files_read: dict[Path, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = self.root.resolve()

    def resolve(self, path: str) -> Path:
        """Resolve a tool-supplied path inside the workspace, or raise."""
        candidate = Path(path)
        resolved = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ToolError(f"path {path!r} is outside the workspace ({self.root})")
        return resolved

    def display(self, path: Path) -> str:
        """Path as the model should see it: relative to the workspace when possible."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def mark_read(self, path: Path) -> None:
        self.files_read[path] = path.stat().st_mtime if path.exists() else time.time()

    def check_readable_before_write(self, path: Path) -> None:
        """Enforce read-before-write. New files are exempt -- nothing to destroy."""
        if not path.exists():
            return
        seen = self.files_read.get(path)
        if seen is None:
            raise ToolError(
                f"{self.display(path)} already exists and has not been read in this session; "
                "read it first so you do not overwrite content you have not seen"
            )
        if path.stat().st_mtime > seen:
            raise ToolError(
                f"{self.display(path)} changed on disk since you read it; read it again first"
            )


class Tool(ABC, Generic[InputT]):
    """Base class for tools. Subclasses set the three attributes and implement `run`."""

    name: str
    description: str
    input_model: type[InputT]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
        )

    @abstractmethod
    async def run(self, args: InputT, ctx: ToolContext) -> str:
        """Do the work. Raise `ToolError` for anything the model should read and retry."""

    async def call(self, raw: dict[str, Any], ctx: ToolContext) -> str:
        """Validate raw model-supplied arguments, then run."""
        try:
            args = self.input_model.model_validate(raw)
        except ValidationError as e:
            raise ToolError(f"invalid arguments for {self.name}: {_format_validation(e)}") from e
        return await self.run(args, ctx)


def _format_validation(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in item['loc']) or '(root)'}: {item['msg']}"
        for item in error.errors()
    )


class ToolRegistry:
    """The set of tools available to a model, by name."""

    def __init__(self, tools: list[Tool[Any]]) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            self._tools[tool.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def execute(self, block: ToolUseBlock, ctx: ToolContext) -> ToolResultBlock:
        """Run one tool call. Never raises for tool-level failures."""
        tool = self._tools.get(block.name)
        if tool is None:
            return ToolResultBlock(
                tool_use_id=block.id,
                content=f"unknown tool {block.name!r}; available: {', '.join(self.names)}",
                is_error=True,
            )
        try:
            content = await tool.call(block.input, ctx)
        except ToolError as e:
            return ToolResultBlock(tool_use_id=block.id, content=str(e), is_error=True)
        except Exception as e:  # a tool bug must not kill the session
            return ToolResultBlock(
                tool_use_id=block.id,
                content=f"{block.name} raised {type(e).__name__}: {e}",
                is_error=True,
            )
        return ToolResultBlock(tool_use_id=block.id, content=content)


def truncate(text: str, limit: int, unit: str = "characters") -> str:
    """Cap tool output so one call cannot eat the whole context window."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[truncated: {len(text)} {unit}, showing first {limit}]"
