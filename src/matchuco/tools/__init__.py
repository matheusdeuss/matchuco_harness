"""The built-in tools and the registry that exposes them to a model."""

from __future__ import annotations

from typing import Any

from matchuco.tools.base import (
    Gate,
    Tool,
    ToolContext,
    ToolError,
    ToolKind,
    ToolRegistry,
    truncate,
)
from matchuco.tools.files import EditTool, ReadTool, WriteTool
from matchuco.tools.plan import ExitPlanModeTool
from matchuco.tools.search import GlobTool, GrepTool
from matchuco.tools.shell import ShellTool

__all__ = [
    "EditTool",
    "ExitPlanModeTool",
    "Gate",
    "GlobTool",
    "GrepTool",
    "ReadTool",
    "ShellTool",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolKind",
    "ToolRegistry",
    "WriteTool",
    "default_registry",
    "default_tools",
    "truncate",
]


def default_tools() -> list[Tool[Any]]:
    """The core tool set, in the order the model sees it.

    Search before read before write is deliberate: tool order is a weak hint,
    and the cheap, safe tools should be the ones that come to mind first.
    """
    return [
        GlobTool(),
        GrepTool(),
        ReadTool(),
        EditTool(),
        WriteTool(),
        ShellTool(),
        ExitPlanModeTool(),
    ]


def default_registry() -> ToolRegistry:
    return ToolRegistry(default_tools())
