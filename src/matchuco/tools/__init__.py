"""The built-in tools and the registry that exposes them to a model."""

from __future__ import annotations

from typing import Any

from matchuco.memory import RememberTool
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
    "RememberTool",
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

    `TaskTool` and `SkillTool` are imported here, not at module import.
    Importing them from the package root pulls `extensions` in, and that
    package reads settings, which imports this package. `SkillTool` is listed
    always, like `exit_plan_mode`: adding it only when a skill file exists
    would change the tool list mid-session and drop the prompt cache.
    """
    from matchuco.extensions.skills import SkillTool
    from matchuco.extensions.subagents import TaskTool

    return [
        GlobTool(),
        GrepTool(),
        ReadTool(),
        EditTool(),
        WriteTool(),
        ShellTool(),
        ExitPlanModeTool(),
        RememberTool(),
        SkillTool(),
        TaskTool(),
    ]


def default_registry() -> ToolRegistry:
    return ToolRegistry(default_tools())
