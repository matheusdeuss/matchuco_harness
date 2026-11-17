"""Optional pieces that sit on the agent loop: subagents, hooks, skills, and MCP."""

from matchuco.extensions.hooks import HookRegistry, hooks_from_settings
from matchuco.extensions.skills import SkillTool
from matchuco.extensions.subagents import TaskTool, child_policy, run_subagent

__all__ = [
    "HookRegistry",
    "SkillTool",
    "TaskTool",
    "child_policy",
    "hooks_from_settings",
    "run_subagent",
]
