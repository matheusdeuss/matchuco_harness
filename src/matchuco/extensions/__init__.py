"""Optional pieces that sit on the agent loop: subagents, hooks, skills, and MCP."""

from matchuco.extensions.hooks import HookRegistry, hooks_from_settings
from matchuco.extensions.mcp_client import McpClient, McpError, attach_mcp_server
from matchuco.extensions.skills import SkillTool
from matchuco.extensions.subagents import TaskTool, child_policy, run_subagent

__all__ = [
    "HookRegistry",
    "McpClient",
    "McpError",
    "SkillTool",
    "TaskTool",
    "attach_mcp_server",
    "child_policy",
    "hooks_from_settings",
    "run_subagent",
]
