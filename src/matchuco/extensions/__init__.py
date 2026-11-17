"""Optional pieces that sit on the agent loop: subagents now, hooks and MCP later."""

from matchuco.extensions.subagents import TaskTool, child_policy, run_subagent

__all__ = ["TaskTool", "child_policy", "run_subagent"]
