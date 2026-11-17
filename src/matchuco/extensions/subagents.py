"""Subagents: one tool call that runs a child agent in its own conversation.

The child does not see the parent's messages. It gets a fresh system prompt
(frozen for the child session, not patched into the parent's) and its own
transcript under `.matchuco/sessions/`. The parent hears only the child's
final text, as the `tool_result` of `task`, and the parent's JSONL records
the child's session id.

Depth stops at 1: a child that calls `task` is refused before another agent
is constructed. The child is read-only unless the caller passes
`readonly=false`, and even then its permission policy is a copy of the
parent's -- same mode, same rules, never a more open one. Read-only means
plan mode, which already denies edits and commands, plus the removal of
`exit_plan_mode` so the child cannot promote itself out of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from matchuco.permissions import PermissionPolicy
from matchuco.sessions import SessionLog
from matchuco.tools.base import Tool, ToolContext, ToolError

if TYPE_CHECKING:
    from matchuco.agent import Agent


class TaskInput(BaseModel):
    description: str = Field(description="Short title of the subtask, a few words.")
    prompt: str = Field(
        description="Everything the subagent needs. It does not see this conversation."
    )
    readonly: bool = Field(
        default=True,
        description="When true (the default), the subagent cannot edit files or run commands.",
    )


class TaskTool(Tool[TaskInput]):
    name = "task"
    description = (
        "Run a subagent on a subtask and get its final answer back as text. The subagent "
        "has its own conversation and does not see this one, so the prompt must stand alone. "
        "It is read-only unless `readonly` is false, and it cannot start a further subagent. "
        "A read-only subagent is still bound by every deny rule of this session."
    )
    input_model = TaskInput
    # Kind is read so plan mode can still delegate exploration. The child, not
    # this tool, is what edits -- and only when readonly is false and the parent
    # policy already allows it.
    kind = "read"

    def preview(self, args: TaskInput, ctx: ToolContext) -> str:
        mode = "read-only" if args.readonly else "same permissions as this session"
        return f"{args.description} ({mode})\n\n{args.prompt}"

    async def run(self, args: TaskInput, ctx: ToolContext) -> str:
        if ctx.depth >= 1:
            raise ToolError(
                "refused: a subagent cannot start another subagent (maximum depth is 1)"
            )
        if ctx.spawn is None:
            raise ToolError("subagents are not available in this session")
        prompt = args.prompt.strip()
        if not prompt:
            raise ToolError("prompt is empty")
        if args.description.strip():
            prompt = f"{args.description.strip()}\n\n{prompt}"
        return await ctx.spawn(prompt, description=args.description, readonly=args.readonly)


def child_policy(parent: PermissionPolicy, *, readonly: bool) -> PermissionPolicy:
    """A policy the child may hold: the parent's rules, and not a looser mode.

    Read-only forces plan mode, which denies edits and commands even when the
    parent is in bypass and even when an allow rule would otherwise match.
    Deny rules are copied either way, so a file the parent must not read stays
    unreadable. The copy does not share the parent's rule lists.
    """
    return PermissionPolicy(
        mode="plan" if readonly else parent.mode,
        allow=list(parent.allow),
        ask=list(parent.ask),
        deny=list(parent.deny),
    )


def subagent_tools() -> list[Tool[Any]]:
    """The parent's tools, without the one that would let a child leave plan mode."""
    from matchuco.tools import default_tools

    return [tool for tool in default_tools() if tool.name != "exit_plan_mode"]


async def run_subagent(parent: Agent, prompt: str, *, description: str, readonly: bool) -> str:
    """Run `prompt` in a child agent. Returns the child's final text.

    `description` is already part of `prompt` when `TaskTool` built it; it is
    accepted so other callers can pass the same shape. The child's session id
    is stored on `parent.context` for the transcript.
    """
    del description  # the tool folds it into `prompt`; kept for the spawner signature
    if parent.depth >= 1:
        raise ToolError("refused: a subagent cannot start another subagent (maximum depth is 1)")

    from matchuco.agent import Agent as AgentCls
    from matchuco.agent import TextDelta, TurnEnd

    child_log = SessionLog.create(parent.context.root)
    parent.context.last_child_session_id = child_log.id
    child = AgentCls(
        parent.provider,
        tools=subagent_tools(),
        root=parent.context.root,
        permissions=child_policy(parent.permissions, readonly=readonly),
        approver=parent.approver,
        max_steps=parent.max_steps,
        context_window=parent.context_window,
        compact_threshold=parent.compact_threshold,
        session=child_log,
        depth=parent.depth + 1,
        hooks=parent.hooks,
    )

    chunks: list[str] = []
    end: TurnEnd | None = None
    async for event in child.run(prompt):
        if isinstance(event, TextDelta):
            chunks.append(event.text)
        elif isinstance(event, TurnEnd):
            end = event
    if end is not None and end.reason == "max_steps":
        raise ToolError(f"subagent stopped after {end.steps} steps without a final answer")
    text = "".join(chunks).strip()
    return text or "(subagent returned no text)"
