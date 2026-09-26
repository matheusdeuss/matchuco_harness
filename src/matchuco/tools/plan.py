"""exit_plan_mode: how the model hands a finished plan to the user.

In plan mode the model may only observe (read, glob, grep). When it has a plan
it calls this tool with the plan as its argument. The tool itself does almost
nothing -- the work happens in the permission gate: calls of kind "plan"
always require the user's approval, and approving one switches the session
out of plan mode. So "the user approved the plan" and "the tool ran" are the
same event, and a rejection comes back to the model as an error result with
the user's feedback, so it can revise the plan.

The tool is always registered, even outside plan mode. Adding and removing it
as the mode changes would change the tool list, which sits at the very start
of every request and would invalidate the provider's prompt cache.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from matchuco.tools.base import Tool, ToolContext


class ExitPlanModeInput(BaseModel):
    plan: str = Field(
        description="The implementation plan, in Markdown: what will change, in which files, "
        "and how it will be verified."
    )


class ExitPlanModeTool(Tool[ExitPlanModeInput]):
    name = "exit_plan_mode"
    description = (
        "Only in plan mode: present your finished plan to the user for approval. If they "
        "approve, plan mode ends and you should start implementing the plan right away. "
        "If they reject it, revise the plan using their feedback. Do not call this to ask "
        "questions -- only when the plan is ready."
    )
    input_model = ExitPlanModeInput
    kind = "plan"

    def preview(self, args: ExitPlanModeInput, ctx: ToolContext) -> str:
        return args.plan

    async def run(self, args: ExitPlanModeInput, ctx: ToolContext) -> str:
        return "The user approved your plan and plan mode is off. Start implementing it now."
