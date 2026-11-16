"""`.matchuco/memory.md`, and the `remember` tool that appends to it.

The file is read while the system prompt is assembled, which happens once per
session. `remember` only touches the file. It does not rebuild the prompt, so
a note saved mid-turn is invisible until the next session (or `/clear`, which
starts one). An empty or missing file adds nothing to the prompt: the agent
behaves exactly as it did before this phase.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from matchuco.tools.base import Tool, ToolContext, ToolError

MEMORY_FILE = "memory.md"


def memory_path(root: Path) -> Path:
    return root / ".matchuco" / MEMORY_FILE


def load_memory(root: Path) -> str:
    """The memory file's text, or "" when there is nothing to inject."""
    path = memory_path(root)
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


class RememberInput(BaseModel):
    note: str = Field(
        description="A durable fact, preference, or convention to keep for future sessions."
    )


class RememberTool(Tool[RememberInput]):
    name = "remember"
    description = (
        "Append a note to auto memory (`.matchuco/memory.md`) so a future session can see it. "
        "Memory is loaded into the system prompt only when a session starts; this call does "
        "not change the prompt of the current session. Use it for facts that should outlive "
        "the conversation, not for task scratch notes."
    )
    input_model = RememberInput
    kind = "edit"

    def permission_subject(self, args: RememberInput, ctx: ToolContext) -> str:
        return f".matchuco/{MEMORY_FILE}"

    def preview(self, args: RememberInput, ctx: ToolContext) -> str:
        return f"append to .matchuco/{MEMORY_FILE}:\n\n{args.note.strip()}"

    async def run(self, args: RememberInput, ctx: ToolContext) -> str:
        note = args.note.strip()
        if not note:
            raise ToolError("note is empty")
        entry = note if note.startswith("- ") else f"- {note}"
        path = memory_path(ctx.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        prefix = existing
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        path.write_text(prefix + entry + "\n", encoding="utf-8", newline="\n")
        return (
            f"remembered in {ctx.display(path)} for future sessions; "
            "the current system prompt is unchanged"
        )
