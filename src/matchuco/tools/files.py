"""File tools: read, write, edit.

The interesting design is not the I/O, it is the guard rails:

- `read` numbers lines, so `edit` can talk about them and the model can quote
  them back accurately.
- `write` and `edit` refuse to touch a file the model has not read this
  session (see `ToolContext.check_readable_before_write`).
- `edit` replaces an exact string and refuses ambiguous matches, instead of
  taking line numbers or a diff. Line numbers drift; a unique snippet does not.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from pydantic import BaseModel, Field

from matchuco.tools.base import Tool, ToolContext, ToolError

MAX_READ_LINES = 2000
MAX_LINE_LENGTH = 2000
MAX_FILE_BYTES = 10 * 1024 * 1024
PREVIEW_LINES = 40
PREVIEW_DIFF_LINES = 80


class ReadInput(BaseModel):
    path: str = Field(description="File to read, relative to the workspace or absolute.")
    offset: int = Field(default=1, ge=1, description="1-based line to start at.")
    limit: int = Field(
        default=MAX_READ_LINES, ge=1, description=f"Max lines to return (default {MAX_READ_LINES})."
    )


class ReadTool(Tool[ReadInput]):
    name = "read"
    description = (
        "Read a text file. Returns the contents with 1-based line numbers. "
        "Long files are truncated: use `offset` and `limit` to page through them. "
        "You must read a file before writing or editing it."
    )
    input_model = ReadInput
    kind = "read"

    def permission_subject(self, args: ReadInput, ctx: ToolContext) -> str:
        return ctx.subject(args.path)

    async def run(self, args: ReadInput, ctx: ToolContext) -> str:
        path = ctx.resolve(args.path)
        if path.is_dir():
            raise ToolError(f"{ctx.display(path)} is a directory; use glob to list its contents")
        if not path.exists():
            raise ToolError(f"{ctx.display(path)} does not exist")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ToolError(f"{ctx.display(path)} is larger than {MAX_FILE_BYTES} bytes")
        if is_binary(path):
            raise ToolError(f"{ctx.display(path)} looks like a binary file")

        ctx.mark_read(path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return f"{ctx.display(path)} is empty"

        start = args.offset - 1
        if start >= len(lines):
            raise ToolError(f"{ctx.display(path)} has only {len(lines)} lines")
        window = lines[start : start + args.limit]

        width = len(str(start + len(window)))
        body = "\n".join(
            f"{n:>{width}}\t{_clip(line)}" for n, line in enumerate(window, start=args.offset)
        )
        shown_to = start + len(window)
        if shown_to < len(lines):
            body += f"\n\n[showing lines {args.offset}-{shown_to} of {len(lines)}]"
        return body


class WriteInput(BaseModel):
    path: str = Field(description="File to write, relative to the workspace or absolute.")
    content: str = Field(description="Full new contents of the file.")


class WriteTool(Tool[WriteInput]):
    name = "write"
    description = (
        "Write a file, creating parent directories as needed. Overwrites the whole file, "
        "so read it first if it already exists. Prefer `edit` for changes to existing files."
    )
    input_model = WriteInput
    kind = "edit"

    def permission_subject(self, args: WriteInput, ctx: ToolContext) -> str:
        return ctx.subject(args.path)

    def preview(self, args: WriteInput, ctx: ToolContext) -> str:
        name = ctx.subject(args.path)
        try:
            path = ctx.resolve(args.path)
            old = path.read_text(encoding="utf-8") if path.is_file() else None
        except (ToolError, OSError, UnicodeDecodeError):
            old = None
        if old is None:
            lines = args.content.splitlines()
            head = "\n".join(lines[:PREVIEW_LINES])
            more = (
                f"\n... +{len(lines) - PREVIEW_LINES} lines" if len(lines) > PREVIEW_LINES else ""
            )
            return f"new file {name}\n\n{head}{more}"
        return unified_diff(old, args.content, name)

    async def run(self, args: WriteInput, ctx: ToolContext) -> str:
        path = ctx.resolve(args.path)
        if path.is_dir():
            raise ToolError(f"{ctx.display(path)} is a directory")
        ctx.check_readable_before_write(path)

        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.content, encoding="utf-8", newline="\n")
        ctx.mark_read(path)  # the model now knows exactly what is on disk

        verb = "updated" if existed else "created"
        return f"{verb} {ctx.display(path)} ({len(args.content.splitlines())} lines)"


class EditInput(BaseModel):
    path: str = Field(description="File to edit.")
    old_string: str = Field(description="Exact text to replace, including indentation.")
    new_string: str = Field(description="Text to replace it with.")
    replace_all: bool = Field(
        default=False, description="Replace every occurrence instead of requiring a unique match."
    )


class EditTool(Tool[EditInput]):
    name = "edit"
    description = (
        "Replace an exact string in a file. `old_string` must match the file byte for byte "
        "(copy it from `read` output without the line-number prefix) and must be unique, "
        "unless `replace_all` is set. Read the file first."
    )
    input_model = EditInput
    kind = "edit"

    def permission_subject(self, args: EditInput, ctx: ToolContext) -> str:
        return ctx.subject(args.path)

    def preview(self, args: EditInput, ctx: ToolContext) -> str:
        note = " (every occurrence)" if args.replace_all else ""
        return unified_diff(args.old_string, args.new_string, ctx.subject(args.path) + note)

    async def run(self, args: EditInput, ctx: ToolContext) -> str:
        path = ctx.resolve(args.path)
        if not path.exists():
            raise ToolError(f"{ctx.display(path)} does not exist; use write to create it")
        if args.old_string == args.new_string:
            raise ToolError("old_string and new_string are identical")
        ctx.check_readable_before_write(path)

        text = path.read_text(encoding="utf-8")
        count = text.count(args.old_string)
        if count == 0:
            raise ToolError(
                f"old_string not found in {ctx.display(path)}; "
                "read the file again and copy the exact text, including whitespace"
            )
        if count > 1 and not args.replace_all:
            raise ToolError(
                f"old_string appears {count} times in {ctx.display(path)}; "
                "add surrounding lines to make it unique, or set replace_all"
            )

        path.write_text(text.replace(args.old_string, args.new_string), encoding="utf-8")
        ctx.mark_read(path)
        where = f"{count} occurrences" if count > 1 else "1 occurrence"
        return f"edited {ctx.display(path)} ({where} replaced)"


def unified_diff(old: str, new: str, name: str) -> str:
    """A unified diff for approval prompts, capped so a huge rewrite stays readable."""
    diff = list(
        difflib.unified_diff(
            old.splitlines(), new.splitlines(), f"a/{name}", f"b/{name}", lineterm=""
        )
    )
    if len(diff) > PREVIEW_DIFF_LINES:
        diff = [*diff[:PREVIEW_DIFF_LINES], f"... +{len(diff) - PREVIEW_DIFF_LINES} diff lines"]
    return "\n".join(diff) or f"(no textual change to {name})"


def is_binary(path: Path) -> bool:
    """A NUL byte in the first block is the cheap, standard heuristic."""
    with path.open("rb") as fh:
        return b"\0" in fh.read(8192)


def _clip(line: str) -> str:
    if len(line) <= MAX_LINE_LENGTH:
        return line
    return f"{line[:MAX_LINE_LENGTH]}... [line truncated, {len(line)} chars]"
