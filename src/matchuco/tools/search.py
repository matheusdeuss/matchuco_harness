"""Search tools: glob (find files by name) and grep (find text inside files).

Both are pure Python rather than shelling out to `fd`/`rg`: the harness must
behave the same on a machine that has neither, and tests must not depend on
what is installed. The cost is speed on very large trees, which we bound with
result limits instead.

Noise control matters more than it looks. A `grep` that walks `.git/` and
`node_modules/` burns context on matches nobody wants, so both tools skip the
usual generated directories.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from matchuco.tools.base import Tool, ToolContext, ToolError
from matchuco.tools.files import is_binary

SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        ".next",
        "target",
    }
)
MAX_GLOB_RESULTS = 200
MAX_GREP_MATCHES = 100
MAX_GREP_FILE_BYTES = 5 * 1024 * 1024


class GlobInput(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py' or 'src/**/test_*.py'.")
    path: str = Field(default=".", description="Directory to search in. Defaults to the root.")


class GlobTool(Tool[GlobInput]):
    name = "glob"
    description = (
        "Find files by name pattern. Returns paths sorted by modification time, newest first, "
        "which puts recently touched files where they are most useful. "
        "Skips .git, node_modules and other generated directories."
    )
    input_model = GlobInput
    kind = "read"

    def permission_subject(self, args: GlobInput, ctx: ToolContext) -> str:
        return ctx.subject(args.path)

    async def run(self, args: GlobInput, ctx: ToolContext) -> str:
        root = ctx.resolve(args.path)
        if not root.is_dir():
            raise ToolError(f"{ctx.display(root)} is not a directory")
        try:
            matches = [p for p in root.glob(args.pattern) if p.is_file() and not _ignored(p, root)]
        except (ValueError, NotImplementedError) as e:
            raise ToolError(f"invalid glob pattern {args.pattern!r}: {e}") from e

        if not matches:
            return f"no files match {args.pattern!r} under {ctx.display(root)}"

        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        shown = matches[:MAX_GLOB_RESULTS]
        out = "\n".join(ctx.display(p) for p in shown)
        if len(matches) > len(shown):
            out += f"\n\n[{len(matches)} matches, showing the {len(shown)} most recent]"
        return out


class GrepInput(BaseModel):
    pattern: str = Field(description="Python regular expression to search for.")
    path: str = Field(default=".", description="File or directory to search. Defaults to root.")
    glob: str | None = Field(
        default=None, description="Only search files matching this glob, e.g. '*.py'."
    )
    case_insensitive: bool = Field(default=False, description="Ignore case.")
    files_only: bool = Field(
        default=False, description="List matching file paths instead of matching lines."
    )


class GrepTool(Tool[GrepInput]):
    name = "grep"
    description = (
        "Search file contents with a regular expression. Returns 'path:line: text' for each "
        "match, or just the paths with `files_only`. Binary files and generated directories "
        "are skipped. Use this to find code before reading whole files."
    )
    input_model = GrepInput
    kind = "read"

    def permission_subject(self, args: GrepInput, ctx: ToolContext) -> str:
        return ctx.subject(args.path)

    async def run(self, args: GrepInput, ctx: ToolContext) -> str:
        target = ctx.resolve(args.path)
        if not target.exists():
            raise ToolError(f"{ctx.display(target)} does not exist")
        try:
            regex = re.compile(args.pattern, re.IGNORECASE if args.case_insensitive else 0)
        except re.error as e:
            raise ToolError(f"invalid regex {args.pattern!r}: {e}") from e

        files = [target] if target.is_file() else _walk(target, args.glob)
        lines: list[str] = []
        hit_files: list[str] = []
        total = 0

        for file in files:
            matched = False
            for number, line in _search(file, regex):
                matched = True
                total += 1
                if len(lines) < MAX_GREP_MATCHES:
                    lines.append(f"{ctx.display(file)}:{number}: {line.strip()[:300]}")
                if args.files_only:
                    break
            if matched:
                hit_files.append(ctx.display(file))

        if not hit_files:
            return f"no matches for {args.pattern!r} in {ctx.display(target)}"
        if args.files_only:
            return "\n".join(hit_files)

        out = "\n".join(lines)
        if total > len(lines):
            out += f"\n\n[{total} matches in {len(hit_files)} files, showing first {len(lines)}]"
        return out


def _walk(root: Path, pattern: str | None) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob(pattern or "*")):
        if path.is_file() and not _ignored(path, root):
            files.append(path)
    return files


def _ignored(path: Path, root: Path) -> bool:
    relative = path.relative_to(root) if path.is_relative_to(root) else path
    return any(part in SKIP_DIRS for part in relative.parts)


def _search(path: Path, regex: re.Pattern[str]) -> list[tuple[int, str]]:
    """Matches in one file as (line number, line). Unreadable files are skipped, not fatal."""
    try:
        if path.stat().st_size > MAX_GREP_FILE_BYTES or is_binary(path):
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [(n, line) for n, line in enumerate(text.splitlines(), 1) if regex.search(line)]
