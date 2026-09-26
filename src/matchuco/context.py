"""Context engineering: deciding what the model sees before it sees anything.

The context window holds everything the model knows about the task: the
system prompt, the tool definitions, and the conversation. This module builds
the parts that are *not* conversation, and measures the whole thing.

**The system prompt is assembled once per session and then frozen.** It is
made of three parts:

    base prompt      how to behave as a coding agent (agent.py)
    environment      OS, shell, date, git branch/status -- a snapshot
    instructions     AGENTS.md / CLAUDE.md files, from general to specific

Freezing matters for two reasons. Providers cache the longest unchanged prefix
of a request (tools -> system -> messages), so a system prompt that changes
every turn pays full price every turn. And newer Claude models bind their
reasoning blocks to the exact prefix that produced them, so editing it
mid-session invalidates them. Anything that changes during a session (the
permission mode, a compaction summary) is *appended* to the conversation
instead.

**Instruction files** are how a project teaches the agent its conventions
("run tests with `uv run pytest`", "never touch generated/"). They are
loaded from most general to most specific, so a later file can refine an
earlier one:

    ~/.matchuco/AGENTS.md                 you, in every project
    <repo root>/AGENTS.md, CLAUDE.md      the team, committed
    ... each directory down to the workspace
    <workspace>/AGENTS.local.md           you, in this project (gitignored)

Both AGENTS.md (the cross-tool convention) and CLAUDE.md are read, so a
repository set up for another agent works here unchanged.

**Measuring** uses two sources. Providers report exactly how many input
tokens a request used, but only after the fact; for everything appended
since, we estimate at ~4 characters per token. The estimate is rough on
purpose -- it only decides *when* to compact, and it is corrected by the next
real number every single turn.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from matchuco.config import user_dir
from matchuco.messages import (
    ContentBlock,
    Message,
    OpaqueBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md")
LOCAL_INSTRUCTION_FILE = "AGENTS.local.md"
MAX_INSTRUCTION_CHARS = 40_000
GIT_STATUS_LINES = 20
CHARS_PER_TOKEN = 4


# --- instruction files -----------------------------------------------------------


@dataclass(frozen=True)
class InstructionFile:
    path: Path
    scope: Literal["user", "project", "local"]
    text: str
    truncated: bool = False


def find_repo_root(start: Path) -> Path | None:
    """The nearest enclosing directory with a `.git`, if any."""
    for directory in (start, *start.parents):
        if (directory / ".git").exists():
            return directory
    return None


def load_instructions(root: Path) -> list[InstructionFile]:
    """Instruction files that apply to `root`, most general first."""
    root = root.resolve()
    found: list[InstructionFile] = []

    def add(path: Path, scope: Literal["user", "project", "local"]) -> None:
        if not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return
        if not text:
            return
        truncated = len(text) > MAX_INSTRUCTION_CHARS
        found.append(InstructionFile(path, scope, text[:MAX_INSTRUCTION_CHARS], truncated))

    add(user_dir() / "AGENTS.md", "user")

    # From the repository root down to the workspace, so deeper files come later
    # and can refine what the top-level ones say.
    top = find_repo_root(root) or root
    chain = [root, *root.parents]
    directories = list(reversed(chain[: chain.index(top) + 1]))
    for directory in directories:
        for name in INSTRUCTION_FILES:
            add(directory / name, "project")
    add(root / LOCAL_INSTRUCTION_FILE, "local")
    return found


def compact_instructions(files: list[InstructionFile]) -> str:
    """Text under any `# Compact Instructions` heading: what a summary must keep."""
    sections = []
    for file in files:
        match = re.search(
            r"^(#+)\s*Compact Instructions\s*$(.*?)(?=^#{1,6}\s|\Z)",
            file.text,
            re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        if match and match.group(2).strip():
            sections.append(match.group(2).strip())
    return "\n\n".join(sections)


# --- environment snapshot ----------------------------------------------------


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    # rstrip only: `git status --short` starts lines with a meaningful space (" M").
    return result.stdout.rstrip() if result.returncode == 0 else None


def shell_name() -> str:
    """The shell `asyncio.create_subprocess_shell` (and so the shell tool) really uses."""
    if os.name == "nt":
        return os.environ.get("COMSPEC", "cmd.exe")
    return "/bin/sh"


def environment_info(root: Path) -> str:
    lines = [
        f"Working directory: {root}",
        f"Platform: {platform.system()} {platform.release()} ({platform.version()})",
        f"Shell used by the shell tool: {shell_name()}",
        f"Today's date: {date.today().isoformat()}",
    ]
    if _git(root, "rev-parse", "--is-inside-work-tree") != "true":
        lines.append("Git: not a git repository")
        return "\n".join(lines)

    # `branch --show-current` also works before the first commit; empty = detached HEAD.
    branch = _git(root, "branch", "--show-current") or "(detached HEAD)"
    lines.append(f"Git branch: {branch}")
    status = (_git(root, "status", "--short") or "").splitlines()
    if status:
        shown = status[:GIT_STATUS_LINES]
        more = len(status) - len(shown)
        lines.append("Git status (at session start):\n" + "\n".join(shown))
        if more:
            lines.append(f"... and {more} more changed files")
    else:
        lines.append("Git status (at session start): clean")
    log = _git(root, "log", "--oneline", "-5")
    if log:
        lines.append("Recent commits:\n" + log)
    return "\n".join(lines)


# --- the system prompt ---------------------------------------------------------


@dataclass(frozen=True)
class SystemPrompt:
    """The frozen system prompt, kept in labelled parts so /context can show them."""

    parts: tuple[tuple[str, str], ...]
    instructions: tuple[InstructionFile, ...] = ()

    @property
    def text(self) -> str:
        return "\n\n".join(text for _, text in self.parts)


def build_system_prompt(base: str, root: Path) -> SystemPrompt:
    parts: list[tuple[str, str]] = [("base prompt", base)]
    parts.append(("environment", "# Environment\n\n" + environment_info(root)))
    instructions = load_instructions(root)
    for file in instructions:
        note = " (truncated)" if file.truncated else ""
        header = f"# Instructions from {file.path} ({file.scope}){note}"
        parts.append((f"instructions: {file.path.name} ({file.scope})", f"{header}\n\n{file.text}"))
    if instructions:
        parts.insert(
            2,
            (
                "instructions preamble",
                "The files below contain instructions from the user and the project. "
                "Follow them; when two conflict, the one listed later is more specific and wins.",
            ),
        )
    return SystemPrompt(tuple(parts), tuple(instructions))


# --- measuring -----------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def estimate_block(block: ContentBlock) -> int:
    match block:
        case TextBlock():
            return estimate_tokens(block.text)
        case ToolUseBlock():
            return estimate_tokens(block.name + json.dumps(block.input))
        case ToolResultBlock():
            return estimate_tokens(block.content)
        case ThinkingBlock():
            return estimate_tokens(block.text)
        case OpaqueBlock():
            return estimate_tokens(json.dumps(block.raw))
    return 0


def estimate_messages(messages: list[Message]) -> int:
    # A few tokens of framing per message (role markers and the like).
    return sum(4 + sum(estimate_block(b) for b in m.content) for m in messages)


def estimate_tools(tools: list[ToolSpec]) -> int:
    return estimate_tokens(json.dumps([t.model_dump() for t in tools]))


def block_category(block: ContentBlock, role: str) -> str:
    match block:
        case ToolUseBlock():
            return "tool calls"
        case ToolResultBlock():
            return "tool results"
        case ThinkingBlock():
            return "reasoning"
        case OpaqueBlock():
            return "other"
    return "user messages" if role == "user" else "assistant messages"
