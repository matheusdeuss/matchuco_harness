"""Skills: the model sees a name and a one-line description, then loads the body.

A skill is a Markdown file at `.matchuco/skills/<name>.md`:

```
# Title

Short description, one line.

The body. It is not part of the first request. The `skill` tool returns it
as a tool_result when the model asks for this name.
```

The title is the first heading. The description is the next non-empty line.
Everything after that is the body. The tool spec lists `name: description`
only, so the body stays out of the prompt until `skill` is called. An unknown
name is a tool error the model can read, not an exception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from matchuco.tools.base import Tool, ToolContext, ToolError

__all__ = ["Skill", "SkillError", "SkillTool", "discover_skills", "parse_skill"]

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_HEADING = re.compile(r"^#{1,6}\s+(\S.*?)\s*$")

_INTRO = (
    "Load a skill's full instructions by name. Until you call this tool you only "
    "know each skill's name and one-line description; the body is not in the prompt."
)


class SkillError(Exception):
    """A skill file exists but cannot be read."""


@dataclass(frozen=True)
class Skill:
    name: str
    title: str
    description: str
    body: str


def discover_skills(root: Path) -> list[Skill]:
    """Read `.matchuco/skills/*.md`. Names come from the file stem."""
    directory = root / ".matchuco" / "skills"
    if not directory.is_dir():
        return []
    found: list[Skill] = []
    for path in sorted(directory.glob("*.md")):
        if not path.is_file():
            continue
        name = path.stem
        if _NAME.fullmatch(name) is None:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SkillError(f"{path}: not valid UTF-8 ({exc})") from exc
        except OSError as exc:
            raise SkillError(f"{path}: {exc}") from exc
        found.append(parse_skill(name, text))
    return found


def parse_skill(name: str, text: str) -> Skill:
    """Split a skill file into title, one-line description, and body."""
    lines = text.splitlines()
    index = _skip_blank(lines, 0)
    title = name
    if index < len(lines) and (heading := _HEADING.match(lines[index].strip())):
        title = heading.group(1).strip()
        index = _skip_blank(lines, index + 1)
    if index < len(lines):
        description = lines[index].strip()
        index = _skip_blank(lines, index + 1)
    else:
        description = ""
    body = "\n".join(lines[index:]).strip()
    return Skill(name=name, title=title, description=description, body=body)


def _skip_blank(lines: list[str], start: int) -> int:
    while start < len(lines) and not lines[start].strip():
        start += 1
    return start


class SkillInput(BaseModel):
    name: str = Field(description="Name of the skill to load, as listed above.")


class SkillTool(Tool[SkillInput]):
    name = "skill"
    description = _INTRO
    input_model = SkillInput
    kind = "read"

    def __init__(self) -> None:
        self._skills: list[Skill] = []

    def bind(self, root: Path) -> None:
        """Load the catalog for this workspace. The body stays off the description."""
        self._skills = discover_skills(root)
        if not self._skills:
            self.description = f"{_INTRO}\n\nAvailable skills:\n(none)"
            return
        listing = "\n".join(f"- {skill.name}: {skill.description}" for skill in self._skills)
        self.description = f"{_INTRO}\n\nAvailable skills:\n{listing}"

    def permission_subject(self, args: SkillInput, ctx: ToolContext) -> str:
        return args.name

    def preview(self, args: SkillInput, ctx: ToolContext) -> str:
        skill = self._find(args.name)
        if skill is None:
            return args.name
        return f"{skill.name}: {skill.description}"

    async def run(self, args: SkillInput, ctx: ToolContext) -> str:
        if _NAME.fullmatch(args.name) is None or (skill := self._find(args.name)) is None:
            known = ", ".join(item.name for item in self._skills) or "(none)"
            raise ToolError(f"unknown skill {args.name!r}; available: {known}")
        if not skill.body:
            raise ToolError(f"skill {args.name!r} has no body beyond its title and description")
        if skill.title:
            return f"# {skill.title}\n\n{skill.body}"
        return skill.body

    def _find(self, name: str) -> Skill | None:
        return next((skill for skill in self._skills if skill.name == name), None)
