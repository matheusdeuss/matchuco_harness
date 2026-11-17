"""Settings, loaded in layers like Claude Code's settings.json files.

    ~/.matchuco/settings.json            user: your preferences, every project
    <root>/.matchuco/settings.json       project: shared with the team, committed
    <root>/.matchuco/settings.local.json local: yours, for this project, gitignored

Rule lists are concatenated across layers (so a project can add a deny rule on
top of your personal allow list); scalar values like `default_mode` are
overridden by the more specific layer. `MATCHUCO_HOME` relocates the user
directory, which is how the tests keep away from your real one.

    {
      "permissions": {
        "default_mode": "default",
        "allow": ["shell(uv run pytest *)", "shell(git status)"],
        "ask":   ["shell(git push *)"],
        "deny":  ["read(.env)", "shell(rm -rf *)"]
      },
      "hooks": {
        "PreToolUse": [{"matcher": "shell", "deny": "no shell in this project"}],
        "PostToolUse": [{"matcher": "read", "note": "file contents may be stale"}]
      }
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from matchuco.permissions import Mode, PermissionPolicy, Rule

PROJECT_DIR = ".matchuco"


class SettingsError(Exception):
    """A settings file exists but cannot be used. Reported with its path."""


class PermissionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_mode: Mode | None = None
    allow: list[str] = []
    ask: list[str] = []
    deny: list[str] = []


class PreToolHook(BaseModel):
    """A PreToolUse hook declared in settings: matching calls are denied."""

    model_config = ConfigDict(extra="forbid")

    matcher: str = "*"
    deny: str

    @field_validator("matcher", "deny")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must not be empty")
        return text


class PostToolHook(BaseModel):
    """A PostToolUse hook declared in settings: matching results gain a note."""

    model_config = ConfigDict(extra="forbid")

    matcher: str = "*"
    note: str

    @field_validator("matcher", "note")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must not be empty")
        return text


class HooksSettings(BaseModel):
    """In-process hooks. A command string is not a hook; register a callable for that."""

    model_config = ConfigDict(extra="forbid")

    PreToolUse: list[PreToolHook] = []
    PostToolUse: list[PostToolHook] = []


class Settings(BaseModel):
    # Unknown top-level keys are ignored so later phases can add sections
    # without older versions refusing to start.
    model_config = ConfigDict(extra="ignore")

    permissions: PermissionSettings = PermissionSettings()
    hooks: HooksSettings = HooksSettings()


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def user_dir() -> Path:
    return Path(os.environ.get("MATCHUCO_HOME") or Path.home() / ".matchuco")


def settings_files(root: Path) -> list[Path]:
    """The layers, least specific first."""
    project = root / PROJECT_DIR
    return [user_dir() / "settings.json", project / "settings.json", local_settings_file(root)]


def local_settings_file(root: Path) -> Path:
    return root / PROJECT_DIR / "settings.local.json"


def load_settings(root: Path) -> Settings:
    merged = Settings()
    for path in settings_files(root):
        if not path.is_file():
            continue
        merged = _merge(merged, _read(path))
    return merged


def _merge(base: Settings, layer: Settings) -> Settings:
    """Rule lists and hook lists concatenate; `default_mode` is overridden."""
    old, new = base.permissions, layer.permissions
    return Settings(
        permissions=PermissionSettings(
            default_mode=new.default_mode or old.default_mode,
            allow=_dedupe(old.allow + new.allow),
            ask=_dedupe(old.ask + new.ask),
            deny=_dedupe(old.deny + new.deny),
        ),
        hooks=HooksSettings(
            PreToolUse=_dedupe_models(base.hooks.PreToolUse + layer.hooks.PreToolUse),
            PostToolUse=_dedupe_models(base.hooks.PostToolUse + layer.hooks.PostToolUse),
        ),
    )


def _read(path: Path) -> Settings:
    """Parse one layer, validating rules here so an error can name the file it is in."""
    try:
        settings = Settings.model_validate_json(path.read_text(encoding="utf-8"))
        perms = settings.permissions
        for text in (*perms.allow, *perms.ask, *perms.deny):
            Rule.parse(text)
    except (ValidationError, ValueError, OSError) as e:
        raise SettingsError(f"{path}: {e}") from e
    return settings


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _dedupe_models(items: list[_ModelT]) -> list[_ModelT]:
    seen: set[str] = set()
    unique: list[_ModelT] = []
    for item in items:
        key = item.model_dump_json()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def build_policy(settings: Settings, mode: Mode | None = None) -> PermissionPolicy:
    """Turn settings into a policy. An explicit `mode` (e.g. --mode) wins over settings."""
    perms = settings.permissions
    try:
        return PermissionPolicy.from_rules(
            mode or perms.default_mode or "default",
            allow=perms.allow,
            ask=perms.ask,
            deny=perms.deny,
        )
    except ValueError as e:
        raise SettingsError(str(e)) from e


def add_local_allow_rule(root: Path, rule: Rule) -> Path:
    """Persist an "always allow" answer to the local (gitignored) settings file."""
    path = local_settings_file(root)
    data: dict[str, object] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SettingsError(f"{path}: {e}") from e
    permissions = data.setdefault("permissions", {})
    if not isinstance(permissions, dict):
        raise SettingsError(f"{path}: 'permissions' must be an object")
    allow = permissions.setdefault("allow", [])
    if str(rule) not in allow:
        allow.append(str(rule))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
