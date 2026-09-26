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
      }
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

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


class Settings(BaseModel):
    # Unknown top-level keys are ignored so later phases can add sections
    # without older versions refusing to start.
    model_config = ConfigDict(extra="ignore")

    permissions: PermissionSettings = PermissionSettings()


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
        layer = _read(path)
        perms, new = merged.permissions, layer.permissions
        merged.permissions = PermissionSettings(
            default_mode=new.default_mode or perms.default_mode,
            allow=_dedupe(perms.allow + new.allow),
            ask=_dedupe(perms.ask + new.ask),
            deny=_dedupe(perms.deny + new.deny),
        )
    return merged


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
