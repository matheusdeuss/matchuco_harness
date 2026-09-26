import json
from pathlib import Path

import pytest

from matchuco.config import (
    SettingsError,
    add_local_allow_rule,
    build_policy,
    load_settings,
)
from matchuco.permissions import Rule


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("MATCHUCO_HOME", str(home))
    return home


def write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_no_files_means_defaults(tmp_path: Path, home: Path) -> None:
    policy = build_policy(load_settings(tmp_path))
    assert policy.mode == "default"
    assert policy.allow == policy.ask == policy.deny == []


def test_layers_concatenate_rules_and_override_mode(tmp_path: Path, home: Path) -> None:
    root = tmp_path / "repo"
    write(home / "settings.json", {"permissions": {"allow": ["shell(ls)"], "default_mode": "plan"}})
    write(root / ".matchuco/settings.json", {"permissions": {"deny": ["read(.env)"]}})
    write(
        root / ".matchuco/settings.local.json",
        {"permissions": {"allow": ["shell(ls)", "shell(pwd)"], "default_mode": "accept_edits"}},
    )

    settings = load_settings(root)
    assert settings.permissions.allow == ["shell(ls)", "shell(pwd)"]  # deduplicated
    assert settings.permissions.deny == ["read(.env)"]
    assert settings.permissions.default_mode == "accept_edits"  # most specific wins
    assert build_policy(settings, "bypass").mode == "bypass"  # --mode beats every file


def test_unknown_top_level_keys_are_ignored(tmp_path: Path, home: Path) -> None:
    write(tmp_path / ".matchuco/settings.json", {"hooks": {}, "permissions": {}})
    assert load_settings(tmp_path).permissions.allow == []


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"permissions": {"default_mode": "yolo"}}),
        json.dumps({"permissions": {"allwo": []}}),  # typo in a permission key
    ],
)
def test_bad_settings_name_the_file(tmp_path: Path, home: Path, content: str) -> None:
    path = tmp_path / ".matchuco/settings.json"
    path.parent.mkdir()
    path.write_text(content)
    with pytest.raises(SettingsError, match=r"settings\.json"):
        load_settings(tmp_path)


def test_bad_rule_is_a_settings_error(tmp_path: Path, home: Path) -> None:
    write(tmp_path / ".matchuco/settings.json", {"permissions": {"allow": ["???"]}})
    with pytest.raises(SettingsError, match=r"settings\.json.*invalid permission rule"):
        load_settings(tmp_path)


def test_always_allow_is_persisted_locally(tmp_path: Path, home: Path) -> None:
    write(tmp_path / ".matchuco/settings.local.json", {"other": 1})
    add_local_allow_rule(tmp_path, Rule("shell", "uv run pytest"))
    add_local_allow_rule(tmp_path, Rule("shell", "uv run pytest"))  # idempotent

    data = json.loads((tmp_path / ".matchuco/settings.local.json").read_text())
    assert data == {"other": 1, "permissions": {"allow": ["shell(uv run pytest)"]}}
    assert load_settings(tmp_path).permissions.allow == ["shell(uv run pytest)"]
