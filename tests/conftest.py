from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Keep every test away from the real ~/.matchuco (settings, AGENTS.md)."""
    home = tmp_path_factory.mktemp("matchuco-home")
    monkeypatch.setenv("MATCHUCO_HOME", str(home))
    return home
