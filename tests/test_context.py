import subprocess
from pathlib import Path

import pytest

from matchuco.agent import (
    PLAN_MODE_REMINDER,
    SUMMARY_PROMPT,
    Agent,
    Compacted,
    Compacting,
    ContextOverflowError,
    TurnEnd,
)
from matchuco.context import (
    MAX_INSTRUCTION_CHARS,
    build_system_prompt,
    compact_instructions,
    environment_info,
    estimate_messages,
    estimate_tokens,
    load_instructions,
)
from matchuco.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from matchuco.permissions import PermissionPolicy
from matchuco.providers import ProviderError
from matchuco.providers.fake import FakeProvider
from matchuco.tools import ReadTool

# --- instruction files ---------------------------------------------------------


def test_instructions_load_from_general_to_specific(tmp_path: Path, isolated_home: Path) -> None:
    repo = tmp_path / "repo"
    workspace = repo / "packages" / "api"
    workspace.mkdir(parents=True)
    (repo / ".git").mkdir()
    (isolated_home / "AGENTS.md").write_text("user rules")
    (repo / "AGENTS.md").write_text("repo rules")
    (repo / "packages" / "CLAUDE.md").write_text("packages rules")
    (workspace / "AGENTS.md").write_text("api rules")
    (workspace / "AGENTS.local.md").write_text("my rules")
    (tmp_path / "AGENTS.md").write_text("outside the repo: ignored")

    files = load_instructions(workspace)

    assert [f.text for f in files] == [
        "user rules",
        "repo rules",
        "packages rules",
        "api rules",
        "my rules",
    ]
    assert [f.scope for f in files] == ["user", "project", "project", "project", "local"]


def test_without_a_repo_only_the_workspace_is_searched(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("parent")
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "AGENTS.md").write_text("ws")
    assert [f.text for f in load_instructions(tmp_path / "ws")] == ["ws"]


def test_huge_and_empty_instruction_files(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("x" * (MAX_INSTRUCTION_CHARS + 10))
    (tmp_path / "CLAUDE.md").write_text("   \n")
    [only] = load_instructions(tmp_path)
    assert only.truncated and len(only.text) == MAX_INSTRUCTION_CHARS


def test_compact_instructions_section(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "# Project\nuse uv\n\n## Compact Instructions\nKeep the list of API endpoints.\n\n"
        "## Style\nblack"
    )
    assert compact_instructions(load_instructions(tmp_path)) == "Keep the list of API endpoints."


def test_system_prompt_parts(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Run tests with uv run pytest.")
    prompt = build_system_prompt("BASE", tmp_path)
    labels = [label for label, _ in prompt.parts]
    assert labels[:2] == ["base prompt", "environment"]
    assert "instructions: AGENTS.md (project)" in labels
    assert prompt.text.startswith("BASE")
    assert "Run tests with uv run pytest." in prompt.text


# --- environment -------------------------------------------------------------------


def test_environment_outside_git(tmp_path: Path) -> None:
    info = environment_info(tmp_path)
    assert "not a git repository" in info
    assert str(tmp_path) in info


def test_environment_in_a_fresh_git_repo(tmp_path: Path) -> None:
    try:
        subprocess.run(
            ["git", "init", "-b", "trunk"], cwd=tmp_path, check=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    (tmp_path / "new.py").write_text("")
    (tmp_path / "staged.py").write_text("")
    subprocess.run(["git", "add", "staged.py"], cwd=tmp_path, check=True)
    info = environment_info(tmp_path)
    assert "Git branch: trunk" in info  # works before the first commit
    assert "?? new.py" in info
    assert "A  staged.py" in info  # status columns survive (first line included)


# --- measuring ---------------------------------------------------------------------


def test_estimates_are_about_four_chars_per_token() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    empty = estimate_messages([Message.user("")])
    big = estimate_messages([Message.user("x" * 4000)])
    assert big - empty == 1000


def small_agent(script: list[Message | str], window: int = 2000, **kw: object) -> Agent:
    """An agent whose fixed overhead is tiny, so tests control what fills the window."""
    provider = FakeProvider(script, context_window=window)
    return Agent(provider, tools=[ReadTool()], system="s", **kw)  # type: ignore[arg-type]


async def test_context_tokens_start_from_the_provider_measurement(tmp_path: Path) -> None:
    agent = small_agent(["one two three"], root=tmp_path)
    before = agent.context_tokens()
    [e async for e in agent.run("hi")]
    # FakeProvider reports 10 input tokens and 3 output tokens.
    assert agent.context_tokens() == 13
    assert before > 0

    report = agent.context_report()
    names = [name for name, _ in report.categories]
    assert names[:2] == ["system: system prompt", "tool definitions"]
    assert "assistant messages" in names and "user messages" in names
    assert report.last_reported == 10
    assert report.threshold == 1600


# --- compaction ----------------------------------------------------------------------


def read_call(path: str) -> Message:
    return Message(
        role="assistant", content=[ToolUseBlock(id="r1", name="read", input={"path": path})]
    )


async def test_auto_compaction_mid_turn(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("line of text\n" * 800)  # ~2.6k tokens with line numbers
    agent = small_agent(
        [read_call("big.txt"), "<summary>Read big.txt; it repeats one line.</summary>", "done"],
        root=tmp_path,
    )

    events = [e async for e in agent.run("summarize big.txt")]

    kinds = [type(e).__name__ for e in events]
    assert kinds.index("Compacting") < kinds.index("Compacted") < kinds.index("TurnEnd")
    compacted = next(e for e in events if isinstance(e, Compacted))
    assert compacted.auto and compacted.tokens_after < compacted.tokens_before

    # The summary request = the conversation + the instructions, same tools.
    summary_request = agent.provider.requests[1]  # type: ignore[attr-defined]
    last_block = summary_request.messages[-1].content[-1]
    assert isinstance(last_block, TextBlock) and last_block.text == SUMMARY_PROMPT
    assert isinstance(summary_request.messages[-1].content[0], ToolResultBlock)
    assert [t.name for t in summary_request.tools] == ["read"]

    # The model continued from a one-message conversation: summary + "continue".
    continued = agent.provider.requests[2].messages  # type: ignore[attr-defined]
    assert len(continued) == 1
    assert "Read big.txt; it repeats one line." in continued[0].text
    assert "Continue the task" in continued[0].text
    assert agent.context.files_read == {}  # must re-read before editing
    assert isinstance(events[-1], TurnEnd)


async def test_auto_compaction_before_a_new_prompt_keeps_the_prompt(tmp_path: Path) -> None:
    agent = small_agent(["x " * 2000, "<summary>We chatted.</summary>", "ok"], root=tmp_path)
    [e async for e in agent.run("first")]
    assert agent.context_tokens() > agent.compact_at

    [e async for e in agent.run("second question")]

    assert [m.role for m in agent.history] == ["user", "assistant"]
    first = agent.history[0].content
    assert "We chatted." in first[0].text  # type: ignore[union-attr]
    assert first[-1] == TextBlock(text="second question")
    # The pending prompt was not part of what got summarized.
    summarized = agent.provider.requests[1].messages  # type: ignore[attr-defined]
    assert "second question" not in "".join(m.text for m in summarized)


async def test_manual_compact_carries_the_summary_into_the_next_prompt(tmp_path: Path) -> None:
    agent = small_agent(
        ["hello!", "<summary>Greeted.</summary>", "fine"],
        root=tmp_path,
        permissions=PermissionPolicy(mode="plan"),
    )
    [e async for e in agent.run("hi")]

    result = await agent.compact("the greeting")

    assert result is not None and not result.auto
    assert agent.history == []
    focus_request = agent.provider.requests[1].messages[-1].text  # type: ignore[attr-defined]
    assert "focus on: the greeting" in focus_request

    [e async for e in agent.run("how are you?")]
    sent = agent.provider.requests[2].messages  # type: ignore[attr-defined]
    assert len(sent) == 1
    blocks = [b.text for b in sent[0].content if isinstance(b, TextBlock)]
    assert "Greeted." in blocks[0]
    assert blocks[1] == PLAN_MODE_REMINDER  # plan mode is re-announced after compaction
    assert blocks[-1] == "how are you?"


async def test_compact_with_nothing_to_compact(tmp_path: Path) -> None:
    assert await small_agent([], root=tmp_path).compact() is None


async def test_project_compact_instructions_reach_the_summarizer(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("## Compact Instructions\nAlways keep the TODO list.")
    provider = FakeProvider(["hello", "<summary>s</summary>"])
    agent = Agent(provider, root=tmp_path, tools=[ReadTool()])
    [e async for e in agent.run("hi")]
    await agent.compact()
    assert "Always keep the TODO list." in provider.requests[1].messages[-1].text


async def test_overflow_that_compaction_cannot_fix(tmp_path: Path) -> None:
    agent = small_agent(["never sent"], root=tmp_path)
    with pytest.raises(ContextOverflowError):
        [e async for e in agent.run("x" * 20_000)]
    assert agent.history == []  # restored


async def test_failure_after_compaction_restores_the_old_conversation(tmp_path: Path) -> None:
    agent = small_agent(["x " * 2000, "<summary>s</summary>"], root=tmp_path)  # then runs out
    [e async for e in agent.run("first")]
    before = list(agent.history)

    with pytest.raises(ProviderError):
        [e async for e in agent.run("second")]

    assert agent.history == before
    events = [e async for e in small_agent(["a"], root=tmp_path).run("q")]
    assert not any(isinstance(e, Compacting) for e in events)


async def test_clear_rereads_instruction_files(tmp_path: Path) -> None:
    agent = Agent(FakeProvider([]), root=tmp_path)
    assert "new rule" not in agent.system
    (tmp_path / "AGENTS.md").write_text("new rule")
    agent.clear()
    assert "new rule" in agent.system


def test_tool_result_blocks_count_toward_the_estimate() -> None:
    small = estimate_messages(
        [Message(role="user", content=[ToolResultBlock(tool_use_id="a", content="")])]
    )
    big = estimate_messages(
        [Message(role="user", content=[ToolResultBlock(tool_use_id="a", content="y" * 400)])]
    )
    assert big - small == 100
