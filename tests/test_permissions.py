from pathlib import Path
from typing import Any

import pytest

from matchuco.agent import Agent, ToolFinished, TurnEnd
from matchuco.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from matchuco.permissions import (
    PermissionPolicy,
    PermissionReply,
    PermissionRequest,
    Rule,
    path_matches,
    split_command,
)
from matchuco.providers.fake import FakeProvider
from matchuco.tools import (
    EditTool,
    ExitPlanModeTool,
    ReadTool,
    ShellTool,
    Tool,
    ToolContext,
    WriteTool,
)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n")
    return ToolContext(root=tmp_path)


def decide(policy: PermissionPolicy, tool: Tool[Any], ctx: ToolContext, **raw: Any) -> str:
    return policy.evaluate(tool, tool.parse(raw), ctx).behavior


class ScriptedApprover:
    """Answers permission requests from a list, and records what it was asked."""

    def __init__(self, *replies: PermissionReply) -> None:
        self.replies = list(replies)
        self.requests: list[PermissionRequest] = []

    async def __call__(self, request: PermissionRequest) -> PermissionReply:
        self.requests.append(request)
        return self.replies.pop(0)


# --- rules ---------------------------------------------------------------------


def test_rule_parsing_roundtrips() -> None:
    assert Rule.parse("shell") == Rule("shell")
    assert Rule.parse(" shell( git status ) ") == Rule("shell", "git status")
    assert Rule.parse("edit()") == Rule("edit")
    assert str(Rule.parse("edit(src/**)")) == "edit(src/**)"
    with pytest.raises(ValueError):
        Rule.parse("not a rule!")


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("src/*.py", "src/app.py", True),
        ("src/*.py", "src/sub/app.py", False),  # * stays inside one directory
        ("src/**", "src/sub/app.py", True),
        ("src/", "src/sub/app.py", True),  # trailing slash = everything under it
        ("**/*.py", "app.py", True),
        ("**/*.py", "a/b/app.py", True),
        (".env", ".env", True),
        (".env", "config/.env", False),
        ("**/.env", "config/.env", True),
    ],
)
def test_path_patterns(pattern: str, path: str, expected: bool) -> None:
    assert path_matches(pattern, path) is expected


def test_split_command() -> None:
    assert split_command("git status") == (["git status"], True)
    assert split_command("git add . && git commit -m 'a && b'") == (
        ["git add .", "git commit -m a && b"],
        True,
    )
    assert split_command("ls | grep x; echo done")[0] == ["ls", "grep x", "echo done"]
    assert split_command(r"dir C:\src")[0] == [r"dir C:\src"]  # backslashes survive


@pytest.mark.parametrize(
    "command",
    ["echo $(whoami)", "echo `id`", "echo hi > out.txt", "cat <(ls)", "a\nb", "echo 'open"],
)
def test_complex_commands_are_not_simple(command: str) -> None:
    assert split_command(command)[1] is False


# --- the decision table ------------------------------------------------------


def test_default_mode(ctx: ToolContext) -> None:
    policy = PermissionPolicy()
    assert decide(policy, ReadTool(), ctx, path="src/app.py") == "allow"
    assert decide(policy, EditTool(), ctx, path="src/app.py", old_string="1", new_string="2") == (
        "ask"
    )
    assert decide(policy, ShellTool(), ctx, command="pytest") == "ask"


def test_accept_edits_mode(ctx: ToolContext) -> None:
    policy = PermissionPolicy(mode="accept_edits")
    assert decide(policy, WriteTool(), ctx, path="new.py", content="") == "allow"
    assert decide(policy, ShellTool(), ctx, command="pytest") == "ask"


def test_plan_mode_is_read_only(ctx: ToolContext) -> None:
    policy = PermissionPolicy.from_rules("plan", allow=["shell", "write"])
    assert decide(policy, ReadTool(), ctx, path="src/app.py") == "allow"
    # Not even an allow rule can make plan mode write.
    assert decide(policy, WriteTool(), ctx, path="new.py", content="") == "deny"
    assert decide(policy, ShellTool(), ctx, command="ls") == "deny"
    assert decide(policy, ExitPlanModeTool(), ctx, plan="do it") == "ask"


def test_exit_plan_mode_outside_plan_mode_is_denied(ctx: ToolContext) -> None:
    assert decide(PermissionPolicy(), ExitPlanModeTool(), ctx, plan="x") == "deny"


def test_bypass_allows_everything_but_deny_and_ask_rules(ctx: ToolContext) -> None:
    policy = PermissionPolicy.from_rules("bypass", ask=["shell(git push *)"], deny=["shell(rm *)"])
    assert decide(policy, ShellTool(), ctx, command="make deploy") == "allow"
    assert decide(policy, ShellTool(), ctx, command="git push origin main") == "ask"
    assert decide(policy, ShellTool(), ctx, command="rm -rf build") == "deny"


def test_deny_rules_beat_allow_rules_and_read_tools(ctx: ToolContext) -> None:
    policy = PermissionPolicy.from_rules(allow=["read"], deny=["read(.env)"])
    (ctx.root / ".env").write_text("SECRET=1")
    assert decide(policy, ReadTool(), ctx, path=".env") == "deny"
    assert decide(policy, ReadTool(), ctx, path="./src/../.env") == "deny"  # normalised first
    assert decide(policy, ReadTool(), ctx, path="src/app.py") == "allow"


def test_allow_rules_for_files(ctx: ToolContext) -> None:
    policy = PermissionPolicy.from_rules(allow=["edit(src/**)"])
    edit = EditTool()
    assert decide(policy, edit, ctx, path="src/app.py", old_string="1", new_string="2") == "allow"
    assert decide(policy, edit, ctx, path="README.md", old_string="a", new_string="b") == "ask"


def test_shell_allow_rules_must_cover_every_subcommand(ctx: ToolContext) -> None:
    policy = PermissionPolicy.from_rules(allow=["shell(git *)", "shell(uv run pytest *)"])
    shell = ShellTool()
    assert decide(policy, shell, ctx, command="git status") == "allow"
    assert decide(policy, shell, ctx, command="git") == "allow"  # `git *` covers bare `git`
    assert decide(policy, shell, ctx, command="git add . && uv run pytest -q") == "allow"
    assert decide(policy, shell, ctx, command="git status && rm -rf ~") == "ask"
    assert decide(policy, shell, ctx, command="git log > leak.txt") == "ask"
    assert decide(policy, shell, ctx, command="git log $(rm -rf ~)") == "ask"


def test_shell_deny_rules_catch_any_subcommand(ctx: ToolContext) -> None:
    policy = PermissionPolicy.from_rules("bypass", deny=["shell(rm *)"])
    assert decide(policy, ShellTool(), ctx, command="ls && rm -rf build") == "deny"


# --- asking the user ---------------------------------------------------------


async def test_check_without_approver_denies_with_guidance(ctx: ToolContext) -> None:
    shell = ShellTool()
    denial = await PermissionPolicy().check(shell, shell.parse({"command": "ls"}), ctx, None)
    assert denial is not None and "non-interactive" in denial


async def test_no_with_feedback_reaches_the_model(ctx: ToolContext) -> None:
    shell = ShellTool()
    approver = ScriptedApprover(PermissionReply("no", feedback="use uv run pytest"))
    denial = await PermissionPolicy().check(
        shell, shell.parse({"command": "pytest"}), ctx, approver
    )
    assert denial is not None and "use uv run pytest" in denial
    request = approver.requests[0]
    assert (request.tool, request.subject, request.preview) == ("shell", "pytest", "$ pytest")
    assert request.always == "always allow shell(pytest)"


async def test_always_on_a_command_adds_an_exact_rule(ctx: ToolContext) -> None:
    saved: list[Rule] = []
    policy = PermissionPolicy(on_new_rule=saved.append)
    shell = ShellTool()
    approver = ScriptedApprover(PermissionReply("always"))

    assert (
        await policy.check(shell, shell.parse({"command": "uv run pytest"}), ctx, approver) is None
    )
    assert saved == [Rule("shell", "uv run pytest")]
    # The next identical call no longer asks; a different one still does.
    assert decide(policy, shell, ctx, command="uv run pytest") == "allow"
    assert decide(policy, shell, ctx, command="uv run pytest -x") == "ask"


async def test_always_is_not_offered_for_complex_commands(ctx: ToolContext) -> None:
    shell = ShellTool()
    args = shell.parse({"command": "ls && rm x"})
    assert PermissionPolicy().suggest_always(shell, args, ctx) is None


async def test_always_on_an_edit_switches_to_accept_edits(ctx: ToolContext) -> None:
    policy = PermissionPolicy()
    write = WriteTool()
    approver = ScriptedApprover(PermissionReply("always"))
    args = write.parse({"path": "new.py", "content": "print(1)\n"})

    assert await policy.check(write, args, ctx, approver) is None
    assert policy.mode == "accept_edits"
    assert approver.requests[0].preview.startswith("new file new.py")


async def test_edit_preview_is_a_diff(ctx: ToolContext) -> None:
    edit = EditTool()
    args = edit.parse({"path": "src/app.py", "old_string": "x = 1", "new_string": "x = 2"})
    preview = edit.preview(args, ctx)
    assert preview.startswith("--- a/src/app.py")
    assert "-x = 1" in preview and "+x = 2" in preview


def test_cycle_skips_bypass() -> None:
    policy = PermissionPolicy()
    assert [policy.cycle_mode() for _ in range(3)] == ["accept_edits", "plan", "default"]
    policy.mode = "bypass"
    assert policy.cycle_mode() == "default"


# --- end to end through the agent loop ------------------------------------------


def call(name: str, call_id: str = "t1", **args: object) -> Message:
    return Message(role="assistant", content=[ToolUseBlock(id=call_id, name=name, input=args)])


async def test_denied_call_becomes_an_error_result_and_nothing_runs(tmp_path: Path) -> None:
    provider = FakeProvider([call("write", path="x.txt", content="hi"), "ok, I will not"])
    agent = Agent(provider, root=tmp_path, approver=ScriptedApprover(PermissionReply("no")))

    events = [e async for e in agent.run("write x.txt")]

    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.failed and "denied" in finished.result.content
    assert not (tmp_path / "x.txt").exists()
    assert isinstance(events[-1], TurnEnd)


async def test_non_interactive_agent_denies_what_needs_approval(tmp_path: Path) -> None:
    provider = FakeProvider([call("shell", command="echo hi"), "cannot"])
    agent = Agent(provider, root=tmp_path)  # no approver, like `matchuco -p`

    [e async for e in agent.run("run it")]

    result = agent.history[2].content[0]
    assert isinstance(result, ToolResultBlock) and result.is_error


async def test_plan_mode_round_trip(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("old\n")
    provider = FakeProvider(
        [
            call("write", "t1", path="a.txt", content="new\n"),  # refused: plan mode
            call("exit_plan_mode", "t2", plan="1. rewrite a.txt"),  # user approves
            call("read", "t3", path="a.txt"),
            call("write", "t4", path="a.txt", content="new\n"),  # now asks, user says yes
            "done",
        ]
    )
    approver = ScriptedApprover(PermissionReply("yes"), PermissionReply("yes"))
    agent = Agent(
        provider, root=tmp_path, permissions=PermissionPolicy(mode="plan"), approver=approver
    )

    [e async for e in agent.run("rewrite a.txt")]

    # The first prompt carried the plan-mode reminder ahead of the user's text.
    first = provider.requests[0].messages[0].content
    assert isinstance(first[0], TextBlock) and "Plan mode is active" in first[0].text
    assert [r.kind for r in approver.requests] == ["plan", "edit"]
    assert agent.permissions.mode == "default"
    assert (tmp_path / "a.txt").read_text() == "new\n"

    # The next prompt tells the model plan mode is over.
    agent.provider = FakeProvider(["ok"])
    [e async for e in agent.run("thanks")]
    assert "Plan mode has ended" in agent.history[-2].content[0].text  # type: ignore[union-attr]
