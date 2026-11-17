"""Hooks run after the permission gate, in process, and cannot overrule a deny."""

import argparse
from pathlib import Path

from matchuco.agent import Agent
from matchuco.cli import build_agent
from matchuco.extensions.hooks import HookRegistry, PostToolUse, PreToolUse
from matchuco.messages import Message, ToolResultBlock, ToolUseBlock
from matchuco.permissions import PermissionPolicy
from matchuco.providers.fake import FakeProvider
from matchuco.sessions import SessionLog, ToolResultEvent


def tool_call(name: str, call_id: str = "t1", **args: object) -> Message:
    return Message(role="assistant", content=[ToolUseBlock(id=call_id, name=name, input=args)])


async def drain(agent: Agent, prompt: str) -> list[object]:
    return [event async for event in agent.run(prompt)]


def _result(agent: Agent, index: int = 2) -> ToolResultBlock:
    block = agent.history[index].content[0]
    assert isinstance(block, ToolResultBlock)
    return block


async def test_pre_hook_denies_after_permission_and_the_loop_continues(tmp_path: Path) -> None:
    calls: list[str] = []

    async def pre(call: PreToolUse) -> str | None:
        calls.append(call.name)
        if call.name == "write" and "blocked" in str(call.args.get("content", "")):
            return "do not write that"
        return None

    def later(call: PreToolUse) -> None:
        calls.append(f"later:{call.name}")
        return None

    hooks = HookRegistry()
    hooks.on_pre(pre)
    hooks.on_pre(later)
    provider = FakeProvider(
        [tool_call("write", path="out.txt", content="blocked\n"), "I will not write it."]
    )
    agent = Agent(
        provider,
        root=tmp_path,
        hooks=hooks,
        permissions=PermissionPolicy(mode="bypass"),
    )
    frozen = agent.system_prompt

    await drain(agent, "write the file")

    assert calls == ["write"]  # the first denial wins; the next hook is not asked
    assert not (tmp_path / "out.txt").exists()
    result = _result(agent)
    assert result.is_error
    assert result.content == "Hook denied: do not write that."
    assert agent.history[-1].text == "I will not write it."
    assert agent.system_prompt is frozen


async def test_permission_deny_is_not_shown_to_the_hook(tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("nope\n")
    seen: list[str] = []

    def pre(call: PreToolUse) -> str | None:
        seen.append(call.name)
        return None  # would allow, if it were consulted

    def post(call: PostToolUse) -> str:
        seen.append(f"post:{call.name}")
        return "should-not-note"

    hooks = HookRegistry()
    hooks.on_pre(pre)
    hooks.on_post(post)
    provider = FakeProvider([tool_call("read", path="secret.txt"), "I was denied."])
    agent = Agent(
        provider,
        root=tmp_path,
        hooks=hooks,
        permissions=PermissionPolicy.from_rules(deny=["read(secret.txt)"]),
    )

    await drain(agent, "read the secret")

    assert seen == []
    result = _result(agent)
    assert result.is_error
    assert "Permission denied" in result.content
    assert "deny rule" in result.content
    assert "nope" not in result.content
    assert "should-not-note" not in result.content
    assert "Hook denied" not in result.content
    assert (tmp_path / "secret.txt").read_text() == "nope\n"


async def test_post_hook_appends_a_note_without_touching_the_system_prompt(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("world\n")

    def post(call: PostToolUse) -> str | None:
        assert call.name == "read"
        assert call.is_error is False
        assert "world" in call.content
        return "HOOK-NOTE-9f3a"

    hooks = HookRegistry()
    hooks.on_post(post)
    provider = FakeProvider([tool_call("read", path="hello.txt"), "The file says world."])
    agent = Agent(provider, root=tmp_path, hooks=hooks)
    frozen = agent.system_prompt

    await drain(agent, "read hello")

    assert agent.system_prompt is frozen
    assert "HOOK-NOTE-9f3a" not in agent.system
    result = _result(agent)
    assert result.is_error is False
    assert "world" in result.content
    assert result.content.endswith("HOOK-NOTE-9f3a")
    followup = provider.requests[1]
    assert "HOOK-NOTE-9f3a" not in followup.system
    rendered = followup.messages[2].model_dump_json()
    assert "HOOK-NOTE-9f3a" in rendered
    assert "world" in rendered


async def test_a_raising_hook_becomes_an_error_result(tmp_path: Path) -> None:
    def pre(call: PreToolUse) -> str | None:
        del call
        raise RuntimeError("boom")

    hooks = HookRegistry()
    hooks.on_pre(pre)
    provider = FakeProvider(
        [tool_call("write", path="out.txt", content="x\n"), "The hook failed, so I stopped."]
    )
    agent = Agent(
        provider,
        root=tmp_path,
        hooks=hooks,
        permissions=PermissionPolicy(mode="bypass"),
    )

    events = await drain(agent, "write it")

    assert not (tmp_path / "out.txt").exists()
    result = _result(agent)
    assert result.is_error
    assert "Hook denied" in result.content
    assert "RuntimeError" in result.content
    assert agent.history[-1].text == "The hook failed, so I stopped."
    assert type(events[-1]).__name__ == "TurnEnd"


async def test_settings_hooks_reach_the_agent_through_config(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("world\n")
    project = tmp_path / ".matchuco"
    project.mkdir()
    (project / "settings.json").write_text(
        """
        {
          "hooks": {
            "PreToolUse": [{"matcher": "write", "deny": "settings blocked the write"}],
            "PostToolUse": [{"matcher": "read", "note": "HOOK-NOTE-from-settings"}]
          }
        }
        """,
        encoding="utf-8",
    )
    provider = FakeProvider(
        [
            tool_call("read", path="hello.txt"),
            tool_call("write", call_id="t2", path="out.txt", content="nope\n"),
            "stopped",
        ]
    )
    args = argparse.Namespace(
        cwd=str(tmp_path),
        mode="bypass",
        max_steps=None,
        context_window=None,
        resume=None,
        fork=None,
    )
    agent = build_agent(args, provider, interactive=False)
    frozen = agent.system_prompt

    await drain(agent, "read, then write")

    assert agent.system_prompt is frozen
    assert "HOOK-NOTE-from-settings" not in agent.system
    read_result = _result(agent)
    assert read_result.is_error is False
    assert read_result.content.endswith("HOOK-NOTE-from-settings")
    write_result = agent.history[4].content[0]
    assert isinstance(write_result, ToolResultBlock)
    assert write_result.is_error
    assert write_result.content == "Hook denied: settings blocked the write."
    assert not (tmp_path / "out.txt").exists()


async def test_child_agent_keeps_the_parent_hooks(tmp_path: Path) -> None:
    def pre(call: PreToolUse) -> str | None:
        if call.name == "shell" and "pwned" in str(call.args.get("command", "")):
            return "shell is hooked"
        return None

    hooks = HookRegistry()
    hooks.on_pre(pre)
    command = "python -c \"open('pwned.txt','w').write('x')\""
    provider = FakeProvider(
        [
            tool_call(
                "task",
                description="run",
                prompt="Run the command.",
                readonly=False,
            ),
            tool_call("shell", command=command),
            "could not run it",
            "the child was stopped",
        ]
    )
    log = SessionLog.create(tmp_path)
    agent = Agent(
        provider,
        root=tmp_path,
        session=log,
        hooks=hooks,
        permissions=PermissionPolicy(mode="bypass"),
    )

    await drain(agent, "delegate the command")

    assert not (tmp_path / "pwned.txt").exists()
    child_turn = provider.requests[2].messages
    denied = child_turn[2].content[0]
    assert isinstance(denied, ToolResultBlock)
    assert denied.is_error
    assert "Hook denied: shell is hooked." in denied.content
    child_ids = [
        event.child_session_id
        for event in log.read_events()
        if isinstance(event, ToolResultEvent) and event.child_session_id
    ]
    assert len(child_ids) == 1
