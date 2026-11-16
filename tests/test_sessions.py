"""Sessions: JSONL transcript, resume, fork, checkpoints, rewind, auto memory."""

import base64
from pathlib import Path

import pytest

from matchuco.agent import Agent, CheckpointSaved
from matchuco.cli import handle_command, parse_args
from matchuco.context import build_system_prompt
from matchuco.memory import load_memory
from matchuco.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from matchuco.permissions import PermissionPolicy
from matchuco.providers import ProviderError
from matchuco.providers.fake import FakeProvider
from matchuco.sessions import (
    RewindEvent,
    SessionError,
    SessionLog,
    UserMessageEvent,
    reconstruct,
)

ACCEPT = PermissionPolicy(mode="accept_edits")


def tool_call(name: str, call_id: str = "t1", **args: object) -> Message:
    return Message(role="assistant", content=[ToolUseBlock(id=call_id, name=name, input=args)])


async def drain(agent: Agent, prompt: str) -> list[object]:
    return [event async for event in agent.run(prompt)]


def test_empty_memory_does_not_change_the_prompt(tmp_path: Path) -> None:
    bare = build_system_prompt("BASE", tmp_path)
    assert all(label != "auto memory" for label, _ in bare.parts)

    (tmp_path / ".matchuco").mkdir()
    (tmp_path / ".matchuco" / "memory.md").write_text("  \n\n")
    assert load_memory(tmp_path) == ""
    again = build_system_prompt("BASE", tmp_path)
    assert [label for label, _ in again.parts] == [label for label, _ in bare.parts]
    assert again.text == bare.text


async def test_remember_updates_the_file_but_not_the_frozen_prompt(tmp_path: Path) -> None:
    agent = Agent(
        FakeProvider([tool_call("remember", note="Run tests with uv."), "noted"]),
        root=tmp_path,
        permissions=ACCEPT,
    )
    frozen = agent.system_prompt
    await drain(agent, "remember the test command")

    assert agent.system_prompt is frozen
    assert "Run tests with uv." not in agent.system
    assert "Run tests with uv." in (tmp_path / ".matchuco" / "memory.md").read_text()

    later = Agent(FakeProvider(["ok"]), root=tmp_path)
    assert "Run tests with uv." in later.system
    assert "auto memory" in [label for label, _ in later.system_prompt.parts]

    agent.clear()
    assert "Run tests with uv." in agent.system
    assert agent.system_prompt is not frozen


async def test_resume_rebuilds_history_and_continues(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("world\n")
    first = Agent(
        FakeProvider([tool_call("read", path="hello.txt"), "It says world."]),
        root=tmp_path,
        permissions=ACCEPT,
        session=SessionLog.create(tmp_path),
    )
    await drain(first, "read hello.txt")
    assert first.session is not None

    provider = FakeProvider(["You already read that it says world."])
    log, state = SessionLog.resume(tmp_path, first.session.id)
    second = Agent(provider, root=tmp_path, permissions=ACCEPT, session=log)
    second.adopt(state)
    await drain(second, "what did it say?")

    sent = "".join(message.model_dump_json() for message in provider.requests[0].messages)
    assert "world" in sent
    assert provider.requests[0].messages[-1].text == "what did it say?"
    # The read is still in effect: an edit does not have to re-read.
    assert (tmp_path / "hello.txt").resolve() in second.context.files_read


async def test_fork_copies_the_log_and_leaves_the_original_intact(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("world\n")
    log = SessionLog.create(tmp_path)
    agent = Agent(
        FakeProvider([tool_call("read", path="hello.txt"), "done"]),
        root=tmp_path,
        permissions=ACCEPT,
        session=log,
    )
    await drain(agent, "read it")
    original = log.path.read_bytes()

    forked, state = SessionLog.fork(tmp_path, log.id)

    assert forked.id != log.id
    assert forked.path.read_bytes() == original
    assert log.path.read_bytes() == original
    assert state.messages  # the copy rebuilt the same conversation
    forked.append(UserMessageEvent(message=Message.user("only on the fork")))
    assert log.path.read_bytes() == original
    assert b"only on the fork" in forked.path.read_bytes()


async def test_failed_turn_is_not_written_to_the_log(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("world\n")
    log = SessionLog.create(tmp_path)
    agent = Agent(
        FakeProvider([tool_call("read", path="hello.txt")]),  # script then runs out
        root=tmp_path,
        permissions=ACCEPT,
        session=log,
    )
    with pytest.raises(ProviderError):
        await drain(agent, "read it")
    assert agent.history == []
    assert log.read_events() == []


async def test_checkpoint_stores_the_previous_bytes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("v1\n")
    log = SessionLog.create(tmp_path)
    agent = Agent(
        FakeProvider(
            [
                tool_call("read", path="a.txt"),
                tool_call("write", path="a.txt", content="v2\n"),
                "ok",
            ]
        ),
        root=tmp_path,
        permissions=ACCEPT,
        session=log,
    )
    events = await drain(agent, "update a.txt")

    saved = [event for event in events if isinstance(event, CheckpointSaved)]
    assert len(saved) == 1
    assert saved[0].path == "a.txt"
    record = agent.checkpoints.load(saved[0].id) if agent.checkpoints else None
    assert record is not None
    assert record.existed
    assert base64.b64decode(record.content_b64) == b"v1\n"
    assert (tmp_path / "a.txt").read_text() == "v2\n"
    # The id is in the JSONL; the previous bytes live in the checkpoint store.
    transcript = log.path.read_text()
    assert saved[0].id in transcript
    assert "content_b64" not in transcript


async def test_rewind_restores_the_file_and_keeps_the_log(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("v1\n")
    log = SessionLog.create(tmp_path)
    provider = FakeProvider(
        [
            tool_call("read", path="a.txt"),
            tool_call("write", call_id="w", path="a.txt", content="v2\n"),
            "wrote v2",
            tool_call("write", call_id="w2", path="a.txt", content="v3\n"),
            "wrote v3",
        ]
    )
    agent = Agent(provider, root=tmp_path, permissions=ACCEPT, session=log)
    await drain(agent, "update a.txt")
    assert (tmp_path / "a.txt").read_text() == "v2\n"
    checkpoint_id = agent.active_checkpoints()[-1].id
    before = log.path.read_bytes()

    agent.rewind(checkpoint_id)

    assert (tmp_path / "a.txt").read_text() == "v1\n"
    after = log.path.read_text(encoding="utf-8")
    assert after.encode("utf-8").startswith(before)
    assert any(isinstance(event, RewindEvent) for event in log.read_events())
    # The read survived; the write's tool call did not stay in the rebuilt history.
    blob = "".join(message.model_dump_json() for message in agent.history)
    assert "v2" not in blob
    assert any(
        isinstance(block, ToolResultBlock) and "v1" in block.content
        for message in agent.history
        for block in message.content
    )
    for index, message in enumerate(agent.history):
        if message.role == "assistant" and message.tool_uses:
            follower = agent.history[index + 1]
            assert any(isinstance(block, ToolResultBlock) for block in follower.content)

    await drain(agent, "write v3 instead")
    assert (tmp_path / "a.txt").read_text() == "v3\n"


async def test_rewind_of_one_grouped_edit_restores_the_whole_batch(tmp_path: Path) -> None:
    both = Message(
        role="assistant",
        content=[
            ToolUseBlock(id="a", name="write", input={"path": "a.txt", "content": "A\n"}),
            ToolUseBlock(id="b", name="write", input={"path": "b.txt", "content": "B\n"}),
        ],
    )
    log = SessionLog.create(tmp_path)
    agent = Agent(
        FakeProvider([both, "created both"]),
        root=tmp_path,
        permissions=ACCEPT,
        session=log,
    )
    await drain(agent, "create two files")
    assert (tmp_path / "a.txt").read_text() == "A\n"
    checkpoints = agent.active_checkpoints()
    assert len(checkpoints) == 2
    assert checkpoints[0].group == checkpoints[1].group

    agent.rewind(checkpoints[1].id)

    assert not (tmp_path / "a.txt").exists()
    assert not (tmp_path / "b.txt").exists()
    assert [message.role for message in agent.history] == ["user"]
    assert agent.history[0].text == "create two files"


async def test_rewind_to_an_unknown_checkpoint_leaves_the_log_alone(tmp_path: Path) -> None:
    log = SessionLog.create(tmp_path)
    agent = Agent(FakeProvider(["hi"]), root=tmp_path, session=log)
    await drain(agent, "hi")
    before = log.path.read_bytes()
    with pytest.raises(SessionError):
        agent.rewind("missing")
    assert log.path.read_bytes() == before


async def test_compaction_is_an_event_and_resume_sees_the_summary(tmp_path: Path) -> None:
    log = SessionLog.create(tmp_path)
    agent = Agent(
        FakeProvider(["hello", "<summary>Greeted.</summary>", "still here"]),
        root=tmp_path,
        system="s",
        tools=[],
        session=log,
    )
    await drain(agent, "hi")
    assert await agent.compact() is not None

    state = reconstruct(log.read_events())
    assert state.messages == []
    assert any(
        isinstance(block, TextBlock) and "Greeted." in block.text for block in state.carryover
    )

    provider = FakeProvider(["continue"])
    resumed, rebuilt = SessionLog.resume(tmp_path, log.id)
    follower = Agent(provider, root=tmp_path, system="s", tools=[], session=resumed)
    follower.adopt(rebuilt)
    await drain(follower, "and now?")
    opening = provider.requests[0].messages[0]
    texts = [block.text for block in opening.content if isinstance(block, TextBlock)]
    assert any("Greeted." in text for text in texts)
    assert texts[-1] == "and now?"


def test_missing_and_invalid_session_ids(tmp_path: Path) -> None:
    with pytest.raises(SessionError):
        SessionLog.resume(tmp_path, "missing")
    with pytest.raises(SessionError):
        SessionLog.resume(tmp_path, "../etc/passwd")
    log = SessionLog.create(tmp_path)
    log.path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SessionError, match=":"):
        log.read_events()


def test_resume_and_fork_flags_are_mutually_exclusive() -> None:
    assert parse_args(["--resume", "abc"]).resume == "abc"
    assert parse_args(["--fork", "abc"]).fork == "abc"
    with pytest.raises(SystemExit):
        parse_args(["--resume", "abc", "--fork", "def"])


async def test_rewind_command_lists_and_restores(tmp_path: Path) -> None:
    log = SessionLog.create(tmp_path)
    agent = Agent(
        FakeProvider([tool_call("write", path="new.txt", content="x\n"), "ok"]),
        root=tmp_path,
        permissions=ACCEPT,
        session=log,
    )
    await drain(agent, "create new.txt")
    checkpoint_id = agent.active_checkpoints()[0].id
    assert await handle_command(agent, "/rewind")
    assert (tmp_path / "new.txt").is_file()
    assert await handle_command(agent, f"/rewind {checkpoint_id}")
    assert not (tmp_path / "new.txt").exists()


async def test_without_a_session_an_edit_leaves_no_checkpoint(tmp_path: Path) -> None:
    agent = Agent(
        FakeProvider([tool_call("write", path="new.txt", content="x\n"), "ok"]),
        root=tmp_path,
        permissions=ACCEPT,
    )
    await drain(agent, "create new.txt")
    assert (tmp_path / "new.txt").is_file()
    assert not (tmp_path / ".matchuco").exists()
