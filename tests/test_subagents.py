"""Subagents: a child with its own transcript, read-only unless asked, depth 1."""

from pathlib import Path

from matchuco.agent import Agent
from matchuco.extensions.subagents import child_policy
from matchuco.messages import Message, ToolResultBlock, ToolUseBlock
from matchuco.permissions import PermissionPolicy
from matchuco.providers.fake import FakeProvider
from matchuco.sessions import SessionLog, ToolResultEvent


def tool_call(name: str, call_id: str = "t1", **args: object) -> Message:
    return Message(role="assistant", content=[ToolUseBlock(id=call_id, name=name, input=args)])


async def drain(agent: Agent, prompt: str) -> list[object]:
    return [event async for event in agent.run(prompt)]


def _child_ids(log: SessionLog) -> list[str]:
    return [
        event.child_session_id
        for event in log.read_events()
        if isinstance(event, ToolResultEvent) and event.child_session_id
    ]


async def test_parent_delegates_a_read_and_only_hears_the_childs_answer(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("world\n")
    parent_task = tool_call(
        "task",
        call_id="p1",
        description="read hello",
        prompt="Read hello.txt and report its contents.",
    )
    provider = FakeProvider(
        [
            parent_task,
            tool_call("read", call_id="c1", path="hello.txt"),
            "hello.txt contains world",
            "The file contains world.",
        ]
    )
    log = SessionLog.create(tmp_path)
    agent = Agent(provider, root=tmp_path, session=log, permissions=PermissionPolicy(mode="bypass"))
    frozen = agent.system_prompt

    await drain(agent, "what does hello.txt say?")

    assert agent.system_prompt is frozen
    # The child never saw the parent's prompt, and the parent never saw the raw read.
    child_request = provider.requests[1]
    assert len(child_request.messages) == 1
    assert "Read hello.txt and report its contents." in child_request.messages[0].text
    assert "what does hello.txt say?" not in child_request.messages[0].text
    parent_followup = provider.requests[3].messages
    result = parent_followup[2].content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.is_error is False
    assert result.content == "hello.txt contains world"
    assert agent.history[-1].text == "The file contains world."

    [child_id] = _child_ids(log)
    child_log = SessionLog(tmp_path, child_id)
    child_text = "".join(event.model_dump_json() for event in child_log.read_events())
    assert "world" in child_text
    assert (tmp_path / ".matchuco" / "sessions" / f"{child_id}.jsonl").is_file()


async def test_readonly_child_cannot_write_or_spawn(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("safe\n")
    nested = Message(
        role="assistant",
        content=[
            ToolUseBlock(id="w", name="write", input={"path": "hello.txt", "content": "pwned\n"}),
            ToolUseBlock(
                id="n",
                name="task",
                input={"description": "deeper", "prompt": "nest", "readonly": False},
            ),
        ],
    )
    provider = FakeProvider(
        [
            tool_call(
                "task",
                call_id="p1",
                description="mutate",
                prompt="Overwrite hello.txt, then start another subagent.",
            ),
            nested,
            "blocked",
            "the child changed nothing",
        ]
    )
    log = SessionLog.create(tmp_path)
    # Bypass on the parent would let the parent write. The child must not inherit that.
    agent = Agent(provider, root=tmp_path, session=log, permissions=PermissionPolicy(mode="bypass"))

    await drain(agent, "change the file")

    assert (tmp_path / "hello.txt").read_text() == "safe\n"
    assert agent.permissions.mode == "bypass"
    [child_id] = _child_ids(log)
    child_results = [
        event.result
        for event in SessionLog(tmp_path, child_id).read_events()
        if isinstance(event, ToolResultEvent)
    ]
    assert len(child_results) == 2
    assert all(result.is_error for result in child_results)
    assert any("Permission denied" in result.content for result in child_results)
    assert any("maximum depth is 1" in result.content for result in child_results)
    # Parent + child. The refused second level never opened a session.
    assert len(list((tmp_path / ".matchuco" / "sessions").glob("*.jsonl"))) == 2


async def test_child_keeps_the_parents_deny_rules(tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("nope\n")
    provider = FakeProvider(
        [
            tool_call("task", description="peek", prompt="Read secret.txt."),
            tool_call("read", path="secret.txt"),
            "could not read it",
            "the child was denied",
        ]
    )
    agent = Agent(
        provider,
        root=tmp_path,
        permissions=PermissionPolicy.from_rules(deny=["read(secret.txt)"]),
    )
    await drain(agent, "look at the secret")

    child_turn = provider.requests[2].messages
    denied = child_turn[2].content[0]
    assert isinstance(denied, ToolResultBlock)
    assert denied.is_error
    assert "deny rule" in denied.content
    assert (tmp_path / "secret.txt").read_text() == "nope\n"


def test_readonly_policy_is_stricter_and_a_copy() -> None:
    parent = PermissionPolicy.from_rules("bypass", allow=["write(docs/**)"], deny=["read(.env)"])
    child = child_policy(parent, readonly=True)
    assert child.mode == "plan"
    assert child.deny == parent.deny
    assert child.allow == parent.allow
    assert child is not parent
    child.allow.append(parent.allow[0])
    assert len(parent.allow) == 1

    opened = child_policy(parent, readonly=False)
    assert opened.mode == "bypass"
    assert opened.deny == parent.deny
