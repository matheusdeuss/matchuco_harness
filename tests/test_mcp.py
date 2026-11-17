"""MCP tools over stdio: the fake server echoes, and a broken frame is a tool error."""

import json
import sys
from pathlib import Path

from matchuco.agent import Agent
from matchuco.config import load_settings
from matchuco.extensions.hooks import HookRegistry, PreToolUse
from matchuco.extensions.mcp_client import attach_mcp_server, connect_mcp_servers
from matchuco.messages import Message, ToolResultBlock, ToolUseBlock
from matchuco.permissions import PermissionPolicy, PermissionReply, PermissionRequest
from matchuco.providers.fake import FakeProvider

SERVER = Path(__file__).with_name("mcp_echo_server.py")


def tool_call(tool: str, call_id: str = "t1", **args: object) -> Message:
    return Message(role="assistant", content=[ToolUseBlock(id=call_id, name=tool, input=args)])


def server_command(log: Path, *, break_json: bool = False) -> list[str]:
    command = [sys.executable, str(SERVER), "--log", str(log)]
    if break_json:
        command.append("--break-json")
    return command


class ScriptedApprover:
    def __init__(self, *replies: PermissionReply) -> None:
        self.replies = list(replies)
        self.requests: list[PermissionRequest] = []

    async def __call__(self, request: PermissionRequest) -> PermissionReply:
        self.requests.append(request)
        return self.replies.pop(0)


async def drain(agent: Agent, prompt: str) -> list[object]:
    return [event async for event in agent.run(prompt)]


def _result(agent: Agent) -> ToolResultBlock:
    block = agent.history[2].content[0]
    assert isinstance(block, ToolResultBlock)
    return block


async def test_fake_provider_receives_the_mcp_echo(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    approver = ScriptedApprover(PermissionReply("yes"))
    provider = FakeProvider([tool_call("mcp_echo_echo", text="hello-mcp"), "heard hello-mcp"])
    agent = Agent(provider, root=tmp_path, approver=approver)
    client = await attach_mcp_server(
        agent.registry, name="echo", command=server_command(log), cwd=tmp_path
    )
    try:
        await drain(agent, "echo something")
    finally:
        await client.close()

    assert [request.tool for request in approver.requests] == ["mcp_echo_echo"]
    assert approver.requests[0].kind == "execute"
    spec = next(item for item in provider.requests[0].tools if item.name == "mcp_echo_echo")
    assert spec.description == "Echo text back."
    assert spec.input_schema["properties"]["text"]["type"] == "string"
    result = _result(agent)
    assert result.is_error is False
    assert result.content == "hello-mcp"
    assert agent.history[-1].text == "heard hello-mcp"
    assert "hello-mcp" in log.read_text(encoding="utf-8")


async def test_plan_mode_denies_the_mcp_call_before_the_server(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    provider = FakeProvider([tool_call("mcp_echo_echo", text="hello-mcp"), "I could not call it."])
    agent = Agent(provider, root=tmp_path, permissions=PermissionPolicy(mode="plan"))
    client = await attach_mcp_server(
        agent.registry, name="echo", command=server_command(log), cwd=tmp_path
    )
    try:
        await drain(agent, "echo")
    finally:
        await client.close()

    result = _result(agent)
    assert result.is_error
    assert "Permission denied" in result.content
    assert "plan mode" in result.content
    assert "hello-mcp" not in result.content
    assert not log.exists()


async def test_default_mode_does_not_run_an_mcp_tool_without_approval(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    provider = FakeProvider([tool_call("mcp_echo_echo", text="hello-mcp"), "denied"])
    agent = Agent(provider, root=tmp_path)
    client = await attach_mcp_server(
        agent.registry, name="echo", command=server_command(log), cwd=tmp_path
    )
    try:
        await drain(agent, "echo")
    finally:
        await client.close()

    result = _result(agent)
    assert result.is_error
    assert "Permission denied" in result.content
    assert not log.exists()


async def test_broken_json_comes_back_as_a_tool_error(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    provider = FakeProvider([tool_call("mcp_echo_echo", text="hello-mcp"), "the tool failed"])
    agent = Agent(provider, root=tmp_path, permissions=PermissionPolicy(mode="bypass"))
    client = await attach_mcp_server(
        agent.registry,
        name="echo",
        command=server_command(log, break_json=True),
        cwd=tmp_path,
    )
    try:
        events = await drain(agent, "echo")
    finally:
        await client.close()

    result = _result(agent)
    assert result.is_error
    assert "invalid JSON" in result.content
    assert agent.history[-1].text == "the tool failed"
    assert type(events[-1]).__name__ == "TurnEnd"
    assert log.is_file()


async def test_a_hook_can_still_deny_after_the_gate_allows(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"

    def pre(call: PreToolUse) -> str | None:
        if call.name.startswith("mcp_"):
            return "no mcp in this test"
        return None

    hooks = HookRegistry()
    hooks.on_pre(pre)
    provider = FakeProvider([tool_call("mcp_echo_echo", text="hello-mcp"), "stopped"])
    agent = Agent(
        provider,
        root=tmp_path,
        hooks=hooks,
        permissions=PermissionPolicy(mode="bypass"),
    )
    client = await attach_mcp_server(
        agent.registry, name="echo", command=server_command(log), cwd=tmp_path
    )
    try:
        await drain(agent, "echo")
    finally:
        await client.close()

    result = _result(agent)
    assert result.is_error
    assert result.content == "Hook denied: no mcp in this test."
    assert not log.exists()


async def test_settings_servers_are_attached_with_the_prefix(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    project = tmp_path / ".matchuco"
    project.mkdir()
    (project / "settings.json").write_text(
        json.dumps({"mcp": {"servers": {"echo": {"command": server_command(log)}}}}),
        encoding="utf-8",
    )
    agent = Agent(FakeProvider(["unused"]), root=tmp_path)
    clients = await connect_mcp_servers(agent.registry, load_settings(tmp_path).mcp, cwd=tmp_path)
    try:
        assert "mcp_echo_echo" in agent.registry
    finally:
        for client in clients:
            await client.close()
