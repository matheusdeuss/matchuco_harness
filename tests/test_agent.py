from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from matchuco.agent import Agent, TextDelta, ToolFinished, ToolStarted, TurnEnd
from matchuco.messages import (
    Message,
    Response,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from matchuco.providers import Done, ProviderError, StreamEvent
from matchuco.providers.fake import FakeProvider


def tool_call(name: str, call_id: str = "t1", **args: object) -> Message:
    return Message(role="assistant", content=[ToolUseBlock(id=call_id, name=name, input=args)])


async def drain(agent: Agent, prompt: str) -> list[object]:
    return [event async for event in agent.run(prompt)]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "hello.txt").write_text("world\n")
    return tmp_path


async def test_loop_runs_a_tool_and_feeds_the_result_back(workspace: Path) -> None:
    provider = FakeProvider([tool_call("read", path="hello.txt"), "The file says world."])
    agent = Agent(provider, root=workspace)

    events = await drain(agent, "what is in hello.txt?")

    started = [e for e in events if isinstance(e, ToolStarted)]
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert [e.name for e in started] == ["read"]
    assert not finished[0].failed
    assert "world" in finished[0].result.content

    # user -> assistant(tool_use) -> user(tool_result) -> assistant(text)
    assert [m.role for m in agent.history] == ["user", "assistant", "user", "assistant"]
    result = agent.history[2].content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.tool_use_id == "t1"
    assert agent.history[-1].text == "The file says world."

    # The second request carried the whole exchange back to the model.
    assert len(provider.requests[1].messages) == 3
    assert [spec.name for spec in provider.requests[0].tools] == agent.registry.names


async def test_turn_ends_when_the_model_stops_asking_for_tools(workspace: Path) -> None:
    agent = Agent(FakeProvider(["just an answer"]), root=workspace)
    events = await drain(agent, "hi")

    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "just an answer"
    end = events[-1]
    assert isinstance(end, TurnEnd)
    assert (end.reason, end.steps) == ("end_turn", 1)
    assert end.usage.output_tokens > 0


async def test_failing_tools_come_back_as_error_results_not_exceptions(workspace: Path) -> None:
    provider = FakeProvider([tool_call("read", path="missing.txt"), "Sorry, no such file."])
    agent = Agent(provider, root=workspace)

    events = await drain(agent, "read missing.txt")

    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.failed
    result = agent.history[2].content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.is_error
    assert isinstance(events[-1], TurnEnd)


async def test_parallel_tool_calls_are_answered_in_one_message(workspace: Path) -> None:
    both = Message(
        role="assistant",
        content=[
            ToolUseBlock(id="a", name="read", input={"path": "hello.txt"}),
            ToolUseBlock(id="b", name="glob", input={"pattern": "*.txt"}),
        ],
    )
    agent = Agent(FakeProvider([both, "done"]), root=workspace)

    events = await drain(agent, "look around")

    assert [e.name for e in events if isinstance(e, ToolStarted)] == ["read", "glob"]
    results = agent.history[2].content
    assert [b.tool_use_id for b in results if isinstance(b, ToolResultBlock)] == ["a", "b"]


async def test_step_limit_stops_a_looping_model(workspace: Path) -> None:
    script = [tool_call("read", call_id=f"t{i}", path="hello.txt") for i in range(10)]
    agent = Agent(FakeProvider(script), root=workspace, max_steps=3)

    end = (await drain(agent, "loop forever"))[-1]

    assert isinstance(end, TurnEnd)
    assert (end.reason, end.steps) == ("max_steps", 3)
    assert len(agent.history) == 1 + 3 * 2  # user + 3 rounds of (assistant, tool results)


class StopReasonProvider:
    """A provider that replays fixed stop reasons -- FakeProvider only emits the common ones."""

    name = "stub"
    model = "stub-model"

    def __init__(self, reasons: list[StopReason]) -> None:
        self._reasons = reasons
        self.calls = 0

    async def stream(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        reason = self._reasons.pop(0)
        message = Message(role="assistant", content=[TextBlock(text=reason)])
        yield Done(Response(message=message, stop_reason=reason, usage=Usage(), model=self.model))


async def test_pause_turn_resends_instead_of_ending_the_turn(workspace: Path) -> None:
    provider = StopReasonProvider(["pause_turn", "end_turn"])
    agent = Agent(provider, root=workspace)

    end = (await drain(agent, "long task"))[-1]

    assert provider.calls == 2
    assert isinstance(end, TurnEnd)
    assert (end.reason, end.steps) == ("end_turn", 2)


async def test_provider_failure_leaves_a_valid_history(workspace: Path) -> None:
    agent = Agent(FakeProvider([]), root=workspace)  # empty script -> ProviderError

    with pytest.raises(ProviderError):
        await drain(agent, "hi")

    assert agent.history == []


async def test_failure_mid_turn_unwinds_the_incomplete_exchange(workspace: Path) -> None:
    provider = FakeProvider([tool_call("read", path="hello.txt")])  # then the script runs out
    agent = Agent(provider, root=workspace)

    with pytest.raises(ProviderError):
        await drain(agent, "read it")

    assert agent.history == []


async def test_clear_forgets_the_conversation_and_the_files_read(workspace: Path) -> None:
    agent = Agent(FakeProvider([tool_call("read", path="hello.txt"), "ok"]), root=workspace)
    await drain(agent, "read hello.txt")
    assert agent.context.files_read

    agent.clear()
    assert agent.history == []
    assert agent.context.files_read == {}
    assert agent.usage.output_tokens > 0  # usage is cumulative across the session


async def test_system_prompt_names_the_workspace(workspace: Path) -> None:
    provider = FakeProvider(["hi"])
    agent = Agent(provider, root=workspace)
    await drain(agent, "hello")
    assert str(workspace.resolve()) in provider.requests[0].system
