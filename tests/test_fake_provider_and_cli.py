import pytest

from matchuco.agent import Agent
from matchuco.cli import _format_args, one_shot, parse_args, run_turn
from matchuco.messages import Message, ToolUseBlock
from matchuco.providers import Done, ProviderError, TextDelta, ToolUseStart, complete
from matchuco.providers.fake import FakeProvider


async def test_fake_streams_text_that_concatenates_back() -> None:
    provider = FakeProvider(["Hello  there, world! "])
    events = [e async for e in provider.stream("sys", [Message.user("hi")])]
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "Hello  there, world! "
    assert isinstance(events[-1], Done)
    assert provider.requests[0].system == "sys"


async def test_fake_tool_call_turn() -> None:
    reply = Message(role="assistant", content=[ToolUseBlock(id="t", name="glob", input={})])
    provider = FakeProvider([reply])
    events = [e async for e in provider.stream("", [Message.user("go")])]
    assert ToolUseStart("t", "glob") in events
    resp = await complete(FakeProvider([reply]), "", [])
    assert resp.stop_reason == "tool_use"


async def test_fake_script_exhausted() -> None:
    with pytest.raises(ProviderError):
        await complete(FakeProvider([]), "", [])


async def test_run_turn_keeps_multi_turn_memory() -> None:
    provider = FakeProvider(["first", "second"])
    agent = Agent(provider)
    await run_turn(agent, "one")
    await run_turn(agent, "two")

    assert [m.role for m in agent.history] == ["user", "assistant", "user", "assistant"]
    assert agent.history[-1].text == "second"
    # The provider saw the full history on the second call (multi-turn memory).
    assert [m.text for m in provider.requests[1].messages] == ["one", "first", "two"]


async def test_one_shot_reports_errors() -> None:
    assert await one_shot(Agent(FakeProvider(["ok"])), "hi") == 0
    assert await one_shot(Agent(FakeProvider([])), "hi") == 1


def test_format_args_stays_on_one_short_line() -> None:
    line = _format_args({"path": "a.py", "content": "x" * 200, "replace_all": True})
    assert "\n" not in line
    assert line.startswith("path=a.py ")
    assert "..." in line
    assert "replace_all=true" in line


def test_cli_defaults_and_overrides() -> None:
    assert parse_args([]).provider in ("anthropic", "openai", "ollama", "fake")
    args = parse_args(["--provider", "fake", "--max-steps", "3", "-p", "hi"])
    assert (args.provider, args.max_steps, args.prompt) == ("fake", 3, "hi")
