import pytest

from matchuco.cli import one_shot, run_turn
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


async def test_run_turn_appends_assistant_message() -> None:
    provider = FakeProvider(["first", "second"])
    history = [Message.user("one")]
    await run_turn(provider, history)
    history.append(Message.user("two"))
    await run_turn(provider, history)

    assert [m.role for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[-1].text == "second"
    # The provider saw the full history on the second call (multi-turn memory).
    assert [m.text for m in provider.requests[1].messages] == ["one", "first", "two"]


async def test_one_shot_reports_errors() -> None:
    assert await one_shot(FakeProvider(["ok"]), "hi") == 0
    assert await one_shot(FakeProvider([]), "hi") == 1
