"""Skills: name and description in the first request, body only after the tool."""

from pathlib import Path

from matchuco.agent import Agent
from matchuco.extensions.skills import parse_skill
from matchuco.messages import Message, ToolResultBlock, ToolUseBlock
from matchuco.providers.fake import FakeProvider

BODY = "SKILL-BODY-9f3a follow the layout before editing anything."
DESCRIPTION = "Check the repository layout before editing."


def tool_call(tool: str, call_id: str = "t1", **args: object) -> Message:
    return Message(role="assistant", content=[ToolUseBlock(id=call_id, name=tool, input=args)])


async def drain(agent: Agent, prompt: str) -> list[object]:
    return [event async for event in agent.run(prompt)]


def _request_blob(system: str, messages: list[Message], tools_json: list[str]) -> str:
    return "\n".join([system, *[message.model_dump_json() for message in messages], *tools_json])


def test_parse_skill_splits_title_description_and_body() -> None:
    skill = parse_skill(
        "layout",
        f"# Layout\n\n{DESCRIPTION}\n\n{BODY}\n\nMore instructions.\n",
    )
    assert skill.title == "Layout"
    assert skill.description == DESCRIPTION
    assert skill.body == f"{BODY}\n\nMore instructions."
    assert BODY not in skill.description


async def test_skill_body_stays_out_of_the_first_request(tmp_path: Path) -> None:
    skills = tmp_path / ".matchuco" / "skills"
    skills.mkdir(parents=True)
    (skills / "layout.md").write_text(
        f"# Layout\n\n{DESCRIPTION}\n\n{BODY}\n",
        encoding="utf-8",
    )
    provider = FakeProvider([tool_call("skill", name="layout"), "I loaded the layout skill."])
    agent = Agent(provider, root=tmp_path, system="system {root}")
    frozen = agent.system_prompt

    await drain(agent, "use the layout skill")

    first = provider.requests[0]
    first_blob = _request_blob(
        first.system, list(first.messages), [spec.model_dump_json() for spec in first.tools]
    )
    assert BODY not in first_blob
    skill_spec = next(spec for spec in first.tools if spec.name == "skill")
    assert "layout" in skill_spec.description
    assert DESCRIPTION in skill_spec.description
    assert BODY not in skill_spec.description

    followup = provider.requests[1]
    assert BODY not in followup.system
    result = followup.messages[2].content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.is_error is False
    assert BODY in result.content
    assert agent.history[-1].text == "I loaded the layout skill."
    assert agent.system_prompt is frozen
    assert BODY not in agent.system


async def test_unknown_skill_is_a_tool_error(tmp_path: Path) -> None:
    provider = FakeProvider([tool_call("skill", name="../secret"), "no such skill"])
    agent = Agent(provider, root=tmp_path)

    events = await drain(agent, "load a missing skill")

    result = agent.history[2].content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.is_error
    assert "unknown skill" in result.content
    assert agent.history[-1].text == "no such skill"
    assert type(events[-1]).__name__ == "TurnEnd"
