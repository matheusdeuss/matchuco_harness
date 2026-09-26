from pathlib import Path

import pytest

from matchuco.messages import ToolUseBlock
from matchuco.tools import ToolContext, ToolError, default_registry, default_tools
from matchuco.tools.files import EditTool, ReadTool, WriteTool
from matchuco.tools.search import GlobTool, GrepTool
from matchuco.tools.shell import ShellTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("import os\n\n\ndef main():\n    return 1\n")
    (tmp_path / "notes.md").write_text("# notes\nalpha\nbeta\n")
    return ToolContext(root=tmp_path)


# --- context and schemas ----------------------------------------------------


def test_paths_outside_the_workspace_are_rejected(ctx: ToolContext) -> None:
    assert ctx.resolve("src/app.py").is_file()
    with pytest.raises(ToolError, match="outside the workspace"):
        ctx.resolve("../secrets.txt")


def test_specs_expose_a_json_schema() -> None:
    specs = {spec.name: spec for spec in default_registry().specs}
    assert set(specs) == {"glob", "grep", "read", "edit", "write", "shell"}
    assert "path" in specs["read"].input_schema["properties"]
    assert specs["read"].input_schema["required"] == ["path"]
    assert all(spec.description for spec in specs.values())


def test_tool_names_are_unique() -> None:
    names = [tool.name for tool in default_tools()]
    assert len(names) == len(set(names))


# --- read -------------------------------------------------------------------


async def test_read_numbers_lines_and_marks_the_file_read(ctx: ToolContext) -> None:
    out = await ReadTool().call({"path": "notes.md"}, ctx)
    assert out.splitlines() == ["1\t# notes", "2\talpha", "3\tbeta"]
    assert ctx.resolve("notes.md") in ctx.files_read


async def test_read_pages_with_offset_and_limit(ctx: ToolContext) -> None:
    out = await ReadTool().call({"path": "notes.md", "offset": 2, "limit": 1}, ctx)
    assert "2\talpha" in out
    assert "beta" not in out
    assert "showing lines 2-2 of 3" in out


async def test_read_reports_useful_failures(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        await ReadTool().call({"path": "nope.txt"}, ctx)
    with pytest.raises(ToolError, match="is a directory"):
        await ReadTool().call({"path": "src"}, ctx)
    with pytest.raises(ToolError, match="only 3 lines"):
        await ReadTool().call({"path": "notes.md", "offset": 9}, ctx)


async def test_read_rejects_binary_files(ctx: ToolContext) -> None:
    (ctx.root / "blob.bin").write_bytes(b"\x89PNG\x00\x01\x02")
    with pytest.raises(ToolError, match="binary"):
        await ReadTool().call({"path": "blob.bin"}, ctx)


async def test_invalid_arguments_are_reported_not_raised(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="invalid arguments for read"):
        await ReadTool().call({"path": "notes.md", "offset": 0}, ctx)


# --- write and the read-before-write rule -----------------------------------


async def test_write_creates_a_new_file_without_reading_it(ctx: ToolContext) -> None:
    out = await WriteTool().call({"path": "deep/new.txt", "content": "hi\n"}, ctx)
    assert "created" in out
    assert (ctx.root / "deep" / "new.txt").read_text() == "hi\n"


async def test_write_refuses_an_unread_existing_file(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="has not been read"):
        await WriteTool().call({"path": "notes.md", "content": "gone"}, ctx)
    assert "alpha" in (ctx.root / "notes.md").read_text()

    await ReadTool().call({"path": "notes.md"}, ctx)
    await WriteTool().call({"path": "notes.md", "content": "gone"}, ctx)
    assert (ctx.root / "notes.md").read_text() == "gone"


async def test_write_refuses_a_file_that_changed_since_it_was_read(ctx: ToolContext) -> None:
    await ReadTool().call({"path": "notes.md"}, ctx)
    ctx.files_read[ctx.resolve("notes.md")] -= 10  # simulate an edit from outside
    with pytest.raises(ToolError, match="changed on disk"):
        await WriteTool().call({"path": "notes.md", "content": "x"}, ctx)


# --- edit -------------------------------------------------------------------


async def test_edit_replaces_a_unique_string(ctx: ToolContext) -> None:
    await ReadTool().call({"path": "src/app.py"}, ctx)
    out = await EditTool().call(
        {"path": "src/app.py", "old_string": "return 1", "new_string": "return 2"}, ctx
    )
    assert "edited" in out
    assert "return 2" in (ctx.root / "src" / "app.py").read_text()


async def test_edit_refuses_ambiguous_and_missing_matches(ctx: ToolContext) -> None:
    (ctx.root / "dup.txt").write_text("a\na\n")
    await ReadTool().call({"path": "dup.txt"}, ctx)
    with pytest.raises(ToolError, match="appears 2 times"):
        await EditTool().call({"path": "dup.txt", "old_string": "a", "new_string": "b"}, ctx)
    with pytest.raises(ToolError, match="not found"):
        await EditTool().call({"path": "dup.txt", "old_string": "zzz", "new_string": "b"}, ctx)


async def test_edit_replace_all(ctx: ToolContext) -> None:
    (ctx.root / "dup.txt").write_text("a\na\n")
    await ReadTool().call({"path": "dup.txt"}, ctx)
    out = await EditTool().call(
        {"path": "dup.txt", "old_string": "a", "new_string": "b", "replace_all": True}, ctx
    )
    assert "2 occurrences" in out
    assert (ctx.root / "dup.txt").read_text() == "b\nb\n"


async def test_edit_rejects_a_no_op(ctx: ToolContext) -> None:
    await ReadTool().call({"path": "notes.md"}, ctx)
    with pytest.raises(ToolError, match="identical"):
        await EditTool().call({"path": "notes.md", "old_string": "x", "new_string": "x"}, ctx)


# --- glob and grep ----------------------------------------------------------


async def test_glob_finds_files_and_skips_generated_dirs(ctx: ToolContext) -> None:
    (ctx.root / "node_modules" / "pkg").mkdir(parents=True)
    (ctx.root / "node_modules" / "pkg" / "index.py").write_text("noise\n")
    out = await GlobTool().call({"pattern": "**/*.py"}, ctx)
    assert out == "src/app.py"


async def test_glob_reports_no_matches_instead_of_failing(ctx: ToolContext) -> None:
    assert "no files match" in await GlobTool().call({"pattern": "**/*.rs"}, ctx)


async def test_grep_returns_path_line_and_text(ctx: ToolContext) -> None:
    out = await GrepTool().call({"pattern": r"def \w+"}, ctx)
    assert out == "src/app.py:4: def main():"


async def test_grep_filters_by_glob_and_case(ctx: ToolContext) -> None:
    assert "no matches" in await GrepTool().call({"pattern": "alpha", "glob": "*.py"}, ctx)
    assert "notes.md" in await GrepTool().call({"pattern": "ALPHA", "case_insensitive": True}, ctx)


async def test_grep_files_only(ctx: ToolContext) -> None:
    out = await GrepTool().call({"pattern": ".", "files_only": True}, ctx)
    assert sorted(out.splitlines()) == ["notes.md", "src/app.py"]


async def test_grep_rejects_a_broken_regex(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="invalid regex"):
        await GrepTool().call({"pattern": "("}, ctx)


# --- shell ------------------------------------------------------------------


async def test_shell_returns_output(ctx: ToolContext) -> None:
    out = await ShellTool().call({"command": "python -c \"print('hello')\""}, ctx)
    assert out.strip() == "hello"


async def test_shell_merges_stderr_and_fails_on_nonzero_exit(ctx: ToolContext) -> None:
    command = "python -c \"import sys; print('boom', file=sys.stderr); sys.exit(3)\""
    with pytest.raises(ToolError, match="exit code 3") as excinfo:
        await ShellTool().call({"command": command}, ctx)
    assert "boom" in str(excinfo.value)


async def test_shell_times_out_instead_of_hanging(ctx: ToolContext) -> None:
    command = 'python -c "import time; time.sleep(30)"'
    with pytest.raises(ToolError, match="timed out"):
        await ShellTool().call({"command": command, "timeout": 1}, ctx)


# --- registry ---------------------------------------------------------------


async def test_registry_turns_failures_into_error_results(ctx: ToolContext) -> None:
    result = await default_registry().execute(
        ToolUseBlock(id="t1", name="read", input={"path": "nope.txt"}), ctx
    )
    assert result.is_error
    assert result.tool_use_id == "t1"
    assert "does not exist" in result.content


async def test_registry_reports_unknown_tools(ctx: ToolContext) -> None:
    result = await default_registry().execute(ToolUseBlock(id="t2", name="teleport", input={}), ctx)
    assert result.is_error
    assert "unknown tool" in result.content
