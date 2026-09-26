"""The shell tool: run a command and give the model what a human would see.

This is the most powerful tool in the harness and the one with no safety net
yet -- it will happily run `rm -rf`. Phase 3 puts it behind permission modes;
until then the only limits are a timeout and an output cap.

Two details that matter for an agent, less so for a human:

- stdout and stderr are merged. The model needs the error message next to the
  output that produced it, not in a separate channel it has to ask for.
- the exit code is always reported, and a non-zero exit is marked as an error
  so the model does not read a failed build as a successful one.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from matchuco.tools.base import Tool, ToolContext, ToolError, truncate

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
MAX_OUTPUT_CHARS = 30_000


class ShellInput(BaseModel):
    command: str = Field(description="Command to run through the system shell.")
    cwd: str | None = Field(
        default=None, description="Directory to run in. Defaults to the workspace root."
    )
    timeout: int = Field(
        default=DEFAULT_TIMEOUT,
        ge=1,
        le=MAX_TIMEOUT,
        description=f"Seconds before the command is killed (max {MAX_TIMEOUT}).",
    )


class ShellTool(Tool[ShellInput]):
    name = "shell"
    description = (
        "Run a shell command in the workspace and return its merged stdout/stderr plus "
        "exit code. Use it for builds, tests, git and anything the other tools do not cover; "
        "prefer glob/grep/read for finding and reading files. Commands are not interactive: "
        "never run something that waits for input or never exits."
    )
    input_model = ShellInput

    async def run(self, args: ShellInput, ctx: ToolContext) -> str:
        cwd = ctx.resolve(args.cwd) if args.cwd else ctx.root
        if not cwd.is_dir():
            raise ToolError(f"{ctx.display(cwd)} is not a directory")

        process = await asyncio.create_subprocess_shell(
            args.command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,  # a prompt should fail fast, not hang
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=args.timeout)
        except TimeoutError:
            await _terminate(process)
            raise ToolError(
                f"command timed out after {args.timeout}s and was killed: {args.command}"
            ) from None
        except asyncio.CancelledError:
            await _terminate(process)
            raise

        output = truncate(stdout.decode("utf-8", errors="replace").strip(), MAX_OUTPUT_CHARS)
        code = process.returncode
        if code:
            raise ToolError(f"exit code {code}\n{output}" if output else f"exit code {code}")
        return output or "(no output, exit code 0)"


async def _terminate(process: asyncio.subprocess.Process) -> None:
    """Kill a runaway process without leaving a zombie behind."""
    if process.returncode is not None:
        return
    try:
        process.kill()
    except ProcessLookupError:
        return
    await process.wait()
