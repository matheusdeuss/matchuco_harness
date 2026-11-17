"""A minimal MCP client: tools over stdio, nothing else.

The server is a subprocess. The client speaks newline-delimited JSON-RPC and
only four messages: `initialize`, the `notifications/initialized` notice,
`tools/list`, and `tools/call`. Resources, prompts, sampling, and HTTP are
out of scope.

Each listed tool is registered as `mcp_<server>_<tool>` with kind `execute`,
so it walks through the same permission gate as `shell`: ask in the default
mode, deny in plan mode. A broken response is a `tool_result` error, not an
exception that kills the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from matchuco import __version__
from matchuco.config import McpSettings
from matchuco.messages import ToolSpec
from matchuco.tools.base import Tool, ToolContext, ToolError, ToolRegistry

__all__ = [
    "McpClient",
    "McpError",
    "McpTool",
    "attach_mcp_server",
    "connect_mcp_servers",
    "mcp_tool_name",
]

PROTOCOL_VERSION = "2024-11-05"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_DEFAULT_TIMEOUT = 10.0


class McpError(Exception):
    """The MCP server could not be started, or its response could not be used."""


@dataclass(frozen=True)
class McpToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


class McpArguments(BaseModel):
    """Arguments the server described. Extra keys are the tool's own fields."""

    model_config = ConfigDict(extra="allow")


class McpClient:
    """One stdio server. `start` performs the handshake and lists tools."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not command:
            raise McpError("MCP server command is empty")
        self.command = [str(part) for part in command]
        self.cwd = cwd
        self.timeout = timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr = bytearray()
        self._next_id = 0

    async def start(self) -> list[McpToolInfo]:
        """Spawn the process, initialize, and return the tools it listed."""
        await self._spawn()
        try:
            await self._request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "matchuco", "version": __version__},
                },
            )
            await self._notify("notifications/initialized")
            listed = await self._request("tools/list", {})
            return _parse_tools(listed)
        except BaseException:
            await self.close()
            raise

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> str:
        """Run `tools/call`. `isError` and a JSON-RPC error become `McpError`."""
        result = await self._request("tools/call", {"name": name, "arguments": dict(arguments)})
        text = _content_text(result)
        if result.get("isError"):
            raise McpError(text or f"MCP tool {name} returned an error")
        return text

    async def close(self) -> None:
        """Stop the subprocess. Safe to call more than once."""
        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.close()
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except TimeoutError:
                proc.kill()
                await proc.wait()
        task = self._stderr_task
        self._stderr_task = None
        if task is not None:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(task, timeout=1)

    async def _spawn(self) -> None:
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=self.cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise McpError(f"could not start MCP server {self.command!r}: {exc}") from exc
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                chunk = await proc.stderr.read(1024)
                if not chunk:
                    return
                self._stderr.extend(chunk)
                if len(self._stderr) > 8000:
                    del self._stderr[:-4000]
        except asyncio.CancelledError:
            return

    async def _notify(self, method: str) -> None:
        await self._write({"jsonrpc": "2.0", "method": method})

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            message = await self._read_message()
            if message.get("id") != request_id:
                continue
            error = message.get("error")
            if error:
                text = error.get("message") if isinstance(error, dict) else str(error)
                raise McpError(f"MCP {method} failed: {text}")
            result = message.get("result")
            if not isinstance(result, dict):
                raise McpError(f"MCP {method} returned no result object")
            return result

    async def _read_message(self) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise McpError("MCP server is not running")
        while True:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=self.timeout)
            except TimeoutError:
                raise McpError(f"MCP server timed out after {self.timeout:g}s") from None
            if not line:
                raise McpError(f"MCP server closed stdout{self._stderr_suffix()}")
            if not line.strip():
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                snippet = line[:200].decode("utf-8", errors="replace")
                raise McpError(f"MCP server returned invalid JSON: {snippet!r}") from exc
            if not isinstance(payload, dict):
                raise McpError("MCP server returned JSON that is not an object")
            return payload

    async def _write(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise McpError("MCP server is not running")
        proc.stdin.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
        await proc.stdin.drain()

    def _stderr_suffix(self) -> str:
        if not self._stderr:
            return ""
        text = self._stderr.decode("utf-8", errors="replace").strip()
        if not text:
            return ""
        return f" (stderr: {text[-500:]})"


def mcp_tool_name(server: str, tool: str) -> str:
    """`mcp_<server>_<tool>`. Both parts must be identifiers, so the name cannot smuggle a path."""
    if _NAME.fullmatch(server) is None or _NAME.fullmatch(tool) is None:
        raise McpError(
            f"MCP server and tool names must be identifiers, got server={server!r} tool={tool!r}"
        )
    return f"mcp_{server}_{tool}"


class McpTool(Tool[McpArguments]):
    """One remote tool. Kind `execute`: ask by default, denied in plan mode."""

    name = "mcp"
    description = "A tool exposed by an MCP server."
    input_model = McpArguments
    kind = "execute"

    def __init__(
        self,
        client: McpClient,
        server: str,
        info: McpToolInfo,
    ) -> None:
        self.name = mcp_tool_name(server, info.name)
        self.description = info.description or f"Tool {info.name} from MCP server {server}."
        self._client = client
        self._remote_name = info.name
        self._schema = info.input_schema

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, input_schema=self._schema)

    def permission_subject(self, args: McpArguments, ctx: ToolContext) -> str:
        del ctx
        payload = json.dumps(args.model_dump(), sort_keys=True, separators=(",", ":"))
        return f"{self._remote_name} {payload}"

    def preview(self, args: McpArguments, ctx: ToolContext) -> str:
        return f"{self.name}\n{self.permission_subject(args, ctx)}"

    async def run(self, args: McpArguments, ctx: ToolContext) -> str:
        del ctx
        try:
            return await self._client.call_tool(self._remote_name, args.model_dump())
        except McpError as exc:
            raise ToolError(str(exc)) from exc


async def attach_mcp_server(
    registry: ToolRegistry,
    *,
    name: str,
    command: Sequence[str],
    cwd: Path | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> McpClient:
    """Start one server and register its tools. Closes the process if registration fails."""
    client = McpClient(command, cwd=cwd, timeout=timeout)
    try:
        listed = await client.start()
        for info in listed:
            registry.add(McpTool(client, name, info))
    except McpError:
        await client.close()
        raise
    except Exception as exc:
        await client.close()
        raise McpError(f"MCP server {name!r} failed: {exc}") from exc
    return client


async def connect_mcp_servers(
    registry: ToolRegistry,
    mcp: McpSettings,
    *,
    cwd: Path | None = None,
) -> list[McpClient]:
    """Start every server in settings. Already-started servers are closed if a later one fails."""
    clients: list[McpClient] = []
    try:
        for name, server in mcp.servers.items():
            clients.append(
                await attach_mcp_server(registry, name=name, command=server.command, cwd=cwd)
            )
    except Exception:
        for client in clients:
            await client.close()
        raise
    return clients


def _parse_tools(result: Mapping[str, Any]) -> list[McpToolInfo]:
    raw = result.get("tools")
    if not isinstance(raw, list):
        raise McpError("MCP tools/list returned no tools array")
    found: list[McpToolInfo] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise McpError("MCP tools/list returned a tool without a name")
        schema = item.get("inputSchema", {"type": "object", "properties": {}})
        if not isinstance(schema, dict):
            raise McpError(f"MCP tool {item['name']} has an inputSchema that is not an object")
        description = item.get("description")
        found.append(
            McpToolInfo(
                name=item["name"],
                description=description if isinstance(description, str) else "",
                input_schema=schema,
            )
        )
    return found


def _content_text(result: Mapping[str, Any]) -> str:
    content = result.get("content", [])
    if not isinstance(content, list):
        raise McpError("MCP tools/call result has no content list")
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)
