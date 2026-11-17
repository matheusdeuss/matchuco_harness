"""Fake MCP server for tests. Speaks initialize, tools/list, and tools/call on stdio.

Stdout is reserved for JSON-RPC lines. `--break-json` answers `tools/call` with
a line that is not JSON. `--log` records each `tools/call` so a test can see
whether the permission gate let the call through.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ECHO_TOOL: dict[str, Any] = {
    "name": "echo",
    "description": "Echo text back.",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--break-json", action="store_true")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    while True:
        line = sys.stdin.readline()
        if line == "":
            return
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict) or "id" not in message:
            continue
        method = message.get("method")
        request_id = message["id"]
        if method == "initialize":
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "echo", "version": "0"},
                    },
                }
            )
        elif method == "tools/list":
            _write({"jsonrpc": "2.0", "id": request_id, "result": {"tools": [ECHO_TOOL]}})
        elif method == "tools/call":
            params = message.get("params")
            params = params if isinstance(params, dict) else {}
            name = params.get("name", "")
            arguments = params.get("arguments")
            arguments = arguments if isinstance(arguments, dict) else {}
            _log_call(args.log, name if isinstance(name, str) else "", arguments)
            if args.break_json:
                sys.stdout.write("{not json\n")
                sys.stdout.flush()
                continue
            text = arguments.get("text", "")
            if not isinstance(text, str):
                text = json.dumps(text)
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": text}],
                        "isError": False,
                    },
                }
            )
        else:
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }
            )


def _log_call(path: Path | None, name: str, arguments: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"call {name} {json.dumps(arguments, sort_keys=True)}\n")
        handle.flush()


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
