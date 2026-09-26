"""Command-line entry point: an interactive agent REPL, or one-shot `-p` mode.

This module is only a renderer. It turns the `AgentEvent` stream into
something readable in a terminal and reads lines from the user; every decision
about what to do lives in `agent.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from rich.console import Console

from matchuco import __version__
from matchuco.agent import (
    Agent,
    TextDelta,
    ThinkingDelta,
    ToolFinished,
    ToolStarted,
    TurnEnd,
)
from matchuco.providers import PROVIDER_NAMES, Provider, ProviderError, create_provider

console = Console()

RESULT_PREVIEW_LINES = 6


async def run_turn(agent: Agent, prompt: str) -> None:
    """Stream one agent turn to the terminal."""
    thinking = False
    async for event in agent.run(prompt):
        match event:
            case ThinkingDelta(text=text):
                if not thinking:
                    console.print("[dim]thinking...[/dim]")
                    thinking = True
                console.out(text, end="", style="dim", highlight=False)
            case TextDelta(text=text):
                if thinking:
                    console.out("\n")
                    thinking = False
                console.out(text, end="", highlight=False)
            case ToolStarted(name=name, args=args):
                console.print(f"\n[cyan]> {name}[/cyan] [dim]{_format_args(args)}[/dim]")
            case ToolFinished(result=result):
                style = "red" if result.is_error else "dim"
                console.print(f"[{style}]{_preview(result.content)}[/{style}]")
            case TurnEnd(reason=reason, steps=steps):
                console.out("")
                if reason == "max_steps":
                    console.print(f"[yellow]stopped after {steps} steps[/yellow]")
                elif reason in ("max_tokens", "refusal"):
                    console.print(f"[yellow]turn ended: {reason}[/yellow]")


def _format_args(args: dict[str, Any]) -> str:
    """One short line describing a call, so the user can follow along."""
    parts = []
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value)
        text = " ".join(text.split())  # collapse multi-line arguments onto one line
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(f"{key}={text}")
    return " ".join(parts)


def _preview(content: str) -> str:
    lines = content.splitlines() or ["(empty)"]
    shown = [f"  {line[:120]}" for line in lines[:RESULT_PREVIEW_LINES]]
    if len(lines) > RESULT_PREVIEW_LINES:
        shown.append(f"  ... +{len(lines) - RESULT_PREVIEW_LINES} lines")
    return "\n".join(shown)


async def repl(agent: Agent) -> None:
    provider = agent.provider
    console.print(
        f"[bold]matchuco[/bold] v{__version__} - {provider.name}/{provider.model} "
        f"- {len(agent.registry)} tools in {agent.context.root}\n"
        "[dim]/clear resets the conversation, /tools lists tools, /usage shows tokens, "
        "/exit quits[/dim]"
    )
    session: PromptSession[str] = PromptSession()
    while True:
        try:
            line = (await session.prompt_async("\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return
        if line == "/clear":
            agent.clear()
            console.print("[dim]conversation cleared[/dim]")
            continue
        if line == "/tools":
            console.print(", ".join(agent.registry.names))
            continue
        if line == "/usage":
            console.print(agent.usage.model_dump())
            continue

        try:
            await run_turn(agent, line)
        except ProviderError as e:
            console.print(f"[red]error:[/red] {e}")
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]interrupted[/yellow]")


async def one_shot(agent: Agent, prompt: str) -> int:
    try:
        await run_turn(agent, prompt)
    except ProviderError as e:
        console.print(f"[red]error:[/red] {e}")
        return 1
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="matchuco", description=__doc__)
    parser.add_argument(
        "--provider",
        choices=PROVIDER_NAMES,
        default=os.environ.get("MATCHUCO_PROVIDER", "anthropic"),
    )
    parser.add_argument("--model", default=os.environ.get("MATCHUCO_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("MATCHUCO_BASE_URL"))
    parser.add_argument("--cwd", default=None, help="workspace root (default: current directory)")
    parser.add_argument("--max-steps", type=int, default=None, help="tool rounds per turn")
    parser.add_argument("-p", "--print", dest="prompt", help="answer one prompt and exit")
    parser.add_argument("--version", action="version", version=f"matchuco {__version__}")
    return parser.parse_args(argv)


def build_agent(args: argparse.Namespace, provider: Provider) -> Agent:
    kwargs: dict[str, Any] = {"root": Path(args.cwd) if args.cwd else None}
    if args.max_steps is not None:
        kwargs["max_steps"] = args.max_steps
    return Agent(provider, **kwargs)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    try:
        provider = create_provider(args.provider, args.model, args.base_url)
    except ProviderError as e:
        console.print(f"[red]error:[/red] {e}")
        sys.exit(1)
    agent = build_agent(args, provider)
    if args.prompt:
        sys.exit(asyncio.run(one_shot(agent, args.prompt)))
    asyncio.run(repl(agent))


if __name__ == "__main__":
    main()
