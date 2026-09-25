"""Command-line entry point: an interactive chat REPL, or one-shot `-p` mode.

Phase 1 scope: plain multi-turn chat with streaming, on any provider. The
agent loop and tools arrive in Phase 2 and plug in behind the same UI.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from rich.console import Console

from matchuco import __version__
from matchuco.messages import Message, Usage
from matchuco.providers import (
    PROVIDER_NAMES,
    Done,
    Provider,
    ProviderError,
    TextDelta,
    ThinkingDelta,
    ToolUseStart,
    create_provider,
)

SYSTEM_PROMPT = "You are matchuco, a helpful assistant running in the user's terminal."

console = Console()


async def run_turn(provider: Provider, history: list[Message]) -> Usage:
    """Stream one assistant turn to the terminal and append it to `history`."""
    thinking = False
    async for event in provider.stream(SYSTEM_PROMPT, history):
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
            case ToolUseStart(name=name):
                console.print(f"\n[cyan]> tool call: {name}[/cyan]")
            case Done(response=response):
                console.out("")
                history.append(response.message)
                if response.stop_reason in ("max_tokens", "refusal"):
                    console.print(f"[yellow]turn ended: {response.stop_reason}[/yellow]")
                return response.usage
    raise ProviderError("stream ended without a response")


async def repl(provider: Provider) -> None:
    console.print(
        f"[bold]matchuco[/bold] v{__version__} - {provider.name}/{provider.model}\n"
        "[dim]/clear resets the conversation, /usage shows tokens, /exit quits[/dim]"
    )
    session: PromptSession[str] = PromptSession()
    history: list[Message] = []
    total = Usage()
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
            history.clear()
            console.print("[dim]conversation cleared[/dim]")
            continue
        if line == "/usage":
            console.print(total.model_dump())
            continue

        history.append(Message.user(line))
        try:
            total += await run_turn(provider, history)
        except ProviderError as e:
            history.pop()  # drop the unanswered user message so history stays valid
            console.print(f"[red]error:[/red] {e}")
        except (KeyboardInterrupt, asyncio.CancelledError):
            history.pop()
            console.print("\n[yellow]interrupted[/yellow]")


async def one_shot(provider: Provider, prompt: str) -> int:
    try:
        await run_turn(provider, [Message.user(prompt)])
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
    parser.add_argument("-p", "--print", dest="prompt", help="answer one prompt and exit")
    parser.add_argument("--version", action="version", version=f"matchuco {__version__}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    try:
        provider = create_provider(args.provider, args.model, args.base_url)
    except ProviderError as e:
        console.print(f"[red]error:[/red] {e}")
        sys.exit(1)
    if args.prompt:
        sys.exit(asyncio.run(one_shot(provider, args.prompt)))
    asyncio.run(repl(provider))


if __name__ == "__main__":
    main()
