"""Command-line entry point: an interactive agent REPL, or one-shot `-p` mode.

This module is only a renderer. It turns the `AgentEvent` stream into
something readable in a terminal and reads lines from the user; every decision
about what to do lives in `agent.py` and `permissions.py`. The one thing it
contributes is the human: `ask_permission` is the `Approver` the permission
system calls when a tool call needs a yes or no.
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
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from matchuco import __version__
from matchuco.agent import (
    Agent,
    Compacted,
    Compacting,
    ContextReport,
    TextDelta,
    ThinkingDelta,
    ToolFinished,
    ToolStarted,
    TurnEnd,
)
from matchuco.config import SettingsError, add_local_allow_rule, build_policy, load_settings
from matchuco.permissions import (
    MODES,
    Approver,
    Mode,
    PermissionReply,
    PermissionRequest,
    Rule,
)
from matchuco.providers import PROVIDER_NAMES, Provider, ProviderError, create_provider

console = Console()

RESULT_PREVIEW_LINES = 6

MODE_STYLES: dict[Mode, str] = {
    "default": "ansiwhite",
    "accept_edits": "ansigreen",
    "plan": "ansicyan",
    "bypass": "ansired",
}


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
                console.print(f"[{style}]{_preview(result.content)}[/{style}]", highlight=False)
            case Compacting(auto=auto, tokens=tokens):
                why = "context is filling up" if auto else "requested"
                console.print(
                    f"\n[yellow]compacting the conversation ({why}, ~{tokens:,} tokens)...[/yellow]"
                )
            case Compacted(tokens_before=before, tokens_after=after):
                console.print(f"[yellow]compacted: ~{before:,} -> ~{after:,} tokens[/yellow]")
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


# --- permission prompts --------------------------------------------------------


def parse_reply(answer: str, always_offered: bool) -> PermissionReply | None:
    """Interpret what the user typed. None means "ask again".

    Anything that is not a recognised answer is taken as a *no with
    instructions*, passed to the model -- "no, run the tests with -x first" is
    more useful to it than a bare refusal.
    """
    text = answer.strip()
    lowered = text.lower()
    if not text:
        return None  # never treat a stray Enter as consent
    if lowered in ("y", "yes", "s", "sim"):
        return PermissionReply("yes")
    if lowered in ("a", "always", "sempre"):
        return PermissionReply("always") if always_offered else None
    if lowered in ("n", "no", "nao", "não"):
        return PermissionReply("no")
    return PermissionReply("no", feedback=text)


def make_approver() -> Approver:
    session: PromptSession[str] = PromptSession()

    async def ask_permission(request: PermissionRequest) -> PermissionReply:
        is_diff = request.preview.startswith("---")
        body: Syntax | str = (
            Syntax(request.preview, "diff", theme="ansi_dark") if is_diff else request.preview
        )
        title = "approve the plan?" if request.kind == "plan" else f"allow {request.tool}?"
        console.print(Panel(body, title=title, subtitle=request.reason, border_style="yellow"))

        options = "[y] yes"
        if request.always:
            options += f"   [a] {request.always}"
        options += "   [n] no   (or type what to do instead)"
        # markup=False: otherwise rich reads "[y]" as a style tag and swallows it.
        console.print(options, style="dim", markup=False, highlight=False)
        while True:
            reply = parse_reply(await session.prompt_async("? "), request.always is not None)
            if reply is not None:
                return reply

    return ask_permission


# --- REPL ------------------------------------------------------------------------


HELP = (
    "/context shows what fills the context window, /compact [focus] summarizes the "
    "conversation, /mode [name] shows or sets the permission mode (shift+tab cycles), "
    "/permissions lists rules, /clear starts over, /tools lists tools, /usage shows tokens, "
    "/exit quits"
)


def render_context(report: ContextReport) -> None:
    """A /context breakdown: a bar, then the categories that fill it."""
    width = 50
    used = report.used
    filled = min(width, round(width * used / report.window))
    limit = min(width, round(width * report.threshold / report.window))
    bar = "".join("#" if i < filled else ("|" if i == limit else ".") for i in range(width))
    console.print(
        f"[bold]context[/bold] [{bar}] ~{used:,} / {report.window:,} tokens "
        f"({used / report.window:.1%}); auto-compacts at {report.threshold:,}",
        highlight=False,
        markup=True,
    )
    table = Table(box=None, show_header=False, padding=(0, 2))
    for name, tokens in report.categories:
        table.add_row(name, f"{tokens:,}", f"{tokens / report.window:.1%}")
    console.print(table)
    note = "estimates at ~4 chars/token"
    if report.last_reported is not None:
        note += f"; the provider reported {report.last_reported:,} input tokens last request"
    console.print(f"[dim]{note}[/dim]")


async def handle_command(agent: Agent, line: str) -> bool:
    """Run a slash command. Returns False when the REPL should exit."""
    name, _, arg = line.partition(" ")
    arg = arg.strip()
    policy = agent.permissions
    if name in ("/exit", "/quit"):
        return False
    if name == "/clear":
        agent.clear()
        console.print("[dim]conversation cleared[/dim]")
    elif name == "/context":
        render_context(agent.context_report())
    elif name == "/compact":
        console.print("[yellow]compacting...[/yellow]")
        result = await agent.compact(arg)
        if result is None:
            console.print("[dim]nothing to compact yet[/dim]")
        else:
            console.print(
                f"[yellow]compacted: ~{result.tokens_before:,} -> ~{result.tokens_after:,} "
                "tokens[/yellow]"
            )
    elif name == "/tools":
        console.print(", ".join(agent.registry.names))
    elif name == "/usage":
        console.print(agent.usage.model_dump())
    elif name == "/mode":
        if arg:
            if arg not in MODES:
                console.print(f"[red]unknown mode {arg!r}[/red]; choose from {', '.join(MODES)}")
            else:
                policy.mode = arg  # `arg in MODES` narrowed it to a Mode
        console.print(f"mode: [bold]{policy.mode}[/bold]")
    elif name == "/permissions":
        console.print(f"mode: [bold]{policy.mode}[/bold]")
        for label, rules in (("deny", policy.deny), ("ask", policy.ask), ("allow", policy.allow)):
            console.print(f"{label}: {', '.join(map(str, rules)) or '(none)'}")
    elif name == "/help":
        console.print(HELP)
    else:
        console.print(f"[red]unknown command {name}[/red]; {HELP}")
    return True


async def repl(agent: Agent) -> None:
    provider = agent.provider
    console.print(
        f"[bold]matchuco[/bold] v{__version__} - {provider.name}/{provider.model} "
        f"- {len(agent.registry)} tools in {agent.context.root}\n[dim]{HELP}[/dim]"
    )

    bindings = KeyBindings()

    @bindings.add("s-tab")
    def _cycle(event: KeyPressEvent) -> None:
        agent.permissions.cycle_mode()
        event.app.invalidate()  # redraw the toolbar

    def toolbar() -> HTML:
        mode = agent.permissions.mode
        used = agent.context_tokens() / agent.context_window
        return HTML(
            f"mode: <style fg='{MODE_STYLES[mode]}'><b>{mode}</b></style>  (shift+tab to cycle)"
            f"   context: {used:.0%}"
        )

    session: PromptSession[str] = PromptSession(key_bindings=bindings, bottom_toolbar=toolbar)
    while True:
        try:
            line = (await session.prompt_async("\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not line:
            continue
        if line.startswith("/"):
            try:
                if not await handle_command(agent, line):
                    return
            except ProviderError as e:
                console.print(f"[red]error:[/red] {e}")
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
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="permission mode (default: from settings, else 'default')",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="override the model's context window in tokens (e.g. to match a local server)",
    )
    parser.add_argument("-p", "--print", dest="prompt", help="answer one prompt and exit")
    parser.add_argument("--version", action="version", version=f"matchuco {__version__}")
    return parser.parse_args(argv)


def build_agent(args: argparse.Namespace, provider: Provider, *, interactive: bool) -> Agent:
    root = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    policy = build_policy(load_settings(root), args.mode)

    def persist(rule: Rule) -> None:
        # The approval already happened; failing to save it must not fail the call.
        try:
            path = add_local_allow_rule(root, rule)
        except (SettingsError, OSError) as e:
            console.print(f"[yellow]allowed {rule} for this session, but could not save it: {e}")
            return
        console.print(f"[dim]saved {rule} to {path}[/dim]")

    policy.on_new_rule = persist
    kwargs: dict[str, Any] = {
        "root": root,
        "permissions": policy,
        # Without a terminal there is nobody to ask: calls that need approval are denied.
        "approver": make_approver() if interactive else None,
    }
    if args.max_steps is not None:
        kwargs["max_steps"] = args.max_steps
    if args.context_window is not None:
        kwargs["context_window"] = args.context_window
    return Agent(provider, **kwargs)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    try:
        provider = create_provider(args.provider, args.model, args.base_url)
        agent = build_agent(args, provider, interactive=not args.prompt)
    except (ProviderError, SettingsError) as e:
        console.print(f"[red]error:[/red] {e}")
        sys.exit(1)
    if args.prompt:
        sys.exit(asyncio.run(one_shot(agent, args.prompt)))
    asyncio.run(repl(agent))


if __name__ == "__main__":
    main()
