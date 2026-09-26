"""Permissions: deciding which tool calls run, which need a human, which never run.

An agent that can run shell commands can do anything you can. The permission
system is the harness's answer to "should this particular call happen?", and it
has three inputs:

1. The **mode** -- the session-wide stance, switchable at any time:
     default       read freely; ask before editing files or running commands
     accept_edits  also edit files freely; still ask before running commands
     plan          read-only: explore and propose a plan, change nothing
     bypass        run everything without asking (deny rules still apply)

2. **Rules** from settings, as `tool` or `tool(pattern)`:
     allow: ["shell(uv run pytest *)", "edit(docs/**)"]
     ask:   ["shell(git push *)"]
     deny:  ["read(.env)", "shell(rm *)"]
   File tools match the pattern as a path glob (`*` stays inside a directory,
   `**` crosses them); `shell` matches it against each sub-command.

3. The **tool's kind** (read / edit / execute / plan), so a new tool gets
   sensible treatment without a special case here.

They are combined in a fixed order, first match wins:

     deny rule                  -> deny   (nothing overrides a deny)
     exit_plan_mode             -> ask in plan mode, deny otherwise
     plan mode, not a read      -> deny   (plan mode is read-only)
     ask rule                   -> ask    (even in bypass mode)
     bypass mode                -> allow
     allow rule                 -> allow
     read tool                  -> allow
     edit tool + accept_edits   -> allow
     anything else              -> ask

`evaluate` is a pure function of (mode, rules, call), which makes the policy
easy to test exhaustively; `check` adds the side effects: asking the user and
remembering "always allow" answers.

What this is *not*: a sandbox. Rules match the text of a command, and a shell
has endless ways to hide what a command does. The splitting below closes the
obvious hole (`git status && rm -rf ~` must not ride on an allow rule for
`git *`) and refuses to auto-approve anything with substitutions or
redirections, but real isolation needs the OS: containers, a restricted user,
or a sandbox profile. Treat allow rules as convenience, deny rules as a guard
rail, and bypass mode as "I trust this machine to be disposable".
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from matchuco.tools.base import Gate, Tool, ToolContext, ToolKind

Mode = Literal["default", "accept_edits", "plan", "bypass"]
MODES: tuple[Mode, ...] = ("default", "accept_edits", "plan", "bypass")
# Shift+Tab cycles through these. bypass is left out on purpose: it should be
# a deliberate choice (`--mode bypass` or `/mode bypass`), not a keypress away.
CYCLE: tuple[Mode, ...] = ("default", "accept_edits", "plan")

Behavior = Literal["allow", "ask", "deny"]

_RULE_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(?:\((.*)\))?\s*$", re.DOTALL)


@dataclass(frozen=True)
class Rule:
    """`tool` or `tool(pattern)`. No pattern means every call to that tool."""

    tool: str
    pattern: str | None = None

    @classmethod
    def parse(cls, text: str) -> Rule:
        match = _RULE_RE.match(text)
        if match is None:
            raise ValueError(
                f"invalid permission rule {text!r}; expected 'tool' or 'tool(pattern)'"
            )
        tool, pattern = match.groups()
        pattern = pattern.strip() if pattern is not None else None
        return cls(tool, pattern or None)

    def __str__(self) -> str:
        return self.tool if self.pattern is None else f"{self.tool}({self.pattern})"

    def matches(self, tool: Tool[Any], subject: str) -> bool:
        if self.tool != tool.name:
            return False
        if self.pattern is None:
            return True
        if tool.kind == "execute":
            # `ls *` should also cover a bare `ls`.
            if self.pattern.endswith(" *") and subject == self.pattern[:-2]:
                return True
            return fnmatchcase(subject, self.pattern)
        return path_matches(self.pattern, subject)


def path_matches(pattern: str, path: str) -> bool:
    """Glob match for workspace-relative POSIX paths.

    `*` and `?` never cross a `/`; `**` does. A pattern ending in `/` means
    "everything under this directory".
    """
    if pattern.endswith("/"):
        pattern += "**"
    regex = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            regex.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            regex.append(".*")
            i += 2
        elif pattern[i] == "*":
            regex.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            regex.append("[^/]")
            i += 1
        else:
            regex.append(re.escape(pattern[i]))
            i += 1
    return re.fullmatch("".join(regex), path) is not None


# Anything that can smuggle a second command or write a file behind the back of
# a rule that only looked at the first word.
_UNSAFE = ("$(", "`", "<(", ">(", ">", "\n", "\r")
_SEPARATORS = {"&&", "||", ";", "|", "&", ";;", "|&"}


def split_command(command: str) -> tuple[list[str], bool]:
    """Split a shell command line into its sub-commands.

    Returns `(parts, simple)`. `simple` is False when the command uses features
    rules cannot reason about (substitution, redirection, multiple lines,
    unbalanced quotes); such a command can still be denied by a rule, but never
    auto-approved by one.

        split_command("git add . && git commit -m 'a && b'")
        -> (["git add .", "git commit -m a && b"], True)
    """
    simple = not any(token in command for token in _UNSAFE)
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.escape = ""  # keep Windows paths like C:\src intact
    try:
        tokens = list(lexer)
    except ValueError:  # unbalanced quotes
        return [command.strip()], False

    parts: list[str] = []
    current: list[str] = []
    for token in tokens:
        if token in _SEPARATORS:
            if current:
                parts.append(" ".join(current))
            current = []
        elif set(token) <= set("<>&|;()"):
            simple = False  # some other operator we do not model
            current.append(token)
        else:
            current.append(token)
    if current:
        parts.append(" ".join(current))
    return parts or [command.strip()], simple


@dataclass(frozen=True)
class Decision:
    behavior: Behavior
    reason: str


@dataclass(frozen=True)
class PermissionRequest:
    """Everything the user needs to decide on one tool call."""

    tool: str
    kind: ToolKind
    subject: str
    preview: str
    reason: str
    # What answering "always" would do, or None when there is no safe "always".
    always: str | None


@dataclass(frozen=True)
class PermissionReply:
    choice: Literal["yes", "always", "no"]
    feedback: str = ""


class Approver(Protocol):
    """Asks the human. The CLI implements it with a prompt; tests with a script."""

    def __call__(self, request: PermissionRequest) -> Awaitable[PermissionReply]: ...


def _rules(texts: Iterable[str | Rule]) -> list[Rule]:
    return [t if isinstance(t, Rule) else Rule.parse(t) for t in texts]


@dataclass
class PermissionPolicy:
    mode: Mode = "default"
    allow: list[Rule] = field(default_factory=list)
    ask: list[Rule] = field(default_factory=list)
    deny: list[Rule] = field(default_factory=list)
    # Called when the user answers "always" and a new allow rule is created,
    # so the CLI can persist it. The policy itself never touches the disk.
    on_new_rule: Callable[[Rule], None] | None = None

    @classmethod
    def from_rules(
        cls,
        mode: Mode = "default",
        *,
        allow: Iterable[str | Rule] = (),
        ask: Iterable[str | Rule] = (),
        deny: Iterable[str | Rule] = (),
    ) -> PermissionPolicy:
        return cls(mode, _rules(allow), _rules(ask), _rules(deny))

    # --- the pure part --------------------------------------------------------

    def evaluate(self, tool: Tool[Any], args: BaseModel, ctx: ToolContext) -> Decision:
        subject = tool.permission_subject(args, ctx)
        parts, simple = split_command(subject) if tool.kind == "execute" else ([subject], True)
        candidates = [subject, *parts] if len(parts) > 1 else parts

        def any_match(rules: list[Rule]) -> Rule | None:
            return next((r for r in rules for c in candidates if r.matches(tool, c)), None)

        if rule := any_match(self.deny):
            return Decision("deny", f"blocked by deny rule {rule}")

        if tool.kind == "plan":
            if self.mode == "plan":
                return Decision("ask", "the plan needs your approval")
            return Decision("deny", "exit_plan_mode can only be used in plan mode")
        if self.mode == "plan" and tool.kind != "read":
            return Decision(
                "deny",
                "plan mode is read-only. Keep exploring with read-only tools, then call "
                "exit_plan_mode with your plan",
            )

        if rule := any_match(self.ask):
            return Decision("ask", f"rule {rule} requires approval")
        if self.mode == "bypass":
            return Decision("allow", "bypass mode")

        # An allow rule must cover *every* sub-command, and only simple commands qualify.
        if simple and all(any(r.matches(tool, p) for r in self.allow) for p in parts):
            return Decision("allow", "allowed by rule")

        if tool.kind == "read":
            return Decision("allow", "read-only tool")
        if tool.kind == "edit" and self.mode == "accept_edits":
            return Decision("allow", "accept_edits mode")
        if tool.kind == "execute" and not simple:
            return Decision("ask", "complex command (substitution, redirection or multi-line)")
        return Decision("ask", f"{tool.kind} tools need approval in {self.mode} mode")

    def suggest_always(self, tool: Tool[Any], args: BaseModel, ctx: ToolContext) -> str | None:
        """Describe what "always" would do for this call, or None if it is not offered."""
        if tool.kind == "plan":
            return "approve and switch to accept_edits mode"
        if tool.kind == "edit":
            return "switch to accept_edits mode for this session"
        if tool.kind == "execute":
            rule = self._rule_for(tool, args, ctx)
            return f"always allow {rule}" if rule else None
        return None

    def _rule_for(self, tool: Tool[Any], args: BaseModel, ctx: ToolContext) -> Rule | None:
        # An exact-command rule: predictable and never broader than what was approved.
        command = tool.permission_subject(args, ctx)
        parts, simple = split_command(command)
        if not simple or len(parts) != 1 or any(c in command for c in "*?["):
            return None
        return Rule(tool.name, parts[0])

    # --- the side-effecting part ---------------------------------------------

    async def check(
        self,
        tool: Tool[Any],
        args: BaseModel,
        ctx: ToolContext,
        approver: Approver | None,
    ) -> str | None:
        """Decide, asking the user if needed. Returns None to run, or a denial message."""
        decision = self.evaluate(tool, args, ctx)
        if decision.behavior == "allow":
            return None
        if decision.behavior == "deny":
            return f"Permission denied: {decision.reason}."
        if approver is None:
            return (
                f"Permission denied: {decision.reason}, and no user is available to approve "
                "it (non-interactive run). Do not retry; finish with what you can do, or tell "
                "the user which permission to grant."
            )

        always = self.suggest_always(tool, args, ctx)
        reply = await approver(
            PermissionRequest(
                tool=tool.name,
                kind=tool.kind,
                subject=tool.permission_subject(args, ctx),
                preview=tool.preview(args, ctx),
                reason=decision.reason,
                always=always,
            )
        )
        if reply.choice == "no":
            message = "The user denied this tool call."
            if reply.feedback:
                message += f" Their instructions: {reply.feedback}"
            else:
                message += " Do not retry it; ask the user how to proceed if you are unsure."
            return message

        if reply.choice == "always" and always is not None:
            self._remember(tool, args, ctx)
        if tool.kind == "plan":
            self.mode = "accept_edits" if reply.choice == "always" else "default"
        return None

    def _remember(self, tool: Tool[Any], args: BaseModel, ctx: ToolContext) -> None:
        if tool.kind == "edit":
            if self.mode == "default":  # never *lower* the mode, e.g. from bypass
                self.mode = "accept_edits"
        elif tool.kind == "execute" and (rule := self._rule_for(tool, args, ctx)):
            self.allow.append(rule)
            if self.on_new_rule is not None:
                self.on_new_rule(rule)

    def gate(self, ctx: ToolContext, approver: Approver | None) -> Gate:
        """Adapt `check` to the registry's `Gate` signature."""

        async def gate(tool: Tool[Any], args: BaseModel) -> str | None:
            return await self.check(tool, args, ctx, approver)

        return gate

    def cycle_mode(self) -> Mode:
        """Next mode in the Shift+Tab cycle. From bypass, go back to the start."""
        index = CYCLE.index(self.mode) if self.mode in CYCLE else -1
        self.mode = CYCLE[(index + 1) % len(CYCLE)]
        return self.mode
