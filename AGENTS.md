# matchuco — agent instructions

A multi-provider agentic harness in Python, built phase by phase as a learning
project. Read `README.md` for the architecture and `docs/` for the per-phase
write-ups (in Portuguese).

## Commands

- Install: `uv sync`
- Tests: `uv run pytest -q` (offline; the fake provider stands in for real models)
- Lint and format: `uv run ruff check .` and `uv run ruff format .`
- Types: `uv run mypy` (strict mode, must stay clean)

Run all four before calling a change done.

## Conventions

- Source lives in `src/matchuco/`; code, comments and the README are in English;
  `docs/NN-*.md` are learning notes in Portuguese.
- The core never imports a provider SDK: everything goes through the neutral
  types in `messages.py`. SDK-specific code stays inside `providers/`.
- The conversation history is append-only. Never edit or delete earlier
  messages; see the docstring at the top of `agent.py`.
- Tool failures are returned as error `tool_result`s, never raised past the registry.
- Every behaviour change gets a test. Prefer `FakeProvider` scripts over mocks.
- Each phase ends with a commit titled `Phase N: ...` and a new `docs/NN-*.md`.

## Compact Instructions

Keep the current phase number, which files were changed, and the state of the
test/lint/mypy runs.
