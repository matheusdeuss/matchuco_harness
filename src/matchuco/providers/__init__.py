"""Model providers and a small factory to build one by name."""

from __future__ import annotations

from matchuco.providers.base import (
    Done,
    Provider,
    ProviderError,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolUseStart,
    complete,
)

PROVIDER_NAMES = ("anthropic", "openai", "ollama", "fake")

__all__ = [
    "PROVIDER_NAMES",
    "Done",
    "Provider",
    "ProviderError",
    "StreamEvent",
    "TextDelta",
    "ThinkingDelta",
    "ToolUseStart",
    "complete",
    "create_provider",
]


def create_provider(name: str, model: str | None = None, base_url: str | None = None) -> Provider:
    """Build a provider by name. SDK imports are deferred so unused ones cost nothing."""
    if name == "anthropic":
        from matchuco.providers.anthropic import DEFAULT_MODEL, AnthropicProvider

        return AnthropicProvider(model or DEFAULT_MODEL)

    if name in ("openai", "ollama"):
        from matchuco.providers.openai_compat import (
            DEFAULT_MODEL,
            OLLAMA_BASE_URL,
            OLLAMA_DEFAULT_MODEL,
            OpenAICompatProvider,
        )

        if name == "ollama":
            return OpenAICompatProvider(
                model or OLLAMA_DEFAULT_MODEL,
                name="ollama",
                base_url=base_url or OLLAMA_BASE_URL,
                api_key="ollama",  # Ollama ignores the key, but the SDK requires one
            )
        return OpenAICompatProvider(model or DEFAULT_MODEL, base_url=base_url)

    if name == "fake":
        from matchuco.providers.fake import FakeProvider

        return FakeProvider(["Hello from the fake provider!"] * 100, model=model or "fake-model")

    raise ValueError(f"unknown provider {name!r}; choose one of {', '.join(PROVIDER_NAMES)}")
