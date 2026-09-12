"""Provider protocol: every client turns a prompt into a timed ModelResponse."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ModelResponse:
    """One call to a hosted model.

    Latency fields are seconds measured wall-clock by the harness, so every
    provider is timed the same way. Token counts come from the provider when
    available (usage object); the mock provider fabricates them.
    """

    text: str
    latency_s: float
    ttft_s: float | None = None  # time to first token, when the provider streams
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class ModelClient(Protocol):
    """Anything that can answer a chat prompt with fixed sampling params."""

    name: str  # display name used in logs and charts
    model_id: str  # provider-side model identifier

    def complete(self, prompt: str, params: dict[str, Any]) -> ModelResponse:
        """Run one completion. Must raise on transport/API errors so the
        runner can apply its retry policy; return partial info via ModelResponse."""
        ...


class Timer:
    """Small wall-clock timer so providers report latency identically."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
