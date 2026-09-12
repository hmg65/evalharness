"""Deterministic mock provider.

Simulates a hosted model with a fixed latency profile, token counts, failure
rate, and answer quality - no network, no keys, no cost. Used by the demo,
the test suite, and as a reference implementation of the ModelClient protocol.

Answers come from a small built-in QA bank: each bank topic has a good, an
ok, and a bad canned answer, and the `quality` knob sets how often the mock
reaches for the good one. Everything is driven by seeded randomness, so a
demo run is reproducible: same config -> same latencies, answers, and scores.
"""
from __future__ import annotations

import random
from typing import Any

from .base import ModelResponse

# match: lowercase substring that maps a prompt to a topic.
# good answers are correct and complete, ok answers are correct but thin,
# bad answers are wrong or refusals - so rubric scorers can tell them apart.
_BANK: list[dict[str, str]] = [
    {
        "match": "6 multiplied by 7",
        "good": "6 multiplied by 7 is 42, since six groups of seven make forty-two.",
        "ok": "It is 42.",
        "bad": "I am not sure, but I think it is 41.",
    },
    {
        "match": "photosynthesis",
        "good": "Photosynthesis converts light energy into chemical energy stored in glucose, releasing oxygen split from water.",
        "ok": "Plants use sunlight to make food.",
        "bad": "Photosynthesis is when plants breathe in oxygen at night.",
    },
    {
        "match": "hash map",
        "good": "Hash map lookup is average O(1): the key is hashed straight to a bucket, and collisions are handled by chaining.",
        "ok": "It is a structure with fast lookups.",
        "bad": "Hash map lookup is always O(n) because hashing causes collisions.",
    },
    {
        "match": "versailles",
        "good": "The Treaty of Versailles was signed in 1919, formally ending the state of war between Germany and the Allied Powers.",
        "ok": "A treaty signed after World War One.",
        "bad": "The Treaty of Versailles ended World War Two in 1945.",
    },
    {
        "match": "jsonl",
        "good": "JSONL lets you append records one line at a time and stream them, without parsing one giant document first.",
        "ok": "It is line-based.",
        "bad": "JSONL requires loading the entire file into memory before reading any record.",
    },
    {
        "match": "p95",
        "good": "p95 captures tail latency - the slowest 5% of calls - which a mean hides when outliers dominate the user experience.",
        "ok": "It is a high percentile.",
        "bad": "p95 is just the mean multiplied by 0.95.",
    },
    {
        "match": "temperature and seed",
        "good": "Fixing temperature and seed makes runs reproducible, so measured differences reflect the model rather than sampling noise.",
        "ok": "For fairness.",
        "bad": "Temperature and seed only change formatting, not the content of outputs.",
    },
    {
        "match": "retry and a timeout",
        "good": "A timeout abandons a call that is taking too long; a retry sends a fresh attempt after a failure, ideally with backoff.",
        "ok": "A retry tries again.",
        "bad": "A retry and a timeout are two names for the same mechanism.",
    },
]

_FALLBACK = {
    "good": "Based on the question, the correct answer follows directly from the definitions involved.",
    "ok": "It depends.",
    "bad": "I am not sure I can answer that.",
}


def _topic(prompt: str) -> dict[str, str]:
    low = prompt.lower()
    for entry in _BANK:
        if entry["match"] in low:
            return entry
    return _FALLBACK


class MockClient:
    """A fake hosted model with a tunable personality.

    latency_ms: base latency; each call adds jitter and a prompt-length term.
    quality: 0..1 probability of a good answer (rest split between ok/bad).
    fail_rate: 0..1 probability of a simulated 500 per call (pre-retry).
    Calls slower than the `mock_max_latency_ms` param raise TimeoutError, so
    the demo exercises the runner's timeout and retry path for real.
    """

    def __init__(
        self,
        name: str,
        model_id: str = "mock-model",
        latency_ms: float = 400.0,
        jitter_ms: float = 150.0,
        quality: float = 0.8,
        fail_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.quality = quality
        self.fail_rate = fail_rate
        self._rng = random.Random(seed)

    def complete(self, prompt: str, params: dict[str, Any]) -> ModelResponse:
        if self._rng.random() < self.fail_rate:
            raise RuntimeError(f"mock 500: simulated provider error from '{self.name}'")

        latency = self.latency_ms + self._rng.uniform(0, self.jitter_ms) + 0.05 * len(prompt)
        cap = params.get("mock_max_latency_ms")
        if cap is not None and latency > cap:
            raise TimeoutError(
                f"mock timeout: {latency:.0f}ms exceeded the {cap:.0f}ms limit for '{self.name}'"
            )

        roll = self._rng.random()
        topic = _topic(prompt)
        if roll < self.quality:
            text = topic["good"]
        elif roll < self.quality + (1 - self.quality) * 0.6:
            text = topic["ok"]
        else:
            text = topic["bad"]
        if params.get("max_tokens") is not None and params["max_tokens"] < 10:
            text = text[: params["max_tokens"] * 4]  # crude truncation to honor the param

        return ModelResponse(
            text=text,
            latency_s=latency / 1000.0,
            prompt_tokens=len(prompt.split()) + 8,
            completion_tokens=len(text.split()),
            model_id=self.model_id,
            raw={"mock": True},
        )
