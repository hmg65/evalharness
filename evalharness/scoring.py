"""Pluggable quality scoring.

A scorer turns (prompt, reference, response_text) into a float in [0, 1] plus
a reason string. Scorers are declared in the run config and composed into a
weighted mean, so the rubric lives next to the run that used it.

Built-ins cover the cheap deterministic cases. For pairwise/LLM-judged
scoring, implement the Scorer protocol (judge.py in the examples shows a
judge that itself calls a model through the same provider layer).
"""
from __future__ import annotations

import re
from typing import Any, Protocol


class Scorer(Protocol):
    name: str
    weight: float

    def score(self, prompt: str, reference: str | None, text: str) -> tuple[float, str]:
        ...


class ContainsScorer:
    """1.0 when every expected substring appears (case-insensitive)."""

    name = "contains"

    def __init__(self, weight: float = 1.0, **_):
        self.weight = weight

    def score(self, prompt: str, reference: str | None, text: str) -> tuple[float, str]:
        if not reference:
            return 0.0, "no reference to check against"
        missing = [s for s in reference.split("|") if s.strip().lower() not in text.lower()]
        if missing:
            return 0.0, f"missing: {missing}"
        return 1.0, "all expected substrings present"


class RegexScorer:
    """1.0 when the pattern matches the response."""

    name = "regex"

    def __init__(self, pattern: str, weight: float = 1.0, **_):
        self.pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        self.weight = weight

    def score(self, prompt: str, reference: str | None, text: str) -> tuple[float, str]:
        hit = bool(self.pattern.search(text))
        return (1.0 if hit else 0.0), f"pattern {'matched' if hit else 'did not match'}"


class LengthScorer:
    """Scores how close the response length (words) is to a target band."""

    name = "length"

    def __init__(self, min_words: int = 5, max_words: int = 120, weight: float = 1.0, **_):
        self.min_words, self.max_words, self.weight = min_words, max_words, weight

    def score(self, prompt: str, reference: str | None, text: str) -> tuple[float, str]:
        n = len(text.split())
        if self.min_words <= n <= self.max_words:
            return 1.0, f"{n} words within [{self.min_words}, {self.max_words}]"
        overshoot = max(self.min_words - n, n - self.max_words)
        score = max(0.0, 1.0 - overshoot / max(self.max_words, 1))
        return score, f"{n} words outside target band"


class NotRefusalScorer:
    """0 when the response looks like a refusal or an 'I don't know'."""

    name = "not_refusal"
    _MARKERS = ("i cannot", "i can't", "i am not sure", "i'm not sure", "as an ai", "i do not know")

    def __init__(self, weight: float = 1.0, **_):
        self.weight = weight

    def score(self, prompt: str, reference: str | None, text: str) -> tuple[float, str]:
        low = text.lower()
        for marker in self._MARKERS:
            if marker in low:
                return 0.0, f"refusal marker: '{marker}'"
        return 1.0, "no refusal markers"


_REGISTRY = {s.name: s for s in (ContainsScorer, RegexScorer, LengthScorer, NotRefusalScorer)}


def build_scorers(specs: list[dict[str, Any]]) -> list[Scorer]:
    scorers: list[Scorer] = []
    for spec in specs:
        spec = dict(spec)
        kind = spec.pop("type")
        if kind not in _REGISTRY:
            raise ValueError(f"unknown scorer type '{kind}'; built-ins: {sorted(_REGISTRY)}")
        scorers.append(_REGISTRY[kind](**spec))
    return scorers


def weighted_score(scorers: list[Scorer], prompt: str, reference: str | None, text: str) -> tuple[float, dict[str, Any]]:
    """Run all scorers; return (weighted mean in [0,1], per-scorer detail)."""
    if not scorers:
        return 0.0, {}
    detail: dict[str, Any] = {}
    total_w = sum(s.weight for s in scorers)
    acc = 0.0
    for s in scorers:
        value, reason = s.score(prompt, reference, text)
        detail[s.name] = {"score": round(value, 4), "weight": s.weight, "reason": reason}
        acc += value * s.weight
    return (acc / total_w if total_w else 0.0), detail
