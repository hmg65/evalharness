"""Latency and reliability statistics over raw call records."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from typing import Any


def percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile on pre-sorted data; p in [0, 100]."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


@dataclass
class ModelStats:
    model: str
    calls: int
    errors: int
    error_rate: float
    latency_mean_s: float
    latency_median_s: float
    latency_p95_s: float
    latency_p99_s: float
    latency_min_s: float
    latency_max_s: float
    quality_mean: float
    total_prompt_tokens: int
    total_completion_tokens: int
    retries: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(model: str, records: list[dict[str, Any]]) -> ModelStats:
    """records: result rows for one model (see runner.RESULT_FIELDS)."""
    lat = sorted(r["latency_s"] for r in records if r["status"] == "ok")
    errors = sum(1 for r in records if r["status"] != "ok")
    qualities = [r["quality"] for r in records if r["status"] == "ok" and r.get("quality") is not None]
    return ModelStats(
        model=model,
        calls=len(records),
        errors=errors,
        error_rate=errors / len(records) if records else 0.0,
        latency_mean_s=statistics.fmean(lat) if lat else 0.0,
        latency_median_s=statistics.median(lat) if lat else 0.0,
        latency_p95_s=percentile(lat, 95),
        latency_p99_s=percentile(lat, 99),
        latency_min_s=lat[0] if lat else 0.0,
        latency_max_s=lat[-1] if lat else 0.0,
        quality_mean=statistics.fmean(qualities) if qualities else 0.0,
        total_prompt_tokens=sum(r.get("prompt_tokens") or 0 for r in records),
        total_completion_tokens=sum(r.get("completion_tokens") or 0 for r in records),
        retries=sum(r.get("attempts", 1) - 1 for r in records),
    )
