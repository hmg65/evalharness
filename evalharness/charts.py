"""Comparison charts (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _models(stats: list[dict[str, Any]]) -> list[str]:
    return [s["model"] for s in stats]


def chart_latency_bars(stats: list[dict[str, Any]], path: Path) -> Path:
    names = _models(stats)
    mean = [s["latency_mean_s"] for s in stats]
    p95 = [s["latency_p95_s"] for s in stats]
    x = range(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], mean, width, label="mean")
    ax.bar([i + width / 2 for i in x], p95, width, label="p95")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("latency (s)")
    ax.set_title("Latency: mean vs p95")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def chart_quality(stats: list[dict[str, Any]], path: Path) -> Path:
    names = _models(stats)
    quality = [s["quality_mean"] for s in stats]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, quality, color="#2b7a4b")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mean quality score (0-1)")
    ax.set_title("Quality: weighted rubric score")
    ax.bar_label(bars, fmt="%.3f")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def chart_error_rate(stats: list[dict[str, Any]], path: Path) -> Path:
    names = _models(stats)
    rates = [100.0 * s["error_rate"] for s in stats]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, rates, color="#b3402e")
    ax.set_ylabel("error rate (%)")
    ax.set_title("Reliability: calls that failed after retries")
    ax.bar_label(bars, fmt="%.1f%%")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def chart_latency_scatter(results: list[dict[str, Any]], path: Path) -> Path:
    """Per-call latency vs quality: shows the latency/quality tradeoff directly."""
    models = sorted({r["model"] for r in results})
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, model in enumerate(models):
        rows = [r for r in results if r["model"] == model and r["status"] == "ok"]
        ax.scatter(
            [r["latency_s"] for r in rows],
            [r["quality"] for r in rows],
            label=model, alpha=0.65, color=colors[i % len(colors)], edgecolors="none",
        )
    ax.set_xlabel("latency (s)")
    ax.set_ylabel("quality score (0-1)")
    ax.set_title("Latency vs quality per call")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_charts(results: list[dict[str, Any]], stats: list[dict[str, Any]],
                 out_dir: str | Path, run_name: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        chart_latency_bars(stats, out / f"{run_name}.latency.png"),
        chart_quality(stats, out / f"{run_name}.quality.png"),
        chart_error_rate(stats, out / f"{run_name}.errors.png"),
        chart_latency_scatter(results, out / f"{run_name}.latency_vs_quality.png"),
    ]
