"""Result tables: JSONL -> per-model summary (JSON, CSV, Markdown)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .metrics import summarize

SUMMARY_FIELDS = [
    "model", "calls", "errors", "error_rate",
    "latency_mean_s", "latency_median_s", "latency_p95_s", "latency_p99_s",
    "quality_mean", "total_prompt_tokens", "total_completion_tokens", "retries",
]


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_model.setdefault(row["model"], []).append(row)
    stats = [summarize(model, rows).to_dict() for model, rows in sorted(by_model.items())]
    for s in stats:
        for k, v in s.items():
            if isinstance(v, float):
                s[k] = round(v, 4)
    return stats


def write_summary(stats: list[dict[str, Any]], out_dir: str | Path, run_name: str) -> tuple[Path, Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{run_name}.summary.json"
    csv_path = out / f"{run_name}.summary.csv"
    md_path = out / f"{run_name}.summary.md"

    json_path.write_text(json.dumps(stats, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows([{k: s.get(k, "") for k in SUMMARY_FIELDS} for s in stats])

    lines = [
        f"# {run_name} - model comparison summary",
        "",
        "| Model | Calls | Errors | Error rate | Mean lat (s) | Median | p95 | p99 | Quality |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        lines.append(
            f"| {s['model']} | {s['calls']} | {s['errors']} | {s['error_rate']:.1%} "
            f"| {s['latency_mean_s']:.3f} | {s['latency_median_s']:.3f} "
            f"| {s['latency_p95_s']:.3f} | {s['latency_p99_s']:.3f} | {s['quality_mean']:.3f} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines))
    return json_path, csv_path, md_path
