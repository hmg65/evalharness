"""The Day 1 lab: a guided first look at how a model endpoint behaves.

One command - ``python -m evalharness lab`` - sends ten requests, one at a
time, to a bundled fake endpoint that behaves like a real streaming one, then
explains the measurements in plain language. No API keys, no network, no
cost. The endpoint is simulated (the output says so, every time); the
measurement method is the same one you will point at a real model later.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .providers.mock import StreamingMockClient
from .runner import write_jsonl

DAY1_MODEL = "mock-stream-7b"

_LONG_PROMPT = (
    "Here is some context from my application logs. Last night the nightly "
    "data pipeline took four hours instead of the usual forty minutes. The "
    "extract step finished on time, the transform step ran six times slower "
    "than normal, and the load step was normal. No errors appear anywhere in "
    "the logs, no retries happened, CPU and memory looked flat, and the only "
    "change deployed yesterday was a new deduplication rule in the transform "
    "step that compares every row against every other row in the same batch. "
    "Batch sizes were unchanged. Given all of this context, what is the most "
    "likely cause of the slowdown, and what is one measurement you would "
    "take tonight to confirm it before changing any code?"
)

# Ten beginner prompts, deliberately mixed. Most are short questions. One is
# a long prompt (r09) so the wait for the first token visibly grows. One asks
# for a long answer (r10, padded by the mock) so the per-token phase visibly
# dominates total time.
DAY1_PROMPTS: list[dict[str, Any]] = [
    {"id": "r01", "prompt": "What is 6 multiplied by 7?"},
    {"id": "r02", "prompt": "Explain photosynthesis in two sentences."},
    {"id": "r03", "prompt": "Why is hash map lookup fast?"},
    {"id": "r04", "prompt": "When was the Treaty of Versailles signed, and what did it end?"},
    {"id": "r05", "prompt": "What is JSONL and why do people use it for logs?"},
    {"id": "r06", "prompt": "What does p95 latency tell you that an average hides?"},
    {"id": "r07", "prompt": "Why should you fix temperature and seed when comparing two models?"},
    {"id": "r08", "prompt": "What is the difference between a retry and a timeout?"},
    {"id": "r09", "prompt": _LONG_PROMPT},
    {"id": "r10", "prompt": "Explain how a hash map handles collisions, step by step, with a worked example.",
     "params": {"pad_to_tokens": 80}},
]

RESULT_FIELDS = [
    "request", "prompt", "prompt_tokens", "completion_tokens",
    "ttft_s", "decode_s", "tpot_ms", "decode_tokens_per_s", "latency_s",
    "status", "error", "started_at",
]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "ok"]
    ttfts = [r["ttft_s"] for r in ok]
    tpots = [r["tpot_ms"] for r in ok]
    speeds = [r["decode_tokens_per_s"] for r in ok]
    latencies = [r["latency_s"] for r in ok]
    total_completion = sum(r["completion_tokens"] for r in ok)
    total_latency = sum(latencies)
    return {
        "model": DAY1_MODEL,
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "ttft_mean_s": statistics.fmean(ttfts) if ttfts else 0.0,
        "ttft_max_s": max(ttfts) if ttfts else 0.0,
        "tpot_mean_ms": statistics.fmean(tpots) if tpots else 0.0,
        "decode_tokens_per_s_mean": statistics.fmean(speeds) if speeds else 0.0,
        "latency_mean_s": statistics.fmean(latencies) if latencies else 0.0,
        "latency_max_s": max(latencies) if latencies else 0.0,
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in ok),
        "total_completion_tokens": total_completion,
        "overall_tokens_per_s": total_completion / total_latency if total_latency else 0.0,
    }


def run_day1(out_dir: str | Path = "results") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Send the ten Day 1 prompts one at a time (concurrency 1) and record
    what a streaming user would experience per request."""
    client = StreamingMockClient(
        name=DAY1_MODEL, model_id="mock-stream-7b-instruct", seed=42,
    )
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(DAY1_PROMPTS, start=1):
        prompt = item["prompt"]
        params = {"temperature": 0.0, "seed": 7, **item.get("params", {})}
        started = time.time()
        try:
            resp = client.complete(prompt, params)
            ttft_s = resp.ttft_s or 0.0
            decode_s = max(resp.latency_s - ttft_s, 0.0)
            out_tokens = resp.completion_tokens or 0
            rows.append({
                "request": i,
                "prompt": prompt if len(prompt) <= 80 else prompt[:77] + "...",
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": out_tokens,
                "ttft_s": round(ttft_s, 4),
                "decode_s": round(decode_s, 4),
                "tpot_ms": round(1000.0 * decode_s / max(out_tokens - 1, 1), 1),
                "decode_tokens_per_s": round((out_tokens - 1) / decode_s, 1) if decode_s > 0 and out_tokens > 1 else None,
                "latency_s": round(resp.latency_s, 4),
                "status": "ok",
                "error": "",
                "started_at": started,
            })
        except Exception as exc:  # provider errors are data, not crashes
            rows.append({
                "request": i,
                "prompt": prompt if len(prompt) <= 80 else prompt[:77] + "...",
                "prompt_tokens": None, "completion_tokens": None,
                "ttft_s": None, "decode_s": None, "tpot_ms": None,
                "decode_tokens_per_s": None, "latency_s": None,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "started_at": started,
            })

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, out / "lab-day1.results.jsonl")
    stats = _summarize(rows)
    chart_day1_timeline(rows, out / "lab-day1.timeline.png")
    (out / "lab-day1.summary.md").write_text(render_day1_markdown(rows, stats))
    return rows, stats


def chart_day1_timeline(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """One stacked bar per request: the wait for the first token, then the
    time spent writing the rest. Makes the two phases visible at a glance."""
    ok = [r for r in rows if r["status"] == "ok"]
    labels = [f"r{r['request']:02d}" for r in ok]
    ttft = [r["ttft_s"] for r in ok]
    decode = [r["decode_s"] for r in ok]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, ttft, label="wait for first token (TTFT)", color="#b3541e")
    ax.bar(labels, decode, bottom=ttft, label="writing the answer (decode)", color="#2b6cb0")
    ax.set_xlabel("request")
    ax.set_ylabel("seconds")
    ax.set_title("Day 1: where the time goes in each request")
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_day1_markdown(rows: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    lines = [
        "# Day 1 lab - what the measurements say",
        "",
        f"{stats['requests']} requests, one at a time, against `{DAY1_MODEL}` "
        "(a simulated endpoint - no real model, no cost).",
        "",
        "| Req | Prompt tokens | Output tokens | First token (s) | Per token (ms) | Total (s) | Tokens/sec |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["status"] == "ok":
            lines.append(
                f"| r{r['request']:02d} | {r['prompt_tokens']} | {r['completion_tokens']} "
                f"| {r['ttft_s']:.3f} | {r['tpot_ms']:.0f} | {r['latency_s']:.3f} "
                f"| {r['decode_tokens_per_s']:.0f} |"
            )
        else:
            lines.append(f"| r{r['request']:02d} | - | - | - | - | - | ERROR |")
    lines += [
        "",
        f"Errors: {stats['errors']} of {stats['requests']}",
        f"Mean wait for the first token: {stats['ttft_mean_s']:.3f} s "
        f"(slowest: {stats['ttft_max_s']:.3f} s)",
        f"Mean time per output token: {stats['tpot_mean_ms']:.0f} ms "
        f"(about {stats['decode_tokens_per_s_mean']:.0f} tokens/sec while writing)",
        f"Mean total time per request: {stats['latency_mean_s']:.3f} s",
        "",
    ]
    return "\n".join(lines)


def print_day1_report(rows: list[dict[str, Any]], stats: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    print()
    print("Day 1 lab: ten requests, one at a time")
    print(f"Endpoint: {DAY1_MODEL} (simulated - no real model, no keys, no cost)")
    print()
    print(f"{'Req':<5}{'Prompt tok':>11}{'Output tok':>11}{'First token':>13}"
          f"{'Per token':>11}{'Total':>9}{'Tok/sec':>9}")
    for r in rows:
        if r["status"] == "ok":
            print(f"r{r['request']:02d}  {r['prompt_tokens']:>10}{r['completion_tokens']:>11}"
                  f"{r['ttft_s']:>11.3f}s{r['tpot_ms']:>9.0f}ms"
                  f"{r['latency_s']:>8.3f}s{r['decode_tokens_per_s']:>9.0f}")
        else:
            print(f"r{r['request']:02d}  {'ERROR: ' + r['error']:>70}")
    print()
    print(f"errors: {stats['errors']} of {stats['requests']} requests failed")
    print()
    print("What those columns mean, in plain terms:")
    print("  Prompt tokens   How much text the model had to READ before answering.")
    print("  Output tokens   How much text it WROTE. Tokens are word pieces;")
    print("                  100 tokens is roughly 75 words.")
    print("  First token     TTFT, time to first token. How long you stare at a")
    print("                  blank screen before anything appears.")
    print("  Per token       TPOT, time per output token. How fast the words")
    print("                  appear once the answer starts coming.")
    print("  Total           Latency: the whole wait, start to finish.")
    print("                  Total = first token + per token x (output tokens - 1).")
    print("  Tok/sec         Writing speed during the answer: 1000 / per-token ms.")
    print()
    print("Three things worth noticing in your own numbers:")
    print(f"  1. r09 had the longest prompt ({max(rows, key=lambda r: r['prompt_tokens'] or 0)['prompt_tokens']} tokens)"
          " and the slowest first token. Long prompts cost time BEFORE the")
    print("     answer starts. That reading phase is called prefill.")
    print("  2. r10 asked for a long answer. Its first token is normal, but the")
    print("     total is the largest. Writing tokens is where long answers cost you.")
    print("  3. Tokens/sec barely moves across requests. The per-token writing")
    print("     speed is a property of the model and the machine, not the prompt.")
    print()
    print(f"Files written: {out}/lab-day1.results.jsonl, lab-day1.summary.md, lab-day1.timeline.png")
    print("Next: open docs/lab/day1.md and answer the five questions from your numbers.")
    print()


def cmd_lab(out_dir: str | Path = "results") -> int:
    rows, stats = run_day1(out_dir)
    print_day1_report(rows, stats, out_dir)
    return 0
