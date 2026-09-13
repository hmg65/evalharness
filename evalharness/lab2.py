"""The Day 2 lab: the same ten prompts, against a real model you are serving.

Day 1 measured a simulated endpoint one request at a time. Day 2 keeps the
prompts, the columns, and the questions, but changes two things: the endpoint
is a real model you are serving yourself, and you choose how many requests
run at once (``--concurrency``). Everything is timed by the harness exactly
as in Day 1 - first token, per token, totals - so the numbers compare
directly.

One command:

    python -m evalharness lab2 --model qwen3:4b-instruct --concurrency 1

The endpoint is any OpenAI-compatible streaming server. The lab setup is a
local server on your own machine (Ollama is the default base URL); later the
same command points at a rented GPU server by changing ``--base-url`` and
``--model`` - the measurements carry over unchanged. Keys come from the
environment, as everywhere in this repo; a local server accepts any
non-empty string, so ``export LAB2_API_KEY=not-a-real-key`` is enough.
"""
from __future__ import annotations

import json
import os
import queue
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .lab import DAY1_PROMPTS
from .metrics import percentile
from .runner import write_jsonl

RESULT_FIELDS = [
    "request", "prompt", "prompt_tokens", "completion_tokens",
    "ttft_s", "decode_s", "tpot_ms", "decode_tokens_per_s", "latency_s",
    "status", "error", "started_at",
]

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY_ENV = "LAB2_API_KEY"


def stream_one(
    prompt: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Send one chat request with ``stream: true`` and time it the way a user
    experiences it: the wait for the first token, then each token after.

    Token counts come from the server's final ``usage`` chunk when it
    provides one (vLLM does, via stream_options.include_usage); otherwise
    streamed content chunks are counted, which overcounts slightly when the
    server batches tokens per chunk.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "seed": 7,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    start = time.time()
    ttft_s: float | None = None
    chunk_tokens = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    end = start
    with httpx.Client(timeout=timeout_s) as client:
        with client.stream(
            "POST", f"{base_url.rstrip('/')}/chat/completions",
            json=body, headers=headers,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                usage = event.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                for choice in event.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        if ttft_s is None:
                            ttft_s = time.time() - start
                        chunk_tokens += 1
        end = time.time()
    latency_s = end - start
    ttft = ttft_s if ttft_s is not None else latency_s
    out_tokens = completion_tokens if completion_tokens is not None else chunk_tokens
    decode_s = max(latency_s - ttft, 0.0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": out_tokens,
        "ttft_s": ttft,
        "decode_s": decode_s,
        "tpot_ms": 1000.0 * decode_s / max((out_tokens or 1) - 1, 1),
        "decode_tokens_per_s": ((out_tokens - 1) / decode_s) if decode_s > 0 and out_tokens and out_tokens > 1 else None,
        "latency_s": latency_s,
        "started_at": start,
    }


def run_day2(
    base_url: str,
    model: str,
    api_key: str,
    concurrency: int,
    max_tokens: int,
    out_dir: str | Path = "results",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Send the ten Day 1 prompts to a real streaming endpoint, up to
    ``concurrency`` at a time, and record the same per-request columns as
    Day 1 plus one new summary number: tokens/sec across the whole batch."""
    work: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()
    for i, item in enumerate(DAY1_PROMPTS, start=1):
        work.put((i, item))
    rows: list[dict[str, Any] | None] = [None] * len(DAY1_PROMPTS)
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                i, item = work.get_nowait()
            except queue.Empty:
                return
            prompt = item["prompt"]
            shown = prompt if len(prompt) <= 80 else prompt[:77] + "..."
            try:
                m = stream_one(
                    prompt, base_url=base_url, model=model,
                    api_key=api_key, max_tokens=max_tokens,
                )
                row: dict[str, Any] = {
                    "request": i,
                    "prompt": shown,
                    "prompt_tokens": m["prompt_tokens"],
                    "completion_tokens": m["completion_tokens"],
                    "ttft_s": round(m["ttft_s"], 4),
                    "decode_s": round(m["decode_s"], 4),
                    "tpot_ms": round(m["tpot_ms"], 1),
                    "decode_tokens_per_s": (
                        round(m["decode_tokens_per_s"], 1)
                        if m["decode_tokens_per_s"] is not None else None
                    ),
                    "latency_s": round(m["latency_s"], 4),
                    "status": "ok",
                    "error": "",
                    "started_at": m["started_at"],
                }
            except Exception as exc:  # provider errors are data, not crashes
                row = {
                    "request": i,
                    "prompt": shown,
                    "prompt_tokens": None, "completion_tokens": None,
                    "ttft_s": None, "decode_s": None, "tpot_ms": None,
                    "decode_tokens_per_s": None, "latency_s": None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "started_at": time.time(),
                }
            with lock:
                rows[i - 1] = row
            work.task_done()

    wall_start = time.time()
    threads = [threading.Thread(target=worker) for _ in range(max(concurrency, 1))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_s = time.time() - wall_start

    final_rows = [r for r in rows if r is not None]
    stats = _summarize(final_rows, model, concurrency, wall_s)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_name = f"lab-day2-c{concurrency}"
    write_jsonl(final_rows, out / f"{run_name}.results.jsonl")
    chart_day2_gantt(final_rows, concurrency, wall_s, out / f"{run_name}.timeline.png")
    (out / f"{run_name}.summary.md").write_text(render_day2_markdown(final_rows, stats))
    return final_rows, stats


def _summarize(rows: list[dict[str, Any]], model: str, concurrency: int, wall_s: float) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "ok"]
    ttfts = sorted(r["ttft_s"] for r in ok)
    tpots = [r["tpot_ms"] for r in ok]
    latencies = [r["latency_s"] for r in ok]
    total_completion = sum(r["completion_tokens"] or 0 for r in ok)
    return {
        "model": model,
        "concurrency": concurrency,
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "wall_s": wall_s,
        "ttft_mean_s": statistics.fmean(ttfts) if ttfts else 0.0,
        "ttft_max_s": max(ttfts) if ttfts else 0.0,
        "ttft_p95_s": percentile(ttfts, 95),
        "tpot_mean_ms": statistics.fmean(tpots) if tpots else 0.0,
        "latency_mean_s": statistics.fmean(latencies) if latencies else 0.0,
        "total_completion_tokens": total_completion,
        "batch_tokens_per_s": total_completion / wall_s if wall_s else 0.0,
    }


def chart_day2_gantt(rows: list[dict[str, Any]], concurrency: int, wall_s: float, path: str | Path) -> Path:
    """One horizontal bar per request, placed at its real start time: the
    orange wait for the first token, then the blue writing phase. At
    concurrency 1 the bars form a staircase; under load they overlap, which
    is the whole lesson."""
    ok = [r for r in rows if r["status"] == "ok"]
    t0 = min(r["started_at"] for r in ok) if ok else 0.0
    labels = [f"r{r['request']:02d}" for r in ok]
    starts = [r["started_at"] - t0 for r in ok]
    ttft = [r["ttft_s"] for r in ok]
    decode = [r["decode_s"] for r in ok]
    y = list(range(len(ok)))
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.barh(y, ttft, left=starts, color="#b3541e", label="wait for first token (TTFT)")
    b2 = ax.barh(y, decode, left=[s + t for s, t in zip(starts, ttft)], color="#2b6cb0", label="writing the answer (decode)")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("seconds since the batch started")
    ax.set_title(f"Day 2 at concurrency {concurrency}: ten requests in {wall_s:.1f} s")
    fig.legend(handles=[b1, b2], loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_day2_markdown(rows: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    lines = [
        "# Day 2 lab - a real model, under load",
        "",
        f"{stats['requests']} requests, {stats['concurrency']} at a time, against "
        f"`{stats['model']}` (a real model served with vLLM).",
        "",
        "| Req | Prompt tokens | Output tokens | First token (s) | Per token (ms) | Total (s) | Tokens/sec |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["status"] == "ok":
            speed = f"{r['decode_tokens_per_s']:.0f}" if r["decode_tokens_per_s"] else "-"
            ptok = r["prompt_tokens"] if r["prompt_tokens"] is not None else "-"
            lines.append(
                f"| r{r['request']:02d} | {ptok} | {r['completion_tokens']} "
                f"| {r['ttft_s']:.3f} | {r['tpot_ms']:.0f} | {r['latency_s']:.3f} "
                f"| {speed} |"
            )
        else:
            lines.append(f"| r{r['request']:02d} | - | - | - | - | - | ERROR |")
    lines += [
        "",
        f"Errors: {stats['errors']} of {stats['requests']}",
        f"Mean wait for the first token: {stats['ttft_mean_s']:.3f} s "
        f"(p95: {stats['ttft_p95_s']:.3f} s, slowest: {stats['ttft_max_s']:.3f} s)",
        f"Mean time per output token: {stats['tpot_mean_ms']:.0f} ms",
        f"Mean total time per request: {stats['latency_mean_s']:.3f} s",
        f"Whole batch: {stats['total_completion_tokens']} tokens in "
        f"{stats['wall_s']:.1f} s = {stats['batch_tokens_per_s']:.0f} tokens/sec overall",
        "",
    ]
    return "\n".join(lines)


def print_day2_report(rows: list[dict[str, Any]], stats: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    c = stats["concurrency"]
    print()
    print(f"Day 2 lab: ten requests, {c} at a time")
    print(f"Endpoint: {stats['model']} (a real model served from your own machine)")
    print()
    print(f"{'Req':<5}{'Prompt tok':>11}{'Output tok':>11}{'First token':>13}"
          f"{'Per token':>11}{'Total':>9}{'Tok/sec':>9}")
    for r in rows:
        if r["status"] == "ok":
            speed = f"{r['decode_tokens_per_s']:.0f}" if r["decode_tokens_per_s"] else "-"
            ptok = r["prompt_tokens"] if r["prompt_tokens"] is not None else "-"
            print(f"r{r['request']:02d}  {ptok:>10}{r['completion_tokens']:>11}"
                  f"{r['ttft_s']:>11.3f}s{r['tpot_ms']:>9.0f}ms"
                  f"{r['latency_s']:>8.3f}s{speed:>9}")
        else:
            print(f"r{r['request']:02d}  {'ERROR: ' + r['error']:>70}")
    print()
    print(f"errors: {stats['errors']} of {stats['requests']} requests failed")
    print()
    print("The new number at this concurrency:")
    print(f"  Batch tokens/sec   {stats['total_completion_tokens']} tokens written in "
          f"{stats['wall_s']:.1f} s wall time = {stats['batch_tokens_per_s']:.0f} tokens/sec.")
    print("  That is the server's total output rate across everyone asking at once,")
    print("  as opposed to any one request's Tok/sec, which is what one user feels.")
    print()
    print("Two things worth noticing in your own numbers:")
    print("  1. Compare First token and Per token against your concurrency-1 run.")
    print("     When several requests arrive together the server works on them in")
    print("     the same forward passes - that is batching. Each person's wait grows")
    print("     a little; the total number of tokens the machine writes per second grows")
    print("     a lot. Check whether your batch tokens/sec went up.")
    print("  2. Open the timeline chart. At concurrency 1 the bars form a")
    print("     staircase - one request finishes before the next starts. Under load")
    print("     the bars overlap. Overlap is throughput.")
    print()
    print(f"Files written: {out}/lab-day2-c{c}.results.jsonl, lab-day2-c{c}.summary.md,")
    print(f"lab-day2-c{c}.timeline.png")
    print("Next: open docs/lab/day2.md and answer the five questions from your numbers.")
    print()


def cmd_lab2(args: Any) -> int:
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(
            f"Environment variable {args.api_key_env} is not set. A local vLLM "
            f"server accepts any non-empty string, e.g.: export {args.api_key_env}=not-a-real-key"
        )
    rows, stats = run_day2(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        out_dir=args.output_dir,
    )
    print_day2_report(rows, stats, args.output_dir)
    return 1 if stats["requests"] and stats["errors"] == stats["requests"] else 0
