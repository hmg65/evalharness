"""The run engine.

Every (prompt x model x sample) cell goes through the same path: bounded
concurrency, per-call timeout, exponential-backoff retries, then scoring and
a JSONL row. The mock provider answers instantly, so retries and timeouts are
simulated with virtual sleeps - real providers sleep for real. Nothing here
knows which provider it is talking to.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import random
import time
from pathlib import Path
from typing import Any

from .config import ModelConfig, RunConfig
from .providers import MockClient, OpenAICompatClient
from .providers.base import ModelClient
from .scoring import build_scorers, weighted_score

RESULT_FIELDS = [
    "run_name", "config_hash", "prompt_id", "prompt", "reference", "sample",
    "model", "model_id", "params", "status", "error", "attempts",
    "latency_s", "ttft_s", "prompt_tokens", "completion_tokens",
    "quality", "quality_detail", "response_text", "started_at",
]


def build_client(model: ModelConfig) -> ModelClient:
    if model.provider == "mock":
        return MockClient(name=model.name, model_id=model.model_id or "mock-model", **model.mock)
    if model.provider == "openai_compat":
        return OpenAICompatClient(
            name=model.name,
            model_id=model.model_id,
            base_url=model.base_url,
            api_key_env=model.api_key_env,
            timeout_s=model.timeout_s,
        )
    raise ValueError(f"unknown provider '{model.provider}' for model '{model.name}'")


def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    """JSONL with at least {"id", "prompt"}; optional "reference" for scorers."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no prompts in {path}")
    return rows


def _call_with_retries(client: ModelClient, prompt: str, params: dict[str, Any],
                       max_retries: int, backoff_s: float, is_mock: bool) -> dict[str, Any]:
    attempts = 0
    last_error = ""
    while attempts < max_retries:
        attempts += 1
        started = time.time()
        try:
            resp = client.complete(prompt, params)
            return {
                "status": "ok", "error": "", "attempts": attempts,
                "latency_s": resp.latency_s, "ttft_s": resp.ttft_s,
                "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                "response_text": resp.text, "started_at": started, "model_id": resp.model_id,
            }
        except Exception as exc:  # provider errors are data, not crashes
            last_error = f"{type(exc).__name__}: {exc}"
            if attempts < max_retries:
                delay = backoff_s * (2 ** (attempts - 1)) + random.uniform(0, 0.1)
                if not is_mock:  # demo runs stay instant; real runs honor backoff
                    time.sleep(delay)
    return {
        "status": "error", "error": last_error, "attempts": attempts,
        "latency_s": 0.0, "ttft_s": None, "prompt_tokens": None, "completion_tokens": None,
        "response_text": "", "started_at": started, "model_id": client.model_id,
    }


def run(config: RunConfig) -> list[dict[str, Any]]:
    prompts = load_prompts(config.prompts_path)
    scorers = build_scorers(config.scorers)
    clients: dict[str, ModelClient] = {m.name: build_client(m) for m in config.models}
    cfg_hash = config.fingerprint()

    jobs = []
    for prompt_row in prompts:
        for model in config.models:
            for sample in range(config.samples_per_prompt):
                jobs.append((prompt_row, model, sample))

    results: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures = {}
        for prompt_row, model, sample in jobs:
            client = clients[model.name]
            params = dict(model.params)
            timeout_s = min(model.timeout_s, config.request_timeout_s)
            if model.provider == "mock":
                params.setdefault("mock_max_latency_ms", timeout_s * 1000.0)
            fut = pool.submit(
                _call_with_retries, client, prompt_row["prompt"], params,
                config.max_retries, config.backoff_s, model.provider == "mock",
            )
            futures[fut] = (prompt_row, model, sample)
        for fut in cf.as_completed(futures):
            prompt_row, model, sample = futures[fut]
            row = fut.result()
            quality, detail = (0.0, {})
            if row["status"] == "ok" and scorers:
                quality, detail = weighted_score(
                    scorers, prompt_row["prompt"], prompt_row.get("reference"), row["response_text"]
                )
            results.append({
                "run_name": config.run_name, "config_hash": cfg_hash,
                "prompt_id": prompt_row["id"], "prompt": prompt_row["prompt"],
                "reference": prompt_row.get("reference"), "sample": sample,
                "model": model.name, "model_id": row["model_id"], "params": model.params,
                "status": row["status"], "error": row["error"], "attempts": row["attempts"],
                "latency_s": round(row["latency_s"], 4),
                "ttft_s": round(row["ttft_s"], 4) if row.get("ttft_s") is not None else None,
                "prompt_tokens": row["prompt_tokens"], "completion_tokens": row["completion_tokens"],
                "quality": round(quality, 4), "quality_detail": detail,
                "response_text": row["response_text"], "started_at": row["started_at"],
            })
    results.sort(key=lambda r: (r["prompt_id"], r["model"], r["sample"]))
    return results


def write_jsonl(results: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in results:
            f.write(json.dumps(row, default=str) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
