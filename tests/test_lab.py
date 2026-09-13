from evalharness.lab import DAY1_PROMPTS, _summarize, render_day1_markdown, run_day1
from evalharness.providers.mock import StreamingMockClient
from evalharness.runner import RESULT_FIELDS, _call_with_retries, run
from evalharness.config import load_config


def test_streaming_mock_splits_ttft_and_decode():
    client = StreamingMockClient(name="m", seed=1)
    resp = client.complete("Why is hash map lookup fast?", {})
    assert resp.ttft_s is not None and resp.ttft_s > 0
    # total latency is first-token wait plus per-token decode
    expected_min = resp.ttft_s + 0.010 * (resp.completion_tokens - 1)
    assert resp.latency_s >= expected_min
    assert resp.raw["streams"] is True


def test_streaming_mock_deterministic():
    a = StreamingMockClient(name="m", seed=7).complete("hello", {})
    b = StreamingMockClient(name="m", seed=7).complete("hello", {})
    assert (a.ttft_s, a.latency_s, a.text) == (b.ttft_s, b.latency_s, b.text)


def test_long_prompt_costs_more_first_token_time():
    short = StreamingMockClient(name="m", seed=3).complete("hi", {})
    long_prompt = "word " * 200 + " what is JSONL?"
    long_ = StreamingMockClient(name="m", seed=3).complete(long_prompt, {})
    assert long_.ttft_s > short.ttft_s


def test_day1_run_shape(tmp_path):
    rows, stats = run_day1(tmp_path)
    assert len(rows) == 10
    assert all(r["status"] == "ok" for r in rows)
    assert stats["requests"] == 10 and stats["errors"] == 0
    for r in rows:
        # latency decomposes into first-token wait + decode time
        assert abs(r["latency_s"] - (r["ttft_s"] + r["decode_s"])) < 1e-3
        assert r["tpot_ms"] > 0
        assert r["decode_tokens_per_s"] > 0
    # sequential execution (concurrency 1): start times are non-decreasing
    starts = [r["started_at"] for r in rows]
    assert starts == sorted(starts)
    # the padded long-answer request should be the slowest overall
    slowest = max(rows, key=lambda r: r["latency_s"])
    assert slowest["request"] == 10
    # artifacts exist
    assert (tmp_path / "lab-day1.results.jsonl").exists()
    assert (tmp_path / "lab-day1.summary.md").exists()
    assert (tmp_path / "lab-day1.timeline.png").stat().st_size > 1000


def test_day1_summary_math():
    fake = [
        {"status": "ok", "ttft_s": 0.2, "tpot_ms": 20.0, "decode_tokens_per_s": 50.0,
         "latency_s": 1.0, "prompt_tokens": 10, "completion_tokens": 41},
        {"status": "ok", "ttft_s": 0.4, "tpot_ms": 40.0, "decode_tokens_per_s": 25.0,
         "latency_s": 2.0, "prompt_tokens": 20, "completion_tokens": 51},
        {"status": "error", "ttft_s": None, "tpot_ms": None, "decode_tokens_per_s": None,
         "latency_s": None, "prompt_tokens": None, "completion_tokens": None},
    ]
    s = _summarize(fake)
    assert s["requests"] == 3 and s["errors"] == 1
    assert abs(s["ttft_mean_s"] - 0.3) < 1e-9
    assert abs(s["tpot_mean_ms"] - 30.0) < 1e-9
    assert s["total_completion_tokens"] == 92
    # overall throughput counts every written token over total latency
    assert abs(s["overall_tokens_per_s"] - 92 / 3.0) < 1e-9


def test_day1_markdown_explains_terms_plainly():
    fake_rows = [{"request": 1, "status": "ok", "prompt_tokens": 12, "completion_tokens": 30,
                  "ttft_s": 0.3, "decode_s": 0.6, "tpot_ms": 20.0,
                  "decode_tokens_per_s": 50.0, "latency_s": 0.9}]
    fake_stats = _summarize(fake_rows)
    md = render_day1_markdown(fake_rows, fake_stats)
    assert "simulated endpoint" in md
    assert "| r01 | 12 | 30 |" in md
    assert "First token (s)" in md


def test_runner_records_ttft_when_provider_streams():
    client = StreamingMockClient(name="m", seed=1)
    out = _call_with_retries(client, "p", {}, max_retries=1, backoff_s=0.01, is_mock=True)
    assert out["status"] == "ok"
    assert out["ttft_s"] is not None and out["ttft_s"] > 0
    assert "ttft_s" in RESULT_FIELDS


def test_demo_run_still_leaves_ttft_none_for_plain_mock():
    cfg = load_config("configs/demo.yaml")
    results = run(cfg)
    ok = [r for r in results if r["status"] == "ok"]
    assert ok and all(r["ttft_s"] is None for r in ok)


def test_day1_prompts_are_ten():
    assert len(DAY1_PROMPTS) == 10
