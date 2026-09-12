from evalharness.metrics import percentile, summarize


def test_percentile_small_samples():
    assert percentile([], 95) == 0.0
    assert percentile([1.0], 95) == 1.0
    vals = sorted([0.1, 0.2, 0.3, 0.4, 0.5])
    assert abs(percentile(vals, 50) - 0.3) < 1e-9
    assert percentile(vals, 100) == 0.5
    assert percentile(vals, 0) == 0.1


def test_summarize_counts_errors_and_retries():
    records = [
        {"status": "ok", "latency_s": 0.2, "quality": 1.0, "attempts": 1,
         "prompt_tokens": 10, "completion_tokens": 5},
        {"status": "ok", "latency_s": 0.4, "quality": 0.5, "attempts": 2,
         "prompt_tokens": 10, "completion_tokens": 6},
        {"status": "error", "latency_s": 0.0, "attempts": 3,
         "prompt_tokens": None, "completion_tokens": None},
    ]
    stats = summarize("m", records)
    assert stats.calls == 3
    assert stats.errors == 1
    assert abs(stats.error_rate - 1 / 3) < 1e-9
    assert abs(stats.latency_median_s - 0.3) < 1e-9
    assert stats.retries == 3  # (1-1)+(2-1)+(3-1)
    assert stats.total_prompt_tokens == 20
    assert abs(stats.quality_mean - 0.75) < 1e-9
