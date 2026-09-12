from evalharness.config import load_config
from evalharness.runner import _call_with_retries, build_client, load_prompts, run


def test_load_prompts():
    prompts = load_prompts("data/prompts/synthetic_qa.jsonl")
    assert len(prompts) == 8
    assert prompts[0]["id"] == "q001"
    assert "reference" in prompts[0]


def test_mock_client_deterministic():
    cfg = load_config("configs/demo.yaml")
    a = build_client(cfg.models[0]).complete("hello", {})
    b = build_client(cfg.models[0]).complete("hello", {})
    # same seed -> same jitter and quality roll -> identical response
    assert a.text == b.text
    assert a.latency_s == b.latency_s
    assert a.latency_s > 0


def test_retries_recover_from_failures():
    cfg = load_config("configs/demo.yaml")
    model = cfg.models[2]  # mock-slow-2b, fail_rate 0.35
    model.mock["fail_rate"] = 1.0
    model.mock["seed"] = 0
    client = build_client(model)
    out = _call_with_retries(client, "p", {}, max_retries=3, backoff_s=0.01, is_mock=True)
    assert out["status"] == "error"
    assert out["attempts"] == 3
    assert "mock 500" in out["error"]


def test_full_demo_run_shape():
    cfg = load_config("configs/demo.yaml")
    results = run(cfg)
    # 8 prompts x 3 models x 3 samples
    assert len(results) == 72
    for row in results:
        assert row["status"] in {"ok", "error"}
        assert row["config_hash"] == cfg.fingerprint()
        if row["status"] == "ok":
            assert row["latency_s"] > 0
            assert 0.0 <= row["quality"] <= 1.0
    # the slow/flaky model must show the failure path at least sometimes
    slow = [r for r in results if r["model"] == "mock-slow-2b"]
    assert any(r["attempts"] > 1 or r["status"] == "error" for r in slow)
