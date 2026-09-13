"""Day 2 lab tests: stream_one timing/usage parsing and run_day2 batch math,
against a local canned SSE server (no model, no cost)."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evalharness.lab2 import RESULT_FIELDS, run_day2, stream_one

CHUNKS = ["The", " answer", " is", " 42", "."]


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        assert body["stream"] is True
        if "explode" in body["messages"][0]["content"]:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"simulated server error")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.05)  # prefill stand-in: the wait for the first token
        for tok in CHUNKS:
            self.wfile.write(_sse({"choices": [{"delta": {"content": tok}}]}))
            self.wfile.flush()
            time.sleep(0.02)  # per-token decode gap
        self.wfile.write(_sse({
            "choices": [{"delta": {}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": len(CHUNKS)},
        }))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def test_stream_one_measures_ttft_tpot_and_usage(server):
    m = stream_one("what is 6 times 7?", base_url=server, model="fake",
                   api_key="x", max_tokens=64)
    assert m["prompt_tokens"] == 12
    assert m["completion_tokens"] == len(CHUNKS)
    # TTFT is the prefill wait, not the whole call
    assert 0.03 < m["ttft_s"] < m["latency_s"]
    # five chunks at 20 ms gaps after the first: TPOT lands near 20 ms
    assert 10 < m["tpot_ms"] < 60
    # Day 1's equation still holds on a real stream
    est = m["ttft_s"] + (m["completion_tokens"] - 1) * m["tpot_ms"] / 1000.0
    assert abs(est - m["latency_s"]) < 0.03


def test_stream_one_falls_back_to_chunk_count_without_usage(server, monkeypatch):
    # a server whose final chunk carries no usage block
    import evalharness.lab2 as lab2

    orig = lab2.json.loads

    def strip_usage(payload):
        event = orig(payload)
        if isinstance(event, dict):
            event.pop("usage", None)
        return event

    monkeypatch.setattr(lab2.json, "loads", strip_usage)
    m = stream_one("hi", base_url=server, model="fake", api_key="x", max_tokens=64)
    assert m["prompt_tokens"] is None
    assert m["completion_tokens"] == len(CHUNKS)


def test_run_day2_ten_rows_batch_throughput_and_files(server, tmp_path):
    rows, stats = run_day2(base_url=server, model="fake", api_key="x",
                           concurrency=4, max_tokens=64, out_dir=tmp_path)
    assert len(rows) == 10
    assert all(list(r.keys()) == RESULT_FIELDS for r in rows)
    assert stats["requests"] == 10 and stats["errors"] == 0
    assert stats["concurrency"] == 4
    assert stats["total_completion_tokens"] == 10 * len(CHUNKS)
    # the batch number Day 2 exists for: server-wide tokens/sec
    assert stats["batch_tokens_per_s"] > 0
    # 10 requests at 4-way parallelism beat strictly serial wall time
    serial_floor = sum(r["latency_s"] for r in rows)
    assert stats["wall_s"] < serial_floor
    for suffix in ("results.jsonl", "summary.md", "timeline.png"):
        assert (tmp_path / f"lab-day2-c4.{suffix}").exists()
    md = (tmp_path / "lab-day2-c4.summary.md").read_text()
    assert "tokens/sec overall" in md


def test_run_day2_records_provider_errors_as_data(server, tmp_path, monkeypatch):
    import evalharness.lab2 as lab2

    boom = {"id": "r05", "prompt": "please explode now"}
    monkeypatch.setattr(lab2, "DAY1_PROMPTS", lab2.DAY1_PROMPTS[:4] + [boom] + lab2.DAY1_PROMPTS[5:])
    rows, stats = run_day2(base_url=server, model="fake", api_key="x",
                           concurrency=2, max_tokens=64, out_dir=tmp_path)
    assert stats["errors"] == 1
    bad = next(r for r in rows if r["status"] == "error")
    assert "500" in bad["error"]
