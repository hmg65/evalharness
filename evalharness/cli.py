"""Command line: run an evaluation, rebuild a report, run the synthetic demo, or run the guided labs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .charts import write_charts
from .config import load_config
from .lab import cmd_lab
from .lab2 import cmd_lab2
from .report import summarize_results, write_summary
from .runner import read_jsonl, run, write_jsonl


def _emit(results_path, stats, out_dir: Path, run_name: str) -> None:
    json_path, csv_path, md_path = write_summary(stats, out_dir, run_name)
    charts = write_charts(read_jsonl(results_path), stats, out_dir, run_name)
    print(f"results:  {results_path}")
    print(f"summary:  {json_path}\n          {csv_path}\n          {md_path}")
    for c in charts:
        print(f"chart:    {c}")
    print()
    print(md_path.read_text())


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run(config)
    results_path = write_jsonl(results, out_dir / f"{config.run_name}.results.jsonl")
    _emit(results_path, summarize_results(results), out_dir, config.run_name)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results = read_jsonl(args.results)
    run_name = args.name or Path(args.results).name.replace(".results.jsonl", "")
    _emit(Path(args.results), summarize_results(results), Path(args.output_dir), run_name)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    demo_cfg = Path(__file__).resolve().parent.parent / "configs" / "demo.yaml"
    args.config = str(demo_cfg)
    return cmd_run(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalharness",
        description="Reproducible evaluation harness for hosted LLM APIs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an evaluation from a YAML config")
    p_run.add_argument("--config", required=True, help="path to the run config YAML")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="rebuild summary + charts from an existing results JSONL")
    p_rep.add_argument("--results", required=True, help="path to a .results.jsonl file")
    p_rep.add_argument("--output-dir", default="results")
    p_rep.add_argument("--name", default=None, help="run name for output files")
    p_rep.set_defaults(func=cmd_report)

    p_demo = sub.add_parser("demo", help="run the bundled synthetic demo (no keys, no cost)")
    p_demo.set_defaults(func=cmd_demo)

    p_lab = sub.add_parser("lab", help="guided Day 1 inference lab: 10 requests, plain-language output")
    p_lab.add_argument("--output-dir", default="results")
    p_lab.set_defaults(func=lambda args: cmd_lab(args.output_dir))

    p_lab2 = sub.add_parser("lab2", help="Day 2 lab: the same 10 requests against a real local model server")
    p_lab2.add_argument("--base-url", default="http://localhost:11434/v1",
                        help="OpenAI-compatible base URL of the local server (default: Ollama)")
    p_lab2.add_argument("--model", required=True,
                        help="model id exactly as the server knows it, e.g. qwen3:4b")
    p_lab2.add_argument("--api-key-env", default="LAB2_API_KEY",
                        help="env var holding the API key; local servers accept any non-empty string")
    p_lab2.add_argument("--concurrency", type=int, default=1,
                        help="how many of the 10 requests run at once")
    p_lab2.add_argument("--max-tokens", type=int, default=256)
    p_lab2.add_argument("--output-dir", default="results")
    p_lab2.set_defaults(func=cmd_lab2)

    args = parser.parse_args(argv)
    return args.func(args)
