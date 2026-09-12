"""Run configuration: YAML in, validated dataclasses out.

Secrets are referenced by environment-variable NAME (api_key_env), never by
value. Any string of the form ${VAR} anywhere in the config is expanded from
the environment, so base URLs and headers can also be parameterized.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            var = match.group(1)
            if var not in os.environ:
                raise RuntimeError(f"Config references ${{{var}}} but it is not set in the environment")
            return os.environ[var]
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass
class ModelConfig:
    name: str
    provider: str = "openai_compat"  # openai_compat | mock
    model_id: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 60.0
    params: dict[str, Any] = field(default_factory=dict)  # temperature, max_tokens, seed, ...
    mock: dict[str, Any] = field(default_factory=dict)    # mock provider personality


@dataclass
class RunConfig:
    run_name: str
    models: list[ModelConfig]
    prompts_path: str
    output_dir: str
    samples_per_prompt: int = 1
    concurrency: int = 4
    max_retries: int = 3
    backoff_s: float = 1.0
    request_timeout_s: float = 60.0
    scorers: list[dict[str, Any]] = field(default_factory=list)

    def fingerprint(self) -> str:
        """Stable hash of everything that defines the comparison, so a results
        file can prove exactly which setup produced it."""
        payload = json.dumps(
            {
                "models": [m.__dict__ for m in self.models],
                "prompts_path": self.prompts_path,
                "samples_per_prompt": self.samples_per_prompt,
                "scorers": self.scorers,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> RunConfig:
    raw = _expand_env(yaml.safe_load(Path(path).read_text()))
    models = [ModelConfig(**m) for m in raw["models"]]
    if not models:
        raise ValueError("config has no models")
    names = [m.name for m in models]
    if len(names) != len(set(names)):
        raise ValueError(f"model names must be unique, got {names}")
    return RunConfig(
        run_name=raw.get("run_name", Path(path).stem),
        models=models,
        prompts_path=raw["prompts_path"],
        output_dir=raw.get("output_dir", "results"),
        samples_per_prompt=int(raw.get("samples_per_prompt", 1)),
        concurrency=int(raw.get("concurrency", 4)),
        max_retries=int(raw.get("max_retries", 3)),
        backoff_s=float(raw.get("backoff_s", 1.0)),
        request_timeout_s=float(raw.get("request_timeout_s", 60.0)),
        scorers=raw.get("scorers", []),
    )
