import os

import pytest

from evalharness.config import load_config


def test_demo_config_loads():
    cfg = load_config("configs/demo.yaml")
    assert cfg.run_name == "demo"
    assert len(cfg.models) == 3
    assert cfg.models[0].provider == "mock"
    assert cfg.samples_per_prompt == 3
    assert len(cfg.fingerprint()) == 16


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_BASE", "https://example.test/v1")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "run_name: t\nprompts_path: data/prompts/synthetic_qa.jsonl\n"
        "models:\n  - name: m\n    provider: mock\n    base_url: ${MY_BASE}\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.models[0].base_url == "https://example.test/v1"


def test_missing_env_var_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_VAR", raising=False)
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "prompts_path: x\nmodels:\n  - name: m\n    provider: mock\n    base_url: ${NOPE_VAR}\n"
    )
    with pytest.raises(RuntimeError, match="NOPE_VAR"):
        load_config(cfg_file)


def test_duplicate_model_names_rejected(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "prompts_path: x\nmodels:\n  - name: m\n    provider: mock\n  - name: m\n    provider: mock\n"
    )
    with pytest.raises(ValueError, match="unique"):
        load_config(cfg_file)
