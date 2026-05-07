from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_compile_sweep_module():
    path = Path(__file__).resolve().parents[1] / "inference" / "scripts" / "compile_sweep.py"
    spec = importlib.util.spec_from_file_location("compile_sweep", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expand_env_defaults_uses_default_when_unset(monkeypatch):
    compile_sweep = _load_compile_sweep_module()

    monkeypatch.delenv("APB_TEST_MODEL_ROOT", raising=False)

    assert compile_sweep.expand_env_defaults("${APB_TEST_MODEL_ROOT:-/models}") == "/models"


def test_render_row_expands_model_root_env_override(monkeypatch):
    compile_sweep = _load_compile_sweep_module()
    manifest = {
        "hosts": {
            "test-host": {"model_root": "${APB_TEST_MODEL_ROOT:-/models}"},
        },
        "models": {
            "Tiny": {"dir": "Tiny-Instruct", "weights_gb": 1},
        },
        "presets": {
            "single": {
                "max_len": 4096,
                "gpu_mem": 0.85,
                "concurrencies": [1],
                "profiles": ["chat-singleturn"],
            },
        },
        "feasibility_ratio": 0.9,
        "cells": [],
    }
    cell = {
        "host": "test-host",
        "model": "Tiny",
        "tp": 1,
        "mode": "single",
        "preset": "single",
    }

    monkeypatch.setenv("APB_TEST_MODEL_ROOT", "/tmp/model-cache")

    row = compile_sweep.render_row(cell, manifest)

    assert row.split("|")[1] == "/tmp/model-cache/Tiny-Instruct"
