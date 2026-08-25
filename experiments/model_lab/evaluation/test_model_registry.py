"""Smoke tests for the Lesson 1.2 model registry."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "models.yaml"


def test_selected_profiles_exist():
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = data["models"]

    assert models["openrouter-free"]["model"] == (
        "nvidia/nemotron-3.5-lightning:free"
    )

    assert models["gemma-4-free"]["model"] == (
        "google/gemma-4-26b-a4b-it:free"
    )

    assert models["gpt-oss-20b-free"]["model"] == (
        "openai/gpt-oss-20b:free"
    )
