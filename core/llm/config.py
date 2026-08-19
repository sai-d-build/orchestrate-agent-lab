"""LLM configuration loading.

Configuration comes from:
    - config/models.yaml
    - environment variables for secrets and runtime selection

Secrets are never stored in YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_CONFIG = ROOT_DIR / "config" / "models.yaml"


@dataclass(frozen=True)
class ModelConfig:
    """Resolved configuration for one model profile."""

    profile: str
    provider: str
    model: str
    api_key_env: str
    purpose: str
    capabilities: tuple[str, ...]


def load_models_config() -> dict:
    """Load the model registry from YAML."""
    with MODELS_CONFIG.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_model_config(profile: str | None = None) -> ModelConfig:
    """Resolve a model profile into a typed configuration."""
    profile = profile or os.getenv("MODEL_PROFILE", "gemini-fast")

    config = load_models_config()
    models = config.get("models", {})

    if profile not in models:
        available = ", ".join(sorted(models))
        raise ValueError(
            f"Unknown model profile '{profile}'. "
            f"Available profiles: {available}"
        )

    model = models[profile]

    return ModelConfig(
        profile=profile,
        provider=model["provider"],
        model=model["model"],
        api_key_env=model["api_key_env"],
        purpose=model["purpose"],
        capabilities=tuple(model.get("capabilities", [])),
    )


def get_api_key(model_config: ModelConfig) -> str:
    """Load the API key from the environment."""
    api_key = os.getenv(model_config.api_key_env)

    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {model_config.api_key_env} "
            "in the environment."
        )

    return api_key
