"""Application-level configuration loading.

Configuration comes from:
    - config/settings.yaml
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

ROOT_DIR = Path(__file__).resolve().parents[1]
SETTINGS_CONFIG = ROOT_DIR / "config" / "settings.yaml"


@dataclass(frozen=True)
class LLMSettings:
    """LLM runtime settings."""

    timeout_seconds: int = 60
    max_retries: int = 2


@dataclass(frozen=True)
class EvaluationSettings:
    """Evaluation settings."""

    enabled: bool = True


@dataclass(frozen=True)
class ObservabilitySettings:
    """Observability settings."""

    enabled: bool = True
    log_prompts: bool = False
    log_responses: bool = False


@dataclass(frozen=True)
class AppSettings:
    """Resolved application settings."""

    name: str = "orchestrate-agent-lab"
    environment: str = "development"
    log_level: str = "INFO"
    llm: LLMSettings = LLMSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    observability: ObservabilitySettings = ObservabilitySettings()


def load_settings() -> dict:
    """Load the application settings from YAML."""
    with SETTINGS_CONFIG.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_settings() -> AppSettings:
    """Resolve application settings into a typed configuration."""
    config = load_settings()

    app = config.get("application", {})
    llm = config.get("llm", {})
    evaluation = config.get("evaluation", {})
    observability = config.get("observability", {})

    return AppSettings(
        name=app.get("name", "orchestrate-agent-lab"),
        environment=app.get("environment", "development"),
        log_level=app.get("log_level", "INFO"),
        llm=LLMSettings(
            timeout_seconds=llm.get("timeout_seconds", 60),
            max_retries=llm.get("max_retries", 2),
        ),
        evaluation=EvaluationSettings(
            enabled=evaluation.get("enabled", True),
        ),
        observability=ObservabilitySettings(
            enabled=observability.get("enabled", True),
            log_prompts=observability.get("log_prompts", False),
            log_responses=observability.get("log_responses", False),
        ),
    )
