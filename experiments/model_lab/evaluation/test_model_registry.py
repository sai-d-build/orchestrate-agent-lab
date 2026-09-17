"""Tests for the Lesson 1.2 model registry and configuration.

Why it exists:
    Ensures the model registry is valid and all profiles have the
    required fields for the experiment to run correctly.

What problem it solves:
    Catches configuration errors (missing fields, typos) before they
    cause runtime failures during experiments.

Alternatives considered:
    - Manual verification: error-prone, not reproducible.
    - Schema validation in YAML: not standard, harder to maintain.

Tradeoffs:
    - Tests are coupled to the YAML structure.
    - Adding a new profile requires updating tests if fields change.

Failure modes:
    - Missing required fields in a profile → test fails.
    - Invalid provider name → test fails.
    - Missing api_key_env → test fails.

Testing:
    - Run with: pytest experiments/model_lab/evaluation/test_model_registry.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from core.config import get_settings
from core.llm.config import get_api_key, get_model_config, load_models_config


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "models.yaml"
SETTINGS = ROOT / "config" / "settings.yaml"

REQUIRED_FIELDS = {"provider", "model", "api_key_env", "purpose", "capabilities"}
VALID_PROVIDERS = {"openrouter", "openai", "anthropic", "gemini", "groq", "mistral"}


def test_models_yaml_exists():
    """Verify the models config file exists."""
    assert CONFIG.exists(), f"Config file not found: {CONFIG}"


def test_models_yaml_is_valid_yaml():
    """Verify the models config is valid YAML."""
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert "models" in data
    assert isinstance(data["models"], dict)
    assert len(data["models"]) > 0


def test_all_profiles_have_required_fields():
    """Verify every model profile has all required fields."""
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = data["models"]

    for profile_name, profile in models.items():
        missing = REQUIRED_FIELDS - set(profile.keys())
        assert not missing, (
            f"Profile '{profile_name}' is missing required fields: {missing}"
        )


def test_all_profiles_have_valid_providers():
    """Verify every model profile has a valid provider."""
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = data["models"]

    for profile_name, profile in models.items():
        assert profile["provider"] in VALID_PROVIDERS, (
            f"Profile '{profile_name}' has invalid provider: "
            f"{profile['provider']}"
        )


def test_openrouter_profiles_have_api_key_env():
    """Verify OpenRouter profiles reference OPENROUTER_API_KEY."""
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = data["models"]

    for profile_name, profile in models.items():
        if profile["provider"] == "openrouter":
            assert profile["api_key_env"] == "OPENROUTER_API_KEY", (
                f"OpenRouter profile '{profile_name}' should use "
                f"OPENROUTER_API_KEY, got: {profile['api_key_env']}"
            )


def test_free_models_are_configured():
    """Verify the three free-tier models for Lesson 1.2 are configured."""
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


def test_free_models_have_enterprise_decision_purpose():
    """Verify free-tier models are tagged for enterprise_decision."""
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = data["models"]

    for profile in ("openrouter-free", "gemma-4-free", "gpt-oss-20b-free"):
        assert models[profile]["purpose"] == "enterprise_decision", (
            f"Profile '{profile}' should have purpose 'enterprise_decision'"
        )


def test_free_models_have_no_data_collection():
    """Verify free-tier models have data_collection: deny."""
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = data["models"]

    for profile in ("openrouter-free", "gemma-4-free", "gpt-oss-20b-free"):
        policy = models[profile].get("provider_policy", {})
        assert policy.get("data_collection") == "deny", (
            f"Profile '{profile}' should have data_collection: deny"
        )


def test_get_model_config_returns_typed_config():
    """Verify get_model_config returns a properly typed ModelConfig."""
    config = get_model_config("openrouter-free")

    assert config.profile == "openrouter-free"
    assert config.provider == "openrouter"
    assert config.model == "nvidia/nemotron-3.5-lightning:free"
    assert config.api_key_env == "OPENROUTER_API_KEY"
    assert config.purpose == "enterprise_decision"
    assert "text" in config.capabilities
    assert "reasoning" in config.capabilities


def test_get_model_config_unknown_profile_raises():
    """Verify get_model_config raises for unknown profiles."""
    with pytest.raises(ValueError, match="Unknown model profile"):
        get_model_config("nonexistent-profile")


def test_get_model_config_default_profile():
    """Verify get_model_config uses MODEL_PROFILE env var or default."""
    with patch.dict(os.environ, {"MODEL_PROFILE": "gemma-4-free"}):
        config = get_model_config()
        assert config.profile == "gemma-4-free"


def test_get_api_key_missing_raises():
    """Verify get_api_key raises when env var is not set."""
    config = get_model_config("openrouter-free")

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENROUTER_API_KEY", None)
        with pytest.raises(RuntimeError, match="Missing API key"):
            get_api_key(config)


def test_get_api_key_returns_env_value():
    """Verify get_api_key returns the env var value."""
    config = get_model_config("openrouter-free")

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-123"}):
        assert get_api_key(config) == "test-key-123"


def test_settings_yaml_exists():
    """Verify the settings config file exists."""
    assert SETTINGS.exists(), f"Settings file not found: {SETTINGS}"


def test_get_settings_returns_typed_config():
    """Verify get_settings returns a properly typed AppSettings."""
    settings = get_settings()

    assert settings.name == "orchestrate-agent-lab"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.llm.timeout_seconds == 60
    assert settings.llm.max_retries == 2
    assert settings.evaluation.enabled is True
    assert settings.observability.enabled is True
