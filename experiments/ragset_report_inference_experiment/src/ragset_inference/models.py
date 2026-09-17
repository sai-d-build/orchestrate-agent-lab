import json
import os
import re
from openai import OpenAI
from pydantic import ValidationError
from .schemas import ReportPrediction, ValidationResult


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from text, handling markdown code fences."""
    text = text.strip()
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON from markdown code fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _create_client(provider: str, api_key: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client for the given provider."""
    if provider == "nvidia":
        return OpenAI(
            api_key=api_key or os.environ.get("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
        )
    elif provider == "openrouter":
        return OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: nvidia, openrouter")


def _detect_provider(model: str) -> str:
    """Detect provider from model name."""
    if model.startswith("nvidia/") or "nemotron" in model.lower():
        return "nvidia"
    return "openrouter"


def _build_extra_body(provider: str, reasoning: bool | None) -> dict | None:
    """Build extra_body for provider-specific parameters."""
    if reasoning is None:
        return None
    if provider == "nvidia":
        # NVIDIA Nemotron supports reasoning parameter
        return {"reasoning": {"enabled": reasoning}}
    elif provider == "openrouter":
        # OpenRouter passes reasoning through to underlying provider
        return {"reasoning": {"enabled": reasoning}}
    return None


class InferenceModel:
    """Inference model using chat.completions API with JSON mode.

    Supports both OpenRouter and NVIDIA Nemotron endpoints.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 4000,
        provider: str | None = None,
        api_key: str | None = None,
        reasoning: bool | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.provider = provider or _detect_provider(model)
        self.reasoning = reasoning
        self.client = _create_client(self.provider, api_key)

    def run(self, system: str, user: str) -> ReportPrediction:
        # Use chat.completions with JSON mode for better free model support
        extra_body = _build_extra_body(self.provider, self.reasoning)
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content
        data = self._extract_json(text)
        if data is None:
            raise RuntimeError(f"Could not extract JSON from inference response: {text[:200]}")
        try:
            return ReportPrediction.model_validate(data)
        except Exception as e:
            raise RuntimeError(f"Could not parse ReportPrediction from JSON: {e}\nRaw: {text[:500]}")

    def _extract_json(self, text: str) -> dict | None:
        """Extract a JSON object from text, handling markdown code fences."""
        text = text.strip()
        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting JSON from markdown code fences
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try finding first JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None


class ValidatorModel:
    """Validator model using chat.completions API with JSON mode.

    Supports both OpenRouter and NVIDIA Nemotron endpoints.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 4000,
        provider: str | None = None,
        api_key: str | None = None,
        reasoning: bool | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.provider = provider or _detect_provider(model)
        self.reasoning = reasoning
        self.client = _create_client(self.provider, api_key)

    def run(self, system: str, user: str) -> ValidationResult:
        extra_body = _build_extra_body(self.provider, self.reasoning)
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content
        data = self._extract_json(text)
        if data is None:
            raise RuntimeError(f"Could not extract JSON from validation response: {text[:200]}")
        try:
            return ValidationResult.model_validate(data)
        except Exception as e:
            raise RuntimeError(f"Could not parse ValidationResult from JSON: {e}\nRaw: {text[:500]}")

    def _extract_json(self, text: str) -> dict | None:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None