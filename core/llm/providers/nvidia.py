"""NVIDIA Nemotron LLM provider.

Why it exists:
    NVIDIA provides Nemotron models via their API endpoint at
    https://integrate.api.nvidia.com/v1, which is OpenAI-compatible.
    This provider enables using Nemotron models directly without OpenRouter.

What problem it solves:
    Abstracts the NVIDIA API into the common LLMProvider interface,
    returning structured responses with token usage metadata.

Alternatives considered:
    - Using OpenRouter to access Nemotron: adds latency, rate limits.
    - Direct HTTP calls: more control but requires manual error handling.

Tradeoffs:
    - Direct NVIDIA API gives better performance and no OpenRouter rate limits.
    - Requires separate NVIDIA API key.
    - Tied to NVIDIA's response format (OpenAI-compatible).

Failure modes:
    - API rate limits → HTTP 429, should be retried.
    - Invalid API key → HTTP 401.
    - Model not available → HTTP 400/404.
    - Network errors → requests.exceptions.ConnectionError.

Testing:
    - Unit tests with mocked HTTP responses.
    - Integration tests require a valid NVIDIA API key.

Production implications:
    - In production, add retry logic with exponential backoff.
    - Consider circuit breakers for repeated failures.
    - Log token usage for cost tracking.
"""

from __future__ import annotations

import time

import requests

from core.llm.client import LLMProvider, LLMResponse
from core.llm.parameters import GenerationParameters


class NVIDIAProvider(LLMProvider):
    """NVIDIA Nemotron provider using OpenAI-compatible API."""

    BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def generate(
        self,
        prompt: str,
        parameters: GenerationParameters | None = None,
    ) -> LLMResponse:

        parameters = parameters or GenerationParameters()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        if parameters.temperature is not None:
            payload["temperature"] = parameters.temperature

        if parameters.max_output_tokens is not None:
            payload["max_tokens"] = parameters.max_output_tokens

        if parameters.reasoning is not None:
            payload["reasoning"] = {
                "enabled": parameters.reasoning
            }

        started = time.perf_counter()

        response = requests.post(
            self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        elapsed = time.perf_counter() - started

        data = response.json()

        content = data["choices"][0]["message"]["content"]

        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_seconds=round(elapsed, 3),
            model=data.get("model", self.model),
            raw_response=data,
        )