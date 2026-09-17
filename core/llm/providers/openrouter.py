"""OpenRouter LLM provider.

Why it exists:
    OpenRouter provides access to multiple LLM providers through a single
    API, including free-tier models useful for benchmarking in Lesson 1.2.

What problem it solves:
    Abstracts the OpenRouter HTTP API into the common LLMProvider interface,
    returning structured responses with token usage metadata.

Alternatives considered:
    - Using the official openrouter Python SDK: adds a dependency, less control.
    - Using provider-specific SDKs: would require separate code paths per provider.

Tradeoffs:
    - Direct HTTP calls give full control but require manual error handling.
    - The provider is tightly coupled to OpenRouter's response format.

Failure modes:
    - API rate limits → HTTP 429, should be retried.
    - Invalid API key → HTTP 401.
    - Model not available → HTTP 400/404.
    - Network errors → requests.exceptions.ConnectionError.

Testing:
    - Unit tests with mocked HTTP responses.
    - Integration tests require a valid API key.

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


class OpenRouterProvider(LLMProvider):
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

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
            "provider": {
                "data_collection": "deny",
                "allow_fallbacks": True,
            },
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
            timeout=60,
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
