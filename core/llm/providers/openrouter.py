"""OpenRouter LLM provider."""

from __future__ import annotations

import requests

from core.llm.client import LLMProvider
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
    ) -> str:

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

        data = response.json()

        return data["choices"][0]["message"]["content"]
