"""OpenRouter LLM provider.

Lesson 1 implementation using the OpenRouter
OpenAI-compatible HTTP API directly with requests.
"""

from __future__ import annotations

import requests

from core.llm.client import LLMProvider


class OpenRouterProvider(LLMProvider):
    """Call models through OpenRouter."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "reasoning": {
                    "enabled": True,
                },
            },
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]
