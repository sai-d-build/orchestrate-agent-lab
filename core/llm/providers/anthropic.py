"""Anthropic provider placeholder."""

from core.llm.client import LLMProvider


class AnthropicProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
