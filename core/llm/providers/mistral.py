"""Mistral provider placeholder."""

from core.llm.client import LLMProvider


class MistralProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
