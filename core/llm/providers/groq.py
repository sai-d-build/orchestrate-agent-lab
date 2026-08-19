"""Groq provider placeholder."""

from core.llm.client import LLMProvider


class GroqProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
