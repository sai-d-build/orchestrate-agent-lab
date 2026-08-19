"""OpenAI provider placeholder."""

from core.llm.client import LLMProvider


class OpenAIProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
