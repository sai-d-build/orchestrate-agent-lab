"""OpenAI provider placeholder."""

from core.llm.client import LLMProvider, LLMResponse
from core.llm.parameters import GenerationParameters


class OpenAIProvider(LLMProvider):
    def generate(
        self,
        prompt: str,
        parameters: GenerationParameters | None = None,
    ) -> LLMResponse:
        raise NotImplementedError
