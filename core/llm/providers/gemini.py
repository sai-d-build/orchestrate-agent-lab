"""Google Gemini provider.

Implemented in Phase 0, Lesson 1.
"""

from core.llm.client import LLMProvider, LLMResponse
from core.llm.parameters import GenerationParameters


class GeminiProvider(LLMProvider):
    """Gemini implementation."""

    def generate(
        self,
        prompt: str,
        parameters: GenerationParameters | None = None,
    ) -> LLMResponse:
        raise NotImplementedError(
            "Gemini provider will be implemented in Lesson 1."
        )
