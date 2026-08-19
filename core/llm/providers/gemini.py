"""Google Gemini provider.

Implemented in Phase 0, Lesson 1.
"""

from core.llm.client import LLMProvider


class GeminiProvider(LLMProvider):
    """Gemini implementation."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            "Gemini provider will be implemented in Lesson 1."
        )
