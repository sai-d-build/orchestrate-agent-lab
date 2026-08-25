"""Provider-independent LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.llm.parameters import GenerationParameters


class LLMProvider(ABC):
    """Common interface implemented by LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        parameters: GenerationParameters | None = None,
    ) -> str:
        """Generate a response from a prompt."""
        raise NotImplementedError
