"""Provider-independent LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface that all LLM providers will implement."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from a text prompt."""
        raise NotImplementedError
