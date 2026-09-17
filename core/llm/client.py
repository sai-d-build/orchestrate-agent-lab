"""Provider-independent LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core.llm.parameters import GenerationParameters


@dataclass(frozen=True)
class LLMResponse:
    """Structured response from an LLM provider.

    Why it exists:
        The original interface returned only a string, losing valuable
        metadata like token usage and latency. This dataclass preserves
        that information for evaluation and cost tracking.

    What problem it solves:
        Metrics like token_usage, estimated_cost, and latency require
        access to provider response metadata, not just the text content.

    Tradeoffs:
        - Slightly more complex return type than a plain string.
        - All providers must populate usage data (or leave it empty).

    Failure modes:
        - Provider doesn't return usage data → fields default to None/0.
    """

    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float | None = None
    model: str | None = None
    raw_response: dict | None = None


class LLMProvider(ABC):
    """Common interface implemented by LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        parameters: GenerationParameters | None = None,
    ) -> LLMResponse:
        """Generate a response from a prompt."""
        raise NotImplementedError
