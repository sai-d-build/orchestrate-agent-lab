"""Common LLM generation parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationParameters:
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning: bool | None = None
