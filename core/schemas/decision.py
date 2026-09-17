"""Pydantic schemas for enterprise decision outputs.

These schemas define the structured output format for the three enterprise
decision types used in Lesson 1.2: support escalation, invoice exception,
and change risk.

Why it exists:
    LLM responses are unstructured text. These schemas enforce a consistent
    output format so that responses can be parsed, validated, and scored
    programmatically.

What problem it solves:
    Without schemas, evaluating model outputs requires fragile regex parsing.
    Pydantic models provide type-safe validation and clear error messages.

Alternatives considered:
    - Raw regex parsing: fragile, no type safety, hard to maintain.
    - JSON schema validation: more verbose, less Pythonic.
    - No validation: responses could be malformed and silently produce
      incorrect evaluation results.

Tradeoffs:
    - Requires the LLM to follow a strict output format (mitigated by
      clear prompting and structured output support in some models).
    - Adds a dependency on pydantic (already a project dependency).

Failure modes:
    - LLM returns malformed output → PydanticValidationError raised.
    - LLM returns unexpected enum values → validation fails.
    - Missing fields → validation fails.

Testing:
    - Unit tests for each schema with valid and invalid inputs.
    - Integration tests with real LLM responses.

Production implications:
    - In production, validation failures should trigger fallback logic
      (e.g., re-prompt, manual review, or default decision).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    """Support ticket priority levels."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class Escalation(str, Enum):
    """Whether a ticket should be escalated."""

    YES = "YES"
    NO = "NO"


class Confidence(str, Enum):
    """Confidence level in the decision."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class InvoiceDecision(str, Enum):
    """Invoice exception decision outcomes."""

    APPROVE = "approve"
    REJECT = "reject"
    MANUAL_REVIEW = "manual_review"


class ChangeRisk(str, Enum):
    """IT change risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MANUAL_REVIEW = "manual_review"


class SupportEscalationDecision(BaseModel):
    """Structured output for support escalation decisions."""

    priority: Priority
    escalation: Escalation
    evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: Confidence
    reasoning_summary: str


class InvoiceExceptionDecision(BaseModel):
    """Structured output for invoice exception decisions."""

    decision: InvoiceDecision
    evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: Confidence
    reasoning_summary: str


class ChangeRiskDecision(BaseModel):
    """Structured output for IT change risk decisions."""

    risk: ChangeRisk
    approval_required: bool
    evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: Confidence
    reasoning_summary: str


class EnterpriseDecision(BaseModel):
    """Generic enterprise decision with a decision_type discriminator.

    This model wraps the specific decision schemas and includes the
    decision_type so that the correct schema can be selected for
    validation and scoring.
    """

    decision_type: str
    priority: Priority | None = None
    escalation: Escalation | None = None
    decision: InvoiceDecision | None = None
    risk: ChangeRisk | None = None
    approval_required: bool | None = None
    evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: Confidence | None = None
    reasoning_summary: str | None = None
