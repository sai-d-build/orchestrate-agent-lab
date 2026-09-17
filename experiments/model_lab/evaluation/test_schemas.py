"""Tests for Pydantic schemas and scoring functions.

Why it exists:
    Ensures the structured output schemas validate correctly and
    the scoring functions produce expected results.

What problem it solves:
    Catches schema validation errors and scoring logic bugs before
    they affect experiment results.

Testing:
    - Run with: pytest experiments/model_lab/evaluation/test_schemas.py -v
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.schemas.decision import (
    ChangeRisk,
    Confidence,
    Escalation,
    InvoiceDecision,
    Priority,
    SupportEscalationDecision,
    InvoiceExceptionDecision,
    ChangeRiskDecision,
)
from experiments.model_lab.evaluation.scoring import (
    score_decision,
    score_consistency,
    estimate_cost,
)


# --- Schema validation tests ---


def test_support_escalation_decision_valid():
    """Verify a valid support escalation decision parses correctly."""
    data = {
        "priority": "P1",
        "escalation": "YES",
        "evidence": ["Production unavailable", "1200 affected users"],
        "missing_information": [],
        "confidence": "HIGH",
        "reasoning_summary": "Production outage with high impact.",
    }
    decision = SupportEscalationDecision(**data)
    assert decision.priority == Priority.P1
    assert decision.escalation == Escalation.YES
    assert decision.confidence == Confidence.HIGH


def test_support_escalation_decision_invalid_priority():
    """Verify invalid priority raises ValidationError."""
    with pytest.raises(ValidationError):
        SupportEscalationDecision(
            priority="P5",
            escalation="YES",
            confidence="HIGH",
            reasoning_summary="test",
        )


def test_support_escalation_decision_invalid_confidence():
    """Verify invalid confidence raises ValidationError."""
    with pytest.raises(ValidationError):
        SupportEscalationDecision(
            priority="P1",
            escalation="YES",
            confidence="VERY_HIGH",
            reasoning_summary="test",
        )


def test_invoice_exception_decision_valid():
    """Verify a valid invoice exception decision parses correctly."""
    data = {
        "decision": "approve",
        "evidence": ["PO matches invoice", "Goods received"],
        "missing_information": [],
        "confidence": "HIGH",
        "reasoning_summary": "All checks passed.",
    }
    decision = InvoiceExceptionDecision(**data)
    assert decision.decision == InvoiceDecision.APPROVE


def test_invoice_exception_decision_invalid():
    """Verify invalid decision value raises ValidationError."""
    with pytest.raises(ValidationError):
        InvoiceExceptionDecision(
            decision="accept",
            confidence="HIGH",
            reasoning_summary="test",
        )


def test_change_risk_decision_valid():
    """Verify a valid change risk decision parses correctly."""
    data = {
        "risk": "high",
        "approval_required": True,
        "evidence": ["Customer impact: high", "No rollback plan"],
        "missing_information": [],
        "confidence": "HIGH",
        "reasoning_summary": "High risk change.",
    }
    decision = ChangeRiskDecision(**data)
    assert decision.risk == ChangeRisk.HIGH
    assert decision.approval_required is True


def test_change_risk_decision_invalid_risk():
    """Verify invalid risk value raises ValidationError."""
    with pytest.raises(ValidationError):
        ChangeRiskDecision(
            risk="critical",
            approval_required=True,
            confidence="HIGH",
            reasoning_summary="test",
        )


def test_decision_with_empty_evidence():
    """Verify decisions with empty evidence lists are valid."""
    decision = SupportEscalationDecision(
        priority="P2",
        escalation="NO",
        evidence=[],
        missing_information=[],
        confidence="MEDIUM",
        reasoning_summary="Low impact issue.",
    )
    assert decision.evidence == []


# --- Scoring tests ---


def test_score_decision_correct():
    """Verify scoring returns 1.0 for a correct decision."""
    expected = {"priority": "P1", "escalation": True}
    actual = {
        "priority": "P1",
        "escalation": "YES",
        "evidence": ["test"],
        "missing_information": [],
        "confidence": "HIGH",
        "reasoning_summary": "test",
    }
    scores = score_decision(expected, actual)
    assert scores["decision_correctness"] == 1.0
    assert scores["structured_output_validity"] == 1.0


def test_score_decision_incorrect():
    """Verify scoring returns 0.0 for an incorrect decision."""
    expected = {"priority": "P1", "escalation": True}
    actual = {
        "priority": "P4",
        "escalation": "NO",
        "evidence": ["test"],
        "missing_information": [],
        "confidence": "HIGH",
        "reasoning_summary": "test",
    }
    scores = score_decision(expected, actual)
    assert scores["decision_correctness"] == 0.0


def test_score_decision_none_response():
    """Verify scoring returns 0.0 for all metrics when response is None."""
    expected = {"priority": "P1", "escalation": True}
    scores = score_decision(expected, None)
    assert scores["decision_correctness"] == 0.0
    assert scores["structured_output_validity"] == 0.0
    assert scores["evidence_correctness"] == 0.0


def test_score_consistency_identical():
    """Verify consistency is 1.0 when all runs produce the same decision."""
    responses = [
        {"priority": "P1", "escalation": "YES"},
        {"priority": "P1", "escalation": "YES"},
        {"priority": "P1", "escalation": "YES"},
    ]
    assert score_consistency(responses) == 1.0


def test_score_consistency_different():
    """Verify consistency is 0.0 when runs differ."""
    responses = [
        {"priority": "P1", "escalation": "YES"},
        {"priority": "P4", "escalation": "NO"},
    ]
    assert score_consistency(responses) == 0.0


def test_score_consistency_with_none():
    """Verify consistency handles None responses."""
    responses = [
        {"priority": "P1", "escalation": "YES"},
        None,
        {"priority": "P1", "escalation": "YES"},
    ]
    assert score_consistency(responses) == 1.0


def test_score_consistency_empty():
    """Verify consistency is 0.0 for empty responses."""
    assert score_consistency([]) == 0.0


def test_estimate_cost_free_model():
    """Verify cost estimation returns 0.0 for free models."""
    assert estimate_cost(1000, "nvidia/nemotron-3.5-lightning:free") == 0.0


def test_estimate_cost_unknown_model():
    """Verify cost estimation returns 0.0 for unknown models."""
    assert estimate_cost(1000, "unknown-model") == 0.0


def test_estimate_cost_none_tokens():
    """Verify cost estimation returns None for missing token count."""
    assert estimate_cost(None, "nvidia/nemotron-3.5-lightning:free") is None


def test_score_injection_resistance_adversarial():
    """Verify injection resistance scores correctly for adversarial cases."""
    expected = {
        "category": "adversarial",
        "priority": "P1",
        "escalation": True,
    }
    actual = {
        "priority": "P1",
        "escalation": "YES",
        "evidence": ["test"],
        "missing_information": [],
        "confidence": "HIGH",
        "reasoning_summary": "test",
    }
    scores = score_decision(expected, actual)
    assert scores["injection_resistance"] == 1.0


def test_score_injection_resistance_non_adversarial():
    """Verify injection resistance is 1.0 for non-adversarial cases."""
    expected = {
        "category": "normal",
        "priority": "P1",
        "escalation": True,
    }
    actual = {
        "priority": "P1",
        "escalation": "YES",
        "evidence": ["test"],
        "missing_information": [],
        "confidence": "HIGH",
        "reasoning_summary": "test",
    }
    scores = score_decision(expected, actual)
    assert scores["injection_resistance"] == 1.0


def test_score_missing_info_detection():
    """Verify missing information detection scores correctly."""
    expected = {"category": "missing_information"}
    actual = {
        "priority": "MANUAL_REVIEW",
        "escalation": "YES",
        "evidence": ["test"],
        "missing_information": ["SLA hours", "Affected users"],
        "confidence": "LOW",
        "reasoning_summary": "Insufficient data.",
    }
    scores = score_decision(expected, actual)
    assert scores["missing_information_detection"] == 1.0


def test_score_missing_info_false_positive():
    """Verify missing information detection penalizes false positives."""
    expected = {"category": "normal"}
    actual = {
        "priority": "P1",
        "escalation": "YES",
        "evidence": ["test"],
        "missing_information": ["Unrelated fact"],
        "confidence": "HIGH",
        "reasoning_summary": "test",
    }
    scores = score_decision(expected, actual)
    assert scores["missing_information_detection"] == 0.5
