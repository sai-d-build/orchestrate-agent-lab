"""Tests for the run_case module functions.

Why it exists:
    Ensures the case loading, prompt building, and response parsing
    functions work correctly without requiring API calls.

What problem it solves:
    Catches bugs in prompt construction and response parsing that
    would otherwise only be discovered during live experiments.

Testing:
    - Run with: pytest experiments/model_lab/evaluation/test_run_case.py -v
"""

from __future__ import annotations

import json

import pytest

from experiments.model_lab.runners.run_case import (
    build_prompt,
    load_case,
    load_prompt_template,
    parse_response,
)


def test_load_prompt_template():
    """Verify the prompt template loads correctly."""
    template = load_prompt_template()
    assert "common" in template
    assert "decision_types" in template
    assert "support_escalation" in template["decision_types"]
    assert "invoice_exception" in template["decision_types"]
    assert "change_risk" in template["decision_types"]


def test_load_case_support_escalation():
    """Verify loading a support escalation case."""
    case = load_case("support_escalation", "SUP-001")
    assert case["id"] == "SUP-001"
    assert "input" in case
    assert "expected" in case
    assert case["expected"]["priority"] == "P1"
    assert case["expected"]["escalation"] is True


def test_load_case_invoice_exception():
    """Verify loading an invoice exception case."""
    case = load_case("invoice_exception", "INV-001")
    assert case["id"] == "INV-001"
    assert case["expected"]["decision"] == "approve"


def test_load_case_change_risk():
    """Verify loading a change risk case."""
    case = load_case("change_risk", "CHG-001")
    assert case["id"] == "CHG-001"
    assert case["expected"]["risk"] == "low"
    assert case["expected"]["approval_required"] is False


def test_load_case_not_found():
    """Verify loading a non-existent case raises ValueError."""
    with pytest.raises(ValueError, match="Case not found"):
        load_case("support_escalation", "NONEXISTENT")


def test_load_case_unknown_dataset():
    """Verify loading from an unknown dataset raises ValueError."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_case("unknown_dataset", "SUP-001")


def test_build_prompt_support_escalation():
    """Verify prompt building for support escalation."""
    template = load_prompt_template()
    case = load_case("support_escalation", "SUP-001")
    prompt = build_prompt(case, "support_escalation", template)

    assert "enterprise decision" in prompt.lower()
    assert "priority" in prompt.lower()
    assert "escalation" in prompt.lower()
    assert "YES" in prompt or "escalation" in prompt.lower()
    assert json.dumps(case["input"]) in prompt or "customer_tier" in prompt


def test_build_prompt_invoice_exception():
    """Verify prompt building for invoice exception."""
    template = load_prompt_template()
    case = load_case("invoice_exception", "INV-001")
    prompt = build_prompt(case, "invoice_exception", template)

    assert "invoice" in prompt.lower()
    assert "approve" in prompt.lower()


def test_build_prompt_change_risk():
    """Verify prompt building for change risk."""
    template = load_prompt_template()
    case = load_case("change_risk", "CHG-001")
    prompt = build_prompt(case, "change_risk", template)

    assert "change risk" in prompt.lower()
    assert "risk" in prompt.lower()


def test_parse_response_valid_json():
    """Verify parsing a valid JSON response."""
    response_text = json.dumps({
        "priority": "P1",
        "escalation": "YES",
        "evidence": ["Production unavailable"],
        "missing_information": [],
        "confidence": "HIGH",
        "reasoning_summary": "High impact outage.",
    })
    parsed = parse_response(response_text, "support_escalation")
    assert parsed is not None
    assert parsed["priority"] == "P1"
    assert parsed["escalation"] == "YES"
    assert parsed["confidence"] == "HIGH"


def test_parse_response_invalid_json():
    """Verify parsing invalid JSON returns None."""
    parsed = parse_response("This is not JSON", "support_escalation")
    assert parsed is None


def test_parse_response_invalid_enum():
    """Verify parsing JSON with invalid enum values returns None."""
    response_text = json.dumps({
        "priority": "P5",
        "escalation": "YES",
        "confidence": "HIGH",
        "reasoning_summary": "test",
    })
    parsed = parse_response(response_text, "support_escalation")
    assert parsed is None


def test_parse_response_invoice_exception():
    """Verify parsing an invoice exception response."""
    response_text = json.dumps({
        "decision": "approve",
        "evidence": ["PO matches invoice"],
        "missing_information": [],
        "confidence": "HIGH",
        "reasoning_summary": "All checks passed.",
    })
    parsed = parse_response(response_text, "invoice_exception")
    assert parsed is not None
    assert parsed["decision"] == "approve"


def test_parse_response_change_risk():
    """Verify parsing a change risk response."""
    response_text = json.dumps({
        "risk": "high",
        "approval_required": True,
        "evidence": ["Customer impact: high"],
        "missing_information": [],
        "confidence": "HIGH",
        "reasoning_summary": "High risk change.",
    })
    parsed = parse_response(response_text, "change_risk")
    assert parsed is not None
    assert parsed["risk"] == "high"
    assert parsed["approval_required"] is True


def test_parse_response_unknown_dataset():
    """Verify parsing with an unknown dataset returns None."""
    response_text = json.dumps({"test": "data"})
    parsed = parse_response(response_text, "unknown_dataset")
    assert parsed is None
