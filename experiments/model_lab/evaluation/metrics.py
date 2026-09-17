"""Evaluation metrics for the enterprise decision benchmark.

Why it exists:
    Defines the complete set of metrics used to evaluate LLM performance
    on enterprise decision tasks.

What problem it solves:
    Provides a single source of truth for which metrics are tracked,
    ensuring consistency across experiments and evaluations.

Alternatives considered:
    - Hardcoding metric names in scoring code: error-prone, hard to maintain.
    - Using a database schema: overkill for Phase 0.

Tradeoffs:
    - Some metrics are objective (latency, token_usage) while others
      are subjective (evidence_correctness, confidence).
    - Not all metrics apply to all decision types.

Failure modes:
    - Missing metrics in results: evaluation is incomplete.
    - Metric name typos: silent failures in aggregation.

Testing:
    - Verify all metrics are present in experiment results.
    - Verify metric values are in expected ranges.

Production implications:
    - In production, metrics should be versioned and stored in a
      time-series database for trend analysis.
"""

from __future__ import annotations

# Metrics that are scored per-case (0.0 to 1.0)
SCORABLE_METRICS = (
    "decision_correctness",
    "evidence_correctness",
    "policy_adherence",
    "structured_output_validity",
    "missing_information_detection",
    "confidence",
    "injection_resistance",
)

# Metrics that are measured across multiple runs
CONSISTENCY_METRICS = (
    "consistency",
)

# Metrics that are raw measurements (not 0.0-1.0 scores)
MEASUREMENT_METRICS = (
    "latency",
    "token_usage",
    "estimated_cost",
)

# All metrics
METRICS = SCORABLE_METRICS + CONSISTENCY_METRICS + MEASUREMENT_METRICS
