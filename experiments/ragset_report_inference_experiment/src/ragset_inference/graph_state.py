from typing import TypedDict, List, Optional
from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import (
    ReportPrediction,
    CritiqueResult,
    JudgeResult,
    TraceRecord,
    LABEL_KEYS,
)


class RagSetState(TypedDict):
    """LangGraph state for RagSet MRI report inference workflow."""

    # Input identifiers
    study_instance_uid: str
    report: str

    # Context (loaded once in initialize node)
    canonical_policy: str
    retrieved_gold_context: str

    # Attempt tracking
    attempt: int
    max_attempts: int

    # Current cycle outputs
    current_prediction: Optional[ReportPrediction]
    current_critique: Optional[CritiqueResult]
    current_judgment: Optional[JudgeResult]

    # History (append-only, one entry per cycle)
    prediction_history: List[ReportPrediction]
    critique_history: List[CritiqueResult]
    judgment_history: List[JudgeResult]

    # Oscillation/stuck detection (computed by inference/critic nodes)
    oscillation_detected: bool
    stuck_detected: bool
    oscillating_labels: List[str]
    stuck_labels: List[str]

    # Final result
    final_status: Optional[str]  # "passed" | "needs_review" | "error"
    final_prediction: Optional[ReportPrediction]
    review_reason: Optional[str]

    # Finalization metadata (authoritative)
    finalization_selected_attempt: Optional[int]
    finalization_reason: Optional[str]
    finalization_terminal_action: Optional[str]
    finalization_terminal_reason_code: Optional[str]
    finalization_conflict_state: Optional[str]

    # Trace records (accumulated per node)
    trace_records: List[TraceRecord]

    # Safety gate failure (deterministic integrity check on Critic output)
    safety_gate_failure: Optional[dict]