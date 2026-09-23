from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


LABEL_KEYS = [
    "ACL", "MCL", "Medial_Meniscus", "Lateral_Meniscus",
    "Medial_OA", "Lateral_OA", "PF_OA", "Effusion",
    "Synovitis", "Bakers", "Contusion", "Fracture",
]

LABEL_LITERAL = Literal[
    "ACL", "MCL", "Medial_Meniscus", "Lateral_Meniscus",
    "Medial_OA", "Lateral_OA", "PF_OA", "Effusion",
    "Synovitis", "Bakers", "Contusion", "Fracture",
]


class LabelValue(BaseModel):
    """Single label prediction with value and evidence."""
    model_config = ConfigDict(extra="forbid")
    value: Literal[0, 1]
    evidence: str | None = None

    # Do NOT coerce null to empty string - null evidence must remain null
    # Per rules: "If no suitable evidence exists: evidence: null"


class ReportPrediction(BaseModel):
    """Flat report prediction with 12 label values and evidence."""
    model_config = ConfigDict(extra="forbid")
    predictions: dict[LABEL_LITERAL, LabelValue]

    @field_validator("predictions")
    @classmethod
    def validate_exactly_12_labels(cls, v: dict[LABEL_LITERAL, LabelValue]) -> dict[LABEL_LITERAL, LabelValue]:
        if len(v) != 12:
            raise ValueError(f"Expected exactly 12 predictions, got {len(v)}")
        expected = set(LABEL_KEYS)
        actual = set(v.keys())
        if actual != expected:
            missing = expected - actual
            extra = actual - expected
            raise ValueError(f"Labels mismatch. Missing: {sorted(missing)}, Extra: {sorted(extra)}")
        return v

    def to_dict(self) -> dict[str, int]:
        """Convert to dict of label -> value for easy comparison."""
        return {label: pred.value for label, pred in self.predictions.items()}


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    predicted: Literal[0, 1]
    corrected: Literal[0, 1] | None = None
    reason: str
    evidence: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["PASS", "FAIL", "AMBIGUOUS"]
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "PASS" and not self.issues


# ============================================================
# Model 2 Critic schemas
# ============================================================

class CritiqueStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    AMBIGUOUS = "AMBIGUOUS"


class CritiqueIssueType(str, Enum):
    CLEAR_POLICY_CONFLICT = "CLEAR_POLICY_CONFLICT"
    CLEAR_REPORT_CONFLICT = "CLEAR_REPORT_CONFLICT"
    UNRESOLVED_POLICY = "UNRESOLVED_POLICY"
    REPORT_AMBIGUITY = "REPORT_AMBIGUITY"
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class CritiqueIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    model1_value: Literal[0, 1]
    proposed_value: Literal[0, 1] | None = None
    issue_type: CritiqueIssueType
    evidence: str | None = None
    policy_rule: str | None = None
    feedback: str


class CritiqueResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: CritiqueStatus
    issues: list[CritiqueIssue] = Field(default_factory=list)
    summary: str = ""
    actionable: bool = False
    affected_labels: list[str] = Field(default_factory=list)

    @property
    def has_actionable_issues(self) -> bool:
        """True if there are CLEAR_POLICY_CONFLICT or CLEAR_REPORT_CONFLICT issues."""
        return any(
            issue.issue_type in (CritiqueIssueType.CLEAR_POLICY_CONFLICT, CritiqueIssueType.CLEAR_REPORT_CONFLICT)
            for issue in self.issues
        )


# ============================================================
# Model 3 Judge schemas
# ============================================================

class JudgeAction(str, Enum):
    PASS = "PASS"
    RETRY_MODEL1 = "RETRY_MODEL1"
    AMBIGUOUS = "AMBIGUOUS"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    STOP = "STOP"


class JudgeReasonCode(str, Enum):
    NO_ACTIONABLE_ISSUE = "NO_ACTIONABLE_ISSUE"
    CLEAR_POLICY_CONFLICT = "CLEAR_POLICY_CONFLICT"
    CLEAR_REPORT_CONFLICT = "CLEAR_REPORT_CONFLICT"
    UNRESOLVED_POLICY = "UNRESOLVED_POLICY"
    REPORT_AMBIGUITY = "REPORT_AMBIGUITY"
    OSCILLATION = "OSCILLATION"
    MODEL1_STUCK = "MODEL1_STUCK"
    REPEATED_CRITIQUE = "REPEATED_CRITIQUE"
    MAX_ATTEMPTS = "MAX_ATTEMPTS"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: JudgeAction
    reason_code: JudgeReasonCode
    rationale: str = Field(max_length=500)
    # No clinical labels, no corrected labels, no suggested labels, no clinical evidence


# ============================================================
# Trace schemas
# ============================================================

class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp_utc: str
    study_instance_uid: str
    graph_node: Literal["initialize", "inference", "critic", "judge", "finalize"]
    stage: Literal["inference", "validation", "judgment"]
    attempt: int
    requested_model: str
    actual_model: str | None = None
    provider: str
    prompt_hash: str
    response_hash: str | None = None
    latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    status: Literal["success", "error", "safety_gate_failure"]
    error: str | None = None
    judge_action: JudgeAction | None = None