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