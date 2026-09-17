from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import run_report
from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import (
    LabelValue, ReportPrediction, ValidationIssue, ValidationResult,
)


def make_prediction(values: dict[str, int] | None = None) -> ReportPrediction:
    """Create a ReportPrediction with all 12 labels."""
    from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import (
        LABEL_KEYS,
    )
    preds = {}
    for label in LABEL_KEYS:
        val = values.get(label, 0) if values else 0
        preds[label] = LabelValue(value=val, evidence="")
    return ReportPrediction(predictions=preds)


def test_retry_then_pass():
    calls = {"v": 0}

    def infer(**kwargs):
        return make_prediction()

    def validate(**kwargs):
        calls["v"] += 1
        return ValidationResult(
            status="PASS" if calls["v"] == 2 else "FAIL",
            issues=[],
        )

    result = run_report(
        "x", "report", infer, validate, {}, max_attempts=3
    )
    assert result["status"] == "passed"
    assert len(result["attempts"]) == 2


def test_labels_passed_to_infer_and_validate():
    """Verify labels are forwarded to infer and validate callbacks."""
    received_labels = []

    def infer(**kwargs):
        received_labels.append(kwargs.get("labels"))
        return make_prediction()

    def validate(**kwargs):
        received_labels.append(kwargs.get("labels"))
        return ValidationResult(status="PASS", issues=[])

    labels = ["ACL", "MCL"]
    result = run_report(
        "x", "report", infer, validate, {}, max_attempts=1, labels=labels
    )
    assert len(received_labels) == 2
    assert received_labels[0] == labels
    assert received_labels[1] == labels


def test_retry_then_fail_needs_review():
    """Verify max attempts reached returns needs_review."""
    def infer(**kwargs):
        return make_prediction({"ACL": 1})

    def validate(**kwargs):
        return ValidationResult(status="FAIL", issues=[
            ValidationIssue(
                label="ACL", predicted=1, corrected=0,
                reason="Not supported", evidence=["evidence"],
            )
        ])

    result = run_report(
        "x", "report", infer, validate, {}, max_attempts=3
    )
    assert result["status"] == "needs_review"
    assert len(result["attempts"]) == 3
    assert result["review_reason"] == "Maximum attempts reached without validator PASS"