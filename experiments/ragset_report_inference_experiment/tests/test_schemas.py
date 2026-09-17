from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import (
    LABEL_KEYS, LABEL_LITERAL, LabelValue, ReportPrediction,
    ValidationIssue, ValidationResult,
)


def test_label_value_valid():
    pred = LabelValue(value=1, evidence="Tear visible")
    assert pred.value == 1
    assert pred.evidence == "Tear visible"


def test_label_value_invalid_value():
    import pytest
    with pytest.raises(Exception):
        LabelValue(value=2, evidence="test")


def test_label_value_extra_field_rejected():
    import pytest
    with pytest.raises(Exception):
        LabelValue(value=1, evidence="test", extra="field")


def test_report_prediction_valid():
    preds = {l: LabelValue(value=0, evidence="") for l in LABEL_KEYS}
    rp = ReportPrediction(predictions=preds)
    assert len(rp.predictions) == 12


def test_report_prediction_exactly_12():
    import pytest
    preds = {l: LabelValue(value=0, evidence="") for l in list(LABEL_KEYS)[:11]}
    with pytest.raises(Exception):
        ReportPrediction(predictions=preds)


def test_report_prediction_missing_label():
    import pytest
    preds = {l: LabelValue(value=0, evidence="") for l in LABEL_KEYS if l != "ACL"}
    with pytest.raises(Exception):
        ReportPrediction(predictions=preds)


def test_report_prediction_duplicate_label():
    import pytest
    # Can't have duplicate keys in a dict, so this tests the validator
    preds = {l: LabelValue(value=0, evidence="") for l in LABEL_KEYS}
    preds["ACL"] = LabelValue(value=1, evidence="")
    # This should work since dict keys are unique
    rp = ReportPrediction(predictions=preds)
    assert rp.predictions["ACL"].value == 1


def test_report_prediction_to_dict():
    preds = {l: LabelValue(value=i % 2, evidence="") for i, l in enumerate(LABEL_KEYS)}
    rp = ReportPrediction(predictions=preds)
    d = rp.to_dict()
    assert len(d) == 12
    assert d["ACL"] == 0


def test_validation_result_pass():
    vr = ValidationResult(status="PASS", issues=[])
    assert vr.passed is True


def test_validation_result_fail():
    vr = ValidationResult(
        status="FAIL",
        issues=[ValidationIssue(label="ACL", predicted=1, corrected=0, reason="test", evidence=["evidence"])],
    )
    assert vr.passed is False


def test_prediction_has_12_labels():
    obj = ReportPrediction.model_validate({
        "predictions": {
            key: {"value": 0, "evidence": ""}
            for key in LABEL_KEYS
        },
    })
    assert len(obj.predictions) == 12


def test_pass_validation():
    assert ValidationResult(status="PASS", issues=[]).passed