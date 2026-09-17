import pandas as pd
from experiments.ragset_report_inference_experiment.src.ragset_inference.evaluate import evaluate
from experiments.ragset_report_inference_experiment.src.ragset_inference.data import LABEL_COLUMNS


def test_evaluate_perfect_match():
    """Verify perfect prediction yields 1.0 for all metrics."""
    gold = pd.DataFrame([
        ["g1", "report1"] + [1] * 12,
        ["g2", "report2"] + [0] * 12,
    ], columns=["StudyInstanceUID", "Report", *LABEL_COLUMNS])
    pred = gold.copy()
    results = evaluate(gold, pred)
    assert results["micro"]["f1"] == 1.0
    assert results["exact_12_label_match"] == 1.0
    for label in LABEL_COLUMNS:
        assert results[label]["f1"] == 1.0


def test_evaluate_all_wrong():
    """Verify completely wrong predictions yield 0.0 for all metrics."""
    gold = pd.DataFrame([
        ["g1", "report1"] + [1] * 12,
        ["g2", "report2"] + [0] * 12,
    ], columns=["StudyInstanceUID", "Report", *LABEL_COLUMNS])
    pred_data = [["g1", "report1"] + [0] * 12, ["g2", "report2"] + [1] * 12]
    pred = pd.DataFrame(pred_data, columns=["StudyInstanceUID", "Report", *LABEL_COLUMNS])
    results = evaluate(gold, pred)
    assert results["micro"]["f1"] == 0.0
    assert results["exact_12_label_match"] == 0.0


def test_evaluate_partial():
    """Verify partial predictions yield intermediate metrics."""
    gold = pd.DataFrame([
        ["g1", "report1"] + [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        ["g2", "report2"] + [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ], columns=["StudyInstanceUID", "Report", *LABEL_COLUMNS])
    # Predict only the first 6 labels correctly for both rows
    pred_data = []
    for i, row in gold.iterrows():
        pred_data.append([row["StudyInstanceUID"], row["Report"]] + [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    pred = pd.DataFrame(pred_data, columns=["StudyInstanceUID", "Report", *LABEL_COLUMNS])
    results = evaluate(gold, pred)
    assert 0.0 < results["micro"]["f1"] < 1.0
    assert results["micro"]["precision"] > 0.0
    assert results["micro"]["recall"] > 0.0


def test_evaluate_different_study_ids():
    """Verify evaluation handles mismatched StudyInstanceUIDs."""
    gold = pd.DataFrame([
        ["g1", "report1"] + [1] * 12,
    ], columns=["StudyInstanceUID", "Report", *LABEL_COLUMNS])
    pred = pd.DataFrame([
        ["g2", "report2"] + [1] * 12,
    ], columns=["StudyInstanceUID", "Report", *LABEL_COLUMNS])
    results = evaluate(gold, pred)
    # No matching StudyInstanceUIDs → all predictions are false positives/negatives
    assert results["micro"]["precision"] == 0.0


def test_evaluate_empty():
    """Verify evaluation handles empty DataFrames."""
    gold = pd.DataFrame(columns=["StudyInstanceUID", "Report", *LABEL_COLUMNS])
    pred = pd.DataFrame(columns=["StudyInstanceUID", "Report", *LABEL_COLUMNS])
    results = evaluate(gold, pred)
    assert "micro" in results
