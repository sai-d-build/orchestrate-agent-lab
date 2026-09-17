import pandas as pd
from experiments.ragset_report_inference_experiment.src.ragset_inference.data import (
    LABEL_COLUMNS, LABEL_COLUMN_ALIASES, split_gold, load_train,
)


def test_split_gold():
    cols = ["StudyInstanceUID", "Report", *LABEL_COLUMNS]
    df = pd.DataFrame([
        ["g", "gold", *([1] * 12)],
        ["u", "unlabeled", *([None] * 12)],
    ], columns=cols)
    gold, unlabeled = split_gold(df)
    assert len(gold) == 1
    assert len(unlabeled) == 1


def test_load_train_renames_spaces():
    """Verify CSV columns with spaces are renamed to underscore labels."""
    csv_cols = ["StudyInstanceUID", "Report", "ACL", "MCL",
                "Medial Meniscus", "Lateral Meniscus",
                "Medial OA", "Lateral OA", "PF OA", "Effusion",
                "Synovitis", "Baker's", "Contusion", "Fracture"]
    df = pd.DataFrame([
        ["s1", "report text", 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    ], columns=csv_cols)
    loaded = load_train(pd.io.common.StringIO(df.to_csv(index=False)))
    for col in LABEL_COLUMNS:
        assert col in loaded.columns, f"Missing column: {col}"


def test_load_train_missing_columns():
    import pytest
    df = pd.DataFrame([["s1", "report", 1, 0]])
    with pytest.raises(ValueError, match="Missing required columns"):
        load_train(pd.io.common.StringIO(df.to_csv(index=False)))


def test_split_gold_unlabeled():
    """Verify rows with any null label are unlabeled."""
    cols = ["StudyInstanceUID", "Report", *LABEL_COLUMNS]
    df = pd.DataFrame([
        ["g1", "gold", *([1] * 12)],
        ["u1", "unlabeled", *([None] * 6) + [1] * 6],
        ["u2", "unlabeled", *([None] * 12)],
    ], columns=cols)
    gold, unlabeled = split_gold(df)
    assert len(gold) == 1
    assert len(unlabeled) == 2
