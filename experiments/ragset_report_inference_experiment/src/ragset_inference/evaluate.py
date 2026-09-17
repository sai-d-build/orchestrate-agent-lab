import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from .data import LABEL_COLUMNS


def evaluate(gold: pd.DataFrame, predicted: pd.DataFrame) -> dict:
    merged = gold[["StudyInstanceUID", *LABEL_COLUMNS]].merge(
        predicted[["StudyInstanceUID", *LABEL_COLUMNS]],
        on="StudyInstanceUID",
        suffixes=("_gold", "_pred"),
    )

    out = {}

    if merged.empty:
        for label in LABEL_COLUMNS:
            out[label] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }
        out["micro"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        out["exact_12_label_match"] = 0.0
        return out

    yt_all, yp_all = [], []

    for label in LABEL_COLUMNS:
        yt = merged[f"{label}_gold"].astype(int)
        yp = merged[f"{label}_pred"].astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0
        )
        out[label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "accuracy": float(accuracy_score(yt, yp)),
        }
        yt_all.extend(yt)
        yp_all.extend(yp)

    p, r, f1, _ = precision_recall_fscore_support(
        yt_all, yp_all, average="micro", zero_division=0
    )
    out["micro"] = {"precision": float(p), "recall": float(r), "f1": float(f1)}

    gold_matrix = merged[[f"{x}_gold" for x in LABEL_COLUMNS]].to_numpy()
    pred_matrix = merged[[f"{x}_pred" for x in LABEL_COLUMNS]].to_numpy()
    out["exact_12_label_match"] = float(
        (gold_matrix == pred_matrix).all(axis=1).mean()
    )
    return out
