from collections import Counter
import pandas as pd
from .data import LABEL_COLUMNS


def analyze_gold(gold: pd.DataFrame) -> dict:
    lengths = gold["Report"].astype(str).str.len()
    return {
        "gold_count": int(len(gold)),
        "label_distribution": {
            label: {
                "positive": int((gold[label] == 1).sum()),
                "negative": int((gold[label] == 0).sum()),
            }
            for label in LABEL_COLUMNS
        },
        "report_length": {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        },
        "duplicate_report_count": int(gold["Report"].duplicated().sum()),
    }


def format_gold_examples(gold: pd.DataFrame) -> str:
    blocks = []
    for _, row in gold.iterrows():
        labels = {label: int(row[label]) for label in LABEL_COLUMNS}
        blocks.append(
            f"STUDY {row['StudyInstanceUID']}\n"
            f"LABELS: {labels}\n"
            f"REPORT:\n{row['Report']}"
        )
    return "\n\n---\n\n".join(blocks)


def label_profile_text(analysis: dict) -> str:
    return "\n".join(
        f"{label}: {v['positive']} positive / {v['negative']} negative"
        for label, v in analysis["label_distribution"].items()
    )
