from pathlib import Path
import pandas as pd

LABEL_COLUMNS = [
    "ACL", "MCL", "Medial_Meniscus", "Lateral_Meniscus",
    "Medial_OA", "Lateral_OA", "PF_OA", "Effusion",
    "Synovitis", "Bakers", "Contusion", "Fracture",
]

# Mapping from CSV column names (with spaces/apostrophes) to
# schema field names (valid Python identifiers).
LABEL_COLUMN_ALIASES = {
    "Medial Meniscus": "Medial_Meniscus",
    "Lateral Meniscus": "Lateral_Meniscus",
    "Medial OA": "Medial_OA",
    "Lateral OA": "Lateral_OA",
    "PF OA": "PF_OA",
    "Baker's": "Bakers",
}


def load_train(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Rename CSV columns with spaces/apostrophes to valid identifiers.
    df = df.rename(columns=LABEL_COLUMN_ALIASES)
    required = {"StudyInstanceUID", "Report", *LABEL_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["StudyInstanceUID"].duplicated().any():
        raise ValueError("StudyInstanceUID must be unique")
    if df["Report"].isna().any():
        raise ValueError("Report contains null values")
    return df


def split_gold(df: pd.DataFrame):
    is_gold = df[LABEL_COLUMNS].notna().all(axis=1)
    return df.loc[is_gold].copy(), df.loc[~is_gold].copy()
