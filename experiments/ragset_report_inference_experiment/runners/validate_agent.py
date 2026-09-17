from pathlib import Path
import pandas as pd
from ..src.ragset_inference.data import load_train, split_gold


def main():
    root = Path(__file__).resolve().parents[3]
    exp = root / "experiments/ragset_report_inference_experiment"
    df = load_train(root / "train.csv")
    gold, _ = split_gold(df)

    heldout = gold.sample(frac=0.20, random_state=42)
    out = exp / "data/validation/heldout.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(out, index=False)

    print(f"Created deterministic held-out set: {len(heldout)} reports")
    print(out)


if __name__ == "__main__":
    main()
