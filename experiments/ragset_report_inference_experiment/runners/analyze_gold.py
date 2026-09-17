from pathlib import Path
import json
from ..src.ragset_inference.data import load_train, split_gold
from ..src.ragset_inference.gold import analyze_gold


def main():
    root = Path(__file__).resolve().parents[3]
    exp = root / "experiments/ragset_report_inference_experiment"
    df = load_train(root / "train.csv")
    gold, unlabeled = split_gold(df)

    (exp / "data/gold").mkdir(parents=True, exist_ok=True)
    (exp / "data/inferred").mkdir(parents=True, exist_ok=True)
    gold.to_csv(exp / "data/gold/gold.csv", index=False)
    unlabeled.to_csv(exp / "data/inferred/unlabeled.csv", index=False)

    analysis = analyze_gold(gold)
    out = exp / "results/gold/basic_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
