#!/usr/bin/env python3
"""Run held-out evaluation separately from full inference.

This script evaluates the pipeline on a held-out subset of gold reports
before running full inference on the unlabeled population.
"""

from pathlib import Path
import json
import os
import yaml
import argparse
from dotenv import load_dotenv

load_dotenv()

import yaml as yaml_mod
from experiments.ragset_report_inference_experiment.src.ragset_inference.data import (
    load_train, split_gold, LABEL_COLUMNS,
)
from experiments.ragset_report_inference_experiment.src.ragset_inference.gold import (
    analyze_gold, format_gold_examples, label_profile_text,
)
from experiments.ragset_report_inference_experiment.src.ragset_inference.retrieval import GoldRetriever
from experiments.ragset_report_inference_experiment.src.ragset_inference.models import InferenceModel, ValidatorModel
from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import run_report
from experiments.ragset_report_inference_experiment.src.ragset_inference.evaluate import evaluate
from experiments.ragset_report_inference_experiment.src.ragset_inference.trace import write_trace


def make_infer_fn(model, inf_cfg, trace_path, labels, gold_analysis):
    """Factory that creates an infer closure with captured dependencies."""

    def infer_fn(**kwargs):
        user = inf_cfg["user_template"].format(
            report=kwargs["report"],
            labels="\n".join(labels),
            gold_analysis=gold_analysis,
            gold_examples=kwargs.get("gold_examples", ""),
            previous_prediction=(
                kwargs["previous_prediction"].model_dump_json()
                if kwargs["previous_prediction"] else "None"
            ),
            validator_feedback=json.dumps(
                kwargs["validator_feedback"], ensure_ascii=False
            ) if kwargs["validator_feedback"] else "None",
        )
        report_id = kwargs["report_id"]
        try:
            prediction = model.run(inf_cfg["system"], user)
            actual_model = getattr(model, 'last_actual_model', model.model)
            write_trace(
                trace_path, report_id=report_id, stage="inference",
                model=model.model, attempt=kwargs["attempt"],
                prompt=inf_cfg["system"] + "\n" + user, status="success",
                provider=model.provider, actual_model=actual_model,
            )
            return prediction
        except Exception as e:
            write_trace(
                trace_path, report_id=report_id, stage="inference",
                model=model.model, attempt=kwargs["attempt"],
                prompt=inf_cfg["system"] + "\n" + user, status="error",
                provider=model.provider, error=str(e),
            )
            raise

    return infer_fn


def make_validate_fn(model, val_cfg, trace_path, labels, gold_analysis):
    """Factory that creates a validate closure with captured dependencies."""

    def validate_fn(**kwargs):
        user = val_cfg["user_template"].format(
            report=kwargs["report"],
            labels="\n".join(labels),
            gold_analysis=gold_analysis,
            prediction=kwargs["prediction"].model_dump_json(),
        )
        report_id = kwargs["report_id"]
        try:
            validation = model.run(val_cfg["system"], user)
            actual_model = getattr(model, 'last_actual_model', model.model)
            write_trace(
                trace_path, report_id=report_id, stage="validation",
                model=model.model, attempt=kwargs["attempt"],
                prompt=val_cfg["system"] + "\n" + user, status="success",
                provider=model.provider, actual_model=actual_model,
            )
            return validation
        except Exception as e:
            write_trace(
                trace_path, report_id=report_id, stage="validation",
                model=model.model, attempt=kwargs["attempt"],
                prompt=val_cfg["system"] + "\n" + user, status="error",
                provider=model.provider, error=str(e),
            )
            raise

    return validate_fn


def main(limit: int | None = None, overwrite: bool = False):
    root = Path(__file__).resolve().parents[3]
    exp = root / "experiments/ragset_report_inference_experiment"

    cfg = yaml.safe_load(
        (exp / "config/experiment.yaml").read_text(encoding="utf-8")
    )
    labels_cfg = yaml.safe_load(
        (exp / "config/labels.yaml").read_text(encoding="utf-8")
    )
    inf_cfg = yaml.safe_load(
        (exp / "prompts/inference.yaml").read_text(encoding="utf-8")
    )
    val_cfg = yaml.safe_load(
        (exp / "prompts/validation.yaml").read_text(encoding="utf-8")
    )

    labels = labels_cfg["labels"]
    df = load_train(root / cfg["source"]["path"])
    gold, _ = split_gold(df)
    analysis = analyze_gold(gold)
    retriever = GoldRetriever(gold, top_k=cfg["retrieval"]["top_k"])

    # Load model profiles from config/models.yaml
    models_cfg = yaml_mod.safe_load(
        (root / "config" / "models.yaml").read_text(encoding="utf-8")
    )
    inf_profile = cfg["models"]["inference_profile"]
    val_profile = cfg["models"]["validator_profile"]
    if inf_profile not in models_cfg.get("models", {}):
        raise ValueError(
            f"Unknown inference profile '{inf_profile}'. "
            f"Available: {list(models_cfg.get('models', {}).keys())}"
        )
    if val_profile not in models_cfg.get("models", {}):
        raise ValueError(
            f"Unknown validator profile '{val_profile}'. "
            f"Available: {list(models_cfg.get('models', {}).keys())}"
        )
    inf_config = models_cfg["models"][inf_profile]
    val_config = models_cfg["models"][val_profile]

    # Check for required API keys based on provider
    inf_provider = inf_config.get("provider", "openrouter")
    val_provider = val_config.get("provider", "openrouter")
    inf_api_key_env = inf_config.get("api_key_env", "OPENROUTER_API_KEY")
    val_api_key_env = val_config.get("api_key_env", "OPENROUTER_API_KEY")

    if not os.environ.get(inf_api_key_env):
        raise RuntimeError(f"{inf_api_key_env} is required for provider '{inf_provider}'.")
    if not os.environ.get(val_api_key_env):
        raise RuntimeError(f"{val_api_key_env} is required for provider '{val_provider}'.")

    model1 = InferenceModel(
        inf_config["model"],
        max_tokens=inf_config.get("parameters", {}).get("max_output_tokens", 4000),
        provider=inf_provider,
        api_key=os.environ.get(inf_api_key_env),
        reasoning=inf_config.get("parameters", {}).get("reasoning"),
    )
    model2 = ValidatorModel(
        val_config["model"],
        max_tokens=val_config.get("parameters", {}).get("max_output_tokens", 4000),
        provider=val_provider,
        api_key=os.environ.get(val_api_key_env),
        reasoning=val_config.get("parameters", {}).get("reasoning"),
    )

    output_dir = root / cfg["runtime"].get("output_dir", "experiments/ragset_report_inference_experiment")
    trace_path = output_dir / "results/inference/model_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and trace_path.exists():
        trace_path.unlink()

    # Load held-out set
    heldout_path = exp / "data/validation/heldout.csv"
    if not heldout_path.exists():
        raise RuntimeError("Held-out set not found. Run validate_agent.py first.")

    heldout_df = load_train(heldout_path)
    if limit is not None:
        heldout_df = heldout_df.head(limit)

    print(f"\n--- Held-out evaluation on {len(heldout_df)} reports ---")

    infer_fn = make_infer_fn(model1, inf_cfg, trace_path, labels, label_profile_text(analysis))
    validate_fn = make_validate_fn(model2, val_cfg, trace_path, labels, label_profile_text(analysis))

    predictions = []
    for _, row in heldout_df.iterrows():
        report_id = str(row["StudyInstanceUID"])
        report = str(row["Report"])
        retrieved = retriever.retrieve_excluding(report, exclude_study_id=report_id)
        relevant = "\n---\n".join(
            f"STUDY {x['study_id']}\nLABELS: {x['labels']}\nREPORT:\n{x['report']}"
            for x in retrieved
        )
        analysis = analyze_gold(gold)
        context = {
            "gold_analysis": label_profile_text(analysis),
            "gold_examples": relevant,
        }
        result = run_report(
            report_id=report_id,
            report=report,
            infer=infer_fn,
            validate=validate_fn,
            context=context,
            max_attempts=cfg["loop"]["max_attempts"],
            labels=labels,
        )
        pred = result["final_prediction"]
        predictions.append({
            "StudyInstanceUID": result["report_id"],
            **pred.to_dict(),
            "Report": report,
        })

    pred_df = type(heldout_df)(predictions)
    eval_results = evaluate(heldout_df, pred_df)
    print(json.dumps(eval_results, indent=2))

    # Save evaluation results
    output_dir = root / cfg["runtime"].get("output_dir", "experiments/ragset_report_inference_experiment")
    eval_path = output_dir / "results/inference/heldout_evaluation.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run held-out evaluation on gold reports."
    )
    parser.add_argument(
        "limit",
        nargs="?",
        type=int,
        default=None,
        help="Maximum number of held-out reports to evaluate (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing trace and start fresh",
    )
    args = parser.parse_args()

    main(limit=args.limit, overwrite=args.overwrite)