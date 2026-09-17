from pathlib import Path
import json
import os
import time
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


def make_infer_fn(model, inf_cfg, trace_path, labels, gold_analysis, gold_examples):
    """Factory that creates an infer closure with captured dependencies."""

    def infer_fn(**kwargs):
        user = inf_cfg["user_template"].format(
            report=kwargs["report"],
            labels="\n".join(labels),
            gold_analysis=gold_analysis,
            gold_examples=gold_examples,
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
            write_trace(
                trace_path, report_id=report_id, stage="inference",
                model=model.model, attempt=kwargs["attempt"],
                prompt=inf_cfg["system"] + "\n" + user, status="success",
            )
            return prediction
        except Exception as e:
            write_trace(
                trace_path, report_id=report_id, stage="inference",
                model=model.model, attempt=kwargs["attempt"],
                prompt=inf_cfg["system"] + "\n" + user, status="error",
                error=str(e),
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
            write_trace(
                trace_path, report_id=report_id, stage="validation",
                model=model.model, attempt=kwargs["attempt"],
                prompt=val_cfg["system"] + "\n" + user, status="success",
            )
            return validation
        except Exception as e:
            write_trace(
                trace_path, report_id=report_id, stage="validation",
                model=model.model, attempt=kwargs["attempt"],
                prompt=val_cfg["system"] + "\n" + user, status="error",
                error=str(e),
            )
            raise

    return validate_fn


def run_heldout_evaluation(
    heldout_df, gold_df, retriever, model1, model2,
    inf_cfg, val_cfg, labels, cfg, trace_path,
):
    """Run the loop on a held-out gold set and evaluate results.

    Uses retrieve_excluding to prevent data leakage:
    a held-out report is never retrieved into its own context.
    """
    print("\n--- Held-out evaluation ---")
    predictions = []
    for _, row in heldout_df.iterrows():
        report_id = str(row["StudyInstanceUID"])
        report = str(row["Report"])
        retrieved = retriever.retrieve_excluding(report, exclude_study_id=report_id)
        relevant = "\n---\n".join(
            f"STUDY {x['study_id']}\nLABELS: {x['labels']}\nREPORT:\n{x['report']}"
            for x in retrieved
        )
        analysis = analyze_gold(gold_df)
        context = {
            "gold_analysis": label_profile_text(analysis),
            "gold_examples": relevant,
        }
        infer_fn = make_infer_fn(model1, inf_cfg, trace_path, labels, context["gold_analysis"], context["gold_examples"])
        validate_fn = make_validate_fn(model2, val_cfg, trace_path, labels, context["gold_analysis"])
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
            "StudyInstanceUID": pred.study_instance_uid,
            **pred.to_dict(),
            "Report": report,
        })

    pred_df = type(heldout_df)(predictions)
    eval_results = evaluate(heldout_df, pred_df)
    print(json.dumps(eval_results, indent=2))
    return eval_results


def load_existing_predictions(result_path: Path) -> set[str]:
    """Load already-processed report IDs from existing predictions file."""
    if not result_path.exists():
        return set()
    processed = set()
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                report_id = record.get("report_id") or record.get("StudyInstanceUID")
                if report_id:
                    processed.add(str(report_id))
            except json.JSONDecodeError:
                continue
    return processed


def append_prediction(result_path: Path, serializable: dict):
    """Append a single prediction to the results file (incremental persistence)."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def main(limit: int | None = None, resume: bool = False, overwrite: bool = False):
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
    gold, unlabeled = split_gold(df)
    analysis = analyze_gold(gold)
    gold_examples = format_gold_examples(gold)
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
    result_path = output_dir / "results/inference/predictions.jsonl"
    trace_path = output_dir / "results/inference/model_trace.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle resume/overwrite logic
    if overwrite:
        if result_path.exists():
            result_path.unlink()
        if trace_path.exists():
            trace_path.unlink()
        processed_ids = set()
        print("Overwrite mode: starting fresh")
    elif resume:
        processed_ids = load_existing_predictions(result_path)
        print(f"Resume mode: skipping {len(processed_ids)} already-processed reports")
    else:
        if result_path.exists():
            result_path.unlink()
        if trace_path.exists():
            trace_path.unlink()
        processed_ids = set()
        print("Fresh run: starting from scratch")

    # Held-out evaluation before full inference
    heldout_path = exp / "data/validation/heldout.csv"
    if heldout_path.exists():
        heldout_df = load_train(heldout_path)
        run_heldout_evaluation(
            heldout_df, gold, retriever, model1, model2,
            inf_cfg, val_cfg, labels, cfg, trace_path,
        )
    else:
        print("No held-out set found. Run validate_agent.py first.")

    output_dir = root / cfg["runtime"].get("output_dir", "experiments/ragset_report_inference_experiment")
    result_path = output_dir / "results/inference/predictions.jsonl"
    trace_path = output_dir / "results/inference/model_trace.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    infer_fn = make_infer_fn(model1, inf_cfg, trace_path, labels, label_profile_text(analysis), gold_examples)
    validate_fn = make_validate_fn(model2, val_cfg, trace_path, labels, label_profile_text(analysis))

    processed_count = 0
    for idx, (_, row) in enumerate(unlabeled.iterrows()):
        report_id = str(row["StudyInstanceUID"])
        report = str(row["Report"])

        # Skip if already processed (resume mode)
        if resume and report_id in processed_ids:
            print(f"  Skipping {report_id} (already processed)")
            continue

        if limit is not None and len(processed_ids) + processed_count >= limit:
            break

        report_id = str(row["StudyInstanceUID"])
        report = str(row["Report"])

        retrieved = retriever.retrieve(report)
        relevant = "\n---\n".join(
            f"STUDY {x['study_id']}\nLABELS: {x['labels']}\nREPORT:\n{x['report']}"
            for x in retrieved
        )

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
        serializable = {
            "report_id": result["report_id"],
            "status": result["status"],
            "review_reason": result["review_reason"],
            "attempts": [
                {
                    "attempt": a.number,
                    "prediction": a.prediction.model_dump(),
                    "validation": a.validation.model_dump(),
                    "elapsed_seconds": a.elapsed_seconds,
                }
                for a in result["attempts"]
            ],
            "final_prediction": pred.model_dump(),
        }

        # Incremental persistence: write immediately after each prediction
        append_prediction(result_path, serializable)
        processed_ids.add(report_id)
        processed_count += 1

        print(f"  Completed {report_id} ({processed_count} total) - status: {result['status']}")

        # 2-second wait between reports to be nice to the API
        time.sleep(2)

    print(f"Inference complete: {processed_count} reports processed")
    print(result_path)
    print(trace_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RagSet report inference with resume/overwrite support."
    )
    parser.add_argument(
        "limit",
        nargs="?",
        type=int,
        default=None,
        help="Maximum number of reports to process (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing predictions.jsonl, skipping already-processed reports",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results and start fresh",
    )
    args = parser.parse_args()

    main(limit=args.limit, resume=args.resume, overwrite=args.overwrite)