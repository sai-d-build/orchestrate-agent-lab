from pathlib import Path
import json
import os
import time
import yaml
import argparse
import queue
from concurrent.futures import ThreadPoolExecutor
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


def make_infer_fn(model, inf_cfg, trace_collector, labels, gold_analysis):
    """Factory that creates an infer closure with captured dependencies.

    gold_examples is now passed per-call via kwargs to ensure target-specific
    retrieval context is used.

    trace_collector: if provided (list), trace records are appended to it instead of writing to file.
                     if None, traces are written directly via write_trace (sequential mode).
    """

    def infer_fn(**kwargs):
        user = inf_cfg["user_prompt_template"].format(
            original_report=kwargs["report"],
            gold_analysis=gold_analysis,
            retrieved_gold_examples=kwargs.get("gold_examples", ""),
        )
        report_id = kwargs["report_id"]
        # Extract validator_feedback for retry awareness (NOT as evidence)
        validator_feedback = kwargs.get("validator_feedback")
        try:
            prediction = model.run(inf_cfg["system_prompt"], user, validator_feedback=validator_feedback)
            # Get actual model from response if available (OpenRouter returns routed model)
            actual_model = getattr(model, 'last_actual_model', model.model)
            trace_record = {
                "report_id": report_id,
                "stage": "inference",
                "model": model.model,
                "attempt": kwargs["attempt"],
                "prompt": inf_cfg["system_prompt"] + "\n" + user,
                "status": "success",
                "provider": model.provider,
                "actual_model": actual_model,
            }
            if trace_collector is not None:
                trace_collector.append(trace_record)
            else:
                # Sequential mode - write directly (requires trace_path global or similar)
                pass  # Not used in parallel mode
            return prediction
        except Exception as e:
            trace_record = {
                "report_id": report_id,
                "stage": "inference",
                "model": model.model,
                "attempt": kwargs["attempt"],
                "prompt": inf_cfg["system_prompt"] + "\n" + user,
                "status": "error",
                "provider": model.provider,
                "error": str(e),
            }
            if trace_collector is not None:
                trace_collector.append(trace_record)
            raise

    return infer_fn


def make_validate_fn(model, val_cfg, trace_collector, labels, gold_analysis):
    """Factory that creates a validate closure with captured dependencies.

    trace_collector: if provided (list), trace records are appended to it instead of writing to file.
                     if None, traces are written directly via write_trace (sequential mode).
    """

    def validate_fn(**kwargs):
        user = val_cfg["user_prompt_template"].format(
            original_report=kwargs["report"],
            gold_analysis=gold_analysis,
            model_1_prediction=kwargs["prediction"].model_dump_json(),
            retrieved_gold_examples="",
        )
        report_id = kwargs["report_id"]
        try:
            validation = model.run(val_cfg["system_prompt"], user)
            actual_model = getattr(model, 'last_actual_model', model.model)
            trace_record = {
                "report_id": report_id,
                "stage": "validation",
                "model": model.model,
                "attempt": kwargs["attempt"],
                "prompt": val_cfg["system_prompt"] + "\n" + user,
                "status": "success",
                "provider": model.provider,
                "actual_model": actual_model,
            }
            if trace_collector is not None:
                trace_collector.append(trace_record)
            return validation
        except Exception as e:
            trace_record = {
                "report_id": report_id,
                "stage": "validation",
                "model": model.model,
                "attempt": kwargs["attempt"],
                "prompt": val_cfg["system_prompt"] + "\n" + user,
                "status": "error",
                "provider": model.provider,
                "error": str(e),
            }
            if trace_collector is not None:
                trace_collector.append(trace_record)
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
        # Use trace collectors for heldout eval (sequential, but with trace collection)
        infer_traces = []
        validate_traces = []
        infer_fn = make_infer_fn(model1, inf_cfg, infer_traces, labels, context["gold_analysis"])
        validate_fn = make_validate_fn(model2, val_cfg, validate_traces, labels, context["gold_analysis"])
        result = run_report(
            report_id=report_id,
            report=report,
            infer=infer_fn,
            validate=validate_fn,
            context=context,
            max_attempts=cfg["loop"]["max_attempts"],
            labels=labels,
        )
        # Write collected traces
        for trace in infer_traces:
            write_trace(trace_path, **trace)
        for trace in validate_traces:
            write_trace(trace_path, **trace)
        
        pred = result["final_prediction"]
        # Extract inferred evidence from final prediction
        inferred_evidence = {
            label: pred.predictions[label].evidence
            for label in pred.predictions
            if pred.predictions[label].evidence
        }
        predictions.append({
            "StudyInstanceUID": result["report_id"],
            **pred.to_dict(),
            "Report": report,
            "original_report": report,
            "inferred_evidence": inferred_evidence,
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


def worker_process_chunk(
    reports_chunk: list,
    model1: InferenceModel,
    model2: ValidatorModel,
    inf_cfg: dict,
    val_cfg: dict,
    labels: list[str],
    gold_analysis_text: str,
    retriever: GoldRetriever,
    max_attempts: int,
    result_queue: queue.Queue,
    worker_id: int,
):
    """Process a chunk of reports in a worker thread.

    Each report goes through the full inference+validation loop.
    Results and traces are sent to result_queue for the main thread to write.
    """
    # Create per-worker trace collectors
    infer_traces = []
    validate_traces = []
    
    # Create per-worker closures with trace collectors
    infer_fn = make_infer_fn(model1, inf_cfg, infer_traces, labels, gold_analysis_text)
    validate_fn = make_validate_fn(model2, val_cfg, validate_traces, labels, gold_analysis_text)

    for idx, row in reports_chunk:
        report_id = str(row["StudyInstanceUID"])
        report = str(row["Report"])

        # Thread-safe retrieval
        retrieved = retriever.retrieve(report)
        relevant = "\n---\n".join(
            f"STUDY {x['study_id']}\nLABELS: {x['labels']}\nREPORT:\n{x['report']}"
            for x in retrieved
        )
        context = {
            "gold_analysis": gold_analysis_text,
            "gold_examples": relevant,
        }

        try:
            result = run_report(
                report_id=report_id,
                report=report,
                infer=infer_fn,
                validate=validate_fn,
                context=context,
                max_attempts=max_attempts,
                labels=labels,
            )

            # Serialize prediction
            pred = result["final_prediction"]
            inferred_evidence = {
                label: pred.predictions[label].evidence
                for label in pred.predictions
                if pred.predictions[label].evidence
            }
            serializable = {
                "report_id": result["report_id"],
                "status": result["status"],
                "review_reason": result["review_reason"],
                "original_report": report,
                "inferred_evidence": inferred_evidence,
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

            # Queue prediction for writer
            result_queue.put({"type": "prediction", "data": serializable})

            # Queue trace records from collectors
            for trace in infer_traces:
                if trace["report_id"] == report_id:
                    result_queue.put({"type": "trace", "data": trace})
            for trace in validate_traces:
                if trace["report_id"] == report_id:
                    result_queue.put({"type": "trace", "data": trace})

            result_queue.put({"type": "progress", "report_id": report_id})

        except Exception as e:
            # Queue error for writer/main thread handling
            result_queue.put({
                "type": "error",
                "report_id": report_id,
                "error": str(e),
                "worker_id": worker_id,
            })

        # Per-worker rate limiting
        time.sleep(2)


def main(limit: int | None = None, resume: bool = False, overwrite: bool = False, retry_review: bool = False):
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

    # Get profile lists from config (support both old single-profile and new multi-profile format)
    inf_profiles = cfg["models"].get("inference_profiles", [cfg["models"].get("inference_profile")])
    val_profiles = cfg["models"].get("validator_profiles", [cfg["models"].get("validator_profile")])

    if len(inf_profiles) != 2 or len(val_profiles) != 2:
        raise ValueError(
            f"Expected 2 inference profiles and 2 validator profiles for parallel inference. "
            f"Got {len(inf_profiles)} inference and {len(val_profiles)} validator profiles."
        )

    # Validate all profiles exist
    for profile_name in inf_profiles + val_profiles:
        if profile_name not in models_cfg.get("models", {}):
            raise ValueError(
                f"Unknown profile '{profile_name}'. "
                f"Available: {list(models_cfg.get('models', {}).keys())}"
            )

    # Load configurations for both workers
    inf_configs = [models_cfg["models"][p] for p in inf_profiles]
    val_configs = [models_cfg["models"][p] for p in val_profiles]

    # Check for required API keys for both workers
    for i, (inf_config, val_config) in enumerate(zip(inf_configs, val_configs)):
        inf_provider = inf_config.get("provider", "openrouter")
        val_provider = val_config.get("provider", "openrouter")
        inf_api_key_env = inf_config.get("api_key_env", "OPENROUTER_API_KEY")
        val_api_key_env = val_config.get("api_key_env", "OPENROUTER_API_KEY")

        if not os.environ.get(inf_api_key_env):
            raise RuntimeError(f"{inf_api_key_env} is required for provider '{inf_provider}' (worker {i}).")
        if not os.environ.get(val_api_key_env):
            raise RuntimeError(f"{val_api_key_env} is required for provider '{val_provider}' (worker {i}).")

    # Create model pairs for both workers
    model_pairs = []
    for inf_config, val_config in zip(inf_configs, val_configs):
        inf_provider = inf_config.get("provider", "openrouter")
        val_provider = val_config.get("provider", "openrouter")
        inf_api_key_env = inf_config.get("api_key_env", "OPENROUTER_API_KEY")
        val_api_key_env = val_config.get("api_key_env", "OPENROUTER_API_KEY")

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
        model_pairs.append((model1, model2))

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

    # Held-out evaluation before full inference (uses first worker's models)
    heldout_path = exp / "data/validation/heldout.csv"
    if heldout_path.exists():
        heldout_df = load_train(heldout_path)
        run_heldout_evaluation(
            heldout_df, gold, retriever, model_pairs[0][0], model_pairs[0][1],
            inf_cfg, val_cfg, labels, cfg, trace_path,
        )
    else:
        print("No held-out set found. Run validate_agent.py first.")

    # Re-initialize paths (heldout eval may have created them)
    output_dir = root / cfg["runtime"].get("output_dir", "experiments/ragset_report_inference_experiment")
    result_path = output_dir / "results/inference/predictions.jsonl"
    trace_path = output_dir / "results/inference/model_trace.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    # --- PARALLEL INFERENCE SETUP ---

    # Collect all unlabeled reports as list of (idx, row) tuples
    all_unlabeled = list(unlabeled.iterrows())

    # Apply resume filter in main thread
    if resume:
        remaining = [(idx, row) for idx, row in all_unlabeled
                     if str(row["StudyInstanceUID"]) not in processed_ids]
    else:
        remaining = all_unlabeled

    # Apply retry-review filter if needed
    if resume and retry_review:
        # Find reports with needs_review status
        needs_review_ids = set()
        if result_path.exists():
            with result_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("status") == "needs_review":
                            rid = record.get("report_id") or record.get("StudyInstanceUID")
                            if rid:
                                needs_review_ids.add(str(rid))
                    except json.JSONDecodeError:
                        continue
        # Include needs_review reports in remaining if they were filtered out
        for idx, row in all_unlabeled:
            report_id = str(row["StudyInstanceUID"])
            if report_id in needs_review_ids and report_id not in [str(r[1]["StudyInstanceUID"]) for r in remaining]:
                remaining.append((idx, row))

    # Apply limit
    if limit is not None:
        remaining = remaining[:limit]

    # Round-robin split into 2 chunks
    chunk_a = remaining[::2]
    chunk_b = remaining[1::2]

    total_expected = len(remaining)
    print(f"Total reports to process: {total_expected}")
    print(f"  Worker A (API_KEY_1): {len(chunk_a)} reports")
    print(f"  Worker B (API_KEY_2): {len(chunk_b)} reports")

    if total_expected == 0:
        print("No reports to process.")
        return

    gold_analysis_text = label_profile_text(analysis)
    max_attempts = cfg["loop"]["max_attempts"]

    # Thread-safe queue for results
    result_queue = queue.Queue()

    # Submit workers
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(
            worker_process_chunk, chunk_a, model_pairs[0][0], model_pairs[0][1],
            inf_cfg, val_cfg, labels, gold_analysis_text,
            retriever, max_attempts, result_queue, 0
        )
        future_b = executor.submit(
            worker_process_chunk, chunk_b, model_pairs[1][0], model_pairs[1][1],
            inf_cfg, val_cfg, labels, gold_analysis_text,
            retriever, max_attempts, result_queue, 1
        )

        # Writer loop in main thread
        completed = 0
        errors = []

        while completed < total_expected:
            item = result_queue.get()  # Blocks until available

            if item["type"] == "prediction":
                append_prediction(result_path, item["data"])
                completed += 1

            elif item["type"] == "trace":
                write_trace(trace_path, **item["data"])

            elif item["type"] == "progress":
                print(f"  Completed {item['report_id']} ({completed}/{total_expected})")

            elif item["type"] == "error":
                errors.append(item)
                print(f"  ERROR in worker {item['worker_id']} for {item['report_id']}: {item['error']}")
                completed += 1  # Count as processed to avoid deadlock

        # Wait for workers to finish
        future_a.result()
        future_b.result()

    if errors:
        print(f"\nCompleted with {len(errors)} errors")
        for e in errors:
            print(f"  Worker {e['worker_id']}: {e['report_id']} - {e['error']}")
    else:
        print(f"\nInference complete: {completed} reports processed")
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
        help="Maximum number of NEW reports to process in this invocation (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing predictions.jsonl, skipping already-processed reports",
    )
    parser.add_argument(
        "--retry-review",
        action="store_true",
        help="With --resume, retry reports that previously ended in needs_review status",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results and start fresh",
    )
    args = parser.parse_args()

    main(limit=args.limit, resume=args.resume, overwrite=args.overwrite, retry_review=args.retry_review)