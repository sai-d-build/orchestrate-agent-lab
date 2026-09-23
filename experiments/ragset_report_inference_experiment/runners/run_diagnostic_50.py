#!/usr/bin/env python3
"""
Run the existing pipeline on 50 diagnostic reports and save complete diagnostic traces.

Uses the production inference path exactly as implemented:
- canonical policy loader
- startup validation gate
- Model 1
- Model 2
- retry loop
- AMBIGUOUS handling
- trace collection
"""

import json
import time
import yaml
import os
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from experiments.ragset_report_inference_experiment.src.ragset_inference.policy_loader import get_canonical_policy_text
from experiments.ragset_report_inference_experiment.src.ragset_inference.data import load_train, split_gold, LABEL_COLUMNS
from experiments.ragset_report_inference_experiment.src.ragset_inference.gold import format_gold_examples
from experiments.ragset_report_inference_experiment.src.ragset_inference.retrieval import GoldRetriever
from experiments.ragset_report_inference_experiment.src.ragset_inference.models import InferenceModel, ValidatorModel
from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import run_report
from experiments.ragset_report_inference_experiment.src.ragset_inference.trace import write_trace
from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import ReportPrediction, ValidationResult, ValidationIssue, LABEL_KEYS


def make_infer_fn(model, inf_cfg, trace_collector, labels, gold_analysis):
    """Factory that creates an infer closure with captured dependencies."""
    def infer_fn(**kwargs):
        template_vars = {
            "original_report": kwargs["report"],
            "gold_analysis": gold_analysis,
            "retrieved_gold_examples": kwargs.get("gold_examples", ""),
        }
        validator_feedback = kwargs.get("validator_feedback")
        if validator_feedback:
            feedback_json = json.dumps(validator_feedback, ensure_ascii=False, indent=2)
            template_vars["validator_feedback_section"] = (
                "VALIDATOR FEEDBACK (for reconsideration, NOT as evidence):\n"
                f"{feedback_json}\n\n"
                "Re-read the ORIGINAL_REPORT from scratch. Validator feedback identifies a disputed interpretation; "
                "it is not automatically ground truth. Recompute the affected labels using the ORIGINAL_REPORT "
                "and GOLD ANNOTATION POLICY. Do not blindly copy the validator correction."
            )
        else:
            template_vars["validator_feedback_section"] = ""
        
        user = inf_cfg["user_prompt_template"].format(**template_vars)
        report_id = kwargs["report_id"]
        attempt = kwargs["attempt"]
        start_time = time.time()
        
        try:
            prediction = model.run(inf_cfg["system_prompt"], user, validator_feedback=validator_feedback)
            actual_model = getattr(model, 'last_actual_model', model.model)
            elapsed = time.time() - start_time
            
            trace_record = {
                "report_id": report_id,
                "stage": "inference",
                "model": model.model,
                "attempt": attempt,
                "prompt": inf_cfg["system_prompt"] + "\n" + user,
                "status": "success",
                "provider": model.provider,
                "actual_model": actual_model,
                "latency_seconds": elapsed,
            }
            if trace_collector is not None:
                trace_collector[report_id] = trace_record
            return prediction
        except Exception as e:
            elapsed = time.time() - start_time
            trace_record = {
                "report_id": report_id,
                "stage": "inference",
                "model": model.model,
                "attempt": attempt,
                "prompt": inf_cfg["system_prompt"] + "\n" + user,
                "status": "error",
                "provider": model.provider,
                "error": str(e),
                "latency_seconds": elapsed,
            }
            if trace_collector is not None:
                trace_collector[report_id] = trace_record
            raise

    return infer_fn


def make_validate_fn(model, val_cfg, trace_collector, labels, gold_analysis):
    """Factory that creates a validate closure with captured dependencies."""
    def validate_fn(**kwargs):
        user = val_cfg["user_prompt_template"].format(
            original_report=kwargs["report"],
            gold_analysis=gold_analysis,
            model_1_prediction=kwargs["prediction"].model_dump_json(),
            retrieved_gold_examples=kwargs.get("gold_examples", ""),
        )
        report_id = kwargs["report_id"]
        attempt = kwargs["attempt"]
        start_time = time.time()
        
        try:
            validation = model.run(val_cfg["system_prompt"], user)
            actual_model = getattr(model, 'last_actual_model', model.model)
            elapsed = time.time() - start_time
            
            trace_record = {
                "report_id": report_id,
                "stage": "validation",
                "model": model.model,
                "attempt": attempt,
                "prompt": val_cfg["system_prompt"] + "\n" + user,
                "status": "success",
                "provider": model.provider,
                "actual_model": actual_model,
                "latency_seconds": elapsed,
            }
            if trace_collector is not None:
                trace_collector[report_id] = trace_record
            return validation
        except Exception as e:
            elapsed = time.time() - start_time
            trace_record = {
                "report_id": report_id,
                "stage": "validation",
                "model": model.model,
                "attempt": attempt,
                "prompt": val_cfg["system_prompt"] + "\n" + user,
                "status": "error",
                "provider": model.provider,
                "error": str(e),
                "latency_seconds": elapsed,
            }
            if trace_collector is not None:
                trace_collector[report_id] = trace_record
            raise

    return validate_fn


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def run_diagnostic_50():
    """Run diagnostic on 50 selected reports."""
    root = Path(__file__).resolve().parents[3]
    exp = root / "experiments/ragset_report_inference_experiment"
    
    # Load configs
    cfg = yaml.safe_load((exp / "config/experiment.yaml").read_text(encoding="utf-8"))
    labels_cfg = yaml.safe_load((exp / "config/labels.yaml").read_text(encoding="utf-8"))
    inf_cfg = yaml.safe_load((exp / "prompts/inference.yaml").read_text(encoding="utf-8"))
    val_cfg = yaml.safe_load((exp / "prompts/validation.yaml").read_text(encoding="utf-8"))
    
    labels = labels_cfg["labels"]
    
    # Load data
    df = load_train(root / cfg["source"]["path"])
    gold, unlabeled = split_gold(df)
    gold_examples = format_gold_examples(gold)
    retriever = GoldRetriever(gold, top_k=cfg["retrieval"]["top_k"])
    
    # Load canonical policy with validation gate
    gold_analysis_text = get_canonical_policy_text(
        csv_path=str(root / "train.csv"),
        gold_analysis_path=str(exp / "results/gold/gold_analysis.json"),
        policy_path=str(root / "config/ragset_label_policy.yaml")
    )
    
    # Load model profiles
    models_cfg = yaml.safe_load((root / "config" / "models.yaml").read_text(encoding="utf-8"))
    
    # Use first worker's profiles for diagnostic (sequential)
    inf_profile = cfg["models"]["inference_profiles"][0]
    val_profile = cfg["models"]["validator_profiles"][0]
    
    inf_config = models_cfg["models"][inf_profile]
    val_config = models_cfg["models"][val_profile]
    
    inf_provider = inf_config.get("provider", "openrouter")
    val_provider = val_config.get("provider", "openrouter")
    inf_api_key_env = inf_config.get("api_key_env", "OPENROUTER_API_KEY")
    val_api_key_env = val_config.get("api_key_env", "OPENROUTER_API_KEY")
    
    if not os.environ.get(inf_api_key_env):
        raise RuntimeError(f"{inf_api_key_env} is required for provider '{inf_provider}'")
    if not os.environ.get(val_api_key_env):
        raise RuntimeError(f"{val_api_key_env} is required for provider '{val_provider}'")
    
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
    
    # Load diagnostic 50 reports
    diagnostic_path = exp / "data/validation/diagnostic_50.csv"
    diagnostic_df = pd.read_csv(diagnostic_path)
    
    # Output paths
    output_dir = exp / "results/validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "diagnostic_50_trace.jsonl"
    results_path = output_dir / "diagnostic_50_results.jsonl"
    
    # Clear previous runs
    if trace_path.exists():
        trace_path.unlink()
    if results_path.exists():
        results_path.unlink()
    
    # Create trace collectors (dict keyed by report_id)
    infer_traces = {}
    validate_traces = {}
    
    # Create closures with trace collectors
    infer_fn = make_infer_fn(model1, inf_cfg, infer_traces, labels, gold_analysis_text)
    validate_fn = make_validate_fn(model2, val_cfg, validate_traces, labels, gold_analysis_text)
    
    max_attempts = cfg["loop"]["max_attempts"]
    
    all_results = []
    
    print(f"Running diagnostic on {len(diagnostic_df)} reports...")
    
    for i, row in diagnostic_df.iterrows():
        report_id = str(row["StudyInstanceUID"])
        report = str(row["Report"])
        boundary_category = row.get("primary_boundary_category", "unknown")
        all_categories = row.get("all_boundary_categories", "unknown")
        
        print(f"\n[{i+1}/{len(diagnostic_df)}] Processing {report_id} (category: {boundary_category})")
        
        # Retrieve gold examples
        retrieved = retriever.retrieve(report)
        relevant = "\n---\n".join(
            f"STUDY {x['study_id']}\nLABELS: {x['labels']}\nREPORT:\n{x['report']}"
            for x in retrieved
        )
        retrieved_study_ids = [x['study_id'] for x in retrieved]
        
        context = {
            "gold_analysis": gold_analysis_text,
            "gold_examples": relevant,
        }
        
        # Run the report through the pipeline
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
            
            # Write collected traces for this report
            if report_id in infer_traces:
                write_trace(trace_path, **infer_traces[report_id])
            if report_id in validate_traces:
                write_trace(trace_path, **validate_traces[report_id])
            
            # Extract detailed attempt information
            attempts_detail = []
            for attempt in result["attempts"]:
                pred = attempt.prediction
                val = attempt.validation
                
                # Get label issues from validation (ValidationResult has 'issues' not 'label_reviews')
                label_reviews = []
                if hasattr(val, 'issues') and val.issues:
                    for issue in val.issues:
                        label_reviews.append({
                            "label": issue.label,
                            "predicted": issue.predicted,
                            "corrected": issue.corrected,
                            "reason": issue.reason,
                            "evidence": issue.evidence,
                        })
                
                attempts_detail.append({
                    "attempt": attempt.number,
                    "prediction": pred.model_dump(),
                    "validation": val.model_dump(),
                    "label_reviews": label_reviews,
                    "elapsed_seconds": attempt.elapsed_seconds,
                    "failure_type": attempt.failure_type.value if attempt.failure_type else None,
                })
            
            final_pred = result["final_prediction"]
            inferred_evidence = {
                label: final_pred.predictions[label].evidence
                for label in final_pred.predictions
                if final_pred.predictions[label].evidence
            }
            
            # Determine policy conventions used (from validation issues)
            policy_conventions_used = []
            unresolved_flags = []
            for attempt in result["attempts"]:
                if hasattr(attempt.validation, 'issues') and attempt.validation.issues:
                    for issue in attempt.validation.issues:
                        if issue.reason and ("UNRESOLVED" in issue.reason or "unresolved" in issue.reason.lower()):
                            unresolved_flags.append(f"{issue.label}: {issue.reason}")
                        if issue.reason and ("SAFE" in issue.reason or "UNSAFE" in issue.reason or "CONTEXTUAL" in issue.reason):
                            policy_conventions_used.append(f"{issue.label}: {issue.reason[:200]}")
            
            # Classify failure category
            failure_category = "NONE"
            if result["status"] == "needs_review":
                review_reason = result.get("review_reason", "")
                if "maximum attempts" in review_reason.lower() and "semantic" in review_reason.lower():
                    failure_category = "MODEL1_STUCK"
                elif "validator instability" in review_reason.lower():
                    failure_category = "VALIDATOR_INSTABILITY"
                elif "ambiguous" in review_reason.lower() or "AMBIGUOUS" in review_reason:
                    failure_category = "TRUE_AMBIGUITY"
                elif "structural" in review_reason.lower() or "transient" in review_reason.lower():
                    failure_category = "SYSTEM_ERROR"
                else:
                    failure_category = "POLICY_AMBIGUITY"
            elif result["status"] == "error":
                failure_category = "SYSTEM_ERROR"
            elif len(result["attempts"]) > 1:
                failure_category = "VALIDATOR_CORRECTION"
            
            # Build comprehensive result record
            result_record = {
                "StudyInstanceUID": report_id,
                "report_hash": compute_hash(report),
                "boundary_category": boundary_category,
                "all_boundary_categories": all_categories,
                "final_labels": {label: final_pred.predictions[label].value for label in final_pred.predictions},
                "final_evidence": inferred_evidence,
                "model1_prediction": result["attempts"][-1].prediction.model_dump() if result["attempts"] else {},
                "model2_validation": result["attempts"][-1].validation.model_dump() if result["attempts"] else {},
                "final_status": result["status"],
                "review_reason": result.get("review_reason"),
                "retry_count": len(result["attempts"]) - 1,
                "failure_category": failure_category,
                "policy_conventions_used": policy_conventions_used,
                "unresolved_policy_flags": unresolved_flags,
                "retrieved_gold_study_ids": retrieved_study_ids,
                "requested_model_inference": model1.model,
                "requested_model_validation": model2.model,
                "actual_model_inference": getattr(model1, 'last_actual_model', model1.model),
                "actual_model_validation": getattr(model2, 'last_actual_model', model2.model),
                "provider_inference": model1.provider,
                "provider_validation": model2.provider,
                "latency_seconds": sum(a.elapsed_seconds for a in result["attempts"]),
                "token_usage": {},  # Would need model support
                "prompt_hash_inference": compute_hash(inf_cfg["system_prompt"]),
                "prompt_hash_validation": compute_hash(val_cfg["system_prompt"]),
                "gold_analysis_hash": compute_hash(gold_analysis_text),
                "attempts": attempts_detail,
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            }
            
            all_results.append(result_record)
            
            # Write result incrementally
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            
            print(f"  Status: {result['status']}, Retries: {len(result['attempts'])-1}, Failure: {failure_category}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            error_record = {
                "StudyInstanceUID": report_id,
                "report_hash": compute_hash(report),
                "boundary_category": boundary_category,
                "all_boundary_categories": all_categories,
                "error": str(e),
                "final_status": "error",
                "failure_category": "SYSTEM_ERROR",
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            }
            all_results.append(error_record)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        
        # Rate limiting
        time.sleep(2)
    
    print(f"\nDiagnostic complete. Results saved to {results_path}")
    print(f"Traces saved to {trace_path}")
    
    return all_results


if __name__ == "__main__":
    run_diagnostic_50()