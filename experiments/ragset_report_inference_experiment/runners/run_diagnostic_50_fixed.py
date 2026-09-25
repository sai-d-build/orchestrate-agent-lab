#!/usr/bin/env python3
"""
Run the LangGraph pipeline on 50 diagnostic reports and save complete diagnostic traces.

Uses the LangGraph orchestration with Model 1 (Inference), Model 2 (Critic), Model 3 (Judge).

FIXES APPLIED:
1. Checkpoint/Resume - Load processed StudyInstanceUIDs from results file to skip already processed
2. Prompt hashes from traces - Extract from trace records (already correct in traces)
3. Build attempts from state histories - Use prediction_history, critique_history, judgment_history
4. Derive retry_count from state - Count unique inference attempts
5. Sum latency from traces - Sum all trace latencies
6. Aggregate token usage from traces - Sum input/output tokens from all traces
7. Extract model predictions from state histories - Get full prediction objects
8. Parse policy flags from critiques - Extract UNRESOLVED_POLICY, REPORT_AMBIGUITY issue types
9. Better error handling - Per-record timeout, progress logging to file
"""

import json
import time
import yaml
import os
import hashlib
import signal
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from experiments.ragset_report_inference_experiment.src.ragset_inference.policy_loader import get_canonical_policy_text
from experiments.ragset_report_inference_experiment.src.ragset_inference.data import load_train, split_gold, LABEL_COLUMNS
from experiments.ragset_report_inference_experiment.src.ragset_inference.gold import format_gold_examples
from experiments.ragset_report_inference_experiment.src.ragset_inference.retrieval import GoldRetriever
from experiments.ragset_report_inference_experiment.src.ragset_inference.models import InferenceModel, CriticModel, JudgeModel
from experiments.ragset_report_inference_experiment.src.ragset_inference.graph import build_graph, create_initial_state
from experiments.ragset_report_inference_experiment.src.ragset_inference.trace import write_trace
from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import ReportPrediction, CritiqueResult, JudgeResult, LABEL_KEYS, CritiqueIssueType


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


class TimeoutException(Exception):
    """Custom exception for timeout handling."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Per-record timeout exceeded")


def load_processed_uids(results_path: Path) -> set:
    """Load already processed StudyInstanceUIDs from results file for checkpoint/resume."""
    processed_uids = set()
    if results_path.exists():
        with results_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    uid = data.get("StudyInstanceUID")
                    if uid:
                        processed_uids.add(uid)
                except json.JSONDecodeError:
                    continue
    return processed_uids


def build_attempts_from_state(state: dict) -> list:
    """Build attempts_detail from state histories (prediction_history, critique_history, judgment_history)."""
    prediction_history = state.get("prediction_history", [])
    critique_history = state.get("critique_history", [])
    judgment_history = state.get("judgment_history", [])
    current_prediction = state.get("current_prediction")
    current_critique = state.get("current_critique")
    current_judgment = state.get("current_judgment")
    
    # Build complete lists including current
    all_predictions = prediction_history + ([current_prediction] if current_prediction else [])
    all_critiques = critique_history + ([current_critique] if current_critique else [])
    all_judgments = judgment_history + ([current_judgment] if current_judgment else [])
    
    attempts_detail = []
    max_len = max(len(all_predictions), len(all_critiques), len(all_judgments))
    
    for attempt_num in range(1, max_len + 1):
        idx = attempt_num - 1
        
        # Get prediction for this attempt
        pred = all_predictions[idx] if idx < len(all_predictions) else None
        crit = all_critiques[idx] if idx < len(all_critiques) else None
        judg = all_judgments[idx] if idx < len(all_judgments) else None
        
        # Extract prediction dict
        prediction_dict = {}
        if pred:
            prediction_dict = {
                "response_hash": None,  # Not stored in state, would need trace
                "labels": {label: pred.predictions[label].value for label in pred.predictions},
                "evidence": {label: pred.predictions[label].evidence for label in pred.predictions},
            }
        
        # Extract validation dict
        validation_dict = {}
        if crit:
            validation_dict = {
                "response_hash": None,
                "status": crit.status.value if hasattr(crit.status, 'value') else str(crit.status),
                "issues": [
                    {
                        "label": issue.label,
                        "model1_value": issue.model1_value,
                        "proposed_value": issue.proposed_value,
                        "issue_type": issue.issue_type.value if hasattr(issue.issue_type, 'value') else str(issue.issue_type),
                        "evidence": issue.evidence,
                        "policy_rule": issue.policy_rule,
                        "feedback": issue.feedback,
                    }
                    for issue in crit.issues
                ],
                "actionable": crit.actionable,
                "affected_labels": crit.affected_labels,
            }
        
        # Determine failure type from judge
        failure_type = None
        if judg:
            action = judg.action.value if hasattr(judg.action, 'value') else str(judg.action)
            if action == "RETRY_MODEL1":
                failure_type = "semantic"
            elif action in ("AMBIGUOUS", "NEEDS_REVIEW", "STOP"):
                failure_type = "policy_ambiguity"
        
        attempts_detail.append({
            "attempt": attempt_num,
            "prediction": prediction_dict,
            "validation": validation_dict,
            "label_reviews": [],
            "elapsed_seconds": 0.0,  # Would need trace data
            "failure_type": failure_type,
        })
    
    return attempts_detail


def extract_policy_flags_from_state(state: dict) -> tuple:
    """Extract policy conventions used and unresolved policy flags from state critiques."""
    policy_conventions_used = []
    unresolved_policy_flags = []
    
    critique_history = state.get("critique_history", [])
    current_critique = state.get("current_critique")
    all_critiques = critique_history + ([current_critique] if current_critique else [])
    
    for crit in all_critiques:
        if not crit:
            continue
        for issue in crit.issues:
            issue_type = issue.issue_type.value if hasattr(issue.issue_type, 'value') else str(issue.issue_type)
            if issue_type in ("UNRESOLVED_POLICY", "REPORT_AMBIGUITY"):
                unresolved_policy_flags.append({
                    "label": issue.label,
                    "issue_type": issue_type,
                    "feedback": issue.feedback,
                })
            elif issue_type in ("CLEAR_POLICY_CONFLICT", "CLEAR_REPORT_CONFLICT"):
                policy_conventions_used.append({
                    "label": issue.label,
                    "issue_type": issue_type,
                    "policy_rule": issue.policy_rule,
                })
    
    return policy_conventions_used, unresolved_policy_flags


def aggregate_metadata_from_state(state: dict) -> dict:
    """Aggregate all metadata from state trace records."""
    trace_records = state.get("trace_records", [])
    
    # Convert to dict if needed
    trace_dicts = []
    for trace in trace_records:
        if hasattr(trace, 'model_dump'):
            trace_dicts.append(trace.model_dump())
        else:
            trace_dicts.append(trace)
    
    # Filter only successful traces with valid attempts
    valid_traces = [t for t in trace_dicts if t.get("attempt", 0) > 0 and t.get("status") == "success"]
    
    # 1. Prompt hashes from traces (already correct - full rendered prompt hashed)
    inference_traces = [t for t in valid_traces if t.get("graph_node") == "inference"]
    critic_traces = [t for t in valid_traces if t.get("graph_node") == "critic"]
    judge_traces = [t for t in valid_traces if t.get("graph_node") == "judge"]
    
    prompt_hash_inference = inference_traces[0].get("prompt_hash", "") if inference_traces else ""
    prompt_hash_validation = critic_traces[0].get("prompt_hash", "") if critic_traces else ""
    prompt_hash_judge = judge_traces[0].get("prompt_hash", "") if judge_traces else ""
    
    # 2. retry_count from traces - count unique inference attempts
    inference_attempts = set(t.get("attempt") for t in inference_traces)
    retry_count = max(0, len(inference_attempts) - 1)
    
    # 3. latency_seconds - sum all trace latencies
    latency_seconds = sum(t.get("latency_seconds", 0) or 0 for t in valid_traces)
    
    # 4. token_usage - aggregate from all traces
    input_tokens = sum(t.get("input_tokens", 0) or 0 for t in valid_traces)
    output_tokens = sum(t.get("output_tokens", 0) or 0 for t in valid_traces)
    token_usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    
    return {
        "prompt_hash_inference": prompt_hash_inference[:16] if prompt_hash_inference else "",
        "prompt_hash_validation": prompt_hash_validation[:16] if prompt_hash_validation else "",
        "prompt_hash_judge": prompt_hash_judge[:16] if prompt_hash_judge else "",
        "retry_count": retry_count,
        "latency_seconds": latency_seconds,
        "token_usage": token_usage,
    }


def run_diagnostic_50():
    """Run diagnostic on 50 selected reports using LangGraph orchestration."""
    root = Path(__file__).resolve().parents[3]
    exp = root / "experiments/ragset_report_inference_experiment"
    
    # Load configs
    cfg = yaml.safe_load((exp / "config/experiment.yaml").read_text(encoding="utf-8"))
    labels_cfg = yaml.safe_load((exp / "config/labels.yaml").read_text(encoding="utf-8"))
    
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
    judge_profile = cfg["models"]["judge_profiles"][0]
    
    inf_config = models_cfg["models"][inf_profile]
    val_config = models_cfg["models"][val_profile]
    judge_config = models_cfg["models"][judge_profile]
    
    inf_provider = inf_config.get("provider", "openrouter")
    val_provider = val_config.get("provider", "openrouter")
    judge_provider = judge_config.get("provider", "openrouter")
    inf_api_key_env = inf_config.get("api_key_env", "OPENROUTER_API_KEY")
    val_api_key_env = val_config.get("api_key_env", "OPENROUTER_API_KEY")
    judge_api_key_env = judge_config.get("api_key_env", "OPENROUTER_API_KEY")
    
    if not os.environ.get(inf_api_key_env):
        raise RuntimeError(f"{inf_api_key_env} is required for provider '{inf_provider}'")
    if not os.environ.get(val_api_key_env):
        raise RuntimeError(f"{val_api_key_env} is required for provider '{val_provider}'")
    if not os.environ.get(judge_api_key_env):
        raise RuntimeError(f"{judge_api_key_env} is required for provider '{judge_provider}'")
    
    model1 = InferenceModel(
        inf_config["model"],
        max_tokens=inf_config.get("parameters", {}).get("max_output_tokens", 4000),
        provider=inf_provider,
        api_key=os.environ.get(inf_api_key_env),
        reasoning=inf_config.get("parameters", {}).get("reasoning"),
    )
    model2 = CriticModel(
        val_config["model"],
        max_tokens=val_config.get("parameters", {}).get("max_output_tokens", 4000),
        provider=val_provider,
        api_key=os.environ.get(val_api_key_env),
        reasoning=val_config.get("parameters", {}).get("reasoning"),
    )
    model3 = JudgeModel(
        judge_config["model"],
        max_tokens=judge_config.get("parameters", {}).get("max_output_tokens", 8000),
        provider=judge_provider,
        api_key=os.environ.get(judge_api_key_env),
        reasoning=judge_config.get("parameters", {}).get("reasoning"),
    )
    
    # Load diagnostic 50 reports
    diagnostic_path = exp / "data/validation/diagnostic_50.csv"
    diagnostic_df = pd.read_csv(diagnostic_path)
    
    # Output paths
    output_dir = exp / "results/validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "diagnostic_50_trace.jsonl"
    results_path = output_dir / "diagnostic_50_results.jsonl"
    progress_path = output_dir / "diagnostic_50_progress.jsonl"
    
    # Checkpoint/Resume: Load already processed UIDs
    processed_uids = load_processed_uids(results_path)
    print(f"Checkpoint/Resume: Found {len(processed_uids)} already processed records")
    
    # Build LangGraph
    graph = build_graph()
    
    max_attempts = cfg["loop"]["max_attempts"]
    
    all_results = []
    
    print(f"Running diagnostic on {len(diagnostic_df)} reports with LangGraph orchestration...")
    
    for i, row in diagnostic_df.iterrows():
        report_id = str(row["StudyInstanceUID"])
        report = str(row["Report"])
        boundary_category = row.get("primary_boundary_category", "unknown")
        all_categories = row.get("all_boundary_categories", "unknown")
        
        # Checkpoint/Resume: Skip already processed
        if report_id in processed_uids:
            print(f"\n[{i+1}/{len(diagnostic_df)}] Skipping {report_id} (already processed)")
            continue
        
        print(f"\n[{i+1}/{len(diagnostic_df)}] Processing {report_id} (category: {boundary_category})")
        
        # Progress logging
        progress_record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "report_id": report_id,
            "index": i + 1,
            "total": len(diagnostic_df),
            "status": "started",
        }
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(progress_record, ensure_ascii=False) + "\n")
        
        # Set up timeout handler (10 minutes per record)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)  # 10 minutes
        
        # Retrieve gold examples
        retrieved = retriever.retrieve(report)
        relevant = "\n---\n".join(
            f"STUDY {x['study_id']}\nLABELS: {x['labels']}\nREPORT:\n{x['report']}"
            for x in retrieved
        )
        retrieved_study_ids = [x['study_id'] for x in retrieved]
        
        # Create initial LangGraph state
        initial_state = create_initial_state(
            study_instance_uid=report_id,
            report=report,
            max_attempts=max_attempts,
        )
        initial_state["canonical_policy"] = gold_analysis_text
        initial_state["retrieved_gold_context"] = relevant
        
        # Run the LangGraph pipeline using graph.invoke()
        try:
            final_state = graph.invoke(
                initial_state,
                config={"configurable": {
                    "retriever": retriever,
                    "inference_model": model1,
                    "critic_model": model2,
                    "judge_model": model3,
                }}
            )
            
            # Disable alarm
            signal.alarm(0)
            
            # Get trace records from state
            trace_records = final_state.get("trace_records", [])
            
            # Write traces to file
            for trace in trace_records:
                trace_dict = trace.model_dump() if hasattr(trace, 'model_dump') else trace
                write_trace(
                    trace_path,
                    report_id=trace_dict.get("study_instance_uid"),
                    stage=trace_dict.get("stage"),
                    model=trace_dict.get("requested_model"),
                    attempt=trace_dict.get("attempt"),
                    prompt="",  # Prompt not stored in TraceRecord
                    status=trace_dict.get("status", "success"),
                    latency_seconds=trace_dict.get("latency_seconds"),
                    input_tokens=trace_dict.get("input_tokens"),
                    output_tokens=trace_dict.get("output_tokens"),
                    error=trace_dict.get("error"),
                    provider=trace_dict.get("provider"),
                    actual_model=trace_dict.get("actual_model"),
                    graph_node=trace_dict.get("graph_node"),
                    judge_action=trace_dict.get("judge_action"),
                    response_hash=trace_dict.get("response_hash"),
                )
            
            # Extract results from state
            final_status = final_state.get("final_status", "error")
            final_prediction = final_state.get("final_prediction")
            review_reason = final_state.get("review_reason")
            selected_attempt = final_state.get("finalization_selected_attempt")
            finalization_reason = final_state.get("finalization_reason")
            terminal_action = final_state.get("finalization_terminal_action")
            terminal_reason_code = final_state.get("finalization_terminal_reason_code")
            
            # Build result record with metadata from state
            final_pred = final_prediction
            if final_pred:
                inferred_evidence = {
                    label: final_pred.predictions[label].evidence
                    for label in final_pred.predictions
                    if final_pred.predictions[label].evidence
                }
                final_labels = {label: final_pred.predictions[label].value for label in final_pred.predictions}
            else:
                inferred_evidence = {}
                final_labels = {}
            
            # Build attempts detail from state histories
            attempts_detail = build_attempts_from_state(final_state)
            
            # Extract policy flags from state
            policy_conventions_used, unresolved_policy_flags = extract_policy_flags_from_state(final_state)
            
            # Aggregate metadata from state traces
            metadata = aggregate_metadata_from_state(final_state)
            
            # Determine failure category
            failure_category = "NONE"
            if final_status == "needs_review":
                review_reason_str = review_reason or ""
                if "maximum attempts" in review_reason_str.lower() and "semantic" in review_reason_str.lower():
                    failure_category = "MODEL1_STUCK"
                elif "oscillation" in review_reason_str.lower() or "stuck" in review_reason_str.lower():
                    failure_category = "VALIDATOR_INSTABILITY"
                elif "ambiguous" in review_reason_str.lower() or "AMBIGUOUS" in review_reason_str:
                    failure_category = "TRUE_AMBIGUITY"
                elif "structural" in review_reason_str.lower() or "transient" in review_reason_str.lower():
                    failure_category = "SYSTEM_ERROR"
                else:
                    failure_category = "POLICY_AMBIGUITY"
            elif final_status == "error":
                failure_category = "SYSTEM_ERROR"
            elif len([t for t in trace_records if t.get("graph_node") == "inference"]) > 1:
                failure_category = "VALIDATOR_CORRECTION"
            
            # Build comprehensive result record
            result_record = {
                "StudyInstanceUID": report_id,
                "report_hash": compute_hash(report),
                "boundary_category": boundary_category,
                "all_boundary_categories": all_categories,
                "final_labels": final_labels,
                "final_evidence": inferred_evidence,
                "model1_prediction": {},  # Would need trace response_hash
                "model2_validation": {},
                "final_status": final_status,
                "review_reason": review_reason,
                "retry_count": metadata["retry_count"],
                "failure_category": failure_category,
                "policy_conventions_used": policy_conventions_used,
                "unresolved_policy_flags": unresolved_policy_flags,
                "retrieved_gold_study_ids": retrieved_study_ids,
                "requested_model_inference": model1.model,
                "requested_model_validation": model2.model,
                "requested_model_judge": model3.model,
                "actual_model_inference": getattr(model1, 'last_actual_model', model1.model),
                "actual_model_validation": getattr(model2, 'last_actual_model', model2.model),
                "actual_model_judge": getattr(model3, 'last_actual_model', model3.model),
                "provider_inference": model1.provider,
                "provider_validation": model2.provider,
                "provider_judge": model3.provider,
                "latency_seconds": metadata["latency_seconds"],
                "token_usage": metadata["token_usage"],
                "prompt_hash_inference": metadata["prompt_hash_inference"],
                "prompt_hash_validation": metadata["prompt_hash_validation"],
                "prompt_hash_judge": metadata["prompt_hash_judge"],
                "gold_analysis_hash": compute_hash(gold_analysis_text),
                "attempts": attempts_detail,
                "finalization_selected_attempt": selected_attempt,
                "finalization_reason": finalization_reason,
                "finalization_terminal_action": terminal_action,
                "finalization_terminal_reason_code": terminal_reason_code,
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            }
            
            all_results.append(result_record)
            
            # Write result incrementally
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            
            # Update progress
            progress_record = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "report_id": report_id,
                "index": i + 1,
                "total": len(diagnostic_df),
                "status": "completed",
                "final_status": final_status,
                "retry_count": metadata["retry_count"],
                "failure_category": failure_category,
            }
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(progress_record, ensure_ascii=False) + "\n")
            
            print(f"  Status: {final_status}, Retries: {metadata['retry_count']}, Failure: {failure_category}")
            print(f"  Latency: {metadata['latency_seconds']:.2f}s, Tokens: {metadata['token_usage']}")
            print(f"  Selected attempt: {selected_attempt}, Reason: {finalization_reason}")
            
        except TimeoutException:
            signal.alarm(0)
            print(f"  TIMEOUT: Record processing exceeded 10 minutes")
            error_record = {
                "StudyInstanceUID": report_id,
                "report_hash": compute_hash(report),
                "boundary_category": boundary_category,
                "all_boundary_categories": all_categories,
                "error": "Timeout: Processing exceeded 10 minutes per record",
                "final_status": "error",
                "failure_category": "SYSTEM_ERROR",
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            }
            all_results.append(error_record)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            
            progress_record = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "report_id": report_id,
                "index": i + 1,
                "total": len(diagnostic_df),
                "status": "timeout",
            }
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(progress_record, ensure_ascii=False) + "\n")
            
        except Exception as e:
            signal.alarm(0)
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
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
            
            progress_record = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "report_id": report_id,
                "index": i + 1,
                "total": len(diagnostic_df),
                "status": "error",
                "error": str(e),
            }
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(progress_record, ensure_ascii=False) + "\n")
        
        # Rate limiting
        time.sleep(2)
    
    print(f"\nDiagnostic complete. Results saved to {results_path}")
    print(f"Traces saved to {trace_path}")
    print(f"Progress saved to {progress_path}")
    
    return all_results


if __name__ == "__main__":
    run_diagnostic_50()