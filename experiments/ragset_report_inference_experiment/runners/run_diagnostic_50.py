#!/usr/bin/env python3
"""
Run the LangGraph pipeline on 50 diagnostic reports and save complete diagnostic traces.

Uses the LangGraph orchestration with Model 1 (Inference), Model 2 (Critic), Model 3 (Judge).
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
from experiments.ragset_report_inference_experiment.src.ragset_inference.models import InferenceModel, CriticModel, JudgeModel
from experiments.ragset_report_inference_experiment.src.ragset_inference.graph import build_graph, create_initial_state
from experiments.ragset_report_inference_experiment.src.ragset_inference.trace import write_trace
from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import ReportPrediction, CritiqueResult, JudgeResult, LABEL_KEYS


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


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
    
    # Clear previous runs
    if trace_path.exists():
        trace_path.unlink()
    if results_path.exists():
        results_path.unlink()
    
    # Build LangGraph
    graph = build_graph()
    
    # Load prompt configs for trace reconstruction
    inf_cfg = yaml.safe_load((exp / "prompts/inference.yaml").read_text(encoding="utf-8"))
    val_cfg = yaml.safe_load((exp / "prompts/validation_critic.yaml").read_text(encoding="utf-8"))
    judge_cfg = yaml.safe_load((exp / "prompts/judge.yaml").read_text(encoding="utf-8"))
    
    max_attempts = cfg["loop"]["max_attempts"]
    
    all_results = []
    
    print(f"Running diagnostic on {len(diagnostic_df)} reports with LangGraph orchestration...")
    
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
        
        # Create initial LangGraph state
        initial_state = create_initial_state(
            study_instance_uid=report_id,
            report=report,
            max_attempts=max_attempts,
        )
        # Add context that the graph nodes need
        initial_state["canonical_policy"] = gold_analysis_text
        initial_state["retrieved_gold_context"] = "\n---\n".join(
            f"STUDY {x['study_id']}\nLABELS: {x['labels']}\nREPORT:\n{x['report']}"
            for x in retrieved
        )
        
        # Create trace collectors for this report
        trace_records = []
        
        # Run the LangGraph pipeline
        try:
            # We need to inject the models and retriever into the graph nodes
            # The graph nodes expect these as kwargs, so we'll use a custom invoke
            from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import (
                initialize_node, inference_node, critic_node, judge_node, finalize_node
            )
            
            # Manually execute the graph nodes in sequence to capture traces
            # This mimics what graph.invoke() would do but with trace collection
            state = initial_state
            
            # Initialize
            state = initialize_node(state, retriever=retriever, inference_model=model1, critic_model=model2, judge_model=model3)
            
            # Run the inference-critic-judge loop
            for attempt_num in range(1, max_attempts + 1):
                state["attempt"] = attempt_num
                
                # Inference
                state = inference_node(state, inference_model=model1)
                
                # Critic
                state = critic_node(state, critic_model=model2)
                
                # Judge
                state = judge_node(state, judge_model=model3)
                
                # Check if we should continue or finalize
                judgment = state.get("current_judgment")
                if judgment:
                    action = judgment.action
                    if action.value == "RETRY_MODEL1" and state["attempt"] < state["max_attempts"]:
                        # Continue to next attempt
                        state["prediction_history"].append(state["current_prediction"])
                        state["critique_history"].append(state["current_critique"])
                        state["judgment_history"].append(state["current_judgment"])
                        continue
                    else:
                        # Finalize
                        state = finalize_node(state)
                        break
                else:
                    # No judgment, finalize
                    state = finalize_node(state)
                    break
            
            # Collect all trace records from the state
            trace_records = state.get("trace_records", [])
            for trace in trace_records:
                trace_dict = trace.model_dump() if hasattr(trace, 'model_dump') else trace
                # Map TraceRecord fields to write_trace parameters
                write_trace_kwargs = {
                    "report_id": trace_dict.get("study_instance_uid"),
                    "stage": trace_dict.get("stage"),
                    "model": trace_dict.get("requested_model"),
                    "attempt": trace_dict.get("attempt"),
                    "prompt": "",  # Prompt not stored in TraceRecord
                    "status": trace_dict.get("status", "success"),
                    "latency_seconds": trace_dict.get("latency_seconds"),
                    "input_tokens": trace_dict.get("input_tokens"),
                    "output_tokens": trace_dict.get("output_tokens"),
                    "error": trace_dict.get("error"),
                    "provider": trace_dict.get("provider"),
                    "actual_model": trace_dict.get("actual_model"),
                    "graph_node": trace_dict.get("graph_node"),
                    "judge_action": trace_dict.get("judge_action"),
                    "response_hash": trace_dict.get("response_hash"),
                }
                # Filter out None values
                write_trace_kwargs = {k: v for k, v in write_trace_kwargs.items() if v is not None}
                write_trace(trace_path, **write_trace_kwargs)
            
            # Extract results
            final_status = state.get("final_status", "error")
            final_prediction = state.get("final_prediction")
            review_reason = state.get("review_reason")
            attempts = state.get("attempts", [])
            
            # Build result record
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
            
            # Build attempts detail
            attempts_detail = []
            for attempt in state.get("attempts", []):
                pred = attempt.prediction
                val = attempt.validation
                
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
            
            # Determine failure category
            failure_category = "NONE"
            if final_status == "needs_review":
                review_reason = state.get("review_reason", "")
                if "maximum attempts" in review_reason.lower() and "semantic" in review_reason.lower():
                    failure_category = "MODEL1_STUCK"
                elif "oscillation" in review_reason.lower() or "stuck" in review_reason.lower():
                    failure_category = "VALIDATOR_INSTABILITY"
                elif "ambiguous" in review_reason.lower() or "AMBIGUOUS" in review_reason:
                    failure_category = "TRUE_AMBIGUITY"
                elif "structural" in review_reason.lower() or "transient" in review_reason.lower():
                    failure_category = "SYSTEM_ERROR"
                else:
                    failure_category = "POLICY_AMBIGUITY"
            elif final_status == "error":
                failure_category = "SYSTEM_ERROR"
            elif len(state.get("attempts", [])) > 1:
                failure_category = "VALIDATOR_CORRECTION"
            
            # Build comprehensive result record
            result_record = {
                "StudyInstanceUID": report_id,
                "report_hash": compute_hash(report),
                "boundary_category": boundary_category,
                "all_boundary_categories": all_categories,
                "final_labels": final_labels,
                "final_evidence": inferred_evidence,
                "model1_prediction": state["attempts"][-1].prediction.model_dump() if state.get("attempts") else {},
                "model2_validation": state["attempts"][-1].validation.model_dump() if state.get("attempts") else {},
                "final_status": final_status,
                "review_reason": review_reason,
                "retry_count": len(state.get("attempts", [])) - 1,
                "failure_category": failure_category,
                "policy_conventions_used": [],
                "unresolved_policy_flags": [],
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
                "latency_seconds": sum(a.elapsed_seconds for a in state.get("attempts", [])),
                "token_usage": {},
                "prompt_hash_inference": compute_hash(inf_cfg["system_prompt"]),
                "prompt_hash_validation": compute_hash(val_cfg["system_prompt"]),
                "prompt_hash_judge": compute_hash(judge_cfg["system_prompt"]),
                "gold_analysis_hash": compute_hash(gold_analysis_text),
                "attempts": attempts_detail,
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            }
            
            all_results.append(result_record)
            
            # Write result incrementally
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            
            print(f"  Status: {final_status}, Retries: {len(state.get('attempts', []))-1}, Failure: {failure_category}")
            
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