#!/usr/bin/env python3
"""
Generate diagnostic audit from the 50-report diagnostic run.

Produces:
- results/validation/diagnostic_50_report.json
- results/validation/diagnostic_50_report.csv

With classifications for each non-clean case as exactly one of:
- MODEL_ERROR
- VALIDATOR_ERROR
- POLICY_ERROR
- RETRIEVAL_ERROR
- GOLD_CONTRADICTION
- TRUE_AMBIGUITY
- UNRESOLVED
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import datetime


def classify_root_cause(record: dict) -> str:
    """Classify the root cause for a non-clean case."""
    status = record.get("final_status", "")
    failure_category = record.get("failure_category", "")
    review_reason = record.get("review_reason", "") or ""
    error = record.get("error", "") or ""
    
    # System errors
    if status == "error" or failure_category == "SYSTEM_ERROR":
        if "API" in error or "rate limit" in error.lower() or "timeout" in error.lower():
            return "MODEL_ERROR"  # API/transient
        if "schema" in error.lower() or "validation" in error.lower() or "json" in error.lower():
            return "MODEL_ERROR"  # Structural output error
        return "SYSTEM_ERROR"
    
    # AMBIGUOUS from validator
    if failure_category == "TRUE_AMBIGUITY" or "ambiguous" in review_reason.lower() or "AMBIGUOUS" in review_reason:
        return "TRUE_AMBIGUITY"
    
    # Validator instability
    if failure_category == "VALIDATOR_INSTABILITY" or "validator instability" in review_reason.lower():
        return "VALIDATOR_ERROR"
    
    # Model 1 stuck
    if failure_category == "MODEL1_STUCK" or "maximum attempts" in review_reason.lower():
        return "MODEL_ERROR"
    
    # Validator corrections (semantic disagreements)
    if failure_category == "VALIDATOR_CORRECTION":
        # Check if it's a policy issue
        unresolved_flags = record.get("unresolved_policy_flags", [])
        if unresolved_flags:
            return "POLICY_ERROR"
        # Check if validator flipped
        attempts = record.get("attempts", [])
        if len(attempts) >= 2:
            # Look for validator corrections that flip
            for attempt in attempts:
                label_reviews = attempt.get("label_reviews", [])
                for lr in label_reviews:
                    if lr.get("corrected") is not None and lr.get("predicted") != lr.get("corrected"):
                        # This is a semantic disagreement - could be model or validator error
                        # If policy is clear, it's model error; if policy unclear, it's policy error
                        pass
        return "MODEL_ERROR"  # Default: model made error that validator caught
    
    # Policy ambiguity
    if failure_category == "POLICY_AMBIGUITY":
        return "POLICY_ERROR"
    
    # Clean PASS
    if status == "passed" and failure_category == "NONE":
        return "CLEAN"
    
    return "UNRESOLVED"


def analyze_label_errors(records: list) -> dict:
    """Analyze errors by label."""
    label_errors = Counter()
    label_corrections = Counter()
    
    for record in records:
        if record.get("final_status") == "error":
            continue
            
        attempts = record.get("attempts", [])
        for attempt in attempts:
            label_reviews = attempt.get("label_reviews", [])
            for lr in label_reviews:
                label = lr.get("label")
                predicted = lr.get("predicted")
                corrected = lr.get("corrected")
                if corrected is not None and predicted != corrected:
                    label_corrections[label] += 1
                    if attempt.get("attempt") == len(attempts):  # Final attempt
                        label_errors[label] += 1
    
    return {
        "corrections_by_label": dict(label_corrections),
        "final_errors_by_label": dict(label_errors),
    }


def analyze_boundary_errors(records: list) -> dict:
    """Analyze errors by boundary category."""
    boundary_stats = {}
    
    for record in records:
        cat = record.get("boundary_category", "unknown")
        if cat not in boundary_stats:
            boundary_stats[cat] = {"total": 0, "passed": 0, "needs_review": 0, "error": 0, "retries": 0}
        
        boundary_stats[cat]["total"] += 1
        status = record.get("final_status", "")
        if status == "passed":
            boundary_stats[cat]["passed"] += 1
        elif status == "needs_review":
            boundary_stats[cat]["needs_review"] += 1
        elif status == "error":
            boundary_stats[cat]["error"] += 1
        
        boundary_stats[cat]["retries"] += record.get("retry_count", 0)
    
    return boundary_stats


def analyze_language_errors(records: list) -> dict:
    """Analyze errors by language (multilingual vs English)."""
    lang_stats = {"english": {"total": 0, "passed": 0, "needs_review": 0, "error": 0},
                  "multilingual": {"total": 0, "passed": 0, "needs_review": 0, "error": 0}}
    
    for record in records:
        cats = record.get("all_boundary_categories", "")
        is_multilingual = "multilingual_reports" in cats
        key = "multilingual" if is_multilingual else "english"
        
        lang_stats[key]["total"] += 1
        status = record.get("final_status", "")
        if status == "passed":
            lang_stats[key]["passed"] += 1
        elif status == "needs_review":
            lang_stats[key]["needs_review"] += 1
        elif status == "error":
            lang_stats[key]["error"] += 1
    
    return lang_stats


def analyze_failure_types(records: list) -> dict:
    """Analyze errors by failure type."""
    failure_counts = Counter()
    for record in records:
        fc = record.get("failure_category", "NONE")
        failure_counts[fc] += 1
    return dict(failure_counts)


def generate_audit():
    """Generate the diagnostic audit report."""
    root = Path(__file__).resolve().parents[3]
    exp = root / "experiments/ragset_report_inference_experiment"
    
    results_path = exp / "results/validation/diagnostic_50_results.jsonl"
    trace_path = exp / "results/validation/diagnostic_50_trace.jsonl"
    
    # Load results
    records = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} result records")
    
    # Classify root causes
    for record in records:
        record["root_cause"] = classify_root_cause(record)
    
    # Summary statistics
    total = len(records)
    passed = sum(1 for r in records if r.get("final_status") == "passed")
    needs_review = sum(1 for r in records if r.get("final_status") == "needs_review")
    errors = sum(1 for r in records if r.get("final_status") == "error")
    
    root_causes = Counter(r.get("root_cause", "UNKNOWN") for r in records)
    failure_categories = analyze_failure_types(records)
    label_analysis = analyze_label_errors(records)
    boundary_analysis = analyze_boundary_errors(records)
    language_analysis = analyze_language_errors(records)
    
    # Validator corrections
    validator_corrections = sum(1 for r in records if r.get("failure_category") == "VALIDATOR_CORRECTION")
    retries = sum(r.get("retry_count", 0) for r in records)
    model1_stuck = sum(1 for r in records if r.get("failure_category") == "MODEL1_STUCK")
    validator_instability = sum(1 for r in records if r.get("failure_category") == "VALIDATOR_INSTABILITY")
    policy_unresolved = sum(1 for r in records if r.get("unresolved_policy_flags"))
    
    # Build audit report
    audit = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_reports": total,
            "diagnostic_file": str(results_path),
            "trace_file": str(trace_path),
        },
        "summary": {
            "total_reports": total,
            "passed": passed,
            "needs_review": needs_review,
            "errors": errors,
            "validator_corrections": validator_corrections,
            "total_retries": retries,
            "model1_stuck": model1_stuck,
            "validator_instability": validator_instability,
            "policy_unresolved_cases": policy_unresolved,
        },
        "root_cause_distribution": dict(root_causes),
        "failure_category_distribution": failure_categories,
        "errors_by_label": label_analysis,
        "errors_by_boundary_category": boundary_analysis,
        "errors_by_language": language_analysis,
        "detailed_records": records,
    }
    
    # Save JSON audit
    json_path = exp / "results/validation/diagnostic_50_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    
    print(f"JSON audit saved to {json_path}")
    
    # Create CSV summary (one row per report)
    csv_rows = []
    for record in records:
        csv_rows.append({
            "StudyInstanceUID": record.get("StudyInstanceUID"),
            "boundary_category": record.get("boundary_category"),
            "all_boundary_categories": record.get("all_boundary_categories"),
            "final_status": record.get("final_status"),
            "review_reason": record.get("review_reason", ""),
            "retry_count": record.get("retry_count", 0),
            "failure_category": record.get("failure_category", ""),
            "root_cause": record.get("root_cause", ""),
            "final_labels": json.dumps(record.get("final_labels", {})),
            "policy_conventions_used": "; ".join(record.get("policy_conventions_used", [])),
            "unresolved_policy_flags": "; ".join(record.get("unresolved_policy_flags", [])),
            "retrieved_gold_study_ids": "; ".join(record.get("retrieved_gold_study_ids", [])),
            "actual_model_inference": record.get("actual_model_inference", ""),
            "actual_model_validation": record.get("actual_model_validation", ""),
            "latency_seconds": record.get("latency_seconds", 0),
            "error": record.get("error", ""),
        })
    
    csv_df = pd.DataFrame(csv_rows)
    csv_path = exp / "results/validation/diagnostic_50_report.csv"
    csv_df.to_csv(csv_path, index=False)
    
    print(f"CSV audit saved to {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("DIAGNOSTIC AUDIT SUMMARY")
    print("="*60)
    print(f"Total reports: {total}")
    print(f"  PASSED: {passed}")
    print(f"  NEEDS_REVIEW: {needs_review}")
    print(f"  ERROR: {errors}")
    print(f"\nValidator corrections: {validator_corrections}")
    print(f"Total retries: {retries}")
    print(f"MODEL1_STUCK: {model1_stuck}")
    print(f"VALIDATOR_INSTABILITY: {validator_instability}")
    print(f"Policy UNRESOLVED cases: {policy_unresolved}")
    
    print("\nRoot cause distribution:")
    for cause, count in root_causes.most_common():
        print(f"  {cause}: {count}")
    
    print("\nFailure category distribution:")
    for cat, count in sorted(failure_categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    print("\nErrors by label (final):")
    for label, count in sorted(label_analysis["final_errors_by_label"].items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")
    
    print("\nErrors by boundary category:")
    for cat, stats in sorted(boundary_analysis.items(), key=lambda x: -x[1]["needs_review"]):
        print(f"  {cat}: total={stats['total']}, passed={stats['passed']}, needs_review={stats['needs_review']}, error={stats['error']}, retries={stats['retries']}")
    
    print("\nErrors by language:")
    for lang, stats in language_analysis.items():
        print(f"  {lang}: total={stats['total']}, passed={stats['passed']}, needs_review={stats['needs_review']}, error={stats['error']}")
    
    return audit


if __name__ == "__main__":
    generate_audit()