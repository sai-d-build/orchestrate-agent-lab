from collections import Counter
import json
import pandas as pd
from pathlib import Path
from .data import LABEL_COLUMNS


def analyze_gold(gold: pd.DataFrame) -> dict:
    """Analyze gold reports - loads persisted LLM analysis if available,
    otherwise falls back to basic statistics."""
    # Try to load persisted LLM analysis first
    root = Path(__file__).resolve().parents[3]
    analysis_path = root / "experiments/ragset_report_inference_experiment/results/gold/gold_analysis.json"

    if analysis_path.exists():
        with analysis_path.open("r", encoding="utf-8") as f:
            llm_analysis = json.load(f)
        # Merge with basic stats for backward compatibility
        basic_stats = _basic_gold_stats(gold)
        return {**basic_stats, **llm_analysis}

    # Fallback to basic statistics
    return _basic_gold_stats(gold)


def _basic_gold_stats(gold: pd.DataFrame) -> dict:
    """Basic statistical analysis of gold reports (fallback)."""
    lengths = gold["Report"].astype(str).str.len()
    return {
        "gold_count": int(len(gold)),
        "label_distribution": {
            label: {
                "positive": int((gold[label] == 1).sum()),
                "negative": int((gold[label] == 0).sum()),
            }
            for label in LABEL_COLUMNS
        },
        "report_length": {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        },
        "duplicate_report_count": int(gold["Report"].duplicated().sum()),
    }


def format_gold_examples(gold: pd.DataFrame) -> str:
    blocks = []
    for _, row in gold.iterrows():
        labels = {label: int(row[label]) for label in LABEL_COLUMNS}
        blocks.append(
            f"STUDY {row['StudyInstanceUID']}\n"
            f"LABELS: {labels}\n"
            f"REPORT:\n{row['Report']}"
        )
    return "\n\n---\n\n".join(blocks)


def label_profile_text(analysis: dict) -> str:
    """Format label distribution for prompts."""
    if "label_distribution" in analysis:
        return "\n".join(
            f"{label}: {v['positive']} positive / {v['negative']} negative"
            for label, v in analysis["label_distribution"].items()
        )
    return "Gold analysis not available"


def format_gold_policy_for_prompt(analysis: dict) -> str:
    """Extract concise policy: SAFE rules, UNSAFE shortcuts, UNRESOLVED per label.
    
    Returns a compact representation suitable for prompt injection.
    """
    if not analysis or "label_specific" not in analysis:
        return label_profile_text(analysis)
    
    parts = []
    label_specific = analysis.get("label_specific", {})
    
    # Check if all 12 labels have analysis
    missing_labels = [label for label in LABEL_COLUMNS if label not in label_specific or not label_specific[label]]
    if missing_labels:
        parts.append(f"NOTE: Insufficient gold evidence for: {', '.join(missing_labels)}")
    
    for label in LABEL_COLUMNS:
        data = label_specific.get(label, {})
        if not data:
            parts.append(f"=== {label} ===\n  INSUFFICIENT_GOLD_EVIDENCE")
            continue
        
        parts.append(f"=== {label} ===")
        
        # SAFE rules
        safe_rules = data.get("safe_rules", [])
        if safe_rules:
            parts.append("  SAFE:")
            for rule in safe_rules:
                rule_text = rule.get("rule", "")
                if rule_text:
                    parts.append(f"    - {rule_text}")
        
        # UNSAFE shortcuts
        unsafe = data.get("unsafe_shortcuts", [])
        if unsafe:
            parts.append("  UNSAFE (do not use as standalone rules):")
            for us in unsafe:
                pattern = us.get("pattern", "")
                reason = us.get("reason", "")
                if pattern:
                    parts.append(f"    - {pattern}: {reason}")
        
        # UNRESOLVED rules
        unresolved = data.get("unresolved_rules", [])
        if unresolved:
            parts.append("  UNRESOLVED:")
            for ur in unresolved:
                question = ur.get("question", "")
                if question:
                    parts.append(f"    - {question}")
        
        # CONTRADICTIONS
        contradictions = data.get("contradictions", [])
        if contradictions:
            parts.append("  CONTRADICTIONS:")
            for c in contradictions:
                pattern = c.get("pattern", "")
                ctype = c.get("contradiction_type", "")
                explanation = c.get("explanation", "")
                if pattern:
                    parts.append(f"    - {pattern} ({ctype}): {explanation}")
    
    return "\n".join(parts) if parts else label_profile_text(analysis)