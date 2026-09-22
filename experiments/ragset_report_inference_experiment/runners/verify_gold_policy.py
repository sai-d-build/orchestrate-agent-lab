#!/usr/bin/env python3
"""Programmatic verification of gold analysis against CSV ground truth."""

import json
import pandas as pd
from pathlib import Path

def verify_gold_analysis():
    """Verify gold_analysis.json counts against CSV ground truth."""
    
    # Load CSV ground truth
    df = pd.read_csv("train.csv")
    label_cols = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
    gold = df[label_cols].notna().all(axis=1)
    gold_df = df[gold]
    
    # Load LLM analysis
    with open("experiments/ragset_report_inference_experiment/results/gold/gold_analysis.json") as f:
        llm_analysis = json.load(f)
    
    label_mapping = {
        'ACL': 'ACL',
        'MCL': 'MCL',
        'Medial_Meniscus': 'Medial Meniscus',
        'Lateral_Meniscus': 'Lateral Meniscus',
        'Medial_OA': 'Medial OA',
        'Lateral_OA': 'Lateral OA',
        'PF_OA': 'PF OA',
        'Effusion': 'Effusion',
        'Synovitis': 'Synovitis',
        'Bakers': "Baker's",
        'Contusion': 'Contusion',
        'Fracture': 'Fracture',
    }
    
    print("=" * 80)
    print("GOLD ANALYSIS VERIFICATION")
    print("=" * 80)
    
    all_match = True
    for llm_label, csv_label in label_mapping.items():
        csv_pos = int((gold_df[csv_label] == 1).sum())
        csv_neg = int((gold_df[csv_label] == 0).sum())
        
        llm_data = llm_analysis.get('label_specific', {}).get(llm_label, {})
        llm_pos = llm_data.get('gold_positive_count', 0)
        llm_neg = llm_data.get('gold_negative_count', 0)
        
        pos_match = csv_pos == llm_pos
        neg_match = csv_neg == llm_neg
        match = pos_match and neg_match
        
        status = "✓" if match else "✗"
        print(f"{status} {llm_label:20s} | CSV: pos={csv_pos:2d} neg={csv_neg:2d} | LLM: pos={llm_pos:2d} neg={llm_neg:2d}")
        
        if not match:
            all_match = False
    
    print("=" * 80)
    if all_match:
        print("✓ ALL COUNTS MATCH")
    else:
        print("✗ SOME COUNTS MISMATCH - review required")
    
    # Verify policy file exists
    policy_path = Path("config/ragset_label_policy.yaml")
    if policy_path.exists():
        print("\n✓ Policy file exists: config/ragset_label_policy.yaml")
    else:
        print("\n✗ Policy file missing!")
    
    return all_match


if __name__ == "__main__":
    verify_gold_analysis()