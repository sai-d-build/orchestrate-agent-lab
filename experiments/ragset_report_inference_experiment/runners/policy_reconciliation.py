#!/usr/bin/env python3
"""
Policy Reconciliation Engine

Audits and reconciles three sources of truth:
1. CSV Ground Truth (authoritative)
2. LLM Gold Analysis (gold_analysis.json) - raw observations
3. Policy YAML (ragset_label_policy.yaml) - candidate operational policy

Detects and reports contradictions rather than silently resolving them.
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ContradictionType(Enum):
    COUNT_MISMATCH = "count_mismatch"
    CONVENTION_CONFLICT = "convention_conflict"
    MISSING_LABEL = "missing_label"
    EXTRA_LABEL = "extra_label"


@dataclass
class Contradiction:
    label: str
    type: ContradictionType
    severity: str  # "ERROR", "WARNING", "INFO"
    message: str
    csv_value: any = None
    llm_value: any = None
    policy_value: any = None
    affected_rules: List[str] = field(default_factory=list)


@dataclass
class LabelPolicy:
    label: str
    csv_positive: int
    csv_negative: int
    llm_positive: int
    llm_negative: int
    policy_positive: int
    policy_negative: int
    conventions: List[Dict] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    status: str = "UNVERIFIED"  # VERIFIED, CONTRADICTION, UNRESOLVED


class PolicyReconciliationEngine:
    """Audits and reconciles gold analysis, policy YAML, and CSV ground truth."""
    
    LABEL_MAPPING = {
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
    
    REQUIRED_LABELS = list(LABEL_MAPPING.keys())
    
    def __init__(self, csv_path: str, gold_analysis_path: str, policy_path: str):
        self.csv_path = Path(csv_path)
        self.gold_analysis_path = Path(gold_analysis_path)
        self.policy_path = Path(policy_path)
        
        self.csv_df = None
        self.gold_analysis = None
        self.policy = None
        self.label_policies: Dict[str, LabelPolicy] = {}
        self.contradictions: List[Contradiction] = []
        
    def load_all(self):
        """Load all three sources."""
        # Load CSV ground truth
        df = pd.read_csv(self.csv_path)
        label_cols = list(self.LABEL_MAPPING.values())
        gold = df[label_cols].notna().all(axis=1)
        self.csv_df = df[gold]
        
        # Load LLM gold analysis
        with open(self.gold_analysis_path) as f:
            self.gold_analysis = json.load(f)
            
        # Load policy YAML
        with open(self.policy_path) as f:
            self.policy = yaml.safe_load(f)
    
    def verify_counts(self) -> List[Contradiction]:
        """Verify count consistency across all three sources."""
        contradictions = []
        
        for llm_label, csv_label in self.LABEL_MAPPING.items():
            csv_pos = int((self.csv_df[csv_label] == 1).sum())
            csv_neg = int((self.csv_df[csv_label] == 0).sum())
            
            llm_data = self.gold_analysis.get('label_specific', {}).get(llm_label, {})
            llm_pos = llm_data.get('gold_positive_count', 0)
            llm_neg = llm_data.get('gold_negative_count', 0)
            
            pol_data = self.policy.get('labels', {}).get(llm_label, {})
            pol_pos = pol_data.get('gold_positive', 0)
            pol_neg = pol_data.get('gold_negative', 0)
            
            # CSV vs Policy (authoritative)
            if csv_pos != pol_pos or csv_neg != pol_neg:
                contradictions.append(Contradiction(
                    label=llm_label,
                    type=ContradictionType.COUNT_MISMATCH,
                    severity="ERROR",
                    message=f"Policy counts differ from CSV ground truth",
                    csv_value={"pos": csv_pos, "neg": csv_neg},
                    policy_value={"pos": pol_pos, "neg": pol_neg},
                    affected_rules=["count_verification"]
                ))
            
            # LLM vs Policy (should match since policy was derived from LLM)
            if llm_pos != pol_pos or llm_neg != pol_neg:
                contradictions.append(Contradiction(
                    label=llm_label,
                    type=ContradictionType.COUNT_MISMATCH,
                    severity="WARNING",
                    message=f"LLM analysis differs from policy YAML",
                    llm_value={"pos": llm_pos, "neg": llm_neg},
                    policy_value={"pos": pol_pos, "neg": pol_neg},
                    affected_rules=["llm_policy_consistency"]
                ))
        
        return contradictions
    
    def verify_conventions(self) -> List[Contradiction]:
        """Verify policy conventions against CSV ground truth."""
        contradictions = []
        
        for llm_label, csv_label in self.LABEL_MAPPING.items():
            pol_data = self.policy.get('labels', {}).get(llm_label, {})
            conventions = pol_data.get('conventions', [])
            
            for conv in conventions:
                status = conv.get('status', '')
                supporting = conv.get('supporting_count', 0)
                contradicting = conv.get('contradicting_count', 0)
                
                # Verify SAFE_POSITIVE claims
                if status == 'SAFE_POSITIVE':
                    if contradicting > 0:
                        contradictions.append(Contradiction(
                            label=llm_label,
                            type=ContradictionType.CONVENTION_CONFLICT,
                            severity="ERROR",
                            message=f"SAFE_POSITIVE rule has {contradicting} contradicting cases",
                            policy_value={"supporting": supporting, "contradicting": contradicting},
                            affected_rules=[f"convention:{conv.get('pattern', 'unknown')}"]
                        ))
                
                # Verify SAFE_NEGATIVE claims
                if status == 'SAFE_NEGATIVE':
                    if supporting > 0:
                        contradictions.append(Contradiction(
                            label=llm_label,
                            type=ContradictionType.CONVENTION_CONFLICT,
                            severity="ERROR",
                            message=f"SAFE_NEGATIVE rule has {supporting} supporting cases (should be 0)",
                            policy_value={"supporting": supporting, "contradicting": contradicting},
                            affected_rules=[f"convention:{conv.get('pattern', 'unknown')}"]
                        ))
                
                # Check UNRESOLVED markers
                if status == 'UNRESOLVED':
                    contradictions.append(Contradiction(
                        label=llm_label,
                        type=ContradictionType.CONVENTION_CONFLICT,
                        severity="WARNING",
                        message=f"Unresolved convention: {conv.get('rationale', 'no rationale')}",
                        policy_value={"pattern": conv.get('pattern', 'unknown')},
                        affected_rules=["unresolved_convention"]
                    ))
        
        return contradictions
    
    def verify_completeness(self) -> List[Contradiction]:
        """Verify all required labels are present in policy."""
        contradictions = []
        
        policy_labels = set(self.policy.get('labels', {}).keys())
        required_labels = set(self.REQUIRED_LABELS)
        
        missing = required_labels - policy_labels
        extra = policy_labels - required_labels
        
        for label in missing:
            contradictions.append(Contradiction(
                label=label,
                type=ContradictionType.MISSING_LABEL,
                severity="ERROR",
                message=f"Required label missing from policy",
                affected_rules=["completeness"]
            ))
        
        for label in extra:
            contradictions.append(Contradiction(
                label=label,
                type=ContradictionType.EXTRA_LABEL,
                severity="WARNING",
                message=f"Extra label in policy not in required set",
                affected_rules=["completeness"]
            ))
        
        return contradictions
    
    def run_full_audit(self) -> Dict:
        """Run complete audit and return results."""
        self.load_all()
        
        all_contradictions = []
        all_contradictions.extend(self.verify_counts())
        all_contradictions.extend(self.verify_conventions())
        all_contradictions.extend(self.verify_completeness())
        
        self.contradictions = all_contradictions
        
        # Build label policies
        for llm_label, csv_label in self.LABEL_MAPPING.items():
            csv_pos = int((self.csv_df[csv_label] == 1).sum())
            csv_neg = int((self.csv_df[csv_label] == 0).sum())
            
            llm_data = self.gold_analysis.get('label_specific', {}).get(llm_label, {})
            llm_pos = llm_data.get('gold_positive_count', 0)
            llm_neg = llm_data.get('gold_negative_count', 0)
            
            pol_data = self.policy.get('labels', {}).get(llm_label, {})
            pol_pos = pol_data.get('gold_positive', 0)
            pol_neg = pol_data.get('gold_negative', 0)
            
            label_contradictions = [c for c in all_contradictions if c.label == llm_label]
            
            status = "VERIFIED"
            if any(c.severity == "ERROR" for c in label_contradictions):
                status = "CONTRADICTION"
            elif any(c.severity == "WARNING" for c in label_contradictions):
                status = "UNRESOLVED"
            
            self.label_policies[llm_label] = LabelPolicy(
                label=llm_label,
                csv_positive=csv_pos,
                csv_negative=csv_neg,
                llm_positive=llm_pos,
                llm_negative=llm_neg,
                policy_positive=pol_pos,
                policy_negative=pol_neg,
                conventions=self.policy.get('labels', {}).get(llm_label, {}).get('conventions', []),
                contradictions=label_contradictions,
                status=status
            )
        
        return {
            "total_contradictions": len(all_contradictions),
            "errors": len([c for c in all_contradictions if c.severity == "ERROR"]),
            "warnings": len([c for c in all_contradictions if c.severity == "WARNING"]),
            "labels": {k: v.__dict__ for k, v in self.label_policies.items()},
            "contradictions": [c.__dict__ for c in all_contradictions]
        }


def main():
    engine = PolicyReconciliationEngine(
        csv_path="train.csv",
        gold_analysis_path="experiments/ragset_report_inference_experiment/results/gold/gold_analysis.json",
        policy_path="config/ragset_label_policy.yaml"
    )
    
    results = engine.run_full_audit()
    
    print(f"\n=== AUDIT SUMMARY ===")
    print(f"Total contradictions: {results['total_contradictions']}")
    print(f"Errors: {results['errors']}")
    print(f"Warnings: {results['warnings']}")
    
    print(f"\n=== LABEL STATUS ===")
    for label, data in results['labels'].items():
        print(f"  {label}: {data['status']} ({len(data['contradictions'])} contradictions)")
    
    print(f"\n=== CONTRADICTIONS ===")
    for c in results['contradictions']:
        print(f"  [{c['severity']}] {c['label']}: {c['message']}")
    
    # Save audit report
    output_path = Path("experiments/ragset_report_inference_experiment/results/gold/policy_audit_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAudit report saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()