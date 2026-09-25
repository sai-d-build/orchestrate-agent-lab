#!/usr/bin/env python3
"""
Canonical Policy Loader

Loads and validates the operational policy for inference.
Ensures consistency between CSV ground truth, gold analysis, and policy YAML.
Provides a single canonical policy for both Model 1 and Model 2.
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PolicyStatus(Enum):
    VERIFIED = "VERIFIED"
    CONTRADICTION = "CONTRADICTION"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class PolicyRule:
    """A single policy rule with metadata."""
    pattern: str
    status: str  # SAFE_POSITIVE, SAFE_NEGATIVE, CONTEXTUAL, UNSAFE, UNRESOLVED
    supporting_count: int = 0
    contradicting_count: int = 0
    rationale: str = ""
    example_uids: List[str] = field(default_factory=list)
    affected_labels: List[str] = field(default_factory=list)


@dataclass
class LabelPolicy:
    """Complete policy for a single label."""
    label: str
    csv_positive: int
    csv_negative: int
    conventions: List[PolicyRule] = field(default_factory=list)
    cross_label_constraints: List[str] = field(default_factory=list)
    evidence_rules: List[str] = field(default_factory=list)
    uncertainty_rules: List[str] = field(default_factory=list)
    temporal_rules: List[str] = field(default_factory=list)
    anatomical_rules: List[str] = field(default_factory=list)
    explicit_evidence_override: List[str] = field(default_factory=list)
    status: PolicyStatus = PolicyStatus.UNRESOLVED
    contradictions: List[str] = field(default_factory=list)


@dataclass
class CanonicalPolicy:
    """Complete canonical policy for inference."""
    labels: Dict[str, LabelPolicy] = field(default_factory=dict)
    cross_label_constraints: List[str] = field(default_factory=list)
    evidence_rules: List[str] = field(default_factory=list)
    uncertainty_rules: List[str] = field(default_factory=list)
    temporal_rules: List[str] = field(default_factory=list)
    anatomical_rules: List[str] = field(default_factory=list)
    explicit_evidence_override: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_prompt_text(self) -> str:
        """Convert to prompt-ready text for Model 1 and Model 2."""
        parts = []
        
        # Label-specific conventions
        for label, policy in self.labels.items():
            if not policy.conventions:
                continue
            parts.append(f"=== {label} ===")
            for conv in policy.conventions:
                status = conv.status
                pattern = conv.pattern
                rationale = conv.rationale
                parts.append(f"  {status}: {pattern}")
                if rationale:
                    parts.append(f"    Rationale: {rationale}")
                if conv.example_uids:
                    parts.append(f"    Examples: {', '.join(conv.example_uids[:3])}")
        
        # Cross-label constraints
        if self.cross_label_constraints:
            parts.append("\n=== CROSS-LABEL CONSTRAINTS ===")
            for constraint in self.cross_label_constraints:
                parts.append(f"  - {constraint}")
        
        # Evidence rules
        if self.evidence_rules:
            parts.append("\n=== EVIDENCE RULES ===")
            for rule in self.evidence_rules:
                parts.append(f"  - {rule}")
        
        # Uncertainty rules
        if self.uncertainty_rules:
            parts.append("\n=== UNCERTAINTY RULES ===")
            for rule in self.uncertainty_rules:
                parts.append(f"  - {rule}")
        
        # Temporal rules
        if self.temporal_rules:
            parts.append("\n=== TEMPORAL/HISTORICAL RULES ===")
            for rule in self.temporal_rules:
                parts.append(f"  - {rule}")
        
        # Anatomical rules
        if self.anatomical_rules:
            parts.append("\n=== ANATOMICAL SPECIFICITY ===")
            for rule in self.anatomical_rules:
                parts.append(f"  - {rule}")
        
        # Explicit evidence override
        if self.explicit_evidence_override:
            parts.append("\n=== EXPLICIT EVIDENCE OVERRIDE ===")
            for rule in self.explicit_evidence_override:
                parts.append(f"  - {rule}")
        
        return "\n".join(parts)
    
    def get_label_section(self, label: str) -> str:
        """Get policy text for a specific label."""
        if label not in self.labels:
            return f"=== {label} ===\n  INSUFFICIENT_GOLD_EVIDENCE"
        
        policy = self.labels[label]
        if not policy.conventions:
            return f"=== {label} ===\n  INSUFFICIENT_GOLD_EVIDENCE"
        
        parts = [f"=== {label} ==="]
        for conv in policy.conventions:
            status = conv.status
            pattern = conv.pattern
            rationale = conv.rationale
            parts.append(f"  {status}: {pattern}")
            if rationale:
                parts.append(f"    Rationale: {rationale}")
            if conv.example_uids:
                parts.append(f"    Examples: {', '.join(conv.example_uids[:3])}")
        return "\n".join(parts)


class PolicyLoader:
    """Loads and validates the canonical operational policy."""
    
    REQUIRED_LABELS = [
        'ACL', 'MCL', 'Medial_Meniscus', 'Lateral_Meniscus',
        'Medial_OA', 'Lateral_OA', 'PF_OA', 'Effusion',
        'Synovitis', 'Bakers', 'Contusion', 'Fracture'
    ]
    
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
    
    def __init__(self, 
                 csv_path: str = "train.csv",
                 gold_analysis_path: str = "experiments/ragset_report_inference_experiment/results/gold/gold_analysis.json",
                 policy_path: str = "config/ragset_label_policy.yaml"):
        self.csv_path = Path(csv_path)
        self.gold_analysis_path = Path(gold_analysis_path)
        self.policy_path = Path(policy_path)
        self.csv_df = None
        self.gold_analysis = None
        self.policy_yaml = None
        self.canonical_policy = None
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
    
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
            self.policy_yaml = yaml.safe_load(f)
    
    def validate_counts(self) -> List[str]:
        """Validate count consistency across CSV, LLM analysis, and policy."""
        errors = []
        
        for llm_label, csv_label in self.LABEL_MAPPING.items():
            csv_pos = int((self.csv_df[csv_label] == 1).sum())
            csv_neg = int((self.csv_df[csv_label] == 0).sum())
            
            pol_data = self.policy_yaml.get('labels', {}).get(llm_label, {})
            pol_pos = pol_data.get('gold_positive', 0)
            pol_neg = pol_data.get('gold_negative', 0)
            
            if csv_pos != pol_pos or csv_neg != pol_neg:
                self.validation_errors.append(
                    f"{llm_label}: Policy counts (pos={pol_pos}, neg={pol_neg}) "
                    f"differ from CSV ground truth (pos={csv_pos}, neg={csv_neg})"
                )
        
        return self.validation_errors
    
    def validate_conventions(self) -> List[str]:
        """Validate policy conventions against CSV ground truth."""
        errors = []
        
        for llm_label, csv_label in self.LABEL_MAPPING.items():
            pol_data = self.policy_yaml.get('labels', {}).get(llm_label, {})
            conventions = pol_data.get('conventions', [])
            
            for conv in conventions:
                status = conv.get('status', '')
                supporting = conv.get('supporting_count', 0)
                contradicting = conv.get('contradicting_count', 0)
                
                if status == 'SAFE_POSITIVE' and contradicting > 0:
                    self.validation_errors.append(
                        f"{llm_label}: SAFE_POSITIVE rule has {contradicting} contradicting cases: "
                        f"'{conv.get('pattern', 'unknown')}'"
                    )
                
                if status == 'SAFE_NEGATIVE' and supporting > 0:
                    self.validation_errors.append(
                        f"{llm_label}: SAFE_NEGATIVE rule has {supporting} supporting cases: "
                        f"'{conv.get('pattern', 'unknown')}'"
                    )
        
        return self.validation_errors
    
    def validate_completeness(self) -> List[str]:
        """Verify all required labels are present."""
        errors = []
        policy_labels = set(self.policy_yaml.get('labels', {}).keys())
        required_labels = set(self.REQUIRED_LABELS)
        
        missing = required_labels - policy_labels
        for label in missing:
            self.validation_errors.append(f"Required label missing from policy: {label}")
        
        return self.validation_errors
    
    def validate_cross_label_constraints(self) -> List[str]:
        """Validate cross-label constraints are present and consistent."""
        errors = []
        constraints = self.policy_yaml.get('cross_label_constraints', [])
        
        # Handle both list of strings and list of dicts with 'rule' key
        constraint_strings = []
        for c in constraints:
            if isinstance(c, dict):
                constraint_strings.append(c.get('rule', ''))
                # Also check examples
                for ex in c.get('examples', []):
                    constraint_strings.append(ex)
            else:
                constraint_strings.append(str(c))
        
        required_constraints = [
            "Do not infer one label merely because another label is positive",
            "ACL tear does not imply MCL tear",
            "Effusion does not imply Synovitis",
            "Fracture does not imply Contusion",
            "Generalized OA does not imply all three compartment OA",
            "Meniscal degeneration does not imply tear",
            "Meniscal extrusion implies tear"
        ]
        
        for req in required_constraints:
            found = any(req.lower() in c.lower() for c in constraint_strings)
            if not found:
                self.validation_warnings.append(f"Missing cross-label constraint: {req}")
        
        return self.validation_errors
    
    def validate_evidence_rules(self) -> List[str]:
        """Validate evidence rules are present."""
        errors = []
        rules = self.policy_yaml.get('evidence_rules', [])
        
        required_rules = [
            "Evidence must be verbatim from ORIGINAL_REPORT",
            "Evidence must be in original language",
            "Null evidence is acceptable for correct negative labels",
            "Evidence quality alone must not change PASS to FAIL",
            "Explicit negative evidence only supports the finding it addresses"
        ]
        
        for req in required_rules:
            found = any(req.lower() in r.lower() for r in rules)
            if not found:
                self.validation_warnings.append(f"Missing evidence rule: {req}")
        
        return self.validation_errors
    
    def validate_uncertainty_rules(self) -> List[str]:
        """Validate uncertainty handling rules."""
        errors = []
        rules = self.policy_yaml.get('uncertainty_rules', [])
        
        required_rules = [
            "Suspected/possible/R-O/suspicious/cannot exclude → classify as UNCERTAIN first",
            "Then apply GOLD_ANALYSIS to map uncertainty to binary label",
            "Do not universally force UNCERTAIN → 0 or UNCERTAIN → 1"
        ]
        
        for req in required_rules:
            found = any(req.lower() in r.lower() for r in rules)
            if not found:
                self.validation_warnings.append(f"Missing uncertainty rule: {req}")
        
        return self.validation_errors
    
    def validate_temporal_rules(self) -> List[str]:
        """Validate temporal/historical rules."""
        errors = []
        rules = self.policy_yaml.get('temporal_rules', [])
        
        required_rules = [
            "Historical/chronic/postoperative findings do not automatically equal current positive",
            "Current explicit imaging findings take precedence unless GOLD_ANALYSIS says otherwise"
        ]
        
        for req in required_rules:
            found = any(req.lower() in r.lower() for r in rules)
            if not found:
                self.validation_warnings.append(f"Missing temporal rule: {req}")
        
        return self.validation_errors
    
    def validate_anatomical_rules(self) -> List[str]:
        """Validate anatomical specificity rules."""
        errors = []
        rules = self.policy_yaml.get('anatomical_rules', [])
        
        required_rules = [
            "Medial/Lateral structures must be evaluated independently",
            "OA compartments require compartment-specific evidence",
            "Patellar/trochlear findings only support PF_OA",
            "MCL/ACL require structure-specific evidence",
            "Medial/Lateral meniscus evaluated independently"
        ]
        
        for req in required_rules:
            found = any(req.lower() in r.lower() for r in rules)
            if not found:
                self.validation_warnings.append(f"Missing anatomical rule: {req}")
        
        return self.validation_errors
    
    def validate_explicit_evidence_override(self) -> List[str]:
        """Validate explicit evidence override rules."""
        errors = []
        rules = self.policy_yaml.get('explicit_evidence_override', [])
        
        required_rules = [
            "When ORIGINAL_REPORT explicitly states a finding, do not negate using weaker indirect inference",
            "Example: 'bone contusion' → Contusion=1, do not change to 0 because of 'marrow edema'",
            "Example: 'ACL sprain' → ACL=1, do not change to 0 because of 'increased signal'",
            "Example: 'Baker's cyst' → Bakers=1, do not change to 0 because of 'popliteal fluid'",
            "Example: 'ACL intact' → ACL=0, do not change to 1 because another ligament is abnormal"
        ]
        
        for req in required_rules:
            found = any(req.lower() in r.lower() for r in rules)
            if not found:
                self.validation_warnings.append(f"Missing explicit evidence override rule: {req}")
        
        return self.validation_errors
    
    def run_full_validation(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validations. Returns (is_valid, errors, warnings)."""
        self.load_all()
        
        self.validate_counts()
        self.validate_conventions()
        self.validate_completeness()
        self.validate_cross_label_constraints()
        self.validate_evidence_rules()
        self.validate_uncertainty_rules()
        self.validate_temporal_rules()
        self.validate_anatomical_rules()
        self.validate_explicit_evidence_override()
        
        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings
    
    def build_canonical_policy(self) -> CanonicalPolicy:
        """Build the canonical policy object from validated YAML."""
        policy = CanonicalPolicy()
        
        # Load label policies
        for llm_label, csv_label in self.LABEL_MAPPING.items():
            pol_data = self.policy_yaml.get('labels', {}).get(llm_label, {})
            csv_pos = int((self.csv_df[csv_label] == 1).sum())
            csv_neg = int((self.csv_df[csv_label] == 0).sum())
            
            label_policy = LabelPolicy(
                label=llm_label,
                csv_positive=csv_pos,
                csv_negative=csv_neg,
                conventions=[],
                status=PolicyStatus.VERIFIED
            )
            
            # Convert conventions
            for conv in pol_data.get('conventions', []):
                rule = PolicyRule(
                    pattern=conv.get('pattern', ''),
                    status=conv.get('status', 'UNRESOLVED'),
                    supporting_count=conv.get('supporting_count', 0),
                    contradicting_count=conv.get('contradicting_count', 0),
                    rationale=conv.get('rationale', ''),
                    example_uids=conv.get('example_uids', []),
                    affected_labels=[llm_label]
                )
                label_policy.conventions.append(rule)
            
            policy.labels[llm_label] = label_policy
        
        # Global rules
        policy.cross_label_constraints = self.policy_yaml.get('cross_label_constraints', [])
        policy.evidence_rules = self.policy_yaml.get('evidence_rules', [])
        policy.uncertainty_rules = self.policy_yaml.get('uncertainty_rules', [])
        policy.temporal_rules = self.policy_yaml.get('temporal_rules', [])
        policy.anatomical_rules = self.policy_yaml.get('anatomical_rules', [])
        policy.explicit_evidence_override = self.policy_yaml.get('explicit_evidence_override', [])
        
        # Metadata
        policy.metadata = {
            'source': 'ragset_label_policy.yaml',
            'csv_ground_truth': 'train.csv',
            'gold_analysis': 'gold_analysis.json',
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.canonical_policy = policy
        return policy
    
    def get_canonical_policy(self) -> CanonicalPolicy:
        """Get the canonical policy, building if necessary."""
        if self.canonical_policy is None:
            self.build_canonical_policy()
        return self.canonical_policy
    
    def get_policy_text_for_prompts(self) -> str:
        """Get the canonical policy text for injection into Model 1 and Model 2 prompts."""
        policy = self.get_canonical_policy()
        return policy.to_prompt_text()
    
    def get_label_policy_text(self, label: str) -> str:
        """Get policy text for a specific label."""
        policy = self.get_canonical_policy()
        return policy.get_label_section(label)


def load_canonical_policy(
    csv_path: str = "train.csv",
    gold_analysis_path: str = "experiments/ragset_report_inference_experiment/results/gold/gold_analysis.json",
    policy_path: str = "config/ragset_label_policy.yaml"
) -> Tuple[CanonicalPolicy, List[str], List[str]]:
    """
    Load and validate the canonical policy.
    
    Returns:
        (canonical_policy, errors, warnings)
        
    Raises:
        RuntimeError: If validation errors exist (policy not valid for production)
    """
    loader = PolicyLoader(csv_path, gold_analysis_path, policy_path)
    is_valid, errors, warnings = loader.run_full_validation()
    
    if not is_valid:
        error_msg = "Policy validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        if warnings:
            error_msg += "\n\nWarnings:\n" + "\n".join(f"  - {w}" for w in warnings)
        raise RuntimeError(error_msg)
    
    if warnings:
        logger.warning("Policy validation warnings:\n" + "\n".join(f"  - {w}" for w in warnings))
    
    policy = loader.get_canonical_policy()
    return policy, errors, warnings


def get_canonical_policy_text(
    csv_path: str = "train.csv",
    gold_analysis_path: str = "experiments/ragset_report_inference_experiment/results/gold/gold_analysis.json",
    policy_path: str = "config/ragset_label_policy.yaml"
) -> str:
    """
    Get the canonical policy text for injection into Model 1 and Model 2 prompts.
    
    This is the single function that should be called by run_inference.py.
    It validates the policy and returns the canonical policy text.
    
    Raises:
        RuntimeError: If policy validation fails (production gate)
    """
    policy, errors, warnings = load_canonical_policy(csv_path, gold_analysis_path, policy_path)
    return policy.to_prompt_text()


def get_label_policy_text(
    label: str,
    csv_path: str = "train.csv",
    gold_analysis_path: str = "experiments/ragset_report_inference_experiment/results/gold/gold_analysis.json",
    policy_path: str = "config/ragset_label_policy.yaml"
) -> str:
    """Get policy text for a specific label."""
    loader = PolicyLoader(csv_path, gold_analysis_path, policy_path)
    loader.load_all()
    loader.run_full_validation()
    policy = loader.build_canonical_policy()
    return policy.get_label_section(label)


def get_policy_hash(policy_path: str = "config/ragset_label_policy.yaml") -> str:
    """Get SHA256 hash of the canonical policy YAML for versioning."""
    import hashlib
    path = Path(policy_path)
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def get_policy_version(policy_path: str = "config/ragset_label_policy.yaml") -> str:
    """Get policy version identifier (hash-based)."""
    return f"policy_{get_policy_hash(policy_path)}"


# Policy precedence hierarchy (authoritative order)
# 1. CSV Ground Truth (authoritative)
# 2. GOLD_ANALYSIS (LLM analysis of gold)
# 3. Canonical Policy YAML (runtime policy)
# 4. Retrieved Examples (reference only, never override)
POLICY_PRECEDENCE = [
    "CSV_GROUND_TRUTH",
    "GOLD_ANALYSIS",
    "CANONICAL_POLICY_YAML",
    "RETRIEVED_EXAMPLES"
]


def get_policy_precedence() -> list[str]:
    """Get the canonical policy precedence hierarchy."""
    return POLICY_PRECEDENCE.copy()


if __name__ == "__main__":
    # Test the loader
    try:
        policy_text = get_canonical_policy_text()
        print("=== CANONICAL POLICY TEXT ===")
        print(policy_text[:3000])
        print("...")
        print(f"\nTotal length: {len(policy_text)} chars")
    except RuntimeError as e:
        print(f"Policy validation failed: {e}")