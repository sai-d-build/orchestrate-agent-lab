from dataclasses import dataclass
from enum import Enum
from time import perf_counter
import json
import yaml
from pathlib import Path
from .schemas import (
    ReportPrediction, ValidationResult, ValidationIssue,
    CritiqueResult, CritiqueIssue, CritiqueIssueType, CritiqueStatus,
    LABEL_KEYS
)


class FailureType(Enum):
    """Classification of failure types for proper retry handling."""
    STRUCTURAL = "structural"           # JSON parse error, schema validation error
    SEMANTIC = "semantic"               # Model 2 identifies genuine binary disagreement
    TRANSIENT = "transient"             # API error, rate limit, None content
    MODEL1_STUCK = "model1_stuck"       # Model 1 repeats same 12-label vector
    VALIDATOR_INSTABILITY = "validator_instability"  # Validator corrections flip across attempts
    POLICY_AMBIGUITY = "policy_ambiguity"  # Validator returns AMBIGUOUS


@dataclass
class Attempt:
    number: int
    prediction: ReportPrediction
    validation: ValidationResult
    elapsed_seconds: float
    failure_type: FailureType | None = None


def _filter_real_issues(validation: ValidationResult) -> ValidationResult:
    """
    Filter validation issues to only keep real discrepancies.

    A real issue is one where:
    - corrected is not None (validator could determine the correct value)
    - predicted != corrected (there's an actual disagreement)

    Issues where corrected is None or predicted == corrected are not real failures.
    
    Preserves explicit validator status (PASS/FAIL/AMBIGUOUS).
    """
    real_issues = [
        issue for issue in validation.issues
        if issue.corrected is not None and issue.predicted != issue.corrected
    ]
    ambiguous_issues = [
        issue for issue in validation.issues
        if issue.corrected is None
    ]
    
    # If validator explicitly set AMBIGUOUS, preserve it
    if validation.status == "AMBIGUOUS":
        return ValidationResult(
            status="AMBIGUOUS",
            issues=ambiguous_issues,
        )
    
    # If no real issues but ambiguous issues exist, mark as AMBIGUOUS
    if not real_issues and ambiguous_issues:
        return ValidationResult(
            status="AMBIGUOUS",
            issues=ambiguous_issues,
        )
    
    return ValidationResult(
        status="PASS" if not real_issues else "FAIL",
        issues=real_issues,
    )


def _classify_failure(validation: ValidationResult, error: Exception | None = None) -> FailureType:
    """Classify the type of failure for proper retry handling."""
    if error is not None:
        # API errors, JSON parse errors, schema validation errors
        return FailureType.TRANSIENT if "None content" in str(error) or "rate limit" in str(error).lower() else FailureType.STRUCTURAL
    
    # Check validator status first
    if validation.status == "AMBIGUOUS":
        return FailureType.POLICY_AMBIGUITY
    
    # Check if validation has real semantic issues
    real_issues = [
        issue for issue in validation.issues
        if issue.corrected is not None and issue.predicted != issue.corrected
    ]
    if real_issues:
        return FailureType.SEMANTIC
    return FailureType.STRUCTURAL


def _detect_validator_oscillation(validator_history: dict[str, list[int]]) -> list[str]:
    """
    Detect deterministic validator oscillation patterns.
    
    Returns list of labels where validator corrections oscillate:
    - 0 → 1 → 0
    - 1 → 0 → 1
    - Any pattern where corrections return to a previous state
    - Immediate flip on second attempt (0→1 or 1→0 where first correction was opposite)
    """
    oscillating = []
    for label, history in validator_history.items():
        if len(history) >= 3:
            # Check for 0→1→0 or 1→0→1 pattern
            if history[-3] == history[-1] and history[-3] != history[-2]:
                oscillating.append(label)
            # Check for any return to a previous state (more general)
            elif len(set(history)) > 1 and history[-1] in history[:-1]:
                oscillating.append(label)
        elif len(history) == 2:
            # Immediate flip on second attempt: first correction was X, second is opposite
            # This indicates the validator is changing its mind based on Model 1's prediction
            if history[0] != history[1]:
                oscillating.append(label)
    return oscillating


def _detect_model1_stuck(model1_history: dict[str, list[int]], disputed_labels: set[str], min_attempts: int = 2) -> list[str]:
    """
    Detect Model 1 STUCK by tracking complete 12-label prediction vectors.
    
    Returns list of labels where Model 1 repeats the same value for a disputed label
    across min_attempts consecutive attempts.
    """
    stuck = []
    for label in disputed_labels:
        history = model1_history[label]
        if len(history) >= min_attempts:
            # Check if last min_attempts values are all the same
            if len(set(history[-min_attempts:])) == 1:
                stuck.append(label)
    return stuck


def _validate_critic_output(
    critique: CritiqueResult,
    report: str,
    model1_prediction: ReportPrediction,
) -> dict | None:
    """
    Deterministic integrity/safety checks on Model 2 (Critic) output.
    
    Does NOT make clinical judgments. Only validates structural integrity
    and consistency with schema rules.
    
    Returns:
        None if all checks pass
        dict with safety_gate_failure details if any check fails
    """
    # 1. Validate Model 2 critique schema - already validated by Pydantic
    
    # 2-4. For each issue, check proposed_value constraints
    for issue in critique.issues:
        # 5. Validate all issue labels against canonical 12 labels
        if issue.label not in LABEL_KEYS:
            return {
                "check": "label_validation",
                "error": f"Invalid label '{issue.label}' in critique issue. Must be one of: {LABEL_KEYS}",
                "issue": issue.model_dump(),
            }
        
        # 2. CLEAR_POLICY_CONFLICT and CLEAR_REPORT_CONFLICT: proposed_value must be 0 or 1
        if issue.issue_type in (CritiqueIssueType.CLEAR_POLICY_CONFLICT, CritiqueIssueType.CLEAR_REPORT_CONFLICT):
            if issue.proposed_value is None:
                return {
                    "check": "proposed_value_required",
                    "error": f"CLEAR_* issue for label '{issue.label}' must have proposed_value 0 or 1, got None",
                    "issue": issue.model_dump(),
                }
            if issue.proposed_value not in (0, 1):
                return {
                    "check": "proposed_value_binary",
                    "error": f"CLEAR_* issue for label '{issue.label}' proposed_value must be 0 or 1, got {issue.proposed_value}",
                    "issue": issue.model_dump(),
                }
            # 4. For CLEAR_*: proposed_value must differ from model1_value
            model1_val = model1_prediction.predictions[issue.label].value
            if issue.proposed_value == model1_val:
                return {
                    "check": "proposed_value_differs",
                    "error": f"CLEAR_* issue for label '{issue.label}' proposed_value ({issue.proposed_value}) must differ from model1_value ({model1_val})",
                    "issue": issue.model_dump(),
                }
        
        # 3. For UNRESOLVED_POLICY, REPORT_AMBIGUITY, INSUFFICIENT_EVIDENCE: proposed_value must be null
        if issue.issue_type in (CritiqueIssueType.UNRESOLVED_POLICY, CritiqueIssueType.REPORT_AMBIGUITY, CritiqueIssueType.INSUFFICIENT_EVIDENCE):
            if issue.proposed_value is not None:
                return {
                    "check": "proposed_value_null_required",
                    "error": f"{issue.issue_type.value} issue for label '{issue.label}' must have proposed_value=null, got {issue.proposed_value}",
                    "issue": issue.model_dump(),
                }
        
        # 6-7. If Model 2 supplies evidence, verify verbatim in ORIGINAL_REPORT
        if issue.evidence is not None and issue.evidence != "":
            if issue.evidence not in report:
                return {
                    "check": "evidence_verbatim",
                    "error": f"Evidence for label '{issue.label}' not found verbatim in ORIGINAL_REPORT: '{issue.evidence[:100]}...'",
                    "issue": issue.model_dump(),
                }
    
    # 8. Validate actionable: true iff at least one CLEAR_* issue exists
    has_clear_issue = any(
        issue.issue_type in (CritiqueIssueType.CLEAR_POLICY_CONFLICT, CritiqueIssueType.CLEAR_REPORT_CONFLICT)
        for issue in critique.issues
    )
    if critique.actionable != has_clear_issue:
        return {
            "check": "actionable_consistency",
            "error": f"actionable={critique.actionable} but has_clear_issue={has_clear_issue}",
            "issue_count": len(critique.issues),
            "clear_issues": [
                i.label for i in critique.issues
                if i.issue_type in (CritiqueIssueType.CLEAR_POLICY_CONFLICT, CritiqueIssueType.CLEAR_REPORT_CONFLICT)
            ],
        }
    
    # 9. Validate status consistency
    clear_issues_exist = has_clear_issue
    unresolved_ambiguity_issues = any(
        issue.issue_type in (CritiqueIssueType.UNRESOLVED_POLICY, CritiqueIssueType.REPORT_AMBIGUITY, CritiqueIssueType.INSUFFICIENT_EVIDENCE)
        for issue in critique.issues
    )
    
    expected_status = None
    if clear_issues_exist:
        expected_status = CritiqueStatus.FAIL
    elif unresolved_ambiguity_issues and not clear_issues_exist:
        expected_status = CritiqueStatus.AMBIGUOUS
    else:
        expected_status = CritiqueStatus.PASS
    
    if critique.status != expected_status:
        return {
            "check": "status_consistency",
            "error": f"status={critique.status.value} but expected={expected_status.value} (clear_issues={clear_issues_exist}, unresolved_ambiguity={unresolved_ambiguity_issues})",
            "issue_types": [i.issue_type.value for i in critique.issues],
        }
    
    return None


def run_model1_attempt(
    report: str,
    gold_context: str,
    policy: str,
    labels: list[str],
    infer,
    previous_prediction=None,
    validator_feedback=None,
    attempt: int = 1,
) -> "ReportPrediction":
    """
    Execute ONE Model 1 inference attempt with optional validator feedback.

    This is the semantic core of the retry mechanism — full 12-label regeneration
    with feedback injection — WITHOUT the multi-attempt orchestration loop.

    Extracted from run_report() for use by LangGraph orchestration.

    Args:
        report: The original MRI report text
        gold_context: Retrieved gold examples as formatted string
        policy: Canonical policy text
        labels: List of 12 label keys (LABEL_KEYS)
        infer: Inference callable (InferenceModel.run)
        previous_prediction: Previous Model 1 prediction for retry context
        validator_feedback: Model 2 critique feedback for retry reconsideration
        attempt: 1-indexed attempt number

    Returns:
        ReportPrediction with all 12 labels regenerated
    """
    # Load inference prompt config
    PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
    with open(PROMPTS_DIR / "inference.yaml", "r", encoding="utf-8") as f:
        INF_CFG = yaml.safe_load(f)

    # Prepare feedback for Model 1 - only for retry attempts
    current_feedback = validator_feedback if attempt > 1 else None

    # Build template variables for the inference prompt
    template_vars = {
        "original_report": report,
        "gold_analysis": policy,
        "retrieved_gold_examples": gold_context,
    }
    if current_feedback:
        feedback_json = json.dumps(current_feedback, ensure_ascii=False, indent=2)
        template_vars["validator_feedback_section"] = (
            "VALIDATOR FEEDBACK (for reconsideration, NOT as evidence):\n"
            f"{feedback_json}\n\n"
            "Re-read the ORIGINAL_REPORT from scratch. Validator feedback identifies a disputed interpretation; "
            "it is not automatically ground truth. Recompute the affected labels using the ORIGINAL_REPORT "
            "and GOLD ANNOTATION POLICY. Do not blindly copy the validator correction."
        )
    else:
        template_vars["validator_feedback_section"] = ""

    # Build user prompt
    user_prompt = INF_CFG["user_prompt_template"].format(**template_vars)

    # Call the inference model
    prediction = infer(
        system=INF_CFG["system_prompt"],
        user=user_prompt,
        validator_feedback=current_feedback,
        attempt=attempt,
    )

    return prediction


def run_report(
    report_id: str,
    report: str,
    infer,
    validate,
    context: dict,
    max_attempts: int,
    labels: list[str] | None = None,
):
    """
    Run the inference-validation loop for a single report.

    Retry logic:
    - Structural/transient failures: retry up to max_attempts
    - Semantic disagreements: retry up to max_attempts with Model 1 reconsidering
    - AMBIGUOUS: immediate needs_review (no retry)
    - After max_attempts: return NEEDS_REVIEW
    """
    previous = None
    feedback = None
    attempts = []
    semantic_disagreement_count = 0
    
    # Track history for instability detection - always use all 12 labels
    model1_history = {label: [] for label in LABEL_KEYS}
    validator_history = {label: [] for label in LABEL_KEYS}
    disputed_labels = set()
    
    # Track complete 12-label prediction vectors for MODEL1_STUCK detection
    model1_vector_history = []

    for number in range(1, max_attempts + 1):
        start = perf_counter()

        # Prepare feedback for Model 1 - only semantic disagreements, not as evidence
        # Validator feedback is for Model 1's awareness, NOT as evidence
        current_feedback = feedback if semantic_disagreement_count > 0 else None

        prediction = infer(
            report_id=report_id,
            report=report,
            context=context,
            previous_prediction=previous,
            validator_feedback=current_feedback,
            attempt=number,
            labels=labels,
            gold_examples=context.get("gold_examples", ""),
        )

        raw_validation = validate(
            report_id=report_id,
            report=report,
            prediction=prediction,
            attempt=number,
            labels=labels,
            gold_examples=context.get("gold_examples", ""),
        )

        # Track validator corrections from RAW validation (before filtering)
        for issue in raw_validation.issues:
            if issue.corrected is not None:
                validator_history[issue.label].append(issue.corrected)
                disputed_labels.add(issue.label)

        # Apply programmatic safety gate: filter to only real issues
        validation = _filter_real_issues(raw_validation)

        # Classify failure type
        failure_type = _classify_failure(validation)

        # Track Model 1 predictions per label
        for label, val in prediction.predictions.items():
            model1_history[label].append(val.value)
        
        # Track complete 12-label prediction vector
        pred_vector = tuple(prediction.predictions[label].value for label in LABEL_KEYS)
        model1_vector_history.append(pred_vector)

        # Detect Model 1 STUCK (same 12-label vector across attempts)
        stuck_labels = []
        if number >= 2:
            # Check if complete prediction vector is identical to previous
            if model1_vector_history[-1] == model1_vector_history[-2]:
                # Find which disputed labels are stuck
                stuck_labels = [
                    label for label in disputed_labels
                    if model1_history[label][-1] == model1_history[label][-2]
                ]
                if stuck_labels:
                    failure_type = FailureType.MODEL1_STUCK

        # Detect Validator Instability (correction flips) - deterministic oscillation
        unstable_labels = _detect_validator_oscillation(validator_history)
        if unstable_labels:
            failure_type = FailureType.VALIDATOR_INSTABILITY

        attempts.append(
            Attempt(
                number=number,
                prediction=prediction,
                validation=validation,
                elapsed_seconds=perf_counter() - start,
                failure_type=failure_type,
            )
        )

        # Handle AMBIGUOUS immediately - no retry
        if validation.status == "AMBIGUOUS":
            return {
                "report_id": report_id,
                "status": "needs_review",
                "attempts": attempts,
                "final_prediction": prediction,
                "review_reason": "Validator AMBIGUOUS - report/gold policy insufficient for confident correction",
            }

        # Validator instability should not result in PASS - escalate to needs_review
        if failure_type == FailureType.VALIDATOR_INSTABILITY:
            return {
                "report_id": report_id,
                "status": "needs_review",
                "attempts": attempts,
                "final_prediction": prediction,
                "review_reason": f"Validator instability detected on labels: {unstable_labels}",
            }

        if validation.passed:
            return {
                "report_id": report_id,
                "status": "passed",
                "attempts": attempts,
                "final_prediction": prediction,
                "review_reason": None,
            }

        # Track semantic disagreements for retry logic
        if failure_type == FailureType.SEMANTIC:
            semantic_disagreement_count += 1
            # Provide Model 2's corrections as feedback for Model 1's reconsideration
            # This is NOT evidence - it's guidance for reconsideration
            feedback = validation.model_dump()
        else:
            # Structural/transient failure - no semantic feedback needed
            feedback = None

        previous = prediction

    # Max attempts reached
    if semantic_disagreement_count > 0:
        return {
            "report_id": report_id,
            "status": "needs_review",
            "attempts": attempts,
            "final_prediction": attempts[-1].prediction,
            "review_reason": "Maximum attempts reached without validator PASS - semantic disagreement persists",
        }
    else:
        return {
            "report_id": report_id,
            "status": "needs_review",
            "attempts": attempts,
            "final_prediction": attempts[-1].prediction,
            "review_reason": "Maximum attempts reached without validator PASS - structural/transient failures",
        }