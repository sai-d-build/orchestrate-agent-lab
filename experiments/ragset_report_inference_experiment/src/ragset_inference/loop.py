from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from .schemas import ReportPrediction, ValidationResult, ValidationIssue, LABEL_KEYS


class FailureType(Enum):
    """Classification of failure types for proper retry handling."""
    STRUCTURAL = "structural"           # JSON parse error, schema validation error
    SEMANTIC = "semantic"               # Model 2 identifies genuine binary disagreement
    TRANSIENT = "transient"             # API error, rate limit, None content
    MODEL1_STUCK = "model1_stuck"       # Model 1 repeats same value for disputed label
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

        # Detect Model 1 STUCK (same value for disputed label across attempts)
        if number > 1 and disputed_labels:
            stuck_labels = [
                label for label in disputed_labels
                if len(set(model1_history[label])) == 1
            ]
            if stuck_labels:
                failure_type = FailureType.MODEL1_STUCK

        # Detect Validator Instability (correction flips)
        unstable_labels = [
            label for label, history in validator_history.items()
            if len(set(history)) > 1
        ]
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