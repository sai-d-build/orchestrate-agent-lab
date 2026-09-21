from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from .schemas import ReportPrediction, ValidationResult, ValidationIssue


class FailureType(Enum):
    """Classification of failure types for proper retry handling."""
    STRUCTURAL = "structural"      # JSON parse error, schema validation error
    SEMANTIC = "semantic"          # Model 2 identifies genuine binary disagreement
    TRANSIENT = "transient"        # API error, rate limit, None content


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
    """
    real_issues = [
        issue for issue in validation.issues
        if issue.corrected is not None and issue.predicted != issue.corrected
    ]
    return ValidationResult(
        status="PASS" if not real_issues else "FAIL",
        issues=real_issues,
    )


def _classify_failure(validation: ValidationResult, error: Exception | None = None) -> FailureType:
    """Classify the type of failure for proper retry handling."""
    if error is not None:
        # API errors, JSON parse errors, schema validation errors
        return FailureType.TRANSIENT if "None content" in str(error) or "rate limit" in str(error).lower() else FailureType.STRUCTURAL
    
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
    - After max_attempts: return NEEDS_REVIEW if semantic disagreement persists
    """
    previous = None
    feedback = None
    attempts = []
    semantic_disagreement_count = 0

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

        validation = validate(
            report_id=report_id,
            report=report,
            prediction=prediction,
            attempt=number,
            labels=labels,
        )

        # Apply programmatic safety gate: filter to only real issues
        validation = _filter_real_issues(validation)

        # Classify failure type
        failure_type = _classify_failure(validation)

        attempts.append(
            Attempt(
                number=number,
                prediction=prediction,
                validation=validation,
                elapsed_seconds=perf_counter() - start,
                failure_type=failure_type,
            )
        )

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