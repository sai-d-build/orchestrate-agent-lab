from dataclasses import dataclass
from time import perf_counter
from .schemas import ReportPrediction, ValidationResult


@dataclass
class Attempt:
    number: int
    prediction: ReportPrediction
    validation: ValidationResult
    elapsed_seconds: float


def run_report(
    report_id: str,
    report: str,
    infer,
    validate,
    context: dict,
    max_attempts: int,
    labels: list[str] | None = None,
):
    previous = None
    feedback = None
    attempts = []

    for number in range(1, max_attempts + 1):
        start = perf_counter()

        prediction = infer(
            report_id=report_id,
            report=report,
            context=context,
            previous_prediction=previous,
            validator_feedback=feedback,
            attempt=number,
            labels=labels,
        )

        validation = validate(
            report_id=report_id,
            report=report,
            prediction=prediction,
            attempt=number,
            labels=labels,
        )

        attempts.append(
            Attempt(
                number=number,
                prediction=prediction,
                validation=validation,
                elapsed_seconds=perf_counter() - start,
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

        previous = prediction
        feedback = validation.model_dump()

    return {
        "report_id": report_id,
        "status": "needs_review",
        "attempts": attempts,
        "final_prediction": attempts[-1].prediction,
        "review_reason": "Maximum attempts reached without validator PASS",
    }
