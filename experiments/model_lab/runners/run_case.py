"""Lesson 1.2: run one enterprise case against one model.

Why it exists:
    Provides a single-case runner for the enterprise decision benchmark.
    Allows testing one model on one case with controlled parameters.

What problem it solves:
    Enables systematic evaluation of LLM performance on enterprise decision
    tasks by running individual cases with configurable parameters.

Alternatives considered:
    - Running all cases at once: harder to debug individual failures.
    - No parameter control: can't test the effect of temperature, reasoning, etc.

Tradeoffs:
    - Single-case runner is simpler but requires a separate experiment runner
      for full matrix evaluation.

Failure modes:
    - API errors: retried with exponential backoff.
    - Malformed LLM response: parsing fails, result marked as invalid.
    - Missing API key: raises RuntimeError.

Testing:
    - Unit tests with mocked provider responses.
    - Integration tests require a valid API key.

Production implications:
    - In production, results should be stored in a database, not just files.
    - Retry logic should use exponential backoff with jitter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
from pydantic import ValidationError

from core.config import get_settings
from core.llm.config import get_api_key, get_model_config
from core.llm.parameters import GenerationParameters
from core.llm.providers.openrouter import OpenRouterProvider
from core.schemas.decision import (
    ChangeRiskDecision,
    InvoiceExceptionDecision,
    SupportEscalationDecision,
)


ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = ROOT / "experiments/model_lab/datasets"
PROMPTS_DIR = ROOT / "experiments/model_lab/prompts"
RESULTS_DIR = ROOT / "experiments/model_lab/results"
PROMPT_TEMPLATE = PROMPTS_DIR / "enterprise_decisions.yaml"

DATASET_FILES = {
    "support_escalation": DATASETS_DIR / "support_escalation.yaml",
    "invoice_exception": DATASETS_DIR / "invoice_exception.yaml",
    "change_risk": DATASETS_DIR / "change_risk.yaml",
}

DECISION_SCHEMAS = {
    "support_escalation": SupportEscalationDecision,
    "invoice_exception": InvoiceExceptionDecision,
    "change_risk": ChangeRiskDecision,
}


def load_prompt_template() -> dict:
    """Load the enterprise decision prompt template."""
    return yaml.safe_load(PROMPT_TEMPLATE.read_text(encoding="utf-8"))


def load_case(dataset: str, case_id: str) -> dict:
    """Load a single case from the specified dataset."""
    if dataset not in DATASET_FILES:
        available = ", ".join(sorted(DATASET_FILES))
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {available}"
        )

    cases_file = DATASET_FILES[dataset]
    data = yaml.safe_load(cases_file.read_text(encoding="utf-8"))

    for case in data["cases"]:
        if case["id"] == case_id:
            return case

    raise ValueError(f"Case not found: {case_id} in dataset '{dataset}'")


def build_prompt(case: dict, dataset: str, template: dict) -> str:
    """Build a prompt using the template and case data."""
    common = template["common"]
    decision_type = template["decision_types"][dataset]

    output_fields = decision_type["output_fields"]
    fields_str = "\n".join(f"{i}. {field}" for i, field in enumerate(output_fields, 1))

    return f"""
{common["instruction"]}

Decision type: {decision_type["description"]}

Case:

{json.dumps(case["input"], indent=2)}

Return your answer as valid JSON with exactly these fields:

{fields_str}
""".strip()


def parse_response(response_text: str, dataset: str) -> dict | None:
    """Parse the LLM response into structured fields.

    Attempts to parse the response as JSON first. If that fails,
    returns None to indicate a parsing failure.
    """
    try:
        data = json.loads(response_text.strip())
    except json.JSONDecodeError:
        return None

    schema = DECISION_SCHEMAS.get(dataset)
    if schema is None:
        return None

    try:
        validated = schema.model_validate(data)
        return validated.model_dump()
    except ValidationError:
        return None


def generate_with_retry(
    provider: OpenRouterProvider,
    prompt: str,
    parameters: GenerationParameters,
    max_retries: int,
) -> str:
    """Call the provider with retry logic.

    Why it exists:
        API calls can fail transiently (rate limits, network issues).
        Retrying with exponential backoff improves reliability.

    Tradeoffs:
        - Adds latency on failures.
        - May amplify load on the API during outages.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = provider.generate(prompt, parameters)
            return response.content
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  Attempt {attempt + 1} failed: {exc}. Retrying in {wait}s...")
                time.sleep(wait)

    raise last_error  # type: ignore[misc]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one enterprise case against one model (Lesson 1.2)."
    )

    parser.add_argument(
        "--profile",
        default="openrouter-free",
        help="Model profile from config/models.yaml",
    )

    parser.add_argument(
        "--dataset",
        default="support_escalation",
        choices=list(DATASET_FILES),
        help="Dataset to load cases from",
    )

    parser.add_argument(
        "--case",
        default="SUP-001",
        help="Case ID to run",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--reasoning",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (default: results/<dataset>/<case_id>.json)",
    )

    args = parser.parse_args()

    settings = get_settings()

    config = get_model_config(args.profile)

    if config.provider != "openrouter":
        raise ValueError(
            f"Lesson 1.2 expects an OpenRouter profile. "
            f"Got: {config.provider}"
        )

    api_key = get_api_key(config)

    case = load_case(args.dataset, args.case)

    template = load_prompt_template()
    prompt = build_prompt(case, args.dataset, template)

    provider = OpenRouterProvider(
        api_key=api_key,
        model=config.model,
    )

    parameters = GenerationParameters(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        reasoning=args.reasoning,
    )

    started = time.perf_counter()

    try:
        answer = generate_with_retry(
            provider,
            prompt,
            parameters,
            max_retries=settings.llm.max_retries,
        )
    except Exception as exc:
        print(f"\nAPI request failed after retries: {exc}")
        raise SystemExit(1)

    elapsed = time.perf_counter() - started

    parsed = parse_response(answer, args.dataset)

    result = {
        "case_id": args.case,
        "dataset": args.dataset,
        "profile": args.profile,
        "model": config.model,
        "parameters": {
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "reasoning": args.reasoning,
        },
        "expected": case["expected"],
        "response": answer,
        "parsed": parsed,
        "parse_success": parsed is not None,
        "latency_seconds": round(elapsed, 3),
    }

    print(json.dumps(result, indent=2))

    # Save results to file
    output_path = Path(args.output) if args.output else (
        RESULTS_DIR / args.dataset / f"{args.case}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
