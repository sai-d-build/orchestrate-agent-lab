"""Enterprise model benchmark runner.

Why it exists:
    Orchestrates the full experiment matrix defined in
    experiments/model_lab/configs/experiment_matrix.yaml.

What problem it solves:
    Runs multiple models across multiple parameter combinations and
    multiple cases, with repeated runs for consistency measurement.

Alternatives considered:
    - Running cases one at a time: tedious, error-prone, no aggregation.
    - Using a workflow engine: overkill for Phase 0.

Tradeoffs:
    - Sequential execution is simpler but slower than parallel.
    - Results are stored as individual JSON files, requiring a
      separate aggregation step.

Failure modes:
    - API errors: individual case failures are logged but don't stop
      the experiment.
    - Missing API keys: experiment fails early with a clear error.

Testing:
    - Unit tests with mocked providers.
    - Integration tests require valid API keys.

Production implications:
    - In production, use a task queue for parallel execution.
    - Store results in a database for easy querying.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import yaml

from core.config import get_settings
from core.llm.config import get_api_key, get_model_config
from core.llm.parameters import GenerationParameters
from core.llm.providers.openrouter import OpenRouterProvider
from experiments.model_lab.evaluation.scoring import (
    estimate_cost,
    score_consistency,
    score_decision,
)
from experiments.model_lab.runners.run_case import (
    DATASET_FILES,
    build_prompt,
    load_case,
    load_prompt_template,
    parse_response,
)


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_CONFIG = ROOT / "experiments/model_lab/configs/experiment_matrix.yaml"
RESULTS_DIR = ROOT / "experiments/model_lab/results"


def load_experiment_config() -> dict:
    """Load the experiment matrix configuration."""
    return yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))


def load_all_cases(dataset: str) -> list[dict]:
    """Load all cases from a dataset."""
    cases_file = DATASET_FILES[dataset]
    data = yaml.safe_load(cases_file.read_text(encoding="utf-8"))
    return data["cases"]


def generate_parameter_combinations(config: dict) -> list[dict]:
    """Generate all parameter combinations from the experiment matrix."""
    params = config.get("parameters", {})

    keys = list(params.keys())
    values = [params[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def run_single_case(
    profile: str,
    dataset: str,
    case_id: str,
    parameters: GenerationParameters,
    max_retries: int,
) -> dict:
    """Run a single case against a single model profile."""
    config = get_model_config(profile)

    if config.provider != "openrouter":
        raise ValueError(
            f"Experiment expects OpenRouter profiles. Got: {config.provider}"
        )

    api_key = get_api_key(config)

    case = load_case(dataset, case_id)
    template = load_prompt_template()
    prompt = build_prompt(case, dataset, template)

    provider = OpenRouterProvider(
        api_key=api_key,
        model=config.model,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = provider.generate(prompt, parameters)
            break
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  Attempt {attempt + 1} failed: {exc}. Retrying in {wait}s...")
                time.sleep(wait)
    else:
        raise last_error  # type: ignore[misc]

    parsed = parse_response(response.content, dataset)

    return {
        "case_id": case_id,
        "dataset": dataset,
        "profile": profile,
        "model": config.model,
        "parameters": {
            "temperature": parameters.temperature,
            "max_output_tokens": parameters.max_output_tokens,
            "reasoning": parameters.reasoning,
        },
        "expected": case["expected"],
        "category": case.get("category", "unknown"),
        "response": response.content,
        "parsed": parsed,
        "parse_success": parsed is not None,
        "latency_seconds": response.latency_seconds,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
    }


def run_experiment(
    config: dict,
    datasets: list[str] | None = None,
    profiles: list[str] | None = None,
) -> list[dict]:
    """Run the full experiment matrix.

    Iterates over all profiles, parameter combinations, datasets, and cases.
    For each combination, runs the case multiple times (runs_per_case)
    to measure consistency.
    """
    settings = get_settings()
    max_retries = settings.llm.max_retries

    if datasets is None:
        datasets = list(DATASET_FILES.keys())

    if profiles is None:
        profiles = config.get("profiles", [])

    runs_per_case = config.get("runs_per_case", 1)
    param_combos = generate_parameter_combinations(config)

    all_results: list[dict] = []

    for profile in profiles:
        for params in param_combos:
            parameters = GenerationParameters(
                temperature=params.get("temperature"),
                max_output_tokens=params.get("max_output_tokens"),
                reasoning=params.get("reasoning"),
            )

            for dataset in datasets:
                cases = load_all_cases(dataset)

                for case in cases:
                    case_id = case["id"]
                    print(f"\nRunning {profile} / {dataset} / {case_id} "
                          f"(temp={params.get('temperature')}, "
                          f"reasoning={params.get('reasoning')})")

                    run_responses: list[dict | None] = []
                    run_results: list[dict] = []

                    for run_idx in range(runs_per_case):
                        try:
                            result = run_single_case(
                                profile=profile,
                                dataset=dataset,
                                case_id=case_id,
                                parameters=parameters,
                                max_retries=max_retries,
                            )
                            run_responses.append(result["parsed"])
                            run_results.append(result)
                            print(f"  Run {run_idx + 1}/{runs_per_case}: "
                                  f"parsed={result['parse_success']}, "
                                  f"latency={result['latency_seconds']}s")
                        except Exception as exc:
                            print(f"  Run {run_idx + 1}/{runs_per_case} FAILED: {exc}")
                            run_responses.append(None)
                            run_results.append({
                                "case_id": case_id,
                                "dataset": dataset,
                                "profile": profile,
                                "error": str(exc),
                                "run_index": run_idx,
                            })

                    # Score the first successful run
                    scored_result = run_results[0] if run_results else {}
                    if scored_result.get("parsed") is not None:
                        scores = score_decision(
                            scored_result["expected"],
                            scored_result["parsed"],
                        )
                        scored_result["scores"] = scores

                        # Consistency score across runs
                        scored_result["consistency_score"] = score_consistency(run_responses)

                        # Cost estimate
                        scored_result["estimated_cost"] = estimate_cost(
                            scored_result.get("total_tokens"),
                            scored_result.get("model", ""),
                        )

                    all_results.extend(run_results)

    return all_results


def save_results(results: list[dict], output_file: str | None = None) -> Path:
    """Save experiment results to a JSON file."""
    if output_file is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = RESULTS_DIR / f"experiment-{timestamp}.json"
    else:
        output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full enterprise model benchmark experiment."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to run (default: all)",
    )

    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Model profiles to run (default: all from experiment matrix)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results",
    )

    args = parser.parse_args()

    config = load_experiment_config()

    print(f"Experiment config: {EXPERIMENT_CONFIG}")
    print(f"Profiles: {config.get('profiles', [])}")
    print(f"Runs per case: {config.get('runs_per_case', 1)}")
    print(f"Datasets: {args.datasets or list(DATASET_FILES.keys())}")

    results = run_experiment(
        config=config,
        datasets=args.datasets,
        profiles=args.profiles,
    )

    output_path = save_results(results, args.output)

    print(f"\n{'='*60}")
    print(f"Experiment complete. {len(results)} results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
