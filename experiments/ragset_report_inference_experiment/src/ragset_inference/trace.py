from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_trace(path, *, report_id, stage, model, attempt, prompt, status,
                latency_seconds=None, input_tokens=None, output_tokens=None,
                error=None, provider=None, actual_model=None):
    """Write a model usage trace record.

    Args:
        path: Path to trace JSONL file
        report_id: StudyInstanceUID of the report
        stage: "inference" or "validation"
        model: Requested model name
        attempt: Attempt number (1-indexed)
        prompt: Full prompt sent to model
        status: "success" or "error"
        latency_seconds: Response latency
        input_tokens: Input token count (if available)
        output_tokens: Output token count (if available)
        error: Error message if status is "error"
        provider: Provider name (e.g., "openrouter", "nvidia")
        actual_model: Actual model used (may differ from requested for OpenRouter)
    """
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_id": report_id,
        "stage": stage,
        "requested_model": model,
        "actual_model": actual_model,
        "provider": provider,
        "attempt": attempt,
        "prompt_hash": prompt_hash(prompt),
        "status": status,
        "latency_seconds": latency_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": error,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
