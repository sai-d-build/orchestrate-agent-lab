from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Optional, Literal


def compute_prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_response_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_trace(
    path,
    *,
    report_id: str,
    stage: Literal["inference", "validation", "judgment"],
    model: str,
    attempt: int,
    prompt: str = "",
    status: Literal["success", "error"],
    latency_seconds: Optional[float] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    error: Optional[str] = None,
    provider: Optional[str] = None,
    actual_model: Optional[str] = None,
    # New fields for LangGraph orchestration
    graph_node: Optional[Literal["initialize", "inference", "critic", "judge", "finalize"]] = None,
    judge_action: Optional[str] = None,
    response_hash: Optional[str] = None,
    timestamp_utc: Optional[str] = None,
    provided_prompt_hash: Optional[str] = None,
):
    """Write a model usage trace record.

    Args:
        path: Path to trace JSONL file
        report_id: StudyInstanceUID of the report
        stage: "inference", "validation", or "judgment"
        model: Requested model name
        attempt: Attempt number (1-indexed)
        prompt: Full prompt sent to model (optional if provided_prompt_hash provided)
        status: "success" or "error"
        latency_seconds: Response latency
        input_tokens: Input token count (if available)
        output_tokens: Output token count (if available)
        error: Error message if status is "error"
        provider: Provider name (e.g., "openrouter", "nvidia")
        actual_model: Actual model used (may differ from requested for OpenRouter)
        graph_node: LangGraph node name ("initialize", "inference", "critic", "judge", "finalize")
        judge_action: Judge action (PASS, RETRY_MODEL1, AMBIGUOUS, NEEDS_REVIEW, STOP)
        response_hash: SHA256 hash of model response text
        timestamp_utc: Optional timestamp (ISO format). If not provided, current time is used.
        provided_prompt_hash: Optional pre-computed prompt hash. If not provided, computed from prompt.
    """
    # Compute prompt_hash: use provided prompt_hash, or compute from prompt if available
    computed_prompt_hash = provided_prompt_hash
    if not computed_prompt_hash and prompt:
        computed_prompt_hash = compute_prompt_hash(prompt)
    
    record = {
        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
        "report_id": report_id,
        "stage": stage,
        "requested_model": model,
        "actual_model": actual_model,
        "provider": provider,
        "attempt": attempt,
        "prompt_hash": computed_prompt_hash,
        "response_hash": response_hash,
        "status": status,
        "latency_seconds": latency_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": error,
        "graph_node": graph_node,
        "judge_action": judge_action,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
