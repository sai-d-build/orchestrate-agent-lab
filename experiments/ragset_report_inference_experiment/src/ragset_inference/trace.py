from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_trace(path, *, report_id, stage, model, attempt, prompt, status,
                latency_seconds=None, input_tokens=None, output_tokens=None,
                error=None):
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_id": report_id,
        "stage": stage,
        "model": model,
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
