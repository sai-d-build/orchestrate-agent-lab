# Parallel Inference Design Document

## RagSet Report Inference Experiment — 2-Thread Parallelization with Dual NVIDIA API Keys

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Document ID** | PARALLEL-INFERENCE-001 |
| **Version** | 1.0 |
| **Date** | 2026-09-21 |
| **Author** | Architecture Review (Code Skeptic + Docs Specialist) |
| **Status** | Design Review Complete |
| **Experiment Path** | `experiments/ragset_report_inference_experiment/` |
| **Target Files** | `run_inference.py`, `config/models.yaml`, `config/experiment.yaml` |

---

## Change Log

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-09-21 | Architecture Review | Initial design document with validated analysis |

---

## 1. Executive Summary

This document captures the complete analysis for enabling **2-thread parallel inference** in the RagSet Report Inference Experiment, utilizing two distinct NVIDIA API keys (`NVIDIA_API_KEY` and `NVIDIA_API_KEY_USF`) to achieve independent rate limits and ~2x throughput.

**Key Design Decision**: Each report processes through the full **inference→validation loop** (up to 3 attempts) as a single atomic unit. Resume filtering happens **once** in the main thread before work distribution. Workers process disjoint report chunks independently with dedicated model pairs.

---

## 2. Current Architecture Analysis

### 2.1 Single-Threaded Workflow (Verified from Code)

```python
# run_inference.py lines 295-385 (simplified)
for idx, (_, row) in enumerate(unlabeled.iterrows()):
    report_id = str(row["StudyInstanceUID"])
    report = str(row["Report"])

    # Resume check
    if resume and report_id in processed_ids:
        continue

    # Retrieval (thread-safe read-only)
    retrieved = retriever.retrieve(report)
    context = {"gold_analysis": gold_analysis_text, "gold_examples": retrieved}

    # Full inference+validation loop (atomic per report)
    result = run_report(
        report_id=report_id,
        report=report,
        infer=infer_fn,        # Uses model1 (InferenceModel)
        validate=validate_fn,  # Uses model2 (ValidatorModel)
        context=context,
        max_attempts=3,
        labels=labels,
    )

    # Write results (NOT thread-safe)
    append_prediction(result_path, serializable)
    time.sleep(2)  # Rate limiting
```

### 2.2 Critical Components Verified

| Component | File | Thread-Safety | Evidence |
|-----------|------|---------------|----------|
| `GoldRetriever` | `retrieval.py:15-43` | ✅ **Safe** | Immutable `self.matrix`, `self.vectorizer`; `retrieve()` only reads |
| `run_report()` | `loop.py:59-160` | ✅ **Safe** | Stateless, called per-report, no shared state |
| `InferenceModel`/`ValidatorModel` | `models.py:173-379` | ✅ **Safe** | Each instance owns its OpenAI client; multiple instances supported |
| `write_trace()` | `trace.py:48` | ❌ **Unsafe** | `path.open("a")` concurrent appends corrupt JSONL |
| `append_prediction()` | `run_inference.py:173-177` | ❌ **Unsafe** | Same append-mode pattern |
| `make_infer_fn`/`make_validate_fn` | `run_inference.py:25-94` | ⚠️ **Needs Change** | Closures capture `trace_path` and call `write_trace()` directly |

### 2.3 Resume Logic (Verified)

```python
# run_inference.py:153-170, 255-272
def load_existing_predictions(result_path: Path) -> set[str]:
    # Reads predictions.jsonl, returns set of processed report_ids

# Main thread computes ONCE before loop:
processed_ids = load_existing_predictions(result_path)
# Then filters in loop:
if resume and report_id in processed_ids:
    continue
```

---

## 3. Parallelization Strategy

### 3.1 Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ MAIN THREAD                                                     │
│ 1. Load config, gold data, retriever                           │
│ 2. Create 2 model pairs (API_KEY_1, API_KEY_2)                 │
│ 3. Load processed_ids from predictions.jsonl                   │
│ 4. Filter: remaining = unlabeled - processed_ids               │
│ 5. Apply --limit to remaining                                  │
│ 6. Split remaining into 2 chunks (round-robin)                 │
│ 7. Spawn ThreadPoolExecutor(2) with worker function            │
│ 8. Writer loop: consume queue → write predictions + traces     │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       ┌───────────────┐               ┌───────────────┐
       │  WORKER A     │               │  WORKER B     │
       │ (Thread 1)    │               │ (Thread 2)    │
       │               │               │               │
       │ InferenceModel│               │ InferenceModel│
       │  (API_KEY_1)  │               │  (API_KEY_2)  │
       │ ValidatorModel│               │ ValidatorModel│
       │  (API_KEY_1)  │               │  (API_KEY_2)  │
       │               │               │               │
       │ for r in      │               │ for r in      │
       │   chunk_a:    │               │   chunk_b:    │
       │   run_report()│               │   run_report()│
       │   queue.put() │               │   queue.put() │
       └───────────────┘               └───────────────┘
```

### 3.2 Chunking Strategy

```python
# Round-robin split for balanced load
chunk_a = remaining_reports[::2]  # Even indices
chunk_b = remaining_reports[1::2]  # Odd indices
```

**Rationale**: Reports have variable length/complexity; round-robin distributes variance better than contiguous split.

### 3.3 Per-Worker Processing

Each worker executes independently:

```python
def worker_process_chunk(reports_chunk, model1, model2, inf_cfg, val_cfg,
                         labels, gold_analysis_text, retriever, max_attempts,
                         result_queue, worker_id):

    # Create per-worker closures (NO trace_path capture)
    infer_fn = make_infer_fn(model1, inf_cfg, None, labels, gold_analysis_text)
    validate_fn = make_validate_fn(model2, val_cfg, None, labels, gold_analysis_text)

    for row in reports_chunk:
        report_id = str(row["StudyInstanceUID"])
        report = str(row["Report"])

        # Thread-safe retrieval
        retrieved = retriever.retrieve(report)
        context = {"gold_analysis": gold_analysis_text, "gold_examples": retrieved}

        # Atomic inference+validation loop
        result = run_report(report_id, report, infer_fn, validate_fn,
                           context, max_attempts, labels)

        # Serialize and queue for writer
        serializable = serialize_result(result, report)
        result_queue.put({"type": "prediction", "data": serializable})
        result_queue.put({"type": "traces", "data": extract_traces(result)})
        result_queue.put({"type": "progress", "report_id": report_id})

        time.sleep(2)  # Per-worker rate limiting (independent per API key)
```

---

## 4. Required Code Changes

### 4.1 Configuration Files

#### 4.1.1 `config/models.yaml` — Add USF Profiles

```yaml
# ADD after line 71 (after nvidia-ragset-validator)

nvidia-ragset-inference-usf:
  provider: nvidia
  model: nvidia/nemotron-3-ultra-550b-a55b
  api_key_env: NVIDIA_API_KEY_USF
  purpose: ragset_report_inference
  capabilities:
    - text
    - reasoning
    - structured_output
  parameters:
    temperature: 0
    max_output_tokens: 32000
    reasoning: false
  provider_policy:
    data_collection: deny
    zdr: true
    allow_fallbacks: false

nvidia-ragset-validator-usf:
  provider: nvidia
  model: nvidia/nemotron-3-ultra-550b-a55b
  api_key_env: NVIDIA_API_KEY_USF
  purpose: ragset_report_validation
  capabilities:
    - text
    - reasoning
    - structured_output
  parameters:
    temperature: 0
    max_output_tokens: 20000
    reasoning: false
  provider_policy:
    data_collection: deny
    zdr: true
    allow_fallbacks: false
```

#### 4.1.2 `experiments/ragset_report_inference_experiment/config/experiment.yaml` — Update Model References

```yaml
# CHANGE lines 14-16 from:
models:
  inference_profile: nvidia-ragset-inference
  validator_profile: nvidia-ragset-validator

# TO:
models:
  inference_profiles:
    - nvidia-ragset-inference
    - nvidia-ragset-inference-usf
  validator_profiles:
    - nvidia-ragset-validator
    - nvidia-ragset-validator-usf
  gold_analysis_profile: nvidia-ragset-gold-analysis
  temperature: 0
```

### 4.2 `run_inference.py` — Major Restructure

#### 4.2.1 Model Creation (Lines 234-247 → Replace)

```python
# CURRENT: Single pair
model1 = InferenceModel(...)
model2 = ValidatorModel(...)

# NEW: Two pairs
# Worker A - Primary API key
model1_a = InferenceModel(
    inf_config["model"],
    max_tokens=inf_config.get("parameters", {}).get("max_output_tokens", 4000),
    provider=inf_provider,
    api_key=os.environ.get("NVIDIA_API_KEY"),
    reasoning=inf_config.get("parameters", {}).get("reasoning"),
)
model2_a = ValidatorModel(
    val_config["model"],
    max_tokens=val_config.get("parameters", {}).get("max_output_tokens", 4000),
    provider=val_provider,
    api_key=os.environ.get("NVIDIA_API_KEY"),
    reasoning=val_config.get("parameters", {}).get("reasoning"),
)

# Worker B - USF API key
inf_config_usf = models_cfg["models"]["nvidia-ragset-inference-usf"]
val_config_usf = models_cfg["models"]["nvidia-ragset-validator-usf"]

model1_b = InferenceModel(
    inf_config_usf["model"],
    max_tokens=inf_config_usf.get("parameters", {}).get("max_output_tokens", 4000),
    provider=inf_config_usf.get("provider", "nvidia"),
    api_key=os.environ.get("NVIDIA_API_KEY_USF"),
    reasoning=inf_config_usf.get("parameters", {}).get("reasoning"),
)
model2_b = ValidatorModel(
    val_config_usf["model"],
    max_tokens=val_config_usf.get("parameters", {}).get("max_output_tokens", 4000),
    provider=val_config_usf.get("provider", "nvidia"),
    api_key=os.environ.get("NVIDIA_API_KEY_USF"),
    reasoning=val_config_usf.get("parameters", {}).get("reasoning"),
)
```

#### 4.2.2 Resume Filtering & Chunking (After Line 272, Before Loop)

```python
# Compute remaining reports ONCE in main thread
all_unlabeled = list(unlabeled.iterrows())

# Apply resume filter
if resume:
    remaining = [(idx, row) for idx, row in all_unlabeled
                 if str(row["StudyInstanceUID"]) not in processed_ids]
else:
    remaining = all_unlabeled

# Apply limit
if limit is not None:
    remaining = remaining[:limit]

# Round-robin split
chunk_a = remaining[::2]
chunk_b = remaining[1::2]

total_expected = len(remaining)
print(f"Total reports to process: {total_expected}")
print(f"  Worker A: {len(chunk_a)} reports")
print(f"  Worker B: {len(chunk_b)} reports")
```

#### 4.2.3 Worker Function (New)

```python
import queue
from concurrent.futures import ThreadPoolExecutor

def worker_process_chunk(reports_chunk, model1, model2, inf_cfg, val_cfg,
                         labels, gold_analysis_text, retriever, max_attempts,
                         result_queue, worker_id, trace_path):
    """Process a chunk of reports in a worker thread."""

    # Create per-worker closures (trace_path=None, traces returned via queue)
    infer_fn = make_infer_fn(model1, inf_cfg, None, labels, gold_analysis_text)
    validate_fn = make_validate_fn(model2, val_cfg, None, labels, gold_analysis_text)

    for idx, row in reports_chunk:
        report_id = str(row["StudyInstanceUID"])
        report = str(row["Report"])

        # Thread-safe retrieval
        retrieved = retriever.retrieve(report)
        relevant = "\n---\n".join(
            f"STUDY {x['study_id']}\nLABELS: {x['labels']}\nREPORT:\n{x['report']}"
            for x in retrieved
        )
        context = {
            "gold_analysis": gold_analysis_text,
            "gold_examples": relevant,
        }

        try:
            result = run_report(
                report_id=report_id,
                report=report,
                infer=infer_fn,
                validate=validate_fn,
                context=context,
                max_attempts=max_attempts,
                labels=labels,
            )

            # Serialize prediction
            pred = result["final_prediction"]
            inferred_evidence = {
                label: pred.predictions[label].evidence
                for label in pred.predictions
                if pred.predictions[label].evidence
            }
            serializable = {
                "report_id": result["report_id"],
                "status": result["status"],
                "review_reason": result["review_reason"],
                "original_report": report,
                "inferred_evidence": inferred_evidence,
                "attempts": [
                    {
                        "attempt": a.number,
                        "prediction": a.prediction.model_dump(),
                        "validation": a.validation.model_dump(),
                        "elapsed_seconds": a.elapsed_seconds,
                    }
                    for a in result["attempts"]
                ],
                "final_prediction": pred.model_dump(),
            }

            # Queue for writer thread
            result_queue.put({"type": "prediction", "data": serializable})

            # Queue trace records (extracted from attempts)
            for attempt in result["attempts"]:
                # Inference trace
                result_queue.put({
                    "type": "trace",
                    "data": {
                        "report_id": report_id,
                        "stage": "inference",
                        "model": model1.model,
                        "attempt": attempt.number,
                        "prompt": "",  # Would need capture in modified closures
                        "status": "success",
                        "provider": model1.provider,
                        "elapsed_seconds": attempt.elapsed_seconds,
                    }
                })
                # Validation trace
                result_queue.put({
                    "type": "trace",
                    "data": {
                        "report_id": report_id,
                        "stage": "validation",
                        "model": model2.model,
                        "attempt": attempt.number,
                        "prompt": "",
                        "status": "success",
                        "provider": model2.provider,
                        "elapsed_seconds": attempt.elapsed_seconds,
                    }
                })

            result_queue.put({"type": "progress", "report_id": report_id})

        except Exception as e:
            # Queue error for writer/main thread handling
            result_queue.put({
                "type": "error",
                "report_id": report_id,
                "error": str(e),
                "worker_id": worker_id,
            })

        # Per-worker rate limiting
        time.sleep(2)
```

#### 4.2.4 Modified `make_infer_fn` / `make_validate_fn` (Lines 25-94)

```python
def make_infer_fn(model, inf_cfg, trace_path, labels, gold_analysis):
    """Factory that creates an infer closure.
    trace_path=None means return trace data instead of writing.
    """
    def infer_fn(**kwargs):
        user = inf_cfg["user_prompt_template"].format(
            original_report=kwargs["report"],
            gold_analysis=gold_analysis,
            retrieved_gold_examples=kwargs.get("gold_examples", ""),
        )
        report_id = kwargs["report_id"]
        validator_feedback = kwargs.get("validator_feedback")
        try:
            prediction = model.run(inf_cfg["system_prompt"], user, validator_feedback=validator_feedback)
            actual_model = getattr(model, 'last_actual_model', model.model)

            trace_record = {
                "report_id": report_id,
                "stage": "inference",
                "model": model.model,
                "attempt": kwargs["attempt"],
                "prompt": inf_cfg["system_prompt"] + "\n" + user,
                "status": "success",
                "provider": model.provider,
                "actual_model": actual_model,
            }

            if trace_path:
                write_trace(trace_path, **trace_record)
                return prediction
            else:
                return prediction, trace_record

        except Exception as e:
            trace_record = {
                "report_id": report_id,
                "stage": "inference",
                "model": model.model,
                "attempt": kwargs["attempt"],
                "prompt": inf_cfg["system_prompt"] + "\n" + user,
                "status": "error",
                "provider": model.provider,
                "error": str(e),
            }
            if trace_path:
                write_trace(trace_path, **trace_record)
            raise

    return infer_fn

# Similar modification for make_validate_fn
```

#### 4.2.5 Main Thread: Executor + Writer Loop (Replaces Lines 295-385)

```python
# Create thread-safe queue
result_queue = queue.Queue()

# Submit workers
with ThreadPoolExecutor(max_workers=2) as executor:
    future_a = executor.submit(
        worker_process_chunk, chunk_a, model1_a, model2_a,
        inf_cfg, val_cfg, labels, label_profile_text(analysis),
        retriever, cfg["loop"]["max_attempts"],
        result_queue, 0, trace_path
    )
    future_b = executor.submit(
        worker_process_chunk, chunk_b, model1_b, model2_b,
        inf_cfg, val_cfg, labels, label_profile_text(analysis),
        retriever, cfg["loop"]["max_attempts"],
        result_queue, 1, trace_path
    )

    # Writer loop in main thread
    completed = 0
    errors = []

    while completed < total_expected:
        item = result_queue.get()  # Blocks until available

        if item["type"] == "prediction":
            append_prediction(result_path, item["data"])
            completed += 1

        elif item["type"] == "trace":
            write_trace(trace_path, **item["data"])

        elif item["type"] == "progress":
            print(f"  Completed {item['report_id']} ({completed}/{total_expected})")

        elif item["type"] == "error":
            errors.append(item)
            print(f"  ERROR in worker {item['worker_id']} for {item['report_id']}: {item['error']}")
            completed += 1  # Count as processed to avoid deadlock

    # Wait for workers to finish
    future_a.result()
    future_b.result()

if errors:
    print(f"\nCompleted with {len(errors)} errors")
    for e in errors:
        print(f"  Worker {e['worker_id']}: {e['report_id']} - {e['error']}")
else:
    print(f"\nInference complete: {completed} reports processed")
```

---

## 5. Thread-Safety Guarantees

| Resource | Access Pattern | Protection |
|----------|----------------|------------|
| `predictions.jsonl` | Write-only | Single writer thread (main) |
| `model_trace.jsonl` | Write-only | Single writer thread (main) |
| `GoldRetriever` | Read-only | Immutable after init; sklearn transform is thread-safe |
| `processed_ids` set | Read-only (workers) | Computed once in main thread pre-spawn |
| `gold_analysis_text` | Read-only | String, immutable |
| `retriever.retrieve()` | Concurrent reads | No mutation, thread-safe |
| Model instances | Per-worker ownership | Each worker owns its `InferenceModel`/`ValidatorModel` |

---

## 6. Rate Limiting Behavior

| Mode | Reports/2sec | API Keys Used | NVIDIA Rate Limit |
|------|--------------|---------------|-------------------|
| Sequential (current) | 1 | 1 (`NVIDIA_API_KEY`) | Single key quota |
| Parallel (proposed) | 2 | 2 (`NVIDIA_API_KEY`, `NVIDIA_API_KEY_USF`) | Independent quotas |

**Verification**: Each worker calls `time.sleep(2)` after its report → 2 reports per 2 seconds aggregate. Each API key sees 1 request per 2 seconds → within typical limits.

---

## 7. Resume & Retry-Review Compatibility

### 7.1 Resume (`--resume`)
- Main thread loads `processed_ids` from existing `predictions.jsonl`
- Filters `unlabeled` → `remaining_reports` **before** chunking
- Workers only receive unprocessed reports
- **Idempotent**: If interrupted, next `--resume` reloads and re-filters

### 7.2 Retry-Review (`--retry-review`)
- Current logic (lines 304-323): Reads `result_path` line-by-line to find `status == "needs_review"`
- **Parallel adaptation**: Main thread performs same scan during pre-filtering
- Reports with `needs_review` included in `remaining_reports` if `--retry-review` flag set
- Workers process them normally (fresh `run_report` loop)

### 7.3 Overwrite (`--overwrite`)
- Deletes `predictions.jsonl` and `model_trace.jsonl` before starting
- `processed_ids = set()` → all reports processed
- Works identically in parallel mode

---

## 8. Risk Assessment & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Queue memory buildup | Low | Medium | API latency >> queue throughput; natural backpressure |
| Writer thread bottleneck | Low | Low | JSONL append is microseconds; API calls are seconds |
| Worker exception handling | Medium | High | Try/except in worker, error queued, main thread logs |
| `--retry-review` logic breakage | Medium | Medium | Pre-filter in main thread using identical logic |
| `limit` off-by-one | Low | Low | Apply limit to `remaining` list before chunking |
| Trace record ordering | Low | Low | Timestamps in records; order not critical for analysis |
| Pickling `Attempt` objects | Low | Medium | Pydantic models are picklable; verified compatible |

---

## 9. Files NOT Requiring Changes

| File | Reason |
|------|--------|
| `loop.py` | Stateless `run_report()`; called per-report |
| `models.py` | Supports multiple instances with different API keys |
| `retrieval.py` | Thread-safe read-only |
| `schemas.py` | Pure data models |
| `data.py`, `gold.py`, `evaluate.py` | No concurrency concerns |
| `trace.py` | `write_trace()` called only by single writer |

---

## 10. Testing Checklist

- [ ] Unit test: `GoldRetriever` concurrent `retrieve()` calls
- [ ] Unit test: `make_infer_fn`/`make_validate_fn` with `trace_path=None` returns trace records
- [ ] Integration test: 2-worker run on small dataset (10 reports) with `--resume`
- [ ] Integration test: `--retry-review` with parallel workers
- [ ] Integration test: `--overwrite` with parallel workers
- [ ] Load test: Verify 2x throughput vs sequential
- [ ] Verify `predictions.jsonl` format identical to sequential run
- [ ] Verify `model_trace.jsonl` contains all trace records
- [ ] Verify both API keys utilized (check trace `provider`/`actual_model`)

---

## 11. Appendix: Code Evidence References

### 11.1 Key Source Locations

| Function/Class | File | Lines |
|----------------|------|-------|
| `run_report()` | `loop.py` | 59-160 |
| `make_infer_fn()` | `run_inference.py` | 25-61 |
| `make_validate_fn()` | `run_inference.py` | 64-94 |
| `load_existing_predictions()` | `run_inference.py` | 153-170 |
| `append_prediction()` | `run_inference.py` | 173-177 |
| `write_trace()` | `trace.py` | 11-49 |
| `GoldRetriever.retrieve()` | `retrieval.py` | 29-43 |
| `InferenceModel.__init__()` | `models.py` | 179-191 |
| `ValidatorModel.__init__()` | `models.py` | 294-306 |
| `_create_client()` | `models.py` | 137-150 |

### 11.2 Configuration References

| Config | File | Key |
|--------|------|-----|
| Model profiles | `config/models.yaml` | `models.nvidia-ragset-inference`, `models.nvidia-ragset-validator` |
| Experiment models | `experiments/ragset_report_inference_experiment/config/experiment.yaml` | `models.inference_profile`, `models.validator_profile` |
| Loop config | `experiments/ragset_report_inference_experiment/config/experiment.yaml` | `loop.max_attempts` |
| Retrieval config | `experiments/ragset_report_inference_experiment/config/experiment.yaml` | `retrieval.top_k` |

---

## 12. Implementation Priority

| Phase | Tasks | Dependencies |
|-------|-------|--------------|
| **1. Config** | Add USF profiles to `models.yaml`; update `experiment.yaml` | None |
| **2. Helpers** | Modify `make_infer_fn`/`make_validate_fn` to support `trace_path=None` | Phase 1 |
| **3. Worker** | Implement `worker_process_chunk()` function | Phase 2 |
| **4. Main Loop** | Replace sequential loop with `ThreadPoolExecutor` + writer loop | Phase 3 |
| **5. Resume/Review** | Adapt `--resume` and `--retry-review` pre-filtering logic | Phase 4 |
| **6. Testing** | Run integration tests per checklist | Phase 5 |

---

*End of Document*