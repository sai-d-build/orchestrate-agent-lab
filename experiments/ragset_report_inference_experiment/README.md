# RagSet Report Label Inference Experiment

Standalone experiment for the existing RagSet `train.csv` — knee MRI report label extraction using a two-agent inference/validation loop with retrieval-augmented context.

---

## Project Evolution

### Phase 1: Initial Design (v1)
- **Goal**: Extract 12 binary labels from knee MRI reports using gold-labeled examples as reference
- **Architecture**: Two-agent loop (Model 1 infers → Pydantic validates → Model 2 validates → retry on failure)
- **Retrieval**: TF-IDF over gold reports with data leakage prevention (`retrieve_excluding`)
- **Model Backend**: OpenAI Responses API with structured Pydantic outputs
- **Key Files**: `run_inference.py`, `analyze_gold.py`, `validate_agent.py`, `loop.py`, `models.py`, `schemas.py`

### Phase 2: OpenRouter Free Model Integration (v2)
- **Migration**: Switched from OpenAI Responses API to OpenRouter `chat.completions` with JSON mode
- **Reason**: OpenAI Responses API `responses.parse` unsupported on free-tier models
- **Models Tested**: `nvidia/nemotron-3-super-120b-a12b`, `meta-llama/llama-3.1-70b-instruct:free`, `mistralai/mistral-7b-instruct:free`
- **Schema Redesign**: Flat JSON with 12 labels (`ACL`, `MCL`, `Medial_Meniscus`, `Lateral_Meniscus`, `Medial_OA`, `Lateral_OA`, `PF_OA`, `Effusion`, `Synovitis`, `Bakers`, `Contusion`, `Fracture`)
- **CLI Enhancements**: Added `--resume`, `--overwrite`, `--limit` flags
- **Resumable Execution**: Incremental persistence after each prediction, skip already-processed reports
- **Config-Driven**: `max_output_tokens` read from `config/models.yaml`
- **Rate Limiting**: 2-second wait between reports

### Phase 3: NVIDIA Nemotron Ultra Integration (v3)
- **New Provider**: Direct NVIDIA API at `https://integrate.api.nvidia.com/v1` (OpenAI-compatible)
- **Model**: `nvidia/nemotron-3-super-120b-a12b` via NVIDIA API key
- **Reasoning Control**: Configurable `reasoning` parameter (enabled/disabled via `extra_body`)
- **Multi-Provider Support**: Auto-detects provider from model name (`nvidia/` prefix or `nemotron`), explicit override supported
- **Schema Hardening**: `evidence` and `corrected` fields accept `null` (coerced to `""` and `None`)
- **YAML Fixes**: Fixed literal block scalar indentation in `validation.yaml`
- **Verification**: All 24 pytest tests pass; inference validated on 1, 3, 5 report runs

### Phase 4: LangGraph Orchestration (v4) — **Current**
- **Orchestration**: LangGraph state machine replaces manual retry loop
- **Nodes**: `initialize` → `inference` → `critic` → `judge` → `increment_attempt` / `finalize`
- **State Management**: Immutable state updates with full history tracking (`prediction_history`, `critique_history`, `judgment_history`)
- **Deterministic Safety Gates**: Critic output validated for structural integrity (evidence verbatim, proposed_value constraints, actionable consistency)
- **Finalization Logic**: 5-tier priority selection from complete attempt history (PASS → policy-supported → resolved → pre-oscillation → terminal fallback)
- **Trace System**: Single-writer pattern with `prompt_hash`, `policy_hash`, `policy_version` for auditability
- **Policy Versioning**: `policy_hash` recorded in every trace for reproducibility
- **Verification**: All 130 pytest tests pass (116 unit + 14 integration)

---

## System Architecture

### Inference Pipeline Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAGSET REPORT INFERENCE PIPELINE                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  train.csv   │────▶│  split_gold  │────▶│  Gold Set    │     │ Inference Set│
│  (source)    │     │  (populated  │     │  (analyzed)  │     │  (unlabeled) │
│              │     │   labels)    │     │              │     │              │
└──────────────┘     └──────────────┘     └──────┬───────┘     └──────┬───────┘
                                                  │                    │
                                                  ▼                    ▼
                                         ┌──────────────────┐  ┌──────────────────┐
                                         │  Gold Analysis   │  │  TF-IDF Retriever│
                                         │  (label profiles,│  │  (top-k=8,       │
                                         │   evidence stats)│  │   exclude self)  │
                                         └────────┬─────────┘  └────────┬─────────┘
                                                  │                    │
                                                  └─────────┬──────────┘
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │   CONTEXT BUILDER    │
                                                 │  (gold_analysis +    │
                                                 │   gold_examples)     │
                                                 └──────────┬───────────┘
                                                            │
                                    ┌───────────────────────┼───────────────────────┐
                                    ▼                       ▼                       ▼
                         ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
                         │   MODEL 1       │      │   PYDANTIC      │      │   MODEL 2       │
                         │   (Inference)   │─────▶│   VALIDATION    │─────▶│   (Validator)   │
                         │                 │      │   (structure)   │      │                 │
                         │ nvidia/nemotron │      │ ReportPrediction│      │ nvidia/nemotron │
                         │ 3-super-120b    │      │ 12 labels flat  │      │ 3-super-120b    │
                         └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
                                  │                        │                        │
                                  │         PASS           │         FAIL           │
                                  └───────────┬────────────┘                        │
                                              ▼                                     │
                                    ┌─────────────────┐                             │
                                    │   RETRY LOOP    │◀────────────────────────────┘
                                    │ (max_attempts=3)│
                                    └────────┬────────┘
                                             │
                                    ┌────────┴────────┐
                                    ▼                 ▼
                         ┌─────────────────┐ ┌─────────────────┐
                         │   PASSED        │ │ NEEDS_REVIEW    │
                         │ (all 12 labels  │ │ (failed after   │
                         │  validated)     │ │  max attempts)  │
                         └────────┬────────┘ └────────┬────────┘
                                  │                   │
                                  └─────────┬─────────┘
                                            ▼
                                 ┌──────────────────────┐
                                 │   OUTPUT WRITERS     │
                                 │  • predictions.jsonl │
                                 │  • model_trace.jsonl │
                                 └──────────────────────┘
```

### Data Flow

1. **Source Loading**: `train.csv` loaded via `load_train()` → split into gold (all 12 labels populated) and inference sets
2. **Gold Analysis**: `analyze_gold()` computes label profiles, evidence patterns, negation/uncertainty statistics
3. **Retrieval**: `GoldRetriever` (TF-IDF) fetches top-k similar gold reports; `retrieve_excluding()` prevents data leakage for held-out evaluation
3. **Context Assembly**: Gold analysis + retrieved examples formatted into prompt context
4. **Inference Loop** (per report):
   - Model 1 generates predictions for all 12 labels with evidence quotes
   - Pydantic validates structure (exactly 12 labels, correct types)
   - Model 2 independently validates each label against ORIGINAL report
   - On FAIL: validator feedback + original report sent back to Model 1 (retry)
   - Max 3 attempts per report
5. **Persistence**: Incremental write to `predictions.jsonl` + `model_trace.jsonl` after each report
6. **Held-Out Evaluation**: Before full inference, evaluate on `data/validation/heldout.csv` using `retrieve_excluding`

### Model Serving Infrastructure

| Component | Implementation | Configuration |
|-----------|----------------|---------------|
| **Provider Abstraction** | `core/llm/client.py` → `LLMProvider` interface | `core/llm/config.py` loads from `config/models.yaml` |
| **OpenRouter Provider** | `core/llm/providers/openrouter.py` | HTTP POST to `https://openrouter.ai/api/v1/chat/completions` |
| **NVIDIA Provider** | `core/llm/providers/nvidia.py` | HTTP POST to `https://integrate.api.nvidia.com/v1/chat/completions` |
| **Experiment Models** | `experiments/ragset_report_inference_experiment/src/ragset_inference/models.py` | `InferenceModel`, `ValidatorModel` with provider auto-detection |
| **Model Profiles** | `config/models.yaml` | `nvidia-ragset-inference`, `nvidia-ragset-validator`, `ragset-inference`, `ragset-validator` |
| **API Keys** | Environment variables via `.env` | `NVIDIA_API_KEY`, `OPENROUTER_API_KEY` |

---

## Execution Instructions

### Prerequisites

```bash
# 1. Install dependencies
pip install -r requirements.txt
# Or: pip install pydantic pyyaml python-dotenv requests scikit-learn pandas openai

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxx
# OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx
```

### Required Files

```
Repository root:
├── train.csv                                    # RagSet training data (required)
├── .env                                         # API keys (required)
└── experiments/ragset_report_inference_experiment/
    ├── config/
    │   ├── experiment.yaml                      # Pipeline config
    │   └── labels.yaml                          # 12 label definitions
    ├── prompts/
    │   ├── inference.yaml                       # Model 1 system + user template
    │   ├── validation.yaml                      # Model 2 system + user template
    │   └── gold_analysis.yaml                   # Gold analysis prompt
    └── data/validation/heldout.csv              # Held-out set (created by validate_agent.py)
```

### Commands

#### 1. Analyze Gold Reports (one-time, or after data changes)
```bash
python -m experiments.ragset_report_inference_experiment.runners.analyze_gold
```
- Reads gold reports from `train.csv`
- Computes label profiles, evidence statistics, negation/uncertainty patterns
- Outputs to console (used by inference pipeline)

#### 2. Create Held-Out Validation Set
```bash
python -m experiments.ragset_report_inference_experiment.runners.validate_agent
```
- Splits gold reports into held-out set (`data/validation/heldout.csv`)
- Required before running inference (used for held-out evaluation)

#### 3. Run Full Inference Pipeline
```bash
# Fresh run (overwrites existing results)
python -m experiments.ragset_report_inference_experiment.runners.run_inference --overwrite

# Resume from previous run (skips already-processed reports)
python -m experiments.ragset_report_inference_experiment.runners.run_inference --resume

# Limit number of reports (useful for testing)
python -m experiments.ragset_report_inference_experiment.runners.run_inference 5 --overwrite

# Combine flags
python -m experiments.ragset_report_inference_experiment.runners.run_inference 10 --resume
```

### Command-Line Flags

| Flag | Description | Default |
|------|-------------|---------|
| `limit` (positional) | Maximum number of reports to process | All |
| `--overwrite` | Delete existing results and start fresh | False |
| `--resume` | Skip already-processed reports from `predictions.jsonl` | False |

### Configuration Options

#### `config/models.yaml` — Model Profiles
```yaml
nvidia-ragset-inference:
  provider: nvidia
  model: nvidia/nemotron-3-super-120b-a12b
  api_key_env: NVIDIA_API_KEY
  purpose: ragset_report_inference
  capabilities: [text, reasoning, structured_output]
  parameters:
    temperature: 0
    max_output_tokens: 20000
    reasoning: false          # ← Disable reasoning for JSON mode compliance
  provider_policy:
    data_collection: deny
    zdr: true
    allow_fallbacks: false
```

#### `config/experiment.yaml` — Pipeline Settings
```yaml
source:
  path: train.csv
  id_column: StudyInstanceUID
  report_column: Report

gold:
  require_all_labels_populated: true

retrieval:
  enabled: true
  method: tfidf
  top_k: 8

models:
  inference_profile: nvidia-ragset-inference   # or ragset-inference for OpenRouter
  validator_profile: nvidia-ragset-validator   # or ragset-validator for OpenRouter
  temperature: 0

loop:
  max_attempts: 3

validation:
  require_validator_pass: true

runtime:
  output_dir: experiments/ragset_report_inference_experiment
  save_model_trace: true
```

### Expected Outputs

#### `results/inference/predictions.jsonl`
One JSON line per report:
```json
{
  "report_id": "1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260",
  "status": "passed",
  "review_reason": null,
  "attempts": [
    {
      "attempt": 1,
      "prediction": {
        "predictions": {
          "ACL": {"value": 0, "evidence": ""},
          "MCL": {"value": 0, "evidence": ""},
          "Medial_Meniscus": {"value": 1, "evidence": "Rotura de menisco interno."},
          "Lateral_Meniscus": {"value": 0, "evidence": ""},
          "Medial_OA": {"value": 1, "evidence": "Artrosis femorotibial medial."},
          "Lateral_OA": {"value": 0, "evidence": ""},
          "PF_OA": {"value": 0, "evidence": ""},
          "Effusion": {"value": 1, "evidence": "Derrame."},
          "Synovitis": {"value": 0, "evidence": ""},
          "Bakers": {"value": 0, "evidence": ""},
          "Contusion": {"value": 0, "evidence": ""},
          "Fracture": {"value": 0, "evidence": ""}
        }
      },
      "validation": {"status": "PASS", "issues": []},
      "elapsed_seconds": 5.78
    }
  ],
  "final_prediction": { ... }
}
```

**Status Values**: `passed` (all labels validated), `needs_review` (failed after max attempts)

#### `results/inference/model_trace.jsonl`
One JSON line per model call:
```json
{
  "report_id": "1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260",
  "stage": "inference",
  "model": "nvidia/nemotron-3-super-120b-a12b",
  "attempt": 1,
  "prompt": "...",
  "status": "success",
  "latency_seconds": 5.78
}
```

---

## Verification Evidence

### Test Suite Results (24/24 Passed)

```bash
$ python -m pytest experiments/ragset_report_inference_experiment/tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.13.2, pytest-9.1.1
collected 24 items

experiments/ragset_report_inference_experiment/tests/test_data.py::test_split_gold PASSED
experiments/ragset_report_inference_experiment/tests/test_data.py::test_load_train_renames_spaces PASSED
experiments/ragset_report_inference_experiment/tests/test_data.py::test_load_train_missing_columns PASSED
experiments/ragset_report_inference_experiment/tests/test_data.py::test_split_gold_unlabeled PASSED
experiments/ragset_report_inference_experiment/tests/test_evaluate.py::test_evaluate_perfect_match PASSED
experiments/ragset_report_inference_experiment/tests/test_evaluate.py::test_evaluate_all_wrong PASSED
experiments/ragset_report_inference_experiment/tests/test_evaluate.py::test_evaluate_partial PASSED
experiments/ragset_report_inference_experiment/tests/test_evaluate.py::test_evaluate_different_study_ids PASSED
experiments/ragset_report_inference_experiment/tests/test_evaluate.py::test_evaluate_empty PASSED
experiments/ragset_report_inference_experiment/tests/test_loop.py::test_retry_then_pass PASSED
experiments/ragset_report_inference_experiment/tests/test_loop.py::test_labels_passed_to_infer_and_validate PASSED
experiments/ragset_report_inference_experiment/tests/test_loop.py::test_retry_then_fail_needs_review PASSED
experiments/ragset_report_inference_experiment/tests/test_schemas.py::test_label_value_valid PASSED
experiments/ragset_report_inference_experiment/tests/test_schemas.py::test_label_value_invalid_value PASSED
experiments/ragset_report_inference_experiment/tests/test_schemas.py::test_label_value_extra_field_rejected PASSED
experiments/ragset_report_inference_experiment/tests/test_schemas.py::test_report_prediction_valid PASSED
experiments/ragset_report_inference_experiment/tests/test_schemas.py::test_report_prediction_exactly_12 PASSED
experiments/ragset_report_inference_experiment/tests/test_schemas.py::test_report_prediction_missing_label PASSED
experiments/ragset_report_inference_experiment/tests/test_schemas.py::test_report_prediction_duplicate_label PASSED
experiments/ragset_report_inference_experiment/tests/test_schemas.py::test_report_prediction_to_dict PASSED
experiments/ragset_report_inference_experiment/tests/test_schemas.py::test_validation_result_pass PASSED
experiments/ragset_report_inference_experiment/tests/test_schemas.py::test_validation_result_fail PASSE
experiments/ragset_report_inference_experiment/tests/test_schemas.py::test_prediction_has_12_labels PASSE
experiments/ragset_report_inference_experiment/tests/test_schemas.py::test_pass_validation PASSED

============================== 24 passed in 2.20s ==============================
```

### Inference Stability Validation

| Run | Reports | Mode | Results |
|-----|---------|------|---------|
| 1 | 1 | `--overwrite` | ✅ Passed (status: passed) |
| 2 | 1 | `--resume` | ✅ Skipped 1, processed 1 (status: needs_review) |
| 3 | 3 | `--resume` | ✅ Skipped 2, processed 1 (status: needs_review) |
| 4 | 5 | `--resume` | ✅ Skipped 2, processed 2 (1 passed, 1 needs_review) |

**Sample Output Verification** (Report `1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260`):
- **Language**: Spanish
- **Correctly Identified**: Medial_Meniscus=1 (Rotura de menisco interno), Medial_OA=1 (Artrosis femorotibial medial), Effusion=1 (Derrame)
- **Correctly Negative**: ACL, MCL, Lateral_Meniscus, Lateral_OA, PF_OA, Synovitis, Bakers, Contusion, Fracture = 0
- **Validation**: PASS (no issues)
- **Latency**: ~5.8 seconds per report

---

## Key Implementation Details

### Schema Design (`schemas.py`)
```python
LABEL_KEYS = ["ACL", "MCL", "Medial_Meniscus", "Lateral_Meniscus",
              "Medial_OA", "Lateral_OA", "PF_OA", "Effusion",
              "Synovitis", "Bakers", "Contusion", "Fracture"]

class LabelValue(BaseModel):
    value: Literal[0, 1]
    evidence: str | None = None  # null → ""

class ReportPrediction(BaseModel):
    predictions: dict[LABEL_LITERAL, LabelValue]  # exactly 12 required

class ValidationIssue(BaseModel):
    label: str
    predicted: Literal[0, 1]
    corrected: Literal[0, 1] | None = None  # null allowed
    reason: str
    evidence: list[str] = []
    schema_version: str = "1.0.0"

class ValidationResult(BaseModel):
    status: Literal["PASS", "FAIL"]
    issues: list[ValidationIssue] = []
```

### Provider Auto-Detection
```python
def _detect_provider(model: str) -> str:
    if model.startswith("nvidia/") or "nemotron" in model.lower():
        return "nvidia"
    return "openrouter"
```

### Reasoning Control (per provider)
```python
def _build_extra_body(provider: str, reasoning: bool | None) -> dict | None:
    if reasoning is None:
        return None
    if provider in ("nvidia", "openrouter"):
        return {"reasoning": {"enabled": reasoning}}
    return None
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `yaml.scanner.ScannerError` | Missing indentation in literal block scalars | Ensure 2-space indent after `\|` in YAML |
| `ValidationError: evidence None` | Model returns `null` for evidence | Schema coerces `null` → `""` via validator |
| `ValidationError: corrected None` | Validator returns `null` for corrected | Schema allows `Literal[0,1] \| None` |
| `503 Service Unavailable` | NVIDIA API transient overload | Retry; not a code issue |
| `KeyError: gold_analysis` | Validator template expects gold_analysis | Pass `gold_analysis` to `make_validate_fn` |
| `OPENROUTER_API_KEY required` | Wrong profile selected | Use `nvidia-ragset-inference` profile or set `OPENROUTER_API_KEY` |

---

## Extending the Pipeline

### Adding a New Provider
1. Implement `LLMProvider` in `core/llm/providers/<name>.py`
2. Add to `_create_client()` in `models.py`
3. Add detection logic to `_detect_provider()`
4. Add profile to `config/models.yaml`

### Adding New Labels
1. Update `LABEL_KEYS` and `LABEL_LITERAL` in `schemas.py`
2. Update `labels.yaml` and prompt templates
3. Run tests to verify schema validation

### Changing Retrieval Method
1. Modify `GoldRetriever` in `retrieval.py`
2. Update `config/experiment.yaml` retrieval config
3. Ensure `retrieve_excluding` prevents data leakage

---

## Related Documentation

- **Architecture**: [`docs/architecture.md`](../../docs/architecture.md)
- **Learning Roadmap**: [`docs/learning-roadmap.md`](../../docs/learning-roadmap.md)
- **Repository Summary**: [`plans/repo-summary.md`](../../plans/repo-summary.md)
- **Agent Instructions**: [`AGENTS.md`](../../AGENTS.md)
- **Handoff Documentation**: [`docs/handoff/`](../../docs/handoff/)