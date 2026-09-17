# Architecture Deep Dive

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATE AGENT LAB ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   EXPERIMENTS   │    │     CORE        │    │    CONFIG       │
│                 │    │                 │    │                 │
│ • lesson01      │    │ • llm/          │    │ • models.yaml   │
│ • model_lab     │◀───│   - client.py   │───▶│ • settings.yaml │
│ • ragset_exp    │    │   - config.py   │    │                 │
│                 │    │   - providers/  │    │                 │
└─────────────────┘    └────────┬────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │OpenRouter│ │  NVIDIA  │ │  Other   │
              │ Provider │ │ Provider │ │Providers │
              └──────────┘ └──────────┘ └──────────┘
```

---

## Core LLM Abstraction Layer

### `core/llm/client.py` — Abstract Interface

```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        parameters: GenerationParameters | None = None,
    ) -> LLMResponse:
        """Generate a response from a prompt."""
        raise NotImplementedError

@dataclass(frozen=True)
class LLMResponse:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float | None = None
    model: str | None = None
    raw_response: dict | None = None
```

### `core/llm/parameters.py` — Generation Parameters

```python
@dataclass
class GenerationParameters:
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning: bool | None = None  # Provider-specific
```

### `core/llm/config.py` — Configuration Loading

```python
def get_model_config(profile: str | None = None) -> ModelConfig:
    """Resolve a model profile into typed configuration."""
    # Loads from config/models.yaml + environment variables

def get_api_key(model_config: ModelConfig) -> str:
    """Load API key from environment variable specified in profile."""
```

---

## Provider Implementations

### OpenRouter Provider (`core/llm/providers/openrouter.py`)

```python
class OpenRouterProvider(LLMProvider):
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, parameters: GenerationParameters | None = None) -> LLMResponse:
        # HTTP POST to OpenRouter with provider policy
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "provider": {"data_collection": "deny", "allow_fallbacks": True},
        }
        # ... apply parameters, make request, return LLMResponse
```

**Features**:
- Full HTTP client with error handling
- Provider policy: `data_collection: deny`, `zdr: true`, `allow_fallbacks: true`
- Token usage extraction from response
- Latency measurement

### NVIDIA Provider (`core/llm/providers/nvidia.py`) — **NEW**

```python
class NVIDIAProvider(LLMProvider):
    BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, parameters: GenerationParameters | None = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if parameters.reasoning is not None:
            payload["reasoning"] = {"enabled": parameters.reasoning}
        # ... make request, return LLMResponse
```

**Features**:
- Direct NVIDIA API (OpenAI-compatible)
- Reasoning control via `reasoning` parameter in payload
- Same interface as OpenRouter provider

### Provider Auto-Detection (Experiment Level)

```python
def _detect_provider(model: str) -> str:
    if model.startswith("nvidia/") or "nemotron" in model.lower():
        return "nvidia"
    return "openrouter"

def _build_extra_body(provider: str, reasoning: bool | None) -> dict | None:
    if reasoning is None:
        return None
    if provider in ("nvidia", "openrouter"):
        return {"reasoning": {"enabled": reasoning}}
    return None
```

---

## RagSet Experiment Architecture

### Pipeline Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAGSET INFERENCE PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

train.csv
    │
    ▼
┌─────────────────────┐
│ load_train()        │  ──▶ pandas DataFrame
│ split_gold()        │  ──▶ gold (all 12 labels) + unlabeled
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ analyze_gold()      │  ──▶ label profiles, evidence stats,
│                     │      negation/uncertainty patterns
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ GoldRetriever       │  ──▶ TF-IDF vectorizer + cosine similarity
│ (top_k=8)           │      retrieve_excluding() for held-out eval
└─────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE LOOP (per report)                  │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  MODEL 1     │───▶│  PYDANTIC    │───▶│  MODEL 2     │      │
│  │  Inference   │    │  Validation  │    │  Validator   │      │
│  │              │    │  (structure) │    │              │      │
│  │ nvidia/      │    │ Report       │    │ nvidia/      │      │
│  │ nemotron-3   │    │ Prediction   │    │ nemotron-3   │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         │         PASS      │         FAIL      │               │
│         └─────────┬─────────┘                   │               │
│                   ▼                             │               │
│         ┌─────────────────┐                     │               │
│         │   RETRY LOOP    │◀────────────────────┘               │
│         │ (max_attempts=3)│                                   │
│         └────────┬────────┘                                   │
│                  │                                            │
│         ┌────────┴────────┐                                   │
│         ▼                 ▼                                   │
│  ┌───────────────┐ ┌───────────────┐                          │
│  │   PASSED      │ │ NEEDS_REVIEW  │                          │
│  │ (validated)   │ │ (max attempts)│                          │
│  └───────┬───────┘ └───────┬───────┘                          │
│          │                 │                                   │
│          └────────┬────────┘                                   │
│                   ▼                                            │
│         ┌─────────────────────┐                                │
│         │   OUTPUT WRITERS    │                                │
│         │ • predictions.jsonl │                                │
│         │ • model_trace.jsonl │                                │
│         └─────────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Modules

| Module | Responsibility |
|--------|----------------|
| `data.py` | `load_train()`, `split_gold()`, `LABEL_COLUMNS` |
| `gold.py` | `analyze_gold()`, `format_gold_examples()`, `label_profile_text()` |
| `retrieval.py` | `GoldRetriever` (TF-IDF), `retrieve()`, `retrieve_excluding()` |
| `models.py` | `InferenceModel`, `ValidatorModel` (provider-agnostic) |
| `schemas.py` | `ReportPrediction`, `ValidationResult`, `LabelValue`, `ValidationIssue` |
| `loop.py` | `run_report()` — orchestrates inference/validation/retry |
| `evaluate.py` | `evaluate()` — compares predictions vs gold |
| `trace.py` | `write_trace()` — JSONL model usage logging |

### Schema Design

```python
# 12 Labels (exact names required)
LABEL_KEYS = [
    "ACL", "MCL", "Medial_Meniscus", "Lateral_Meniscus",
    "Medial_OA", "Lateral_OA", "PF_OA", "Effusion",
    "Synovitis", "Bakers", "Contusion", "Fracture",
]

class LabelValue(BaseModel):
    value: Literal[0, 1]
    evidence: str | None = None  # null → "" via validator

class ReportPrediction(BaseModel):
    predictions: dict[LABEL_LITERAL, LabelValue]  # exactly 12 required

class ValidationIssue(BaseModel):
    label: str
    predicted: Literal[0, 1]
    corrected: Literal[0, 1] | None = None  # null allowed
    reason: str
    evidence: list[str] = []

class ValidationResult(BaseModel):
    status: Literal["PASS", "FAIL"]
    issues: list[ValidationIssue] = []
```

---

## Data Flow Details

### 1. Source Loading & Splitting
```python
df = load_train("train.csv")           # pandas DataFrame
gold, unlabeled = split_gold(df)       # gold: all 12 labels populated
```

### 2. Gold Analysis
```python
analysis = analyze_gold(gold)          # label profiles, evidence patterns
gold_examples = format_gold_examples(gold)  # formatted for prompts
```

### 3. Retrieval
```python
retriever = GoldRetriever(gold, top_k=8)
# For held-out eval (prevents data leakage):
retrieved = retriever.retrieve_excluding(report, exclude_study_id=report_id)
# For inference:
retrieved = retriever.retrieve(report)
```

### 4. Context Assembly
```python
context = {
    "gold_analysis": label_profile_text(analysis),
    "gold_examples": relevant,  # formatted retrieved examples
}
```

### 5. Inference Loop (`loop.py`)
```python
def run_report(report_id, report, infer, validate, context, max_attempts, labels):
    for attempt in 1..max_attempts:
        prediction = infer(report_id, report, context, attempt, ...)
        validation = validate(report_id, report, prediction, attempt, ...)
        if validation.passed:
            return {"status": "passed", "final_prediction": prediction, ...}
    return {"status": "needs_review", "final_prediction": prediction, ...}
```

### 6. Persistence
```python
# Incremental write after each report
append_prediction(result_path, serializable)
write_trace(trace_path, report_id, stage, model, attempt, prompt, status)
```

---

## Configuration System

### `config/models.yaml` — Model Profiles

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
    reasoning: false
  provider_policy:
    data_collection: deny
    zdr: true
    allow_fallbacks: false
```

### `config/experiment.yaml` — Pipeline Settings

```yaml
source:
  path: train.csv
  id_column: StudyInstanceUID
  report_column: Report

retrieval:
  enabled: true
  method: tfidf
  top_k: 8

models:
  inference_profile: nvidia-ragset-inference
  validator_profile: nvidia-ragset-validator
  temperature: 0

loop:
  max_attempts: 3

validation:
  require_validator_pass: true

runtime:
  output_dir: experiments/ragset_report_inference_experiment
  save_model_trace: true
```

---

## Model Serving Infrastructure

| Layer | Component | Technology |
|-------|-----------|------------|
| **Abstraction** | `LLMProvider` interface | Python ABC |
| **Config** | `ModelConfig` dataclass | YAML + env vars |
| **OpenRouter** | HTTP client | `requests` → `https://openrouter.ai/api/v1` |
| **NVIDIA** | HTTP client | `requests` → `https://integrate.api.nvidia.com/v1` |
| **Experiment** | `InferenceModel`/`ValidatorModel` | OpenAI SDK with `extra_body` |
| **API Keys** | Environment variables | `.env` → `NVIDIA_API_KEY`, `OPENROUTER_API_KEY` |

---

## Extensibility Points

| Extension | Location | Method |
|-----------|----------|--------|
| New Provider | `core/llm/providers/` | Implement `LLMProvider`, add to `_create_client()` |
| New Labels | `schemas.py` | Update `LABEL_KEYS`, `LABEL_LITERAL` |
| New Retrieval | `retrieval.py` | Subclass/extend `GoldRetriever` |
| New Metrics | `evaluate.py` | Add to `evaluate()` return dict |
| New Experiment | `experiments/` | Follow RagSet pattern |
| New Challenge | `challenges/` | Copy template, implement 7-step workflow |

---

## Failure Modes & Mitigations

| Failure Mode | Detection | Mitigation |
|--------------|-----------|------------|
| Provider API 503 | HTTP status check | Retry with backoff (configurable) |
| Invalid JSON from model | Pydantic validation error | `_extract_json()` with markdown fence handling |
| Null evidence/corrected | Schema validation error | Validators coerce `null` → `""`/`None` |
| Data leakage | Held-out eval mismatch | `retrieve_excluding()` prevents self-retrieval |
| Reasoning text instead of JSON | JSON parse failure | `reasoning: false` in config, `extra_body` control |
| Missing API key | `RuntimeError` at startup | Clear error message with env var name |

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Latency per report | ~5-6 seconds | NVIDIA Nemotron 3 Ultra |
| Token usage (inference) | ~2,000-4,000 | Depends on report length |
| Token usage (validation) | ~1,500-3,000 | Depends on prediction complexity |
| Max attempts per report | 3 | Configurable in `experiment.yaml` |
| Rate limiting | 2 sec between reports | Respects API limits |
| Memory | Low | Streaming JSONL writes |