# Extension Guide

Guide for adding new providers, labels, retrieval methods, experiments, and other extensions.

---

## Adding a New LLM Provider

### 1. Implement Provider Class

Create `core/llm/providers/<name>.py`:

```python
"""<Name> LLM Provider.

Why it exists:
    <Provider> provides access to <models> via <API>.

What problem it solves:
    Abstracts the <Provider> HTTP API into the common LLMProvider interface.

Alternatives considered:
    - Using official SDK: adds dependency
    - Direct HTTP: more control, manual error handling

Tradeoffs:
    - Direct HTTP gives full control
    - Tied to provider's response format

Failure modes:
    - API rate limits → HTTP 429
    - Invalid API key → HTTP 401
    - Model not available → HTTP 400/404

Testing:
    - Unit tests with mocked HTTP
    - Integration tests require valid API key
"""

from __future__ import annotations
import time
import requests
from core.llm.client import LLMProvider, LLMResponse
from core.llm.parameters import GenerationParameters


class <Name>Provider(LLMProvider):
    BASE_URL = "https://api.<provider>.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def generate(
        self,
        prompt: str,
        parameters: GenerationParameters | None = None,
    ) -> LLMResponse:
        parameters = parameters or GenerationParameters()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if parameters.temperature is not None:
            payload["temperature"] = parameters.temperature
        if parameters.max_output_tokens is not None:
            payload["max_tokens"] = parameters.max_output_tokens
        if parameters.reasoning is not None:
            payload["reasoning"] = {"enabled": parameters.reasoning}

        started = time.perf_counter()
        response = requests.post(
            self.BASE_URL,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - started
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_seconds=round(elapsed, 3),
            model=data.get("model", self.model),
            raw_response=data,
        )
```

### 2. Register Provider

Update `experiments/ragset_report_inference_experiment/src/ragset_inference/models.py`:

```python
def _create_client(provider: str, api_key: str | None = None) -> OpenAI:
    if provider == "nvidia":
        return OpenAI(api_key=api_key or os.environ.get("NVIDIA_API_KEY"),
                      base_url="https://integrate.api.nvidia.com/v1")
    elif provider == "openrouter":
        return OpenAI(api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
                      base_url="https://openrouter.ai/api/v1")
    elif provider == "<name>":  # ADD HERE
        return OpenAI(api_key=api_key or os.environ.get("<NAME>_API_KEY"),
                      base_url="https://api.<provider>.com/v1")
    else:
        raise ValueError(f"Unknown provider: {provider}")

def _detect_provider(model: str) -> str:
    if model.startswith("nvidia/") or "nemotron" in model.lower():
        return "nvidia"
    elif model.startswith("<prefix>/"):  # ADD HERE
        return "<name>"
    return "openrouter"

def _build_extra_body(provider: str, reasoning: bool | None) -> dict | None:
    if reasoning is None:
        return None
    if provider in ("nvidia", "openrouter", "<name>"):  # ADD HERE
        return {"reasoning": {"enabled": reasoning}}
    return None
```

### 3. Add Model Profile

Update `config/models.yaml`:

```yaml
<name>-ragset-inference:
  provider: <name>
  model: <provider>/<model-id>
  api_key_env: <NAME>_API_KEY
  purpose: ragset_report_inference
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

### 4. Add Environment Variable

Update `.env.example`:
```bash
<NAME>_API_KEY=your-<name>-api-key-here
```

### 5. Test

```bash
python -m pytest experiments/ragset_report_inference_experiment/tests/ -v
python -m experiments.ragset_report_inference_experiment.runners.run_inference 1 --overwrite
```

---

## Adding New Labels

### 1. Update Schema (`schemas.py`)

```python
LABEL_KEYS = [
    "ACL", "MCL", "Medial_Meniscus", "Lateral_Meniscus",
    "Medial_OA", "Lateral_OA", "PF_OA", "Effusion",
    "Synovitis", "Bakers", "Contusion", "Fracture",
    "NEW_LABEL_1", "NEW_LABEL_2",  # ADD HERE
]

LABEL_LITERAL = Literal[
    "ACL", "MCL", "Medial_Meniscus", "Lateral_Meniscus",
    "Medial_OA", "Lateral_OA", "PF_OA", "Effusion",
    "Synovitis", "Bakers", "Contusion", "Fracture",
    "NEW_LABEL_1", "NEW_LABEL_2",  # ADD HERE
]
```

### 2. Update Labels Config (`config/labels.yaml`)

```yaml
labels:
  - ACL
  - MCL
  - Medial_Meniscus
  - Lateral_Meniscus
  - Medial_OA
  - Lateral_OA
  - PF_OA
  - Effusion
  - Synovitis
  - Bakers
  - Contusion
  - Fracture
  - NEW_LABEL_1
  - NEW_LABEL_2
```

### 3. Update Prompts

**`prompts/inference.yaml`** — Add to LABELS list and OUTPUT FORMAT example
**`prompts/validation.yaml`** — Add to LABELS list and examples

### 4. Update Gold Analysis (if needed)

The `analyze_gold()` function automatically computes statistics for all labels in `LABEL_KEYS`.

### 5. Run Tests

```bash
python -m pytest experiments/ragset_report_inference_experiment/tests/test_schemas.py -v
python -m experiments.ragset_report_inference_experiment.runners.run_inference 1 --overwrite
```

---

## Changing Retrieval Method

### Current: TF-IDF (`retrieval.py`)

```python
class GoldRetriever:
    def __init__(self, gold_df, top_k=8):
        self.vectorizer = TfidfVectorizer(...)
        self.tfidf_matrix = self.vectorizer.fit_transform(gold_reports)
```

### Option 1: BM25

```python
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, gold_df, top_k=8):
        self.corpus = [doc.split() for doc in gold_reports]
        self.bm25 = BM25Okapi(self.corpus)
```

### Option 2: Embeddings + Vector Search

```python
from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingRetriever:
    def __init__(self, gold_df, top_k=8):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(gold_reports)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
```

### Option 3: Hybrid (TF-IDF + Embeddings)

```python
class HybridRetriever:
    def __init__(self, gold_df, top_k=8, alpha=0.5):
        self.tfidf = TfidfVectorizer(...)
        self.embedding_model = SentenceTransformer(...)
        self.alpha = alpha
```

### Integration

Update `config/experiment.yaml`:
```yaml
retrieval:
  enabled: true
  method: bm25  # or embeddings, hybrid
  top_k: 8
```

Update `retrieval.py` to factory pattern:
```python
def create_retriever(method: str, gold_df, top_k: int):
    if method == "tfidf":
        return GoldRetriever(gold_df, top_k)
    elif method == "bm25":
        return BM25Retriever(gold_df, top_k)
    elif method == "embeddings":
        return EmbeddingRetriever(gold_df, top_k)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
```

---

## Adding New Evaluation Metrics

### 1. Define Metric Function (`evaluate.py`)

```python
def compute_new_metric(gold_df, pred_df) -> float:
    """Compute new metric comparing gold vs predictions."""
    # Implementation
    return score
```

### 2. Add to Evaluation Output

```python
def evaluate(gold_df, pred_df) -> dict:
    # ... existing metrics ...
    results["new_metric"] = compute_new_metric(gold_df, pred_df)
    return results
```

### 3. Update Test

```python
def test_evaluate_new_metric():
    result = evaluate(gold_df, pred_df)
    assert "new_metric" in result
    assert 0 <= result["new_metric"] <= 1
```

---

## Creating a New Experiment

### 1. Directory Structure

```
experiments/<new_experiment>/
├── README.md
├── requirements.txt
├── config/
│   ├── experiment.yaml
│   └── labels.yaml
├── prompts/
│   ├── inference.yaml
│   ├── validation.yaml
│   └── <task>.yaml
├── data/
│   ├── gold/
│   ├── inferred/
│   └── validation/
├── runners/
│   ├── analyze_gold.py
│   ├── validate_agent.py
│   └── run_inference.py
├── src/<experiment_name>/
│   ├── __init__.py
│   ├── data.py
│   ├── gold.py
│   ├── retrieval.py
│   ├── models.py
│   ├── schemas.py
│   ├── loop.py
│   ├── evaluate.py
│   └── trace.py
├── results/
└── tests/
    ├── test_data.py
    ├── test_evaluate.py
    ├── test_loop.py
    └── test_schemas.py
```

### 2. Follow RagSet Pattern

Key patterns to replicate:
- Two-agent loop with Pydantic validation
- Retrieval with data leakage prevention
- Incremental JSONL persistence
- Model trace logging
- Config-driven via YAML
- Resume/overwrite CLI flags

### 3. Register in Model Config

Add profiles to `config/models.yaml` for the new experiment.

---

## Adding a New Challenge

### 1. Copy Template

```bash
cp -r challenges/may26 challenges/<new_challenge>
```

### 2. Update Files

- `README.md` — Challenge description, dataset, evaluation criteria
- `postmortem.md` — Fill in template after completion
- `code/` — Starter code or empty
- `data/` — Challenge dataset
- `evaluation/` — Evaluation scripts

### 3. Follow 7-Step Workflow

1. **Understand** — Read problem, explore data
2. **Design** — Architecture, approach, success criteria
3. **Baseline** — Implement minimal working version
4. **Evaluate** — Run baseline, measure metrics
5. **Improve** — Iterate on model/prompt/retrieval
6. **Alternatives** — Try different approaches
7. **Postmortem** — Document learnings, failures, decisions

---

## Configuration Best Practices

### Model Profiles

```yaml
# Good: Explicit, complete profile
my-experiment-inference:
  provider: nvidia
  model: nvidia/nemotron-3-super-120b-a12b
  api_key_env: NVIDIA_API_KEY
  purpose: my_experiment_inference
  capabilities: [text, reasoning, structured_output]
  parameters:
    temperature: 0
    max_output_tokens: 8000
    reasoning: false
  provider_policy:
    data_collection: deny
    zdr: true
    allow_fallbacks: false

# Avoid: Incomplete, relies on defaults
bad-profile:
  provider: openrouter
  model: some-model
```

### Experiment Config

```yaml
# Good: Explicit, documented
source:
  path: data.csv
  id_column: id
  text_column: text

retrieval:
  enabled: true
  method: tfidf
  top_k: 8

models:
  inference_profile: my-exp-inference
  validator_profile: my-exp-validator

loop:
  max_attempts: 3

validation:
  require_validator_pass: true
```

---

## Testing New Extensions

### Checklist

- [ ] Unit tests for new component
- [ ] Integration test with full pipeline
- [ ] Schema validation tests
- [ ] Edge case handling (null, empty, malformed)
- [ ] Performance benchmark
- [ ] Documentation updated

### Run Commands

```bash
# Unit tests
python -m pytest experiments/<new_exp>/tests/ -v

# Integration test
python -m experiments.<new_exp>.runners.run_inference 1 --overwrite

# Full test suite
python -m pytest experiments/ -v
```

---

## Common Pitfalls

| Pitfall | Prevention |
|---------|------------|
| Forgetting `extra=forbid` in schemas | Always use `ConfigDict(extra="forbid")` |
| Missing indentation in YAML literals | Use 2 spaces after `\|` |
| Not handling `null` from models | Add validators to coerce `null` |
| Hardcoding API keys | Always use `api_key_env` from config |
| Skipping `retrieve_excluding` | Always use for held-out evaluation |
| Not testing resume logic | Test `--resume` after every change |
| Forgetting provider in `extra_body` | Update `_build_extra_body()` |

---

## Code Style Guidelines

- Type hints on all public functions
- Docstrings with Why/What/Alternatives/Tradeoffs/Failure modes
- Pydantic models with `extra="forbid"`
- Validators for coercion (null → default)
- Incremental persistence (write after each item)
- Structured logging (JSONL traces)
- Config-driven (no hardcoded values)