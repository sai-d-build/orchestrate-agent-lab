# Project Overview

## Orchestrate Agent Lab

**Learning lab + reusable hackathon starter for reliable AI/data engineering.**

### Vision

Build a cloud-first AI/GenAI learning environment where capabilities are introduced incrementally — only after underlying concepts are understood and tested. No frameworks (LangChain, LangGraph, smolagents) until primitives are mastered.

### Current Phase

**Phase 0 — Foundation** | **Current Step: Lesson 1.2 — Model Selection + Enterprise Decision Benchmark**

```
Phase 0: Foundation     → LLM APIs, Pydantic, structured outputs, env vars, testing, Git
Phase 1: Retrieval      → Keyword search, BM25, embeddings, vector retrieval, hybrid, reranking
Phase 2: Agents         → Tool calling, agent loop, smolagents, LangChain, LangGraph
Phase 3: Reliability    → Guardrails, prompt injection defense, observability, cost/latency
Phase 4: Multimodal     → Vision, OCR, audio, modality routing
Phase 5: Orchestrate    → Historical challenges, timed simulations, postmortems, interview prep
```

---

## Repository Purpose

| Goal | Description |
|------|-------------|
| **Learning** | Understand AI engineering primitives by building them |
| **Reusable Starter** | Hackathon-ready codebase with working LLM abstraction |
| **Experiment Platform** | Structured experiments with evaluation, persistence, tracing |
| **Challenge Framework** | Timed labs with postmortem templates for skill building |

---

## Project Structure Summary

```
orchestrate-agent-lab/
├── config/                 # Configuration (no secrets)
│   ├── models.yaml         # Model registry: profiles, providers, capabilities
│   └── settings.yaml       # App settings: timeouts, retries, observability
├── core/                   # Reusable AI capabilities (grows incrementally)
│   └── llm/                # LLM abstraction layer
│       ├── client.py       # Abstract LLMProvider interface
│       ├── config.py       # Model config loading (YAML + env vars)
│       ├── parameters.py   # GenerationParameters dataclass
│       └── providers/      # Provider implementations
├── experiments/            # Learning experiments
│   ├── lesson01_llm_api.py # Lesson 1: Basic LLM API call
│   ├── model_lab/          # Lesson 1.2: Enterprise decision benchmark
│   └── ragset_report_inference_experiment/  # RagSet knee MRI labels
├── challenges/             # Timed challenge labs (4, not started)
├── docs/                   # Documentation
│   ├── architecture.md     # Architecture progression
│   ├── cloud-setup.md      # Setup instructions
│   ├── learning-roadmap.md # 6-phase roadmap
│   └── handoff/            # Consolidated handoff docs
├── prompts/                # Versioned reusable prompts
├── src/orchestrate_agent_lab/  # Package source
├── tests/                  # Test suite
├── AGENTS.md               # AI coding agent instructions
├── pyproject.toml          # Python project config
├── requirements.txt        # Dependencies
└── .gitignore
```

---

## Completed Work (Phase 0)

### Lesson 1: Basic LLM API (`experiments/lesson01_llm_api.py`)
- Transaction analysis with structured Pydantic output
- OpenRouter free model benchmarking
- Basic prompt engineering patterns

### Lesson 1.2: Enterprise Decision Benchmark (`experiments/model_lab/`)
- **3 Datasets** (6 cases each): Support Escalation, Invoice Exception, Change Risk
- **Case Categories**: Normal, Ambiguous, Conflicting, Missing Info, Adversarial
- **11 Evaluation Metrics**: Decision correctness, evidence correctness, policy adherence, structured output validity, missing info detection, confidence, consistency, injection resistance, latency, token usage, estimated cost
- **Experiment Matrix**: Model profiles × parameter combinations × runs per case
- **Status**: Case runner implemented; experiment runner & scoring pending

### RagSet Report Inference Experiment (`experiments/ragset_report_inference_experiment/`)
- **Task**: Extract 12 binary labels from knee MRI reports
- **Labels**: ACL, MCL, Medial_Meniscus, Lateral_Meniscus, Medial_OA, Lateral_OA, PF_OA, Effusion, Synovitis, Bakers, Contusion, Fracture
- **Architecture**: Two-agent loop (Inference → Pydantic → Validator → Retry)
- **Retrieval**: TF-IDF over gold reports with data leakage prevention
- **Model Backend**: NVIDIA Nemotron 3 Ultra (120B) via direct API or OpenRouter
- **Status**: ✅ Complete — 24 tests passing, stable inference validated

---

## Key Design Decisions

### 1. Provider-Independent LLM Interface
- `core/llm/client.py` defines abstract `LLMProvider` with `generate()` method
- Only OpenRouter and NVIDIA fully implemented; others are placeholders
- **Rationale**: Start with one working provider, expand incrementally

### 2. Configuration via YAML + Environment Variables
- `config/models.yaml` defines model profiles (provider, model ID, API key env var, capabilities, parameters)
- `config/settings.yaml` defines app-level settings
- API keys **never** in YAML — loaded from environment via `api_key_env`
- `core/llm/config.py` provides `get_model_config()` and `get_api_key()`

### 3. Model Registry (Lesson 1.2)
- Three OpenRouter free-tier models configured for benchmarking
- Direct provider configs (Gemini, OpenAI, Anthropic, Groq, Mistral) as placeholders
- Each profile: `data_collection: deny`, `zdr: true`, `allow_fallbacks: false`

### 4. Enterprise Decision Benchmark Datasets
- Three YAML datasets, 6 cases each covering different decision categories
- Adversarial cases test prompt injection resistance

### 5. Evaluation Metrics (11 Total)
`decision_correctness`, `evidence_correctness`, `policy_adherence`, `structured_output_validity`, `missing_information_detection`, `confidence`, `consistency`, `injection_resistance`, `latency`, `token_usage`, `estimated_cost`

---

## Current State Summary

| Component | Status | Details |
|-----------|--------|---------|
| LLM Abstraction Layer | ✅ | Abstract interface defined |
| OpenRouter Provider | ✅ | Full HTTP client with structured outputs |
| NVIDIA Provider | ✅ | Direct API, reasoning control via `extra_body` |
| Other Providers | 🚧 | Placeholders only |
| Model Registry | ✅ | 3 free + 5 direct provider configs |
| Lesson 1 (LLM API) | ✅ | Experiment script working |
| Lesson 1.2 (Model Lab) | 🟡 | Case runner done; experiment runner & scoring pending |
| RagSet Experiment | ✅ | Full pipeline, 24 tests passing, stable |
| Evaluation Metrics | 🟡 | Defined; scoring implementation pending |
| Challenges | 🚧 | 4 templates, not started |
| Tests | 🟡 | RagSet: 24 passing; core tests minimal |

---

## Dependencies

| Category | Packages |
|----------|----------|
| **Runtime** | pydantic>=2, PyYAML>=6, python-dotenv>=1, requests>=2, scikit-learn, pandas, openai |
| **Dev** | pytest>=8 |
| **Python** | 3.11+ |

---

## Architecture Progression

```
LLM API → Pydantic/structured output → retrieval → evaluation → guardrails → agents → multimodal
```

Frameworks (smolagents, LangChain, LangGraph) introduced only after underlying concepts are understood.

---

## Principles (from AGENTS.md)

1. Start simple
2. Prefer deterministic code/SQL when appropriate
3. Measure before adding complexity
4. Keep secrets out of source control
5. Learn frameworks by understanding underlying primitives first

---

## Next Phase Targets (Phase 1 — Retrieval)

- Keyword search & BM25
- Embeddings & vector retrieval
- Metadata filtering & hybrid retrieval
- Reranking & retrieval evaluation