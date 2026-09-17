# Handoff Documentation Index

Consolidated documentation for the next developer to follow up on maintenance and extensions of the Orchestrate Agent Lab.

---

## Quick Navigation

| Document | Purpose | Audience |
|----------|---------|----------|
| [Project Overview](./project-overview.md) | High-level project structure, phases, and current state | All developers |
| [Architecture Deep Dive](./architecture.md) | System architecture, data flow, component interactions | Backend/ML engineers |
| [Execution Guide](./execution-guide.md) | Step-by-step commands, configuration, troubleshooting | DevOps/Engineers |
| [Verification Evidence](./verification-evidence.md) | Test results, stability validation, performance metrics | QA/Engineers |
| [Extension Guide](./extension-guide.md) | Adding providers, labels, retrieval methods, new experiments | All developers |
| [RagSet Experiment README](../experiments/ragset_report_inference_experiment/README.md) | Complete experiment-specific documentation | RagSet experiment maintainers |

---

## Repository Structure (Key Paths)

```
orchestrate-agent-lab/
├── config/
│   ├── models.yaml              # Model registry (profiles, providers, capabilities)
│   └── settings.yaml            # Application settings
├── core/
│   ├── llm/
│   │   ├── client.py            # LLMProvider abstract interface
│   │   ├── config.py            # Model config loading (YAML + env)
│   │   ├── parameters.py        # GenerationParameters dataclass
│   │   └── providers/           # Provider implementations
│   │       ├── nvidia.py        # ✅ NVIDIA Nemotron (NEW)
│   │       ├── openrouter.py    # ✅ OpenRouter (full HTTP client)
│   │       ├── openai.py        # 🚧 Placeholder
│   │       ├── anthropic.py     # 🚧 Placeholder
│   │       ├── gemini.py        # 🚧 Placeholder
│   │       ├── groq.py          # 🚧 Placeholder
│   │       └── mistral.py       # 🚧 Placeholder
│   ├── evaluation/              # 🚧 Planned
│   ├── guardrails/              # 🚧 Planned
│   ├── multimodal/              # 🚧 Planned
│   ├── retrieval/               # 🚧 Planned
│   └── schemas/                 # 🚧 Planned
├── experiments/
│   ├── lesson01_llm_api.py      # ✅ Lesson 1: Basic LLM API
│   ├── model_lab/               # ✅ Lesson 1.2: Enterprise decision benchmark
│   └── ragset_report_inference_experiment/  # ✅ RagSet knee MRI labels
│       ├── README.md            # ← Comprehensive experiment docs
│       ├── config/
│       ├── prompts/
│       ├── runners/
│       ├── src/ragset_inference/
│       └── tests/               # 24 tests passing
├── challenges/                  # Timed challenge labs (4, not started)
├── docs/
│   ├── architecture.md          # Architecture progression
│   ├── cloud-setup.md           # Setup instructions
│   ├── learning-roadmap.md      # 6-phase roadmap
│   ├── decisions/               # ADR placeholder
│   ├── interview/               # Interview prep
│   ├── lessons/                 # Lesson notes
│   └── handoff/                 # ← THIS FOLDER
├── prompts/                     # Versioned reusable prompts
├── src/orchestrate_agent_lab/   # Package source
├── tests/                       # Test suite
├── AGENTS.md                    # AI coding agent instructions
├── pyproject.toml               # Python project config
├── requirements.txt             # Dependencies
└── .gitignore
```

---

## Current Phase & Status

| Area | Status | Notes |
|------|--------|-------|
| **Phase** | Phase 0 — Foundation | Lesson 1.2 complete |
| **LLM Abstraction** | ✅ Core interface defined | `core/llm/client.py` |
| **OpenRouter Provider** | ✅ Fully implemented | HTTP client with structured outputs |
| **NVIDIA Provider** | ✅ Fully implemented | Direct API, reasoning control |
| **Other Providers** | 🚧 Placeholders only | OpenAI, Anthropic, Gemini, Groq, Mistral |
| **Model Registry** | ✅ 3 free + 5 direct configs | `config/models.yaml` |
| **Lesson 1 (LLM API)** | ✅ Experiment script | `experiments/lesson01_llm_api.py` |
| **Lesson 1.2 (Model Lab)** | ✅ Case runner | `experiments/model_lab/runners/run_case.py` |
| **RagSet Experiment** | ✅ Full pipeline | 24 tests passing, stable inference |
| **Evaluation Metrics** | ✅ Defined (11) | Scoring implementation pending |
| **Challenges** | 🚧 Templates only | 4 challenges, not started |
| **Tests** | ✅ RagSet: 24 passing | Core tests minimal |

---

## Key Principles (from AGENTS.md)

1. **Inspect before changing** — Understand existing code first
2. **Explain intended change** — Document before implementing
3. **Identify affected files** — Map impact radius
4. **Smallest useful change** — Incremental, testable steps
5. **Run tests after changes** — Never skip validation
6. **Report failures honestly** — Diagnose root cause
7. **Update documentation** — When architecture/behavior changes
8. **Meaningful commits** — Clear messages, atomic changes
9. **No frameworks without need** — Learn primitives first
10. **Never commit secrets** — Use `.env`, `.env.example`

---

## Learning Rule

For every important implementation, explain:
- **Why it exists** — Problem being solved
- **What problem it solves** — Specific use case
- **Alternatives considered** — And why rejected
- **Tradeoffs** — Performance, complexity, maintainability
- **Failure modes** — How it breaks, how to detect
- **Testing** — Unit, integration, validation
- **Evaluation** — Metrics, benchmarks
- **Production implications** — Scaling, monitoring, cost

---

## Immediate Next Steps (Priority Order)

### High Priority
1. **Implement `scoring.py`** for Model Lab — Compare model output against expected for 11 metrics
2. **Implement `run_experiment.py`** — Full experiment matrix (models × params × runs)
3. **Add `--dataset` argument** to `run_case.py` — Support all 3 datasets
4. **Create Pydantic schemas** in `core/schemas/` for enterprise decisions

### Medium Priority
5. **Parse model responses** — Extract structured fields for evaluation
6. **Save results to `results/`** — JSON persistence instead of stdout only
7. **Capture token usage** — From API responses for cost/latency metrics
8. **Implement multi-run consistency** — Run each case 3× per experiment matrix
9. **Use prompt template** — Load from `enterprise_decisions.yaml`
10. **Add retry logic** — Use `settings.yaml` `max_retries: 2`

### Low Priority
11. **Implement other providers** — OpenAI, Anthropic, Gemini, Groq, Mistral
12. **Add guardrails** — Prompt injection defense, evidence validation
13. **Add retrieval** — BM25, embeddings, vector search
14. **Start challenges** — May26, June26, August26, September26

---

## Environment Setup

```bash
# Clone and setup
git clone <repo>
cd orchestrate-agent-lab

# Install dependencies
pip install -r requirements.txt
# pydantic>=2, pyyaml>=6, python-dotenv>=1, requests>=2, scikit-learn, pandas, openai, pytest

# Configure API keys
cp .env.example .env
# Edit .env:
# NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxx
# OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx

# Verify setup
python -m pytest experiments/ragset_report_inference_experiment/tests/ -v
# Should show 24 passed
```

---

## Contact & Context

- **Repository**: Orchestrate Agent Lab — Learning lab + hackathon starter
- **Philosophy**: Learn by building; frameworks after primitives
- **Current Focus**: Phase 0 Foundation → Phase 1 Retrieval
- **Key Experiment**: RagSet knee MRI report label extraction (12 binary labels)
- **Model**: NVIDIA Nemotron 3 Ultra (120B) via direct API or OpenRouter

---

*Last updated: 2026-09-17 | Phase 0 — Lesson 1.2 Complete*