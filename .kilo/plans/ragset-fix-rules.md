# RagSet Fix Implementation Rules

**Mandatory rules for all implementation work across Phases 1-9**

---

## 1. Git & Commit Discipline

| Rule | Enforcement |
|------|-------------|
| **No result files in commits** | `git status` must show only source/code/test/doc changes before any commit |
| **No generated/cache files** | `__pycache__/`, `*.pyc`, `.DS_Store`, `results/**/*.jsonl`, `results/**/*.json`, `data/validation/*.csv` must be in `.gitignore` |
| **No backup prompts in active directory** | `*_backup*.yaml`, `*_bkp*.yaml`, `validation.yaml` (legacy) → move to `archive/` or delete |
| **One logical change per commit** | Each phase = separate commit; no mixing runtime fixes with test additions |
| **Conventional commit messages** | `fix:`, `feat:`, `test:`, `docs:`, `refactor:`, `chore:` prefixes |

---

## 2. Code Quality Standards

### 2.1 Python Style
- **Type hints mandatory** on all public functions
- **Pydantic v2** for all data models (no raw dicts for structured data)
- **No silent fallbacks** — fail fast with descriptive errors
- **No mutable defaults** — use `None` + factory pattern
- **Max function length**: 50 lines (split if exceeded)

### 2.2 Error Handling
```python
# GOOD: Explicit, typed, traceable
raise ValueError(f"Policy contradiction for {label}: {detail}")

# BAD: Silent, untyped, untraceable
return None  # or pass
```

### 2.3 Logging
- **Structured logging only** — use `structlog` or stdlib with consistent fields
- **No secrets in logs** — API keys, tokens, PII must be redacted
- **Trace context** — include `report_id`, `attempt`, `node`, `model` in all log lines

---

## 3. LangGraph-Specific Rules

### 3.1 State Management
```python
# GOOD: Immutable updates, explicit returns
def node(state: RagSetState) -> RagSetState:
    new_state = dict(state)
    new_state["field"] = value
    return new_state

# BAD: In-place mutation
state["field"] = value
return state
```

### 3.2 History Semantics (CRITICAL)
| Rule | Rationale |
|------|-----------|
| **Never duplicate current item in history** | `state["history"]` already contains current; pass directly |
| **Only Model 1 execution increments `attempt`** | Model 2/3 never increment attempt counter |
| **History arrays are append-only** | No reordering, no deletion, no in-place modification |
| **Current prediction/critique/judgment = last element of history** | Single source of truth |

### 3.3 Node Contracts
```python
# Every node MUST:
# 1. Accept RagSetState
# 2. Return RagSetState (new dict)
# 3. Append to trace_records if LLM call made
# 4. Never write files directly (worker owns I/O)
```

---

## 4. Trace System Rules

### 4.1 Trace Record Requirements
Every LLM call produces **exactly one** `TraceRecord` with:
```python
{
    "timestamp_utc": ISO8601,
    "report_id": str,
    "stage": "inference" | "validation" | "judgment",
    "requested_model": str,
    "actual_model": str,
    "provider": str,
    "attempt": int,           # 1-indexed, only Model 1 increments
    "prompt_hash": str,       # SHA256, mandatory
    "response_hash": str,     # SHA256, mandatory
    "status": "success" | "error",
    "latency_seconds": float,
    "input_tokens": int | None,
    "output_tokens": int | None,
    "error": str | None,
    "graph_node": "initialize" | "inference" | "critic" | "judge" | "finalize",
    "judge_action": str | None,
    "policy_hash": str,       # NEW: mandatory
    "prompt_version": str,    # NEW: mandatory
}
```

### 4.2 Trace Writing
- **Single writer**: Only `worker_process_chunk` → main process writes `model_trace.jsonl`
- **No trace writes in graph nodes** — nodes only append to `state["trace_records"]`
- **Prompt hash mandatory** — never write full prompt to JSONL (size/security)

---

## 5. Contract & Schema Rules

### 5.1 Judge Reason Code Contract
**Single source of truth** — one enum shared by:
- `schemas.py` `JudgeReasonCode`
- `prompts/judge.yaml` allowed outputs
- `graph_nodes.py` finalization mapping
- Tests

```python
# Canonical contract (example - finalize in Phase 3)
class JudgeReasonCode(str, Enum):
    SUPPORTED_PASS = "SUPPORTED_PASS"
    CLEAR_ACTIONABLE_CONFLICT = "CLEAR_ACTIONABLE_CONFLICT"
    MODEL2_UNSUPPORTED_CORRECTION = "MODEL2_UNSUPPORTED_CORRECTION"
    UNRESOLVED_POLICY = "UNRESOLVED_POLICY"
    REPORT_AMBIGUITY = "REPORT_AMBIGUITY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    MODEL1_STUCK = "MODEL1_STUCK"
    OSCILLATION = "OSCILLATION"
    REPEATED_CRITIQUE = "REPEATED_CRITIQUE"
    MAX_ATTEMPTS = "MAX_ATTEMPTS"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    HARD_STOP = "HARD_STOP"
```

### 5.2 Critic Validation Rules (Enforced in `_validate_critic_output`)
| Rule | Check |
|------|-------|
| `model1_value` matches actual Model 1 prediction | `issue.model1_value == current_prediction.predictions[issue.label].value` |
| `proposed_value` is 0/1 for CLEAR_* issues | `issue.proposed_value in (0, 1)` |
| `proposed_value` is `None` for UNRESOLVED/AMBIGUOUS/INSUFFICIENT | `issue.proposed_value is None` |
| `evidence` is verbatim from report for CLEAR_REPORT_CONFLICT | `issue.evidence in original_report` |
| `evidence` non-null for CLEAR_REPORT_CONFLICT | `issue.evidence is not None` |
| `actionable` == True iff CLEAR_* issues exist | `issue.actionable == any(CLEAR_* in issues)` |
| `affected_labels` matches CLEAR_* issue labels | `set(affected_labels) == set(CLEAR_* labels)` |

### 5.3 Schema Versioning
All output schemas include:
```python
class BaseSchema(BaseModel):
    schema_version: str = "1.0.0"  # Semantic version
```

---

## 6. Policy Rules

### 6.1 Contradiction Classification
Every contradiction in `policy_audit_report.json` must be classified:
| Classification | Meaning | Runtime Behavior |
|----------------|---------|------------------|
| `RESOLVED` | Deterministic fix applied | Becomes SAFE rule |
| `TRUE_CONTRADICTION` | Genuine GOLD vs text conflict | Quarantined; never becomes SAFE |
| `UNRESOLVED` | Insufficient evidence | Marked UNRESOLVED_POLICY |

### 6.2 Policy Precedence (Runtime Authority)
```
1. CSV gold outcome (ground truth)
2. GOLD_ANALYSIS (LLM analysis of gold)
3. Canonical policy YAML (runtime policy)
4. Retrieved examples (reference only, never override)
```

### 6.3 Policy Versioning
- `policy_hash` = SHA256 of canonical policy YAML
- Recorded in every trace record
- Inference refuses to start if policy has unresolved structural errors

---

## 7. Testing Rules

### 7.1 Test Categories
| Category | Location | When to Run |
|----------|----------|-------------|
| Unit | `tests/`, `experiments/.../tests/` | Every commit |
| Integration | `test_e2e_regression.py` | Pre-merge |
| Contract | `test_langgraph.py` (new contract tests) | Every commit |
| E2E (real LLM) | Manual / CI nightly | Phase 9 gate |

### 7.2 Mandatory Test Patterns
```python
# Every new test MUST:
# 1. Use descriptive name: test_<behavior>_<condition>_<expected>
# 2. Test ONE behavior per test
# 3. Use fixtures for common state (create_initial_state, make_prediction, etc.)
# 4. Assert exact enum values, not string containment
# 5. Include negative cases (what should NOT happen)
```

### 7.3 Required Test Coverage (Phase 7)
- No-retry on unsupported Model 2 correction
- Retry convergence: M1 wrong → M2 clear → M1 correct → M2 PASS
- Unresolved policy → AMBIGUOUS (never RETRY)
- Mixed critique combinations
- Deterministic attempt semantics
- Trace exactly-once
- Trace provenance (all fields preserved)
- Resume/idempotency
- Parallel worker isolation

---

## 8. Prompt Rules

### 8.1 Active Prompts Only
**Production prompt directory contains exactly:**
```
prompts/
├── inference.yaml          # Model 1
├── validation_critic.yaml  # Model 2
├── judge.yaml              # Model 3
└── gold_analysis.yaml      # Research only
```

### 8.2 Prompt Versioning
- Every prompt file has `version:` and `prompt_hash:` in YAML frontmatter
- Hash recorded in trace records
- Runtime validates prompt hash matches expected version

### 8.3 Prompt Context Budget
- Max context: **80% of model limit** (leave 20% for output)
- Runtime check before each LLM call
- Fail fast if overflow predicted

---

## 9. Model & Provider Rules

### 9.1 Model Configuration
- **Explicit config over defaults** — `config/models.yaml` is authoritative
- **Provider detection** — explicit config takes precedence over heuristic
- **Model identifier verification** — validate exact NVIDIA Nemotron identifier at startup

### 9.2 Retry & Backoff
| Error Type | Retry | Backoff |
|------------|-------|---------|
| 429 Rate limit | Yes | Exponential (1s, 2s, 4s, max 60s) |
| 404 Not found | No | — |
| Timeout | Yes | Exponential (max 3 retries) |
| Malformed JSON | Yes | Immediate (max 2 retries) |
| Schema validation failure | No | — (log, escalate) |
| Semantic failure | No | — (log, escalate) |

### 9.3 Concurrency Limits
- **Max parallel workers**: Configurable (default 4)
- **Per-provider rate limits**: Respect provider headers
- **Token budget tracking**: Log input/output tokens per report

---

## 10. Phase-Specific Rules

### Phase 1 (Runtime Blockers)
- Fix must not change behavior, only fix crashes
- Add regression test for each bug fixed

### Phase 2 (History)
- All history fixes must pass `test_*_not_duplicated` tests
- Oscillation detection must use full `critique_history`

### Phase 3 (Contracts)
- Judge enum change = atomic commit with prompt + schema + tests
- No free-form reason codes anywhere

### Phase 4 (Policy)
- Every contradiction classified with written justification
- Policy audit must pass (0 structural errors) before Phase 9

### Phase 7 (Tests)
- Tests written BEFORE implementation where possible (TDD)
- Each test maps to a specific rule in this document

### Phase 9 (E2E)
- Blind adjudication set created BEFORE evaluation
- Clinical error taxonomy applied to every error
- Results documented with taxonomy breakdown

---

## 11. Definition of Done (Per Phase)

| Phase | Done When |
|-------|-----------|
| 1 | All 4 runtime bugs fixed + regression tests pass |
| 2 | All 5 history duplication tests pass + oscillation/stuck tests pass |
| 3 | Judge contract unified across 4 artifacts + Critic validation tests pass |
| 4 | Policy audit: 0 structural errors, all 15 contradictions classified, policy_hash in traces |
| 5 | Requirements updated, startup validation passes, model verification works |
| 6 | `git status` clean (only source/code/test/doc), legacy path removed |
| 7 | All 18 new tests pass + existing 114 tests pass |
| 8 | Docs match implementation, schema versioning in all outputs |
| 9 | Production gate criteria met on 100-case blind evaluation |

---

## 12. Escalation Triggers

**Stop and escalate if:**
- Any test fails after implementation
- Policy audit shows new structural errors
- Trace duplication detected in output
- Context overflow predicted for any report
- Model identifier mismatch at startup
- API error rate > 5% in E2E run

---

## 13. Reference Files (Keep Updated)

| File | Purpose | Update Frequency |
|------|---------|------------------|
| `.kilo/plans/ragset-fix-plan.md` | Phase plan | Per phase completion |
| `.kilo/plans/ragset-fix-rules.md` | This file | As rules evolve |
| `config/models.yaml` | Model config | Per model change |
| `config/ragset_label_policy.yaml` | Canonical policy | Per policy change |
| `prompts/*.yaml` | Active prompts | Per prompt change |
| `schemas.py` | Data contracts | Per contract change |

---

**These rules are binding. Deviations require explicit approval and documentation.**