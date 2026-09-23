# RagSet — Reviewed LangGraph + Model 3 Judge Implementation Plan

## Review decision

**READY FOR IMPLEMENTATION**

The proposed architecture is directionally correct and all identified contradictions have been resolved:

```text
Model 1 — Inference
        ↓
Model 2 — Critic / Validator
        ↓
Model 3 — Judge
        ↓
PASS / RETRY / AMBIGUOUS / REVIEW / STOP
        ↓
possibly back to Model 1
```

LangGraph is appropriate for orchestration/state and Pydantic for typed model contracts.

However, the submitted plan contains several contradictions that should be fixed before coding.

## 1. Critical contradiction: retry mechanism

The plan simultaneously says:

- existing retry mechanism must remain unchanged
- LangGraph should call the existing retry mechanism
- but Phase H says the existing `loop.py` retry mechanism is **not used directly** and LangGraph handles iteration

These cannot all be true.

### Correct design

Use **one semantic retry implementation** and **one orchestration layer**:

```text
Existing retry semantics (loop.py run_report behavior)
        ↓
callable Model-1 attempt/retry operation (exposed from loop.py)
        ↓
LangGraph controls whether it is called again (orchestration only)
```

Do not maintain:

```text
LangGraph retry loop
+
loop.py retry loop
```

That would create nested retry controllers.

The existing behavior must remain **unchanged**:

- full 12-label regeneration
- validator feedback injection
- attempt numbering
- max-attempt semantics
- MODEL1_STUCK behavior
- existing trace semantics

**LangGraph replaces the orchestration layer only** — it does not reimplement retry logic. The `run_report` function's **single-attempt semantic behavior** (feedback injection, 12-label regeneration, attempt tracking, stuck detection) is extracted into a callable `run_model1_attempt()` and invoked by the LangGraph `inference` node. The multi-attempt orchestration loop in `run_report` is **replaced** by LangGraph.

## 2. Correct role of Model 2

Model 2 is a **Critic of Model 1**, not an independent classifier.

It must see:

```text
ORIGINAL_REPORT
CANONICAL_POLICY
RETRIEVED_GOLD_CONTEXT
MODEL_1_CURRENT_PREDICTION
MODEL_1_EVIDENCE
CURRENT_ATTEMPT
```

The submitted plan says to "keep independent derivation mandate." That wording is wrong for this architecture.

**Correct prompt instruction for Model 2:**

> First inspect the original report and canonical policy. Then inspect Model 1's prediction. Critique Model 1 against the report and policy. Do not assume Model 1 is correct or incorrect.

Do not tell Model 2 to derive a hidden independent answer and then compare. Model 2's role is to **critique Model 1's specific predictions** using the report and policy as the standard.

## 3. Model 2 schema

The proposed schema is appropriate:

```python
class CritiqueStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    AMBIGUOUS = "AMBIGUOUS"


class CritiqueIssueType(str, Enum):
    CLEAR_POLICY_CONFLICT = "CLEAR_POLICY_CONFLICT"
    CLEAR_REPORT_CONFLICT = "CLEAR_REPORT_CONFLICT"
    UNRESOLVED_POLICY = "UNRESOLVED_POLICY"
    REPORT_AMBIGUITY = "REPORT_AMBIGUITY"
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class CritiqueIssue(BaseModel):
    label: str
    model1_value: Literal[0, 1]
    proposed_value: Literal[0, 1] | None
    issue_type: CritiqueIssueType
    evidence: str | None
    policy_rule: str | None
    feedback: str


class CritiqueResult(BaseModel):
    status: CritiqueStatus
    issues: list[CritiqueIssue]
    summary: str
    actionable: bool  # True iff status == "FAIL" and at least one issue has issue_type in {CLEAR_POLICY_CONFLICT, CLEAR_REPORT_CONFLICT}
    affected_labels: list[str]
```

`proposed_value=None` is correct for ambiguity. Do not force a binary correction for unresolved cases.

## 4. Model 3 role

Model 3 is a **workflow judge**, not a third clinical classifier.

It receives:

```text
ORIGINAL_REPORT
CANONICAL_POLICY
RETRIEVED_GOLD_CONTEXT
MODEL_1_CURRENT_PREDICTION
MODEL_2_CRITIQUE
PREDICTION_HISTORY
CRITIQUE_HISTORY
JUDGMENT_HISTORY
CURRENT_ATTEMPT
MAX_ATTEMPTS
```

It may output **only** these workflow actions:

```text
PASS
RETRY_MODEL1
AMBIGUOUS
NEEDS_REVIEW
STOP
```

**Model 3 must never output the 12 clinical labels.** Its schema strictly forbids clinical label fields. Any output containing clinical labels is a schema validation error.

## 5. Model 3 rules

**Decision priority (highest to lowest):**

1. **STOP** — unrecoverable workflow/system state
2. **NEEDS_REVIEW** — max attempts reached (`attempt >= max_attempts`) — **OVERRIDES ALL OTHER**
3. **NEEDS_REVIEW** — oscillation detected (A→B→A pattern in prediction_history)
4. **NEEDS_REVIEW** — MODEL1_STUCK (identical 12-label vector across ≥2 attempts despite actionable critique)
5. **NEEDS_REVIEW** — repeated critique cycles (same critique pattern repeats)
6. **RETRY_MODEL1** — critique has `CLEAR_POLICY_CONFLICT` or `CLEAR_REPORT_CONFLICT` **AND** `attempt < max_attempts`
7. **AMBIGUOUS** — policy is UNRESOLVED, report genuinely ambiguous, or disagreement cannot be resolved from canonical policy
8. **PASS** — critique status PASS, no actionable issues

### PASS

Model 2 finds no actionable clear conflict (status PASS, no issues with CLEAR_* type).

### RETRY_MODEL1

Only for a clear actionable Model 1 error supported by report + policy:

- `CLEAR_POLICY_CONFLICT`
- `CLEAR_REPORT_CONFLICT`

**AND** `attempt < max_attempts` (enforced by graph routing as hard guard).

### AMBIGUOUS

Use when:

- policy is UNRESOLVED
- report is genuinely ambiguous
- disagreement cannot be resolved from the canonical policy

**Does not apply if max attempts reached** (see priority 2).

### NEEDS_REVIEW

Use when (in priority order):

1. Max attempts reached (`attempt >= max_attempts`)
2. Oscillation detected (A→B→A pattern in prediction_history)
3. MODEL1_STUCK (identical 12-label vector across ≥2 attempts despite actionable critique)
4. Repeated critique cycles (same critique pattern repeats)
5. Stable resolution cannot be established

### STOP

Only for unrecoverable workflow/system state.

Model 3 must never silently turn `UNRESOLVED` into `RETRY_MODEL1`.

## 6. Model configuration correction

The submitted plan specifies:

```yaml
nvidia/nemotron-3-ultra-550b-a55b
```

but the requested architecture calls for the NVIDIA **Nemotron 3 Super 120B model**.

Do not silently substitute 550B.

The implementation must use the **exact available 120B model identifier** from the NVIDIA API / OpenRouter catalog.

Do not invent an identifier.

**Correct model identifier:** `nvidia/nemotron-3-super-120b-a12b` (Nemotron 3 Super 120B)

Conceptually:

```yaml
nvidia-ragset-judge:
  provider: nvidia
  model: nvidia/nemotron-3-super-120b-a12b
  api_key_env: NVIDIA_API_KEY
  purpose: ragset_workflow_judge
  capabilities:
    - text
    - reasoning
    - structured_output
  parameters:
    temperature: 0
    max_output_tokens: 8000
    reasoning: false
  provider_policy:
    data_collection: deny
    zdr: true
    allow_fallbacks: false
```

If a USF profile already exists in the project, reuse its established convention. Do not invent a new provider abstraction merely to create a USF variant.

**Action required:** Add the judge profile to `config/models.yaml` using the exact identifier `nvidia/nemotron-3-super-120b-a12b` and reference it in `experiment.yaml` under `judge_profiles`.

## 7. LangGraph state

Recommended state:

```python
class RagSetState(TypedDict):
    study_instance_uid: str
    report: str
    canonical_policy: str
    retrieved_gold_context: str

    attempt: int
    max_attempts: int

    current_prediction: ReportPrediction | None
    current_critique: CritiqueResult | None
    current_judgment: JudgeResult | None

    prediction_history: list[ReportPrediction]
    critique_history: list[CritiqueResult]
    judgment_history: list[JudgeResult]

    # Oscillation/stuck detection (computed by inference/critic nodes from loop.py utilities)
    oscillation_detected: bool
    stuck_detected: bool
    oscillating_labels: list[str]
    stuck_labels: list[str]

    final_status: str | None          # "passed" | "needs_review" | "error"
    final_prediction: ReportPrediction | None
    review_reason: str | None

    trace_records: list[TraceRecord]  # Structured, not raw dict
```

**State design principles:**
- No duplicate state — reuse existing Pydantic models (`ReportPrediction`, `CritiqueResult`, `JudgeResult`)
- History lists are append-only — each cycle adds one entry
- `attempt` is the current 1-indexed attempt number
- `trace_records` accumulates structured `TraceRecord` objects (see §14) for final write

## 8. LangGraph nodes

Create:

```text
initialize
inference
critic
judge
finalize
```

Responsibilities:

- `initialize`:
  - Load canonical policy via `get_canonical_policy_text()` (policy_loader.py)
  - Retrieve gold context via `GoldRetriever.retrieve(report)`
  - Initialize all state fields (attempt=1, empty histories, etc.)
- `inference`: invoke `run_model1_attempt()` with current state
- `critic`: invoke Model 2 (CriticModel) with Model 1 output; update oscillation/stuck detection flags
- `judge`: invoke Model 3 (JudgeModel) with Model 1 + Model 2 + history + detection flags
- `finalize`: produce existing final result and trace; write accumulated trace_records

## 9. Graph

```text
START
  ↓
initialize
  ↓
inference
  ↓
critic
  ↓
judge
```

**Conditional routing from `judge` node:**

| Judge Action | Condition | Next Node |
|--------------|-----------|-----------|
| `PASS` | Critique status PASS, no actionable issues | `finalize` → END |
| `AMBIGUOUS` | Critique has UNRESOLVED_POLICY/REPORT_AMBIGUITY issues | `finalize` → END |
| `NEEDS_REVIEW` | Oscillation, MODEL1_STUCK, max attempts reached, repeated cycles | `finalize` → END |
| `STOP` | Unrecoverable workflow/system state | `finalize` → END |
| `RETRY_MODEL1` | Critique has CLEAR_POLICY_CONFLICT or CLEAR_REPORT_CONFLICT **AND** `attempt < max_attempts` | `inference` (loop) |

**Graph structure:**

```text
finalize → END
```

**Max-attempts guard:** The conditional routing function **MUST** check `state["attempt"] < state["max_attempts"]` before allowing `RETRY_MODEL1`. If `attempt >= max_attempts`, route to `finalize` with `NEEDS_REVIEW` regardless of judge output.

The `RETRY_MODEL1` edge creates the loop: `judge` → `inference` → `critic` → `judge` → ...

## 10. Retry integration

Do not create a second retry implementation.

**Extract the single-attempt logic from `run_report` into a reusable callable:**

```python
def run_model1_attempt(
    report: str,
    gold_context: str,
    policy: str,
    labels: list[str],
    previous_prediction: ReportPrediction | None = None,
    validator_feedback: dict | None = None,
    attempt: int = 1,
) -> ReportPrediction:
    """
    Execute ONE Model 1 inference attempt with optional validator feedback.
    
    This is the semantic core of the retry mechanism — full 12-label regeneration
    with feedback injection — WITHOUT the multi-attempt orchestration loop.
    
    Extracted from loop.py:run_report() lines 162-178 (inference call) and
    the feedback preparation logic lines 165-177.
    """
    ...
```

**Signature requirements:**
- `labels`: LABEL_KEYS (12 labels) — required for schema validation
- `gold_context`: retrieved gold examples + canonical policy text
- `policy`: canonical policy text (same as Model 2 and Model 3 receive)
- `previous_prediction`: Model 1's prior prediction (for retry context)
- `validator_feedback`: Model 2's critique (for retry reconsideration)
- `attempt`: 1-indexed attempt number

**Preserved semantics:**
- Full 12-label regeneration (not patch)
- Validator feedback injected as reconsideration guidance, NOT evidence
- Attempt numbering for prompt context
- Same Model 1 prompt template and inference call

LangGraph controls:

```text
judge → RETRY_MODEL1 → inference
```

The `inference` node invokes `run_model1_attempt()` with current state.

Add a regression test proving:

- exactly one retry occurs per RETRY_MODEL1 decision
- attempt number increments once per LangGraph cycle
- existing feedback reaches Model 1 via `validator_feedback` parameter
- all 12 labels are regenerated (schema validation)
- no nested retry loop exists (`run_report` NOT called recursively)

## 11. Do not put clinical logic in Python

Do not add clinical label rules to graph routing.

The graph routes workflow actions only.

Clinical interpretation remains in:

- Model 1
- Model 2
- canonical policy
- Model 3's workflow judgment

## 12. Oscillation and MODEL1_STUCK

**Keep the existing detection mechanisms in `loop.py`** (`_detect_validator_oscillation`, `_detect_model1_stuck`) as the authoritative implementation.

**Pass detection results to Model 3 via state:**

Add to `RagSetState`:
```python
oscillation_detected: bool
stuck_detected: bool
oscillating_labels: list[str]
stuck_labels: list[str]
```

The `inference` and `critic` nodes update these fields by calling the existing detection functions on the accumulated history. Model 3 receives them as structured input.

**Model 3 prompt addition (User prompt):**

```text
OSCILLATION_DETECTED: {oscillation_detected}
OSCILLATING_LABELS: {oscillating_labels}
STUCK_DETECTED: {stuck_detected}
STUCK_LABELS: {stuck_labels}
```

**Do not create two competing definitions.** Model 3 uses the pre-computed flags; the existing functions remain the single source of truth.

Example oscillation (A→B→A):
```text
Attempt 1: A
Attempt 2: B
Attempt 3: A
```
→ `oscillation_detected=true` → Model 3 returns `NEEDS_REVIEW`

Example MODEL1_STUCK (A→A→A despite feedback):
```text
Attempt 1: A
Attempt 2: A
Attempt 3: A
```
→ `stuck_detected=true` → Model 3 returns `NEEDS_REVIEW`

Do not keep retrying indefinitely.

## 13. Model 3 prompt

### System

```text
You are the RagSet workflow judge.

You do not generate clinical labels.
You do not modify Model 1's labels.

You determine only what the workflow should do next.

Allowed actions:
PASS
RETRY_MODEL1
AMBIGUOUS
NEEDS_REVIEW
STOP

Use the ORIGINAL_REPORT and CANONICAL_POLICY as authoritative context.

Treat Model 2 as a critique, not unquestionable truth.

Choose RETRY_MODEL1 only when there is a clear actionable Model 1 error supported by the report and policy.

Choose AMBIGUOUS when the relevant convention is unresolved or the evidence cannot establish a clear correction.

Choose NEEDS_REVIEW when the workflow is oscillating, Model 1 is stuck, attempts are exhausted, or stable resolution cannot be established.

Do not invent clinical evidence.
Do not output clinical labels.
Do not override the canonical policy.
```

### User

```text
ORIGINAL_REPORT:
{original_report}

CANONICAL_POLICY:
{canonical_policy}

RETRIEVED_GOLD_CONTEXT:
{retrieved_gold_examples}

MODEL_1_CURRENT_PREDICTION:
{model_1_prediction}

MODEL_2_CRITIQUE:
{model_2_critique}

PREDICTION_HISTORY:
{prediction_history}

CRITIQUE_HISTORY:
{critique_history}

JUDGMENT_HISTORY:
{judgment_history}

CURRENT_ATTEMPT_NUMBER:
{attempt_number}

MAX_ATTEMPTS:
{max_attempts}

OSCILLATION_DETECTED: {oscillation_detected}
OSCILLATING_LABELS: {oscillating_labels}
STUCK_DETECTED: {stuck_detected}
STUCK_LABELS: {stuck_labels}

Determine the next workflow action.

**HARD CONSTRAINT:** If CURRENT_ATTEMPT_NUMBER >= MAX_ATTEMPTS, you MUST return NEEDS_REVIEW. Do not return RETRY_MODEL1.
```

## 14. Trace

Extend the existing trace rather than replacing it.

**Keep (existing fields in `trace.py:write_trace`):**

```text
study_instance_uid
stage
attempt
requested_model
actual_model
provider
prompt_hash          # computed via trace.prompt_hash()
latency
token_usage          # input_tokens, output_tokens (when available)
status
error
```

**Note:** `response_hash` is NOT currently in `trace.py` — add it as new field.

**Add (new fields):**

```text
graph_node              # "initialize" | "inference" | "critic" | "judge" | "finalize"
judge_action            # PASS | RETRY_MODEL1 | AMBIGUOUS | NEEDS_REVIEW | STOP
response_hash           # SHA256 of model response text
prediction_history      # list of 12-label vectors per attempt
critique_history        # list of critique summaries per attempt
judgment_history        # list of judge actions per attempt
```

**Structured `TraceRecord` (Pydantic model for `trace_records` in state):**

```python
class TraceRecord(BaseModel):
    timestamp_utc: str
    study_instance_uid: str
    graph_node: Literal["initialize", "inference", "critic", "judge", "finalize"]
    stage: Literal["inference", "validation", "judgment"]
    attempt: int
    requested_model: str
    actual_model: str | None
    provider: str
    prompt_hash: str
    response_hash: str | None
    latency_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    status: Literal["success", "error"]
    error: str | None
    judge_action: Literal["PASS", "RETRY_MODEL1", "AMBIGUOUS", "NEEDS_REVIEW", "STOP"] | None
```

**Per-attempt trace bundle (written at each judge node):**

```json
{
  "study_instance_uid": "...",
  "attempt": 1,
  "model1_prediction": {...},
  "model2_critique": {...},
  "model3_judgment": {...},
  "judge_action": "RETRY_MODEL1",
  "graph_node": "judge",
  "timestamp": "...",
  "latency": {...},
  "token_usage": {...},
  "requested_model": "...",
  "actual_model": "...",
  "provider": "..."
}
```

Do not store hidden chain-of-thought.

## 15. Required tests

In addition to the submitted 15 tests, add:

16. Existing retry function is actually called.
17. Existing retry semantics remain unchanged.
18. **No nested retry loops** — LangGraph loop is the only loop; `run_report` is not called recursively.
19. Attempt number increments exactly once per LangGraph cycle.
20. Validator feedback reaches Model 1.
21. Retry regenerates all 12 labels.
22. Graph terminates after PASS.
23. Graph terminates after REVIEW.
24. Graph cannot exceed configured maximum.
25. Model 3 malformed output is handled.
26. Model 2 malformed output is handled.
27. Model 1 malformed output is handled.

**Explicit nested-retry prevention tests:**

28. `run_report` is invoked exactly once per report (not per retry).
29. LangGraph `inference` node calls the exposed Model 1 attempt operation, not `run_report`.
30. No code path creates a second retry controller inside the graph.

Most important test:

```text
Model 3 = RETRY_MODEL1
        ↓
exactly one Model 1 retry
        ↓
no second internal retry loop
```

## 16. Parallel workers

Existing parallel workers may remain.

Each worker should execute an isolated graph state:

```text
worker
  ↓
graph.invoke(report_state)
  ↓
final result
```

Do not share mutable per-report state between workers.

**GoldRetriever thread-safety:** The shared `GoldRetriever` instance (created once in `run_inference.py`) is read-only after `fit()` — `self.matrix` and `self.vectorizer` are immutable. Concurrent `retrieve()` calls are thread-safe. Each worker receives the same retriever instance; no per-worker copy needed.

## 17. Diagnostics

Use exactly the same 50 reports as V1/V2.

Do not select a new sample.

Record:

- PASS
- NEEDS_REVIEW
- AMBIGUOUS
- VALIDATOR_INSTABILITY
- MODEL1_STUCK
- MODEL_ERROR
- VALIDATOR_ERROR
- POLICY_ERROR
- GOLD_CONTRADICTION
- RETRIEVAL_ERROR
- API errors
- retry count
- total attempts
- Model 1 latency
- Model 2 latency
- Model 3 latency

Also compare V2 → V3.

## 18. Oscillation gate — require no unexplained instability

The correct requirement is:

> **No unexplained instability remains.**

A legitimate unresolved case may correctly terminate as:

```text
NEEDS_REVIEW
```

The objective is **stable and auditable routing**, not forcing every case to PASS.

**Distinction:**
- **Explained instability** → policy is UNRESOLVED, report is genuinely ambiguous → routes to AMBIGUOUS/NEEDS_REVIEW with clear reason
- **Unexplained instability** → validator flips without policy basis, Model 1 oscillates without feedback → indicates a bug to fix

The gate checks that every instability case has a documented root cause in the audit.

## 19. Corrected file plan

### New

```text
src/ragset_inference/graph_state.py
src/ragset_inference/graph_nodes.py
src/ragset_inference/graph.py
prompts/judge.yaml
tests/test_langgraph.py
```

### Modified

```text
src/ragset_inference/schemas.py
src/ragset_inference/models.py
prompts/validation.yaml
config/models.yaml
config/experiment.yaml
src/ragset_inference/trace.py
runners/run_inference.py
```

**WILL modify `loop.py`** to:
1. Extract single-attempt logic into `run_model1_attempt()` function (see §10) with full signature including `labels` parameter
2. Keep `_detect_validator_oscillation()`, `_detect_model1_stuck()`, `_filter_real_issues()`, `_classify_failure()` as reusable utilities
3. Keep `run_report()` for backward compatibility (diagnostic runner) but mark as legacy
4. Do not rewrite semantic behavior — only extract and expose

## 20. Implementation order

### Phase 1 — Contracts (schemas)
1. Add `CritiqueResult` and `CritiqueIssue` to `schemas.py`.
2. Add `JudgeResult` and `JudgeAction` to `schemas.py`.
3. Run schema tests (`test_schemas.py`).

### Phase 2 — Model 2 Critic
4. Convert `ValidatorModel` → `CriticModel` in `models.py` (return `CritiqueResult`).
5. Update `prompts/validation.yaml` to Critic prompt + CritiqueResult output format.
6. Verify Model 2 prompt includes `MODEL_1_CURRENT_PREDICTION` and `MODEL_1_EVIDENCE`.
7. Run existing tests — ensure no regression.

### Phase 3 — Model 3 Judge
8. Add `JudgeModel` class in `models.py` (return `JudgeResult`).
9. Create `prompts/judge.yaml` with system + user prompt.
10. **Add judge profile to `config/models.yaml` using exact identifier `nvidia/nemotron-3-super-120b-a12b`** (Nemotron 3 Super 120B).
11. Reference judge profile in `experiment.yaml` under `judge_profiles`.
12. Test judge schema validation.

### Phase 4 — Retry boundary (critical)
13. Identify the reusable Model 1 attempt operation in `loop.py` (extract from `run_report`).
14. Expose it as a callable: `run_model1_attempt(report, gold_context, policy, labels, previous_prediction, validator_feedback, attempt)` with full signature per §10.
15. Prove semantic behavior is unchanged: full 12-label regeneration, feedback injection, attempt tracking.
16. Add **no-nested-retry test** proving `run_report` is not called recursively.

### Phase 5 — LangGraph
17. Create `graph_state.py` with `RagSetState` TypedDict **including oscillation/stuck detection fields** (`oscillation_detected`, `stuck_detected`, `oscillating_labels`, `stuck_labels`).
18. Create `graph_nodes.py` with 5 nodes: `initialize`, `inference`, `critic`, `judge`, `finalize` per §8 responsibilities.
19. Create `graph.py` with StateGraph, conditional routing table with max-attempts guard.
20. Integrate trace fields in `trace.py` and node trace writes; add `TraceRecord` Pydantic model.

### Phase 6 — Tests
21. Run all existing tests (`test_loop.py`, `test_e2e_regression.py`, etc.) — must pass.
22. Run new `test_langgraph.py` (28+ tests including nested-retry prevention).
23. Run retry integration tests.

### Phase 7 — Diagnostic
24. Run the **exact same 50 reports** (seed 42, same CSV selection).
25. Generate V3 audit: `diagnostic_50_report.json/.csv`, `diagnostic_50_trace.jsonl`.
26. Compare V2 → V3 metrics table.

### Phase 8 — Gate
27. Produce:
   - `results/validation/V3_LANGGRAPH_IMPLEMENTATION.md`
   - `results/validation/GATE_DECISION_V3.md`
28. **Do not launch the 4,349-report run automatically.**

## 21. V3 gate

### READY only if

- all existing tests pass
- all LangGraph tests pass
- no nested retry controller exists
- existing retry semantics remain unchanged
- Model 2 genuinely critiques Model 1
- Model 3 controls workflow only
- Model 3 outputs no clinical labels
- all three models receive the same canonical policy
- Model 3 sees complete attempt history
- oscillation/stuck cases terminate correctly
- unresolved cases route to AMBIGUOUS/review
- no unexplained new regressions
- complete traceability exists
- API failures are separated from semantic failures
- the same 50-report diagnostic is reproducible

### NOT_READY if

- retry semantics unexpectedly change
- nested retry loops exist
- Model 3 becomes a clinical classifier
- Model 2 does not receive Model 1 output
- Model 3 lacks history
- graph can loop indefinitely
- state leaks between workers
- unexplained oscillation remains
- unexplained MODEL1_STUCK remains
- policy contradictions appear
- new semantic regressions appear

## 22. Final architecture

```text
                         LANGGRAPH
                       STATE GRAPH
                           │
                           ▼
                    ┌──────────────┐
                    │ INITIALIZE   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   MODEL 1    │
                    │  INFERENCE   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   MODEL 2    │
                    │    CRITIC    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   MODEL 3    │
                    │    JUDGE     │
                    │ NVIDIA 120B  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
            PASS         RETRY        REVIEW
              │            │            │
              ▼            ▼            ▼
             END        MODEL 1        END
                          │
                          └────→ MODEL 2
                                  │
                                  └────→ MODEL 3
```

## Final review decision

**IMPLEMENT THE REVISED PLAN.**

All mandatory corrections have been applied:

1. ✅ Retry contradiction resolved — LangGraph replaces orchestration only; single-attempt logic extracted to `run_model1_attempt()`
2. ✅ No nested retry controllers — graph routing is the only loop; `run_report` not called recursively
3. ✅ "Independent derivation" wording removed — Model 2 is a Critic of Model 1
4. ✅ Model 2 explicitly critiques Model 1 — prompt and schema updated
5. ✅ Model 3 is workflow judge only — schema forbids clinical labels
6. ✅ Exact NVIDIA 120B model identifier — `nvidia/nemotron-3-super-120b-a12b`
7. ✅ No invented USF config — reuse existing convention if available
8. ✅ No-nested-retry tests added — tests 28-30
9. ✅ No unexplained instability — explained vs unexplained distinction with gate criteria
10. ✅ Same 50-report diagnostic — seed 42, same CSV selection

### Recommended stack

```text
Pydantic
   ↓
typed model contracts

LangGraph
   ↓
state + orchestration + conditional loop

Model 1
   ↓
clinical inference

Model 2
   ↓
critique / validation

Model 3
   ↓
workflow judgment

Existing retry semantics
   ↓
full Model 1 regeneration

Canonical policy
   ↓
shared clinical convention
```
