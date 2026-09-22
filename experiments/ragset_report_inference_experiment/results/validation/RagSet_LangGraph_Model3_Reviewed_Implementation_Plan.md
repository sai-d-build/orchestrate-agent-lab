# RagSet — Reviewed LangGraph + Model 3 Judge Implementation Plan

## Review decision

**REVISE BEFORE IMPLEMENTATION**

The proposed architecture is directionally correct:

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
Existing retry semantics
        ↓
callable Model-1 attempt/retry operation
        ↓
LangGraph controls whether it is called again
```

Do not maintain:

```text
LangGraph retry loop
+
loop.py retry loop
```

That would create nested retry controllers.

The existing behavior must remain:

- full 12-label regeneration
- validator feedback injection
- attempt numbering
- max-attempt semantics
- MODEL1_STUCK behavior
- existing trace semantics

LangGraph should replace only the surrounding orchestration.

## 2. Correct role of Model 2

Model 2 is a **critic of Model 1**, not an independent classifier.

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

Use:

> First inspect the original report and canonical policy. Then inspect Model 1's prediction. Critique Model 1 against the report and policy. Do not assume Model 1 is correct or incorrect.

Do not tell Model 2 to derive a hidden independent answer and then compare unless that is intentionally part of the design.

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
    actionable: bool
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

It may output only:

```text
PASS
RETRY_MODEL1
AMBIGUOUS
NEEDS_REVIEW
STOP
```

It must never output the 12 clinical labels.

## 5. Model 3 rules

### PASS

Model 2 finds no actionable clear conflict.

### RETRY_MODEL1

Only for a clear actionable Model 1 error supported by report + policy:

- `CLEAR_POLICY_CONFLICT`
- `CLEAR_REPORT_CONFLICT`

### AMBIGUOUS

Use when:

- policy is UNRESOLVED
- report is genuinely ambiguous
- disagreement cannot be resolved from the canonical policy

### NEEDS_REVIEW

Use when:

- oscillation is present
- Model 1 is stuck
- repeated critique cycles occur
- attempts are exhausted
- stable resolution cannot be established

### STOP

Only for unrecoverable workflow/system state.

Model 3 must never silently turn `UNRESOLVED` into `RETRY_MODEL1`.

## 6. Model configuration correction

The submitted plan specifies:

```yaml
nvidia/nemotron-3-ultra-550b-a55b
```

but the requested architecture calls for the NVIDIA **1200B-class model**.

Do not silently substitute 550B.

The implementation must inspect the existing NVIDIA/OpenRouter configuration and use the exact available 1200B model identifier.

Do not invent an identifier.

Conceptually:

```yaml
nvidia-ragset-judge:
  provider: nvidia
  model: <EXACT_AVAILABLE_NVIDIA_1200B_MODEL>
  api_key_env: NVIDIA_API_KEY
  purpose: ragset_workflow_judge
  capabilities:
    - text
    - reasoning
    - structured_output
  parameters:
    temperature: 0
    max_output_tokens: 8000
```

If a USF profile already exists in the project, reuse its established convention. Do not invent a new provider abstraction merely to create a USF variant.

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

    final_status: str | None
    final_prediction: ReportPrediction | None
    review_reason: str | None

    trace_records: list[dict]
```

Avoid duplicating existing state unnecessarily.

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

- `initialize`: initialize state and load/retrieve context
- `inference`: invoke Model 1 through the existing inference operation
- `critic`: invoke Model 2 with Model 1 output
- `judge`: invoke Model 3 with Model 1 + Model 2 + history
- `finalize`: produce existing final result and trace

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

Conditional routing:

```text
PASS            → finalize
AMBIGUOUS       → finalize
NEEDS_REVIEW    → finalize
STOP            → finalize
RETRY_MODEL1    → inference
```

Then:

```text
finalize → END
```

## 10. Retry integration

Do not create a second retry implementation.

Refactor only enough of the existing loop to expose one reusable operation such as:

```python
run_model1_attempt(
    report,
    gold_context,
    policy,
    previous_prediction=None,
    validator_feedback=None,
    attempt=1,
)
```

It must preserve the current semantics.

LangGraph controls:

```text
judge → RETRY_MODEL1 → inference
```

The inference node invokes the existing Model 1 attempt behavior.

Add a regression test proving:

- exactly one retry occurs
- attempt number increments once
- existing feedback reaches Model 1
- all 12 labels are regenerated
- no nested retry loop exists

## 11. Do not put clinical logic in Python

Do not add clinical label rules to graph routing.

The graph routes workflow actions only.

Clinical interpretation remains in:

- Model 1
- Model 2
- canonical policy
- Model 3's workflow judgment

## 12. Oscillation and MODEL1_STUCK

Keep the existing mechanisms.

Use their state/history as information available to Model 3.

Do not create two competing definitions of oscillation or stuck behavior.

Example:

```text
Attempt 1: A
Attempt 2: B
Attempt 3: A
```

Model 3 should recognize the cycle and return:

```text
NEEDS_REVIEW
```

Likewise:

```text
Attempt 1: A
Attempt 2: A
Attempt 3: A
```

after actionable feedback should terminate as `NEEDS_REVIEW`.

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

Determine the next workflow action.
```

## 14. Trace

Extend the existing trace rather than replacing it.

Keep:

```text
study_instance_uid
stage
attempt
requested_model
actual_model
provider
prompt_hash
response_hash
latency
token_usage
status
error
```

Add:

```text
graph_node
judge_action
prediction_history
critique_history
judgment_history
```

Do not store hidden chain-of-thought.

## 15. Required tests

In addition to the submitted 15 tests, add:

16. Existing retry function is actually called.
17. Existing retry semantics remain unchanged.
18. No nested retry loops.
19. Attempt number increments exactly once.
20. Validator feedback reaches Model 1.
21. Retry regenerates all 12 labels.
22. Graph terminates after PASS.
23. Graph terminates after REVIEW.
24. Graph cannot exceed configured maximum.
25. Model 3 malformed output is handled.
26. Model 2 malformed output is handled.
27. Model 1 malformed output is handled.

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

## 18. Do not require zero instability as the only criterion

The correct requirement is:

> No unexplained instability remains.

A legitimate unresolved case may correctly terminate as:

```text
NEEDS_REVIEW
```

The objective is stable and auditable routing, not forcing every case to PASS.

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

Potentially modify `loop.py` only to extract/reuse the existing retry operation. Do not rewrite its semantic behavior.

## 20. Implementation order

### Phase 1 — Contracts
1. Add CritiqueResult.
2. Add JudgeResult.
3. Test schemas.

### Phase 2 — Model 2
4. Convert ValidatorModel → CriticModel.
5. Update critique prompt.
6. Verify Model 2 sees Model 1.
7. Run tests.

### Phase 3 — Model 3
8. Add JudgeModel.
9. Add judge.yaml.
10. Configure exact available NVIDIA 1200B model.
11. Test judge schema.

### Phase 4 — Retry boundary
12. Identify current retry operation.
13. Expose it for LangGraph if necessary.
14. Prove semantic behavior is unchanged.
15. Add no-nested-retry test.

### Phase 5 — LangGraph
16. Add state.
17. Add nodes.
18. Add graph.
19. Add conditional routing.
20. Add trace integration.

### Phase 6 — Tests
21. Run all existing tests.
22. Run all LangGraph tests.
23. Run retry integration tests.

### Phase 7 — Diagnostic
24. Run the exact same 50 reports.
25. Generate V3 audit.
26. Compare V2 → V3.

### Phase 8 — Gate
27. Produce:
   - `results/validation/V3_LANGGRAPH_IMPLEMENTATION.md`
   - `results/validation/GATE_DECISION_V3.md`
28. Do not launch the 4,349-report run automatically.

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
                    │ NVIDIA 1200B │
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

**Do not implement the submitted plan verbatim.**

Mandatory corrections:

1. Resolve the retry contradiction.
2. Do not maintain two retry controllers.
3. Remove "independent derivation" wording from Model 2.
4. Model 2 explicitly critiques Model 1.
5. Model 3 is workflow judge only.
6. Use the exact available NVIDIA 1200B model identifier rather than silently using 550B.
7. Do not invent a USF provider configuration if the project does not already support it.
8. Add no-nested-retry tests.
9. Require no **unexplained** instability, rather than mechanically requiring zero.
10. Run the same 50-report diagnostic before considering the 4,349-report run.

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
