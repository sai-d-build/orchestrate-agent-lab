# RagSet Inference Experiment - Phase-by-Phase Fix Plan

**Status:** RED for commit → Target: GREEN after Phase 4, PRODUCTION-READY after Phase 9

---

## Phase 1: Critical Runtime Blockers (Day 1-2)
*Must fix before any commit - these cause crashes or data corruption*

### 1.1 Fix `JudgeReasonCode.AMBIGUOUS` Runtime Bug
**File:** `src/ragset_inference/graph_nodes.py` → `_determine_finalization_metadata()`
**Issue:** Code references `JudgeReasonCode.AMBIGUOUS` which doesn't exist
**Fix:** Replace with valid enum: `JudgeReasonCode.REPORT_AMBIGUITY` or `JudgeReasonCode.NO_EVIDENCE_AMBIGUOUS`
**Test:** Add `test_ambiguous_finalization_does_not_raise`

### 1.2 Fix `trace.py` `prompt_hash` Name Shadowing
**File:** `src/ragset_inference/trace.py` → `write_trace()`
**Issue:** Parameter `prompt_hash` shadows function `prompt_hash()`
**Fix:** Rename parameter to `provided_prompt_hash` or rename function to `compute_prompt_hash()`
**Test:** Add `test_trace_prompt_hash_preserved`

### 1.3 Stop Duplicate Trace Writes
**Files:** 
- `src/ragset_inference/graph_nodes.py` → `finalize_node()` (remove `write_trace` calls)
- `runners/run_inference.py` → `worker_process_chunk()` (keep as single writer)
**Issue:** Traces written twice → duplicated `model_trace.jsonl` entries
**Fix:** Single owner pattern - only worker/main writer writes traces
**Test:** Add `test_trace_written_once`

### 1.4 Preserve `prompt_hash` in Trace Pipeline
**Files:**
- `runners/run_inference.py` → `worker_process_chunk()` valid keys (add `"prompt_hash"`)
- `src/ragset_inference/graph_nodes.py` → pass existing hash instead of `prompt=""`
**Issue:** Hash discarded → auditability broken
**Test:** Verify hash appears in `model_trace.jsonl`

---

## Phase 2: Graph History Duplication Fixes (Day 2-3)
*Core correctness - affects Judge context, oscillation, stuck detection*

### 2.1 Fix Prediction History Duplication
**File:** `src/ragset_inference/graph_nodes.py` → `critic_node()`
**Issue:** `state["prediction_history"] + [prediction]` but current already appended by `inference_node()`
**Fix:** Use `state["prediction_history"]` directly
**Test:** Add `test_prediction_history_not_duplicated`

### 2.2 Fix Critique History Duplication
**File:** `src/ragset_inference/graph_nodes.py` → `judge_node()`
**Issue:** `state["critique_history"] + [state["current_critique"]]` but current already appended by `critic_node()`
**Fix:** Pass `state["critique_history"]` directly
**Test:** Add `test_critic_history_not_duplicated`

### 2.3 Fix Judgment History Duplication
**File:** `src/ragset_inference/graph_nodes.py` → `judge_node()` / graph routing
**Issue:** Same pattern - current judgment duplicated in history passed to next cycle
**Fix:** Pass `state["judgment_history"]` directly
**Test:** Add `test_judgment_history_not_duplicated`

### 2.4 Fix Graph-Level Oscillation Detection
**File:** `src/ragset_inference/graph_nodes.py` → `critic_node()`
**Issue:** `validator_history` built only from current critique, not `critique_history`
**Fix:** Aggregate `proposed_value` from `state["critique_history"] + [current_critique]`
**Test:** Add `test_oscillation_uses_prior_critiques`

### 2.5 Fix Stuck Detection False Positives
**File:** `src/ragset_inference/loop.py` → `_detect_model1_stuck()` / `graph_nodes.py` caller
**Issue:** Current prediction counted twice → false stuck signal on first retry
**Fix:** Use deduplicated history; require `min_attempts >= 2` with actual changes
**Test:** Add `test_model1_stuck_not_triggered_on_first_attempt`

---

## Phase 3: Contract Alignment (Day 3-4)
*Schema/prompt/enum consistency - prevents silent contract violations*

### 3.1 Define Canonical Judge Output Contract
**Decision needed:** Choose ONE enum set for Judge output
**Recommended contract:**
```
SUPPORTED_PASS
CLEAR_ACTIONABLE_CONFLICT
MODEL2_UNSUPPORTED_CORRECTION
UNRESOLVED_POLICY
REPORT_AMBIGUITY
INSUFFICIENT_EVIDENCE
MODEL1_STUCK
OSCILLATION
REPEATED_CRITIQUE
MAX_ATTEMPTS
INTEGRITY_FAILURE
HARD_STOP
```

### 3.2 Align All Four Artifacts to Contract
| Artifact | Action |
|----------|--------|
| `schemas.py` `JudgeReasonCode` | Update enum to match contract |
| `prompts/judge.yaml` | Update allowed reason codes in prompt |
| `graph_nodes.py` | Update `_determine_finalization_metadata()` mapping |
| `tests/test_langgraph.py` | Update test expectations |

### 3.3 Add Critic `model1_value` Validation
**File:** `src/ragset_inference/loop.py` → `_validate_critic_output()`
**Issue:** Safety gate doesn't verify `model1_value` matches actual Model 1 prediction
**Fix:** Add check: `issue.model1_value == current_prediction.predictions[issue.label].value`
**Test:** Add `test_critic_model1_value_matches_actual_prediction`

### 3.4 Add `affected_labels` Consistency Check
**File:** `src/ragset_inference/loop.py` → `_validate_critic_output()`
**Issue:** `affected_labels` should exactly match labels with CLEAR_* issues
**Fix:** Validate set equality
**Test:** Add `test_affected_labels_matches_clear_issues`

---

## Phase 4: Policy Reconciliation (Day 4-5)
*Clinical correctness - policy must match gold standard*

### 4.1 Run Policy Reconciliation
**File:** `runners/policy_reconciliation.py`
**Action:** Execute to identify exact discrepancies between CSV, gold_analysis, canonical policy

### 4.2 Fix 12 Count Errors
**Data from audit:**
| Label | CSV | Policy | Delta |
|-------|-----|--------|-------|
| ACL | 24/34 | 24/37 | +3 neg |
| MCL | 9/49 | 9/52 | +3 neg |
| Medial Meniscus | 26/32 | 27/34 | +1 pos, +2 neg |
| Lateral Meniscus | 23/35 | 24/37 | +1 pos, +2 neg |
| Medial OA | 15/43 | 16/45 | +1 pos, +2 neg |
| Lateral OA | 11/47 | 11/50 | +3 neg |
| PF OA | 21/37 | 23/37 | +2 pos |
| Effusion | 35/23 | 36/24 | +1 pos, +1 neg |
| Synovitis | 27/31 | 28/32 | +1 pos, +1 neg |
| Bakers | 12/46 | 12/48 | +2 neg |
| Contusion | 19/39 | 19/41 | +2 neg |
| Fracture | 18/40 | 18/42 | +2 neg |

**Fix:** Reconcile each label - determine source of truth (CSV gold labels)

### 4.3 Resolve 15 Contradictions
**File:** `config/ragset_label_policy.yaml` / `runners/policy_reconciliation.py`
**Action:** Review each contradiction, decide canonical rule

### 4.4 Validate Policy Loader
**File:** `src/ragset_inference/policy_loader.py`
**Test:** `runners/verify_gold_policy.py` must pass with zero errors

---

## Phase 5: Dependencies & Requirements (Day 5)
*Basic reproducibility*

### 5.1 Update `requirements.txt`
**Add:**
```
langgraph>=0.2.0
python-dotenv>=1.0.0
```

### 5.2 Verify Import Compatibility
**Test:** `python -c "import langgraph; from dotenv import load_dotenv"`

---

## Phase 6: Clean Generated/Backup Files (Day 5)
*Commit hygiene - before any commit*

### 6.1 Remove from Git Tracking (if tracked)
```bash
git rm --cached experiments/ragset_report_inference_experiment/results/inference/model_trace.jsonl
git rm --cached experiments/ragset_report_inference_experiment/results/inference/predictions.jsonl
```

### 6.2 Verify `.gitignore` Coverage
**Ensure ignored:**
```
results/**/*.json
results/**/*.jsonl
results/**/*.csv
data/validation/*.csv
__pycache__/
*.pyc
.DS_Store
*.bkp.*
*_backup*
```

### 6.3 Archive/Remove Backup Prompts
**Move to `archive/` or delete:**
- `prompts/inference_backup2.yaml`
- `prompts/validation_back2.yaml`
- `prompts/validation.yaml` (legacy, not used by graph)

---

## Phase 7: Test Coverage Gaps (Day 5-6)
*Critical integration tests missing*

### 7.1 Add Missing Tests to `test_langgraph.py`
```python
test_critic_history_not_duplicated
test_prediction_history_not_duplicated
test_judgment_history_not_duplicated
test_oscillation_uses_prior_critiques
test_model1_stuck_not_triggered_on_first_attempt
test_judge_reason_code_matches_prompt_contract
test_ambiguous_finalization_does_not_raise
test_trace_written_once
test_trace_prompt_hash_preserved
test_critic_model1_value_matches_actual_prediction
test_affected_labels_matches_clear_issues
```

### 7.2 Add E2E Graph Integration Test
**File:** `test_e2e_regression.py`
**Test:** Full graph run with mocked LLMs covering all terminal paths

---

## Phase 8: Documentation Updates (Day 6)
*Sync docs with implementation*

### 8.1 Update `PARALLEL_INFERENCE_DESIGN.md`
- Document `increment_attempt` node
- Document finalization metadata fields
- Document history flow (no duplication)

### 8.2 Update `README.md`
- Current architecture diagram
- Prompt contract summary
- How to run inference/validation

### 8.3 Move Design Docs from `results/validation/` to `docs/`
- `RagSet_LangGraph_Model3_Reviewed_Implementation_Plan.md`
- `V3_LANGGRAPH_IMPLEMENTATION.md`
- `GATE_DECISION*.md`

---

## Phase 9: E2E Validation (Day 7+)
*Real LLM validation before production*

### 9.1 Small-Scale Real LLM Test (10-20 reports)
**Command:** `python -m experiments.ragset_report_inference_experiment.runners.run_inference --limit 20`
**Validate:** No crashes, trace integrity, reasonable PASS/NEEDS_REVIEW rates

### 9.2 Full 50-Case Diagnostic Re-run
**Command:** `python -m experiments.ragset_report_inference_experiment.runners.run_diagnostic_50_fixed.py`
**Compare:** Results vs previous diagnostic (validator instability, label error concentration)

### 9.3 Production Readiness Gate
**Criteria:**
- [ ] Zero runtime errors on 50-case diagnostic
- [ ] Policy audit: 0 errors, 0 contradictions
- [ ] Trace deduplication verified
- [ ] All 114+ new tests pass
- [ ] Judge contract aligned across prompt/schema/tests

---

## Dependency Graph

```
Phase 1 (Runtime) ──────┐
                        ├──→ Phase 2 (History) ──────┐
Phase 3 (Contract) ─────┘                            ├──→ Phase 4 (Policy) ──→ Phase 5 (Deps) ──→ Phase 6 (Clean) ──→ Phase 7 (Tests) ──→ Phase 8 (Docs) ──→ Phase 9 (E2E)
                        └──→ Phase 3 (Contract) ─────┘
```

**Critical Path:** Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 9

**Parallelizable:** Phase 5, 6, 7, 8 can overlap with later phases

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Judge enum fix breaks existing tests | Update tests in same PR (Phase 3) |
| History fix changes finalization behavior | Phase 7 tests lock expected behavior |
| Policy reconciliation changes clinical labels | Phase 4 must complete before Phase 9 |
| Real LLM behavior differs from mocks | Phase 9 is mandatory gate |

---

## Success Criteria for Commit

After Phase 6:
- [ ] `git status` shows only source/code/test/doc changes
- [ ] No result files staged
- [ ] All 114+ tests pass
- [ ] Policy audit: 0 errors
- [ ] Requirements complete

After Phase 9:
- [ ] 50-case diagnostic runs clean
- [ ] Label error concentration improved vs baseline
- [ ] Ready for 4,349 production reports