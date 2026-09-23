# Diagnostic 50 Validation Pipeline - Fix Plan

## Executive Summary
The diagnostic_50 validation run produced only 41/50 results with critical metadata fields empty/incorrect. Root causes identified in `run_diagnostic_50.py` runner code.

---

## Issues & Fixes

### 1. Missing 9 Records (41/50 processed)
**Root Cause**: Script likely interrupted externally (timeout/OOM) after record 41. No error records for missing 9.
**Location**: `run_diagnostic_50.py` lines 138-368
**Fix**: 
- Add checkpoint/resume capability (track processed StudyInstanceUIDs)
- Add per-record timeout with retry
- Add progress logging to file (not just stdout)
- Wrap entire loop in try/except with graceful degradation

### 2. Identical Prompt Hashes (all 41 records)
**Root Cause**: Lines 335-337 hash only static system prompt, not full rendered prompt with report content.
```python
"prompt_hash_inference": compute_hash(inf_cfg["system_prompt"]),  # WRONG
```
**Fix**: Compute hash on actual full prompt per attempt. Options:
- Extract from trace records (trace has `prompt_hash` field)
- Reconstruct full prompt in runner using same template logic as graph_nodes.py

### 3. Empty `attempts` Arrays
**Root Cause**: Lines 263-286 build `attempts_detail` from `state.get("attempts", [])` which is always empty. The manual graph execution (lines 180-215) doesn't populate `state["attempts"]` - only legacy `run_report()` in loop.py does.
**Fix**: Aggregate from `trace_records` by grouping on `attempt` field:
- For each attempt number, find inference/validation/judgment traces
- Reconstruct prediction/validation from trace `response_hash` or re-parse
- Populate `elapsed_seconds` from trace `latency_seconds`

### 4. `retry_count: -1` (all records)
**Root Cause**: Line 319: `len(state.get("attempts", [])) - 1` = `0 - 1 = -1`
**Fix**: Derive from trace records: count unique `attempt` values on inference-stage traces.

### 5. Zero `latency_seconds` & Empty `token_usage`
**Root Cause**: Line 333 sums from empty `attempts`; line 334 hardcoded `{}`
**Fix**: Sum `latency_seconds` from all trace records; extract token counts from API response (requires model classes to return token usage).

### 6. Empty `model1_prediction` / `model2_validation`
**Root Cause**: Lines 315-316 index into empty `state["attempts"]`
**Fix**: Extract from last attempt's trace records or final_prediction in state.

### 7. Empty `policy_conventions_used` / `unresolved_policy_flags`
**Root Cause**: Lines 321-322 hardcoded `[]`
**Fix**: Parse from critique issues (issue_type = UNRESOLVED_POLICY, REPORT_AMBIGUITY) and judge rationale.

---

## Implementation Priority

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Fix prompt hashes, attempts, retry_count, latency | Medium |
| P0 | Add checkpoint/resume for missing records | Medium |
| P1 | Populate token_usage from API responses | Low |
| P1 | Extract policy_conventions_used from critiques | Low |
| P2 | Add per-record timeout & better error handling | Medium |

---

## Testing Strategy
1. Run fixed runner on all 50 records
2. Verify: 50 results, non-empty attempts, correct retry_count (0-2), unique prompt hashes
3. Cross-validate: trace records match result metadata
4. Run existing test suite: `pytest tests/test_langgraph.py tests/test_loop.py -v`

---

## Files to Modify
- `experiments/ragset_report_inference_experiment/runners/run_diagnostic_50.py` (primary)
- `experiments/ragset_report_inference_experiment/src/ragset_inference/models.py` (add token usage return)
- `experiments/ragset_report_inference_experiment/src/ragset_inference/trace.py` (if schema changes needed)

---

## Risk Mitigation
- Backup current results before re-run: `cp diagnostic_50_results.jsonl diagnostic_50_results.jsonl.bak`
- Run in batches of 10 with progress checkpoints
- Monitor API quota/rate limits