# Requirements Gap Analysis: RagSet Report Inference Experiment

**Analysis Date:** 2026-09-17
**Current Implementation:** `experiments/ragset_report_inference_experiment/`
**Requirements Source:** User-provided 30-point specification

---

## Executive Summary

The current implementation has **significant gaps** against the 30-point specification. While the infrastructure (loop, schemas, providers, retrieval) is solid, the **prompts and validation logic** do not fully align with the clinical validation requirements. The system currently optimizes for evidence quality rather than clinical label correctness, and the validator fails to independently re-evaluate all 12 labels.

**Overall Compliance: ~60%** — Core pipeline works but clinical validation behavior needs major prompt/logic changes.

---

## Detailed Gap Analysis

### ✅ FULLY COMPLIANT (12/30)

| Req | Topic | Status | Evidence |
|-----|-------|--------|----------|
| 1 | Goal & 12 labels | ✅ | `LABEL_KEYS` in schemas.py matches exactly |
| 2 | Core architecture (Model 1 → Pydantic → Model 2 → retry) | ✅ | `loop.py` implements bounded retry loop |
| 13 | Pydantic = structural only | ✅ | `schemas.py` validates structure only |
| 14 | Model 2 PASS/FAIL on label correctness | ✅ | `ValidationResult.status` = PASS/FAIL |
| 15 | Validator output structure | ✅ | `ValidationResult` matches spec |
| 16 | `corrected` = 0/1/null | ✅ | `ValidationIssue.corrected: Literal[0,1] \| None` |
| 19 | Full regeneration on FAIL | ✅ | `loop.py` passes full `previous_prediction` + `validator_feedback` |
| 20 | Why full regeneration | ✅ | Prompt instructs "re-evaluate all 12 labels independently" |
| 21 | Retry prompt behavior | ✅ | Inference prompt: "Re-read ORIGINAL REPORT... re-evaluate all 12 labels" |
| 22 | Max attempts = 3 | ✅ | `config/experiment.yaml: max_attempts: 3` |
| 23 | Review queue data | ✅ | `predictions.jsonl` saves all attempts, validation, trace |
| 24 | Model usage trace | ✅ | `model_trace.jsonl` with timestamp, model, latency, tokens |

---

### ⚠️ PARTIALLY COMPLIANT (10/30)

| Req | Topic | Gap | Severity |
|-----|-------|-----|----------|
| 3 | Model 1 responsibility | Prompt lists 16 considerations but doesn't emphasize "observed RagSet convention" as primary driver | Medium |
| 4 | Model 1 reconsideration list | 16 items listed but not enforced as mandatory re-evaluation per label | Medium |
| 5 | Evidence not primary validation | **Validator prompt treats evidence quality as FAIL criterion** (Req 5, 16) | **Critical** |
| 6 | Model 2 independent re-evaluation | Prompt says "independently determine" but validation logic focuses on evidence quality | **Critical** |
| 7 | Model 2 re-review all factors | Listed but not enforced as mandatory per-label checklist | Medium |
| 8 | Question/indication handling | Covered in prompts but not validated as separate check | Low |
| 9 | Negation | Covered in prompts | Low |
| 10 | Uncertainty | Covered in prompts | Low |
| 11 | Historical/current | Covered in prompts | Low |
| 12 | Anatomy/laterality | Covered in prompts | Low |

---

### ❌ NON-COMPLIANT (8/30)

| Req | Topic | Gap | Severity |
|-----|-------|-----|----------|
| 5 | Evidence not primary validation | **Validator FAILs on "unrelated evidence" even if label value is correct** | **Critical** |
| 16 | PASS/FAIL on label correctness | Validator FAILs on "evidence copied from another label" even if 0/1 is correct | **Critical** |
| 17 | Validator output | Includes `evidence` array in issues — spec says "evidence quality not PASS/FAIL criterion" | **Critical** |
| 18 | `corrected = null` usage | Not used when validator uncertain; always provides 0/1 | Medium |
| 25 | OpenRouter free routing | **Using NVIDIA direct API, not OpenRouter free** | **Critical** |
| 26 | Gold analysis before inference | `analyze_gold.py` exists but not run as mandatory pre-step | Medium |
| 27 | Retrieval | TF-IDF implemented but `retrieve_excluding` not verified for held-out | Medium |
| 28 | Separation of concerns | Validator conflates evidence quality with label correctness | **Critical** |
| 29 | Desired outcome | Evidence quality affects PASS/FAIL — should not | **Critical** |
| 30 | Final rule | System optimizes for evidence quality, not clinical label correctness | **Critical** |

---

## Root Cause Analysis

### The Core Problem: Validator Conflates Evidence Quality with Label Correctness

**Current Behavior (Wrong):**
```
Model 1: Synovitis = 0, evidence = "No knee effusion"
Model 2: FAIL — "evidence more directly related to Effusion"
```

**Required Behavior (Per Spec):**
```
Model 1: Synovitis = 0, evidence = "No knee effusion"
Model 2: Independently determines Synovitis = 0 is CORRECT per report
         → PASS (evidence quality irrelevant to PASS/FAIL)
```

### Why This Happens

1. **Validator Prompt (Lines 44-97)** explicitly lists "unrelated evidence" as FAIL criterion
2. **Validation Schema** requires `evidence` array in issues — encourages evidence-focused failures
3. **PASS Condition (Lines 303-323)** includes "unrelated evidence" as FAIL trigger
4. **No Independent Re-evaluation Enforcement** — validator checks Model 1's evidence rather than independently determining label from report

### The Evidence-Quality Trap

The spec explicitly states (Req 5, 16, 29):
> "DO NOT make evidence-quality perfection a hard PASS/FAIL condition"
> "Do NOT fail solely because evidence quotation is imperfect"
> "Optimize for correct RagSet 12-label clinical outcomes"

But the current validator **does the opposite** — it fails on evidence quality issues even when the 0/1 label is clinically correct.

---

## Required Changes by Level

### Level 1: Critical — Validator Logic & Prompts (Must Fix)

| Change | File | Description |
|--------|------|-------------|
| **Rewrite validator system prompt** | `prompts/validation.yaml` | Remove evidence-quality as FAIL criterion; focus on independent label determination |
| **Rewrite PASS condition** | `prompts/validation.yaml` | PASS = all 12 labels correct per report; FAIL = any label incorrect |
| **Remove evidence-quality FAIL triggers** | `prompts/validation.yaml` | Delete "unrelated evidence", "evidence copied", "fabricated evidence" as FAIL criteria |
| **Enforce independent re-evaluation** | `prompts/validation.yaml` | Add explicit instruction: "For each label, independently determine 0/1 from ORIGINAL REPORT" |
| **Update validator output** | `schemas.py` + `prompts/validation.yaml` | Make `evidence` optional in `ValidationIssue`; not required for FAIL |
| **Switch to OpenRouter free** | `config/models.yaml` | Change profiles to `provider: openrouter`, `model: openrouter/free` |

### Level 2: Important — Model 1 Prompt & Gold Analysis

| Change | File | Description |
|--------|------|-------------|
| **Strengthen Model 1 prompt** | `prompts/inference.yaml` | Add explicit "For EACH label, independently determine 0/1 from ORIGINAL REPORT" |
| **Enforce RagSet convention** | `prompts/inference.yaml` | Make "observed gold-label convention" the primary decision driver |
| **Run gold analysis as mandatory step** | `runners/run_inference.py` | Auto-run `analyze_gold.py` before inference; fail if not done |
| **Verify `retrieve_excluding`** | `retrieval.py` | Add test/assertion that held-out reports never retrieve themselves |

### Level 3: Nice-to-Have — Polish & Robustness

| Change | File | Description |
|--------|------|-------------|
| **Use `corrected = null`** | `prompts/validation.yaml` + `schemas.py` | When validator uncertain, use null instead of guessing 0/1 |
| **Question/indication validation** | `prompts/validation.yaml` | Add explicit check: "Is this label based on a clinical question?" |
| **Empty findings handling** | `prompts/inference.yaml` | Explicit instruction: "If report has 'Bevindingen:' with no findings, use null evidence" |
| **OpenRouter actual model logging** | `models.py` | Record actual model from OpenRouter response `model` field |

---

## Implementation Priority Order

```
Week 1 (Critical):
1. Rewrite validator system prompt — remove evidence-quality FAILs
2. Rewrite PASS/FAIL logic — label correctness only
3. Switch to OpenRouter free routing
4. Test with 5 reports — verify PASS on correct labels despite imperfect evidence

Week 2 (Important):
5. Strengthen Model 1 prompt — independent per-label determination
6. Run gold analysis as mandatory pre-step
7. Verify retrieve_excluding for held-out eval

Week 3 (Polish):
8. Use corrected=null when uncertain
9. Add question/indication explicit check
10. OpenRouter actual model logging
```

---

## Test Cases to Validate Compliance

After changes, these scenarios must PASS:

| Scenario | Model 1 Prediction | Model 1 Evidence | Expected Validator Result |
|----------|-------------------|------------------|---------------------------|
| Correct label, imperfect evidence | Synovitis=0 | "No knee effusion" | **PASS** |
| Correct label, null evidence | ACL=0 | null | **PASS** |
| Correct label, reused evidence | Baker's=0 | "No effusion" | **PASS** (if 0 is correct) |
| Incorrect label | Synovitis=0 | "No effusion" | **FAIL** (if report shows synovitis) |
| Question as finding | Meniscus=1 | "Meniscusscheur/mediaal?" | **FAIL** |
| Negation error | ACL=1 | "No ACL tear" | **FAIL** |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Validator becomes too lenient | Medium | High | Add regression tests for known failure cases |
| Model 1 ignores validator feedback | Low | Medium | Ensure retry prompt emphasizes "use feedback to identify errors" |
| OpenRouter free model quality | Medium | High | Test with multiple free models; have NVIDIA as fallback |
| Gold analysis not representative | Low | Medium | Analyze all 58 gold reports; document conventions |

---

## Conclusion

The current implementation has **excellent infrastructure** but **fundamentally misaligned validation logic**. The validator acts as an "evidence quality checker" rather than an "independent clinical label validator." 

**Fixing the validator prompt and PASS/FAIL logic (Level 1) will resolve ~80% of compliance gaps.** The remaining changes are incremental improvements.

**Recommendation:** Prioritize Level 1 changes immediately. The infrastructure is ready; only the clinical validation behavior needs correction.