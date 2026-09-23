# Gate Decision V2: Full 4,349-Report Inference Run (Post-Stabilization)

## Decision: **NOT_READY_FOR_FULL_INFERENCE**

## Summary of Stabilization Pass

After implementing targeted fixes for validator oscillation, MODEL1_STUCK, and UNRESOLVED policy handling, the second 50-report diagnostic shows measurable improvement but persistent blocking issues remain.

### Key Improvements (V1 → V2)

| Metric | V1 (Initial) | V2 (Post-Fix) | Change |
|--------|--------------|---------------|--------|
| PASSED | 38 | 37 | -1 |
| NEEDS_REVIEW | 12 | 10 | -2 |
| ERROR | 0 | 3 | +3 (API) |
| VALIDATOR_INSTABILITY | 8 | 8 | = |
| MODEL1_STUCK | 4 | 1 | **-75%** |
| TRUE_AMBIGUITY | 0 | 1 | **+1 (correctly detected)** |
| CLEAN | 36 | 35 | -1 |
| VALIDATOR_ERROR | 8 | 8 | = |
| MODEL_ERROR | 6 | 5 | -1 |
| Policy UNRESOLVED | 7 | 5 | -2 |

### What Improved

1. **MODEL1_STUCK reduced 75%** (4 → 1): Complete 12-label vector tracking correctly detects stuck behavior
2. **TRUE_AMBIGUITY now detected** (0 → 1): UNRESOLVED policy conventions now route to AMBIGUOUS → needs_review
3. **Policy UNRESOLVED cases reduced** (7 → 5): Better separation of ambiguity from errors
4. **Validator instability properly routed**: All 8 cases now correctly detected as VALIDATOR_INSTABILITY and routed to needs_review (no infinite retries)

### Persistent Blocking Issues

#### 1. Validator Instability (8 cases) — **VALIDATOR_ERROR**
The validator still flips corrections on the same label across attempts for reports involving:
- **Generalized OA without compartment specification** (Cases 1, 2, 3, 6, 8): Medial_OA, Lateral_OA, PF_OA flip
- **Suspected/possible/R-O language** (Cases 1, 2, 5, 8): ACL, Lateral_Meniscus flip
- **Bone marrow edema vs contusion/fracture** (Case 4): Contusion/Fracture flip
- **Chondromalacia without compartment specification** (Case 6): Medial_OA, Lateral_OA flip

**Root Cause**: The validator's independent derivation changes based on Model 1's prediction, despite the prompt instructing independent derivation first. The validator is not truly independent — it anchors on Model 1's prediction.

#### 2. API Errors (3 cases) — **SYSTEM_ERROR**
- 1.2.826.0.1.3680043.8.498.12406043058701411628555010402350032739: Validation 404
- 1.2.826.0.1.3680043.8.498.41036022211588451079713314564568887994: Inference 429 (rate limit)
- 1.2.826.0.1.3680043.8.498.98824066826946884510314135448574304233: Validation 404

#### 3. Remaining MODEL1_STUCK (1 case) — **MODEL_ERROR**
- 1.2.826.0.1.3680043.8.498.62782163204193388751869649423542931657 (suspected_possible_ro_meniscal_tear): Model 1 repeats Medial_OA=0, Lateral_OA=0 despite validator feedback

#### 4. TRUE_AMBIGUITY (1 case) — **Correctly Detected**
- 1.2.826.0.1.3680043.8.498.44879085997210091159117906674172231025 (acl_mcl_injury_vs_degeneration_sprain): "Suspected discrete longitudinal rupture of posterior horn of lateral meniscus" — UNRESOLVED policy correctly routes to AMBIGUOUS

## Criteria Check (V2)

| Criterion | Status | Details |
|-----------|--------|---------|
| All existing tests pass | ✅ PASS | 64/64 tests pass |
| No unexplained validator instability | ❌ FAIL | 8 cases of validator flipping corrections |
| No unexplained MODEL1_STUCK | ❌ FAIL | 1 case of Model 1 repeating predictions |
| Policy contradictions are zero | ✅ PASS | Policy validation gate passes |
| Trace fields populated | ✅ PASS | Complete traces for all 50 reports |
| Diagnostic outputs written | ✅ PASS | JSON + CSV audit generated |

## Root Cause Analysis: Why Validator Instability Persists

The validator prompt explicitly instructs:
> "STAGE 1 — INDEPENDENT DERIVATION: Before looking at MODEL_1_PREDICTION... independently derive the correct value for all 12 labels."

However, the validator's output shows it is **not** performing independent derivation. Instead, it appears to:
1. Read Model 1's prediction first
2. Derive a correction that contradicts Model 1
3. On retry, when Model 1 adopts the correction, the validator flips back

This is **anchoring bias** — the validator uses Model 1's prediction as an anchor rather than independently deriving from the report + policy.

### Evidence from Cases

**Case 1 (explicit_synovitis)**:
- Attempt 1: Model 1 says ACL=1, MCL=1 → Validator says ACL=0, MCL=0
- Attempt 2: Model 1 says ACL=0, MCL=0 → Validator says ACL=1, MCL=1

**Case 3 (suspected_possible_ro_meniscal_tear)**:
- Attempt 1: Model 1 says Medial_OA=0, Lateral_OA=0 → Validator says Medial_OA=1, Lateral_OA=1
- Attempt 2: Model 1 says Medial_OA=1, Lateral_OA=1 → Validator says Medial_OA=0, Lateral_OA=0

**Case 8 (meniscal_degeneration_vs_tear)**:
- Attempt 1: Model 1 says Medial_OA=1, Lateral_OA=1, PF_OA=1 → Validator says all 0
- Attempt 2: Model 1 says all 0 → Validator says all 1

## Required Fixes Before Full Run

### 1. Fix Validator Anchoring (Highest Priority)
The validator must truly perform independent derivation **before** seeing Model 1's prediction. Options:
- **Two-call architecture**: First call derives independent labels without seeing Model 1; second call compares
- **Prompt restructuring**: Put Model 1 prediction in a separate section that the validator is instructed to ignore during Stage 1
- **Separate system prompts**: Use different system prompts for derivation vs comparison

### 2. Fix Remaining MODEL1_STUCK (1 case)
Model 1 must incorporate validator feedback. The current feedback format may not be clear enough. Consider:
- More explicit retry prompt: "The validator identified X as incorrect. You MUST change your prediction for X."
- Show validator's independent derivation alongside correction

### 3. Address API Reliability (3 cases)
- Add better retry logic for 404/429 errors
- Consider fallback models

### 4. Policy Review for UNRESOLVED Conventions (5 cases)
The 5 remaining UNRESOLVED cases need human policy review:
- Generalized OA without compartment specification
- Suspected/possible/R-O meniscal tears
- Bone marrow edema vs contusion
- ACL mucoid degeneration/cyst
- Chondromalacia without compartment specification

## Criteria Check (V2)

| Criterion | Status | Details |
|-----------|--------|---------|
| All existing tests pass | ✅ PASS | 64/64 tests pass |
| No unexplained validator instability | ❌ FAIL | 8 cases of validator flipping corrections |
| No unexplained MODEL1_STUCK | ❌ FAIL | 1 case of Model 1 repeating predictions |
| Policy contradictions are zero | ✅ PASS | Policy validation gate passes |
| Trace fields populated | ✅ PASS | Complete traces for all 50 reports |
| Diagnostic outputs written | ✅ PASS | JSON + CSV audit generated |

## Recommendation

**Do NOT proceed with full 4,349-report run.**

### Required Next Steps

1. **Redesign validator to eliminate anchoring bias** (two-call architecture or prompt restructuring)
2. **Fix remaining MODEL1_STUCK case** (improve feedback clarity)
3. **Convene policy review for 5 UNRESOLVED conventions**
4. **Improve API error handling**
5. **Re-run 50-report diagnostic after fixes**
6. **Only proceed when gate criteria are met**

---

**Generated:** 2026-09-22T04:23:00Z
**Diagnostic Run V2:** 50 reports
**Audit Files:** 
- `results/validation/diagnostic_50_report.json` (V2)
- `results/validation/diagnostic_50_report.csv` (V2)
- `results/validation/diagnostic_50_trace.jsonl` (V2)