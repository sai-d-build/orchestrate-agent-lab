# Gate Decision: Full 4,349-Report Inference Run

## Decision: **NOT_READY_FOR_FULL_INFERENCE**

## Blocking Issues

### 1. Validator Instability (8 cases)
**Root Cause: VALIDATOR_ERROR**

The validator flips corrections across attempts on the same report/label:
- Attempt 1: Label X = 0 → 1
- Attempt 2: Label X = 1 → 0
- Attempt 3: Label X = 1 → 1

**Affected Reports:**
- 1.2.826.0.1.3680043.8.498.12938500386086171684983091332136233036 (explicit_synovitis)
- 1.2.826.0.1.3680043.8.498.13269841239774089690227054060975903636 (explicit_synovitis)
- 1.2.826.0.1.3680043.8.498.62782163204193388751869649423542931657 (suspected_possible_ro_meniscal_tear)
- 1.2.826.0.1.3680043.8.498.40223239401390462911823193489179244754 (chondromalacia_chondropathy)
- 1.2.826.0.1.3680043.8.498.10659781998801845781246073928620795082 (effusion_vs_synovitis)
- 1.2.826.0.1.3680043.8.498.18199465156572252531439880765733639584 (explicit_negative_findings)
- 1.2.826.0.1.3680043.8.498.87064372531225943854092598173945043861 (meniscal_degeneration_vs_tear)
- 1.2.826.0.1.3680043.8.498.97353980946516912216109392482235435606 (meniscal_degeneration_vs_tear)

**Impact:** Validator cannot provide consistent guidance, causing infinite flip-flop that triggers VALIDATOR_INSTABILITY detection and routes to needs_review.

### 2. Model 1 STUCK (4 cases)
**Root Cause: MODEL_ERROR**

Model 1 repeats the same prediction for a disputed label despite validator feedback:
- 1.2.826.0.1.3680043.8.498.88523737187772411911621088333701306485 (suspected_possible_ro_meniscal_tear) - Medial_OA, Lateral_OA
- 1.2.826.0.1.3680043.8.498.13245409855393847268063261203253424169 (chondromalacia_chondropathy) - Lateral_OA, PF_OA
- 1.2.826.0.1.3680043.8.498.44879085997210091159117906674172231025 (acl_mcl_injury_vs_degeneration_sprain) - Lateral_Meniscus
- 1.2.826.0.1.3680043.8.498.27339999385861467858422021400023533486 (meniscal_degeneration_vs_tear) - Contusion, Fracture

**Impact:** Model 1 fails to incorporate validator feedback, causing MODEL1_STUCK detection and routes to needs_review.

### 3. Policy UNRESOLVED Cases (7 cases)
**Root Cause: POLICY_ERROR / TRUE_AMBIGUITY**

Gold analysis has UNRESOLVED conventions that cause ambiguity:
- Suspected/possible/R-O meniscal tear → UNRESOLVED
- Generalized OA without compartment specification → CONTEXTUAL
- Bone marrow edema without explicit contusion → SAFE_NEGATIVE
- ACL mucoid degeneration/cyst → UNSAFE
- MCL grade 1 sprain with intact contour → UNSAFE
- Effusion without explicit synovitis → SAFE_NEGATIVE for Synovitis
- Chondromalacia without compartment specification → SAFE_NEGATIVE for compartment OA

**Impact:** These are genuine policy ambiguities that require human adjudication or more gold examples.

## Criteria Check

| Criterion | Status | Details |
|-----------|--------|---------|
| All existing tests pass | ✅ PASS | 64/64 tests pass |
| No unexplained validator instability | ❌ FAIL | 8 cases of validator flipping corrections |
| No unexplained MODEL1_STUCK | ❌ FAIL | 4 cases of Model 1 repeating predictions |
| Policy contradictions are zero | ✅ PASS | Policy validation gate passes |
| Trace fields populated | ✅ PASS | Complete traces for all 50 reports |
| Diagnostic outputs written | ✅ PASS | JSON + CSV audit generated |

## Recommendation

**Do NOT proceed with full 4,349-report run.**

### Required Fixes Before Full Run:

1. **Fix Validator Instability**: The validator must provide consistent corrections. Options:
   - Add validator self-consistency check (validator reviews its own previous corrections)
   - Use higher temperature/seed for validator to reduce randomness
   - Add explicit instruction: "Do not contradict your previous correction for the same label on the same report"

2. **Fix Model 1 STUCK**: Model 1 must incorporate validator feedback. Options:
   - Strengthen retry prompt to emphasize "Re-read the ORIGINAL_REPORT from scratch"
   - Add explicit instruction: "If validator says X, you must change your prediction for X"
   - Consider using a different model for retries

3. **Resolve Policy UNRESOLVED**: Need human review of the 7 UNRESOLVED conventions:
   - Add more gold examples for suspected/possible/R-O tears
   - Clarify generalized OA mapping rules
   - Clarify bone marrow edema → contusion mapping
   - Clarify ACL/MCL indirect evidence rules

## Next Steps

1. Address validator instability (highest priority - affects 8/50 reports)
2. Address Model 1 STUCK (affects 4/50 reports)
3. Convene policy review for UNRESOLVED conventions
4. Re-run diagnostic 50 after fixes
5. Only proceed to full run when gate criteria are met

---

**Generated:** 2026-09-22T02:37:00Z
**Diagnostic Run:** 50 reports
**Audit Files:** 
- `results/validation/diagnostic_50_report.json`
- `results/validation/diagnostic_50_report.csv`
- `results/validation/diagnostic_50_trace.jsonl`