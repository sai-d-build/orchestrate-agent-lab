# RagSet Report Inference Pipeline - Diagnostic Report

**Date:** 2026-09-18
**Pipeline Version:** Current (post-fixes)
**Total Studies Analyzed:** 45
**Pass Rate:** 88.9% (40/45 passed, 5 needs_review)

---

## PART 1: CURRENT PIPELINE ARCHITECTURE

### 1.1 Data Loading (`data.py`)
- **Source:** `train.csv` with 12 label columns + `StudyInstanceUID` + `Report`
- **Column Mapping:** CSV columns with spaces/apostrophes → valid Python identifiers
- **Split:** `split_gold()` separates rows where ALL 12 labels are populated (gold) vs unlabeled
- **Gold Reports:** 58 reports with complete labels (used for GOLD_ANALYSIS and retrieval)

### 1.2 Gold Analysis (`gold.py`)
- **Basic Stats Fallback:** Label distribution, report length stats
- **LLM Analysis (Optional):** `analyze_gold_llm.py` generates rich patterns from gold reports
- **Output:** `gold_analysis.json` (not yet generated - file missing)
- **Format:** `format_gold_examples()` creates STUDY/LABELS/REPORT blocks for retrieval

### 1.3 Retrieval (`retrieval.py`)
- **Method:** TF-IDF with character n-grams (2-4), multilingual support
- **Top-K:** 8 (configurable)
- **Leakage Prevention:** `retrieve_excluding()` for held-out evaluation
- **Output:** Top-K gold studies with labels and full reports

### 1.4 Model 1 Inference (`models.py` - InferenceModel)
- **Provider:** NVIDIA Nemotron 3 Ultra (direct API) or OpenRouter
- **Mode:** JSON mode with `response_format={"type": "json_object"}`
- **Temperature:** 0 (deterministic)
- **Max Tokens:** 4000 (configurable)
- **Retry Logic:** 3 retries with exponential backoff for transient failures
- **PF_OA Flattening:** Post-processing to handle nested PF_OA hallucination

### 1.5 Model 2 Validation (`models.py` - ValidatorModel)
- **Provider:** OpenRouter free tier (different from Model 1)
- **Same API structure** as InferenceModel
- **Retry Logic:** 3 retries with exponential backoff

### 1.6 Loop Logic (`loop.py`)
- **Max Attempts:** 3 (configurable)
- **Failure Classification:** STRUCTURAL, SEMANTIC, TRANSIENT
- **Programmatic Safety Gate:** `_filter_real_issues()` filters issues where `corrected=None` or `predicted==corrected`
- **Feedback Mechanism:** Validator feedback passed to Model 1 on semantic disagreements
- **Statuses:** `passed`, `needs_review`

### 1.7 Model 2 Validation Output
- **Status:** PASS/FAIL
- **Issues:** List of `{label, predicted, corrected, reason, evidence[]}`
- **Programmatic Safety Gate:** `_filter_real_issues()` applied before PASS/FAIL determination

### 1.7 Output Generation
- **predictions.jsonl:** Incremental append with full provenance
- **model_trace.jsonl:** Per-stage tracing (inference/validation)

---

## PART 2: EXPERIMENTAL PRINCIPLES (Preserved)

- 12 fixed labels, multilingual reports
- Direct inference from ORIGINAL_REPORT (no translation)
- English only for label names
- Evidence verbatim in original language
- Retrieved examples = reference only
- GOLD_ANALYSIS = dataset convention authority

---

## PART 3: STUDY TRACES & FAILURE MATRIX

### Overall Statistics (45 studies)
- **Passed:** 40 (88.9%)
- **Needs Review:** 5 (11.1%)
- **Total Issues Across All Attempts:** 45 issues

### Failure Distribution by Label
| Label | Issues | Primary Pattern |
|-------|--------|-----------------|
| PF_OA | 9 | pred=1 → corr=0 (over-generalization) |
| Medial_OA | 8 | pred=1 → corr=0 (over-generalization) |
| Lateral_OA | 8 | pred=1 → corr=0 (over-generalization) |
| Contusion | 10 | pred=0→corr=1 (7), pred=1→corr=0 (3) |
| Lateral_Meniscus | 4 | pred=1 → corr=0 (uncertain/suspected tears) |
| Effusion | 3 | pred=0 → corr=1 |
| Bakers | 2 | pred=0 → corr=1 |
| Synovitis | 1 | pred=0 → corr=1 |

### Needs Review Studies (5/45)
| Study | Primary Issues | Attempts | Validator Stability |
|-------|---------------|----------|---------------------|
| 100257... | OA over-generalization (3 compartments) | 3 | STABLE |
| 100611... | Lat_Meniscus (uncertain), PF_OA, Contusion | 3 | STABLE |
| 101124... | Contusion (explicit bone infarct) | 3 | STABLE |
| 101208... | Contusion (unstable!), Bakers, Effusion | 3 | **UNSTABLE** (Contusion) |
| 101240... | OA over-generalization (chondromalacia), Effusion | 3 | STABLE |

---

## PART 4: KNOWN FAILURE PATTERNS

### OA Over-Generalization (25 issues: Medial_OA 8, Lateral_OA 8, PF_OA 9)
**Pattern:** Model 1 maps generalized OA statements ("osteoarthritis of knee", "wearing of articular cartilage") to ALL three compartments.
**Root Cause:** Model 1 ignores compartment specificity requirement; reuses generic evidence across compartments.
**Validator:** Correctly identifies as over-generalization (corr=0).
**Model 1 Behavior:** STUCK - repeats same over-generalized prediction across all 3 attempts.

### Meniscus Uncertainty (4 issues: Lateral_Meniscus)
**Pattern:** Model 1 treats "suspected/R-O/possible tear" as confirmed tear (pred=1).
**Validator:** Correctly maps uncertain language to 0 per GOLD_ANALYSIS.
**Model 1 Behavior:** STUCK - repeats same error across attempts.

### Contusion Ambiguity (10 issues: 7 pred=0→corr=1, 3 pred=1→corr=0)
**Pattern A (pred=0→corr=1):** Explicit "bone contusion/bone bruise/infarct" missed by Model 1.
**Pattern B (pred=1→corr=0):** Model 1 equates bone marrow edema with contusion; Validator corrects per GOLD_ANALYSIS.
**Validator Stability:** ONE UNSTABLE case (Study 101208 - Contusion flipped 1→0→1).

### Effusion/Synovitis Confusion (3 Effusion, 1 Synovitis)
**Pattern:** Model 1 misses explicit effusion; Validator catches it.
**Synovitis:** Validator infers from Baker's cyst fluid (debatable per rules).

### Baker's Cyst (2 issues)
**Pattern:** Model 1 misses explicit Baker's cyst; or misclassifies other cysts as Baker's.

---

## PART 5: EVIDENCE AUDIT

### Evidence Quality
- **Null Evidence:** Properly handled as `null` (not empty string) ✓
- **Verbatim:** Evidence appears verbatim from original report ✓
- **Language Preservation:** Evidence in original language (Spanish, Bulgarian, Turkish, etc.) ✓
- **Relevance:** Some cross-label evidence reuse observed (e.g., same OA evidence for 3 compartments)

### Evidence Format Compliance
- ✅ `evidence: null` for missing (not empty string)
- ✅ Verbatim quotes in original language
- ⚠️ Some cross-label evidence reuse (OA compartments sharing same generic evidence)

---

## PART 6: MODEL 1 AUDIT

### Strengths
- Good at explicit findings (explicit tears, explicit fractures, explicit Baker's cysts)
- Proper null evidence handling
- Multilingual capability (Spanish, Bulgarian, Turkish, Croatian, French, German)

### Weaknesses
1. **OA Over-Generalization:** Maps generalized OA to all 3 compartments (25 issues)
2. **Uncertainty Handling:** Treats R/O/suspected/possible as confirmed (4 Lateral_Meniscus issues)
3. **Contusion Confusion:** Both false negatives (misses explicit contusion) and false positives (marrow edema → contusion)
3. **Retry Behavior - STUCK:** Repeats identical predictions across all 3 attempts (no fresh analysis)
4. **Evidence Reuse:** Reuses same generic OA evidence for all 3 compartments

### Root Causes
- Missing explicit "OA compartment specificity" rule enforcement in retry
- Missing "uncertainty classification" step before binary assignment
- Missing "contusion vs marrow edema" distinction in prompt
- Retry mechanism doesn't force fresh analysis (Model 1 sees same prompt + feedback but doesn't re-analyze)

---

## PART 7: MODEL 2 AUDIT

### Strengths
- Correctly identifies OA over-generalization (25/25)
- Correctly handles uncertainty (R/O tears → 0)
- Correctly identifies explicit contusion/bone infarct (7/7)
- Programmatic safety gate filters bogus issues (corrected=None, predicted==corrected)

### Weaknesses
1. **Validator Instability:** ONE case of Contusion flip (1→0→1) in Study 101208
2. **Synovitis Inference:** Infers synovitis from Baker's cyst fluid (Study 101208) - questionable per rules
3. **Contusion Flip:** Model 2 changed Contusion correction 1→0→1 in Study 101208

### Root Causes
- Model 2 sometimes uses effusion/Baker's fluid as synovitis evidence (violates "Effusion ≠ Synovitis")
- Contusion instability suggests GOLD_ANALYSIS not definitive enough for marrow edema vs contusion

---

## PART 8: VALIDATOR STABILITY AUDIT

| Study | Label | Attempt 1 | Attempt 2 | Attempt 3 | Stability |
|-------|-------|-----------|-----------|-----------|-----------|
| 100257 | OA (3) | 0 | 0 | 0 | ✅ STABLE |
| 100611 | Lat_Men, PF_OA, Contusion | 0,0,1 | 0,0,1 | 0,0,1 | ✅ STABLE |
| 101124 | Contusion | 1 | 1 | 1 | ✅ STABLE |
| 101208 | Contusion | 0 | 1 | 1 | ❌ **UNSTABLE** (flip 1→0→1) |
| 101240 | OA (3), Effusion | 0,0,0,1 | 0,0,1 | 0,0,0,1 | ✅ STABLE |

**Validator Instability Rate:** 1/5 studies (20%) with at least one unstable label.

---

## PART 9-14: RETRY/ADJUDICATION DESIGN REVIEW

### Current Retry Logic
- **Max Attempts:** 3
- **Feedback:** Validator corrections passed as `validator_feedback` to Model 1
- **Feedback Usage:** Passed as `validator_feedback` parameter to Model 1 (not in prompt)
- **Model 1 Retry Behavior:** STUCK - repeats same prediction despite feedback

### Missing Adjudication Layer
- No third-model adjudicator for unresolved disagreements
- No label-level adjudication (full report re-processed)
- No confidence scoring
- No GOLD_ANALYSIS deterministic resolution step

### Retry Effectiveness
- **Structural Failures:** Handled by model-level retries (3 attempts)
- **Semantic Disagreements:** Model 1 STUCK - doesn't change despite feedback
- **Result:** 5/45 studies reach NEEDS_REVIEW after 3 identical attempts

---

## PART 15-24: SAFETY GATES & PROVENANCE

### Programmatic Safety Gate ✅
- `_filter_real_issues()` correctly filters:
  - `corrected=None` issues
  - `predicted == corrected` issues
- Status recomputed from filtered issues

### Contradiction Check ❌ MISSING
- No check for validator corrections contradicting explicit ORIGINAL_REPORT evidence
- Example: Validator could negate explicit "bone contusion" using weaker "marrow edema" inference

### Compartment Safety ❌ WEAK
- No programmatic check for OA compartment specificity
- Model 1 reuses same evidence for all 3 OA compartments
- No validation that OA evidence mentions specific compartment

### Uncertainty Safety ⚠️ PARTIAL
- Prompt mentions uncertainty classification but not enforced
- Model 1 treats R/O/suspected as confirmed
- Validator correctly applies GOLD_ANALYSIS but Model 1 doesn't learn

### Provenance ⚠️ PARTIAL
- predictions.jsonl has: original_report, inferred_evidence, attempts with predictions/validations
- Missing: label-level resolution provenance (MODEL_AGREEMENT, ACCEPTED_BY_GOLD, etc.)
- Missing: adjudicator trail (not implemented)

---

## PART 25: DIAGNOSTIC SUMMARY

### Critical Bugs (Must Fix)
1. **Model 1 STUCK on retries** - Doesn't change predictions despite validator feedback
2. **OA Over-Generalization** - 25 issues, Model 1 STUCK across all retries
3. **Validator Instability** - 1/5 needs_review studies has unstable corrections
3. **No GOLD_ANALYSIS** - File missing, fallback to basic stats only

### Missing Rules in Prompts
1. **OA Compartment Specificity** - Not enforced in retry
2. **Uncertainty Classification** - Not enforced as explicit step
3. **Contusion vs Marrow Edema** - Missing distinction
4. **Explicit Evidence Override** - Validator can negate explicit findings
5. **Explicit Evidence Override for Model 2** - Missing in prompt

### Missing Pipeline Components
1. **GOLD_ANALYSIS** - Not generated (file missing)
2. **Third-Model Adjudicator** - Not implemented
3. **Label-Level Adjudication** - Not implemented
3. **Contradiction Check** - Missing programmatic safety gate
3. **Compartment Safety Gate** - Missing programmatic check
3. **Provenance Tracking** - Missing label-level resolution tracking

---

## PART 26: PROPOSED CHANGES (Priority Order)

### 1. CRITICAL BUGS (Must Fix)
1. **Fix Model 1 STUCK behavior** - Ensure Model 1 actually re-analyzes on retry
   - Pass validator feedback IN THE PROMPT (not just as parameter)
   - Add explicit "RE-ANALYZE" instruction in retry prompt
   
2. **Generate GOLD_ANALYSIS** - Run `analyze_gold_llm.py` to create gold_analysis.json
   - Required for dataset-convention mapping

3. **Fix Validator Instability** - Add consistency check in ValidatorModel
   - Track corrections across attempts, flag instability

### 2. PROGRAMMATIC SAFETY FIXES
1. **Add Contradiction Check** - Reject validator corrections that contradict explicit ORIGINAL_REPORT evidence
2. **Add Compartment Safety Gate** - Validate OA predictions have compartment-specific evidence
3. **Add Contradiction Check for Adjudicator** (when implemented)

### 3. MODEL 1 PROMPT FIXES
1. **Add OA Compartment Specificity Rule** with explicit retry checklist
2. **Add Uncertainty Classification Step** (explicit step before binary assignment)
3. **Add Contusion vs Marrow Edema Distinction** with explicit terminology
4. **Add Explicit Evidence Override Rule** - "Do not negate explicit findings with weaker inferences"
5. **Add Retry Checklist** - Explicit re-analysis steps for disputed labels

### 4. MODEL 2 PROMPT FIXES
1. **Add Explicit Evidence Override Rule** - "Do not negate explicit findings"
2. **Add OA Compartment Rule** - Explicit compartment requirement
3. **Add Uncertainty Rule** - Explicit UNCERTAIN classification step
4. **Add Explicit Evidence Override** - "Do not negate explicit findings with weaker inferences"

### 5. RETRY-LOOP FIXES
1. **Distinguish Structural vs Semantic Retries** - Already done but Model 1 doesn't use feedback
2. **Add Model 1 Stuck Detection** - If Model 1 repeats same prediction, escalate
3. **Add Validator Instability Detection** - Track correction changes across attempts

### 6. ADJUDICATOR DESIGN (Future)
1. **Third-Model Adjudicator** - Different model, label-level only
2. **Label-Level Adjudication** - Only disputed labels
3. **Confidence Scoring** - HIGH/MEDIUM/LOW + RESOLVED/UNRESOLVED
4. **Provenance Tracking** - Label-level resolution states

### 7. ROUTING LOGIC
1. **Decision Tree:** Model 1 → Structure Validation → Model 2 → Programmatic Filter → GOLD_ANALYSIS Resolution → Adjudicator → NEEDS_REVIEW
2. **Label-Level Resolution** - Only disputed labels to adjudicator
3. **NEEDS_REVIEW Criteria** - Specific conditions (not just "max attempts")

### 8. PROVENANCE & LOGGING
1. **Label-Level Resolution States** - MODEL_AGREEMENT, ACCEPTED_BY_GOLD, ACCEPTED_BY_ADJUDICATOR, NEEDS_REVIEW
2. **Enhanced Trace Format** - Include resolution source per label
3. **Validator Stability Metrics** - Track correction changes per label per study

---

## PART 27: RISK ASSESSMENT

| Change | Risk | Mitigation |
|--------|------|------------|
| Model 1 retry fix | May introduce new errors if re-analysis worse | Keep original as fallback; compare |
| GOLD_ANALYSIS generation | LLM analysis quality variable | Fallback to basic stats; human review |
| Validator instability fix | May mask genuine ambiguity | Flag instability, don't auto-resolve |
| Adjudicator model | Cost, latency, new failure modes | Label-level only; different model; confidence scoring |

---

## PART 28: NEXT STEPS

1. **Immediate:** Fix Model 1 STUCK bug (pass feedback in prompt)
2. **Immediate:** Generate GOLD_ANALYSIS (run analyze_gold_llm.py)
3. **Immediate:** Add contradiction check and compartment safety gate
4. **Short-term:** Update Model 1/2 prompts with missing rules
5. **Short-term:** Add validator instability detection
6. **Medium-term:** Design and implement adjudicator model
7. **Medium-term:** Implement label-level adjudication and provenance tracking
8. **Ongoing:** Monitor validator stability metrics, adjust GOLD_ANALYSIS as needed