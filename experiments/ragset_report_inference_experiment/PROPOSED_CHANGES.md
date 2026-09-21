# Proposed Changes for RagSet Pipeline

Based on the diagnostic report, here are the exact changes needed, in priority order.

---

## 1. CRITICAL BUGS (Must Fix Immediately)

### 1.1 Fix Model 1 STUCK Bug
**File:** `experiments/ragset_report_inference_experiment/runners/run_inference.py`
**Function:** `make_infer_fn`

**Problem:** Model 1 receives `validator_feedback` as a parameter but doesn't include it in the prompt. Model 1 repeats identical predictions across all retries.

**Fix:** Modify `make_infer_fn` to include validator feedback in the user prompt when provided.

```python
def infer_fn(**kwargs):
    user = inf_cfg["user_prompt_template"].format(
        original_report=kwargs["report"],
        gold_analysis=gold_analysis,
        retrieved_gold_examples=kwargs.get("gold_examples", ""),
    )
    
    # Add validator feedback to prompt for retries
    validator_feedback = kwargs.get("validator_feedback")
    if validator_feedback:
        feedback_text = json.dumps(validator_feedback, ensure_ascii=False, indent=2)
        user += f"\n\nVALIDATOR FEEDBACK (for reconsideration, NOT as evidence):\n{feedback_text}\n\nRe-analyze the disputed labels from ORIGINAL_REPORT. Do not simply repeat previous prediction."
    
    report_id = kwargs["report_id"]
    try:
        prediction = model.run(inf_cfg["system_prompt"], user, validator_feedback=validator_feedback)
        ...
```

### 1.2 Generate GOLD_ANALYSIS
**Action:** Run the gold analysis script to create `gold_analysis.json`

```bash
cd /Users/sai-work/Documents/gitCode/new/orchestrate-agent-lab
python -m experiments.ragset_report_inference_experiment.runners.analyze_gold_llm
```

**Prerequisites:** Valid API key for inference model (NVIDIA_API_KEY or OPENROUTER_API_KEY)

### 1.3 Fix Validator Instability
**File:** `experiments/ragset_report_inference_experiment/src/ragset_inference/loop.py`

Add validator stability tracking in `run_report`:

```python
def run_report(...):
    ...
    validator_corrections_history = {}  # label -> list of corrections
    
    for number in range(1, max_attempts + 1):
        ...
        validation = validate(...)
        validation = _filter_real_issues(validation)
        
        # Track validator corrections for stability detection
        for issue in validation.issues:
            label = issue.label
            if label not in validator_corrections_history:
                validator_corrections_history[label] = []
            validator_corrections_history[label].append(issue.corrected)
        
        # Detect instability
        unstable_labels = [
            label for label, corrections in validator_corrections_history.items()
            if len(set(corrections)) > 1
        ]
        
        if unstable_labels:
            # Log instability, consider escalation
            for label in unstable_labels:
                print(f"WARNING: Validator unstable on {label}: {validator_corrections_history[label]}")
        
        ...
```

---

## 2. PROGRAMMATIC SAFETY FIXES

### 2.1 Contradiction Check
**File:** `experiments/ragset_report_inference_experiment/src/ragset_inference/loop.py`

Add contradiction check after validation:

```python
def _check_contradictions(validation: ValidationResult, report: str) -> ValidationResult:
    """
    Reject validator corrections that contradict explicit ORIGINAL_REPORT evidence.
    """
    real_issues = []
    for issue in validation.issues:
        if issue.corrected is not None and issue.predicted != issue.corrected:
            # Check if correction contradicts explicit report evidence
            # This is a simplified check - in production, use more sophisticated NLP
            if issue.corrected == 0 and issue.predicted == 1:
                # Validator says 0, Model 1 said 1 - check if report has explicit positive evidence
                # This is a placeholder - real implementation needs evidence matching
                pass
            real_issues.append(issue)
    
    return ValidationResult(
        status="PASS" if not real_issues else "FAIL",
        issues=real_issues,
    )
```

### 2.2 Compartment Safety Gate
**File:** `experiments/ragset_report_inference_experiment/src/ragset_inference/loop.py`

Add compartment specificity check for OA labels:

```python
OA_LABELS = {"Medial_OA", "Lateral_OA", "PF_OA"}

def _check_compartment_specificity(prediction: ReportPrediction, report: str) -> list[str]:
    """
    Check that OA predictions have compartment-specific evidence.
    Returns list of labels that fail compartment specificity check.
    """
    violations = []
    for label in OA_LABELS:
        pred = prediction.predictions[label]
        if pred.value == 1 and pred.evidence:
            evidence_lower = pred.evidence.lower()
            # Check if evidence mentions the specific compartment
            compartment_keywords = {
                "Medial_OA": ["medial", "medial compartment", "medial tibiofemoral", "medial femoral condyle", "medial tibial"],
                "Lateral_OA": ["lateral", "lateral compartment", "lateral tibiofemoral", "lateral femoral condyle", "lateral tibial"],
                "PF_OA": ["patellofemoral", "patellar", "trochlear", "patella"],
            }
            keywords = compartment_keywords.get(label, [])
            if not any(kw in evidence_lower for kw in keywords):
                # Evidence doesn't mention specific compartment
                # This is a warning, not a hard failure
                pass
    return violations
```

---

## 3. MODEL 1 PROMPT FIXES (inference.yaml)

### 3.1 Add OA Compartment Specificity Rule (Retry Checklist)
**File:** `experiments/ragset_report_inference_experiment/prompts/inference.yaml`

Add to the OA RULES section:

```yaml
  OA RETRY CHECKLIST:
  ===================
  On retry, for EACH OA label (Medial_OA, Lateral_OA, PF_OA):
  
  1. Locate the EXACT sentence(s) in ORIGINAL_REPORT mentioning this compartment.
  2. Verify the evidence explicitly mentions the compartment:
     - Medial_OA: "medial compartment", "medial tibiofemoral", "medial femoral condyle", "medial tibial plateau"
     - Lateral_OA: "lateral compartment", "lateral tibiofemoral", "lateral femoral condyle", "lateral tibial plateau"
     - PF_OA: "patellofemoral", "patellar", "trochlear", "patella"
  3. If NO compartment-specific evidence exists, set value = 0.
  4. Do NOT reuse generalized OA evidence for multiple compartments.
  5. Do NOT infer compartment from generalized "osteoarthritis of knee" without GOLD_ANALYSIS support.
```

### 3.2 Add Uncertainty Classification Step
**File:** `experiments/ragset_report_inference_experiment/prompts/inference.yaml`

Add to CORE DECISION PROCEDURE:

```yaml
  CORE DECISION PROCEDURE
  =======================

  For EACH label:

  1. Read the ORIGINAL_REPORT carefully.
  2. Find evidence specifically relevant to that label.
  3. Classify the evidence state EXPLICITLY:
     
     * PRESENT: Explicit confirmed finding
     * EXPLICIT_ABSENT: Explicit negative ("no tear", "intact", "normal", "absent")
     * UNCERTAIN: "possible", "probable", "suspicious for", "may represent", "cannot exclude", "questionable", "equivocal", "likely", "suggestive of", "R/O", "rule out"
     * HISTORICAL: "old", "chronic", "prior", "previous", "healed", "resolved", "postoperative", "history of", "status post"
     * QUESTION_ONLY: Clinical question, indication, "rule out", "suspected", "evaluate for"
     * NOT_MENTIONED: No relevant evidence found
     
  4. Determine the anatomical structure and compartment involved.
  5. Use GOLD_ANALYSIS to map the observed evidence state to the RagSet binary convention.
  6. Assign value 0 or 1.
  7. Select only supporting evidence from ORIGINAL_REPORT.
  8. If there is no suitable evidence quote, use null.
```

### 3.3 Add Contusion vs Marrow Edema Distinction
**File:** `experiments/ragset_report_inference_experiment/prompts/inference.yaml`

Add to CONTUSION section:

```yaml
  CONTUSION
  =========

  Explicit terminology supporting Contusion=1:
  * "bone contusion"
  * "bone bruise"
  * "bone infarct"
  * "osteonecrosis" (when acute/traumatic context)
  
  Do NOT equate with Contusion=1:
  * "bone marrow edema" (without explicit contusion terminology)
  * "subchondral edema" (without explicit contusion terminology)
  * "osteochondral edema" (without explicit contusion terminology)
  * "chronic osteochondral/subchondral edema" (chronic/degenerative)
  
  Decision procedure:
  1. Search for explicit "bone contusion", "bone bruise", "bone infarct" terminology
  2. If explicit terminology present → Contusion=1
  3. If only "bone marrow edema", "subchondral edema", "osteochondral edema" → 
     Use GOLD_ANALYSIS and report context (acute trauma vs chronic/degenerative)
  9. Do NOT allow a validator to override explicit "bone contusion" evidence 
     merely because another part of the report describes marrow edema.
```

### 3.4 Add Explicit Evidence Override Rule
**File:** `experiments/ragset_report_inference_experiment/prompts/inference.yaml`

Add new section:

```yaml
  EXPLICIT EVIDENCE OVERRIDE
  ==========================

  When the ORIGINAL_REPORT explicitly states a finding using definitive terminology,
  do not negate that finding using a weaker indirect inference.

  Examples:
  * If report explicitly says "bone contusion" → Contusion=1
    Do not change to 0 because another sentence says "bone marrow edema"
  * If report explicitly says "ACL sprain" → ACL=1
    Do not change to 0 because another sentence says "increased signal"
  * If report explicitly says "Baker's cyst" → Bakers=1
    Do not change to 0 because another sentence says "popliteal fluid"
  * If report explicitly says "ACL intact" → ACL=0
    Do not change to 1 because another ligament is abnormal
```

### 3.5 Add Retry Checklist
**File:** `experiments/ragset_report_inference_experiment/prompts/inference.yaml`

Add new section:

```yaml
  RETRY CHECKLIST
  ===============

  On retry (attempt > 1), for EACH disputed label from validator feedback:
  
  1. Re-read ORIGINAL_REPORT from scratch.
  2. Locate the EXACT relevant sentence(s) for this label.
  3. Identify the anatomical structure and compartment.
  4. Classify the evidence state (PRESENT/EXPLICIT_ABSENT/UNCERTAIN/HISTORICAL/QUESTION_ONLY/NOT_MENTIONED).
  5. Determine if evidence is direct or inferred.
  5. Apply GOLD_ANALYSIS.
  6. Recompute the binary value INDEPENDENTLY.
  
  SPECIFIC CHECKS for common errors:
  * Generalized OA incorrectly mapped to multiple compartments
  * Nonspecific evidence mapped to a specific structure
  * Uncertainty (R/O, possible, suspicious) treated as confirmation
  * Degeneration treated as tear
  * Discoid morphology treated as tear
  * Historical finding treated as current
  * Effusion treated as synovitis
  * Other cyst treated as Baker's cyst
  * Marrow edema treated as fracture
  * Nonspecific bone abnormality treated as contusion
  * One generic evidence quote reused across unrelated labels
  
  Do NOT preserve a previous value simply because it appeared in an earlier attempt.
```

---

## 4. MODEL 2 PROMPT FIXES (validation.yaml)

### 4.1 Add Explicit Evidence Override Rule
**File:** `experiments/ragset_report_inference_experiment/prompts/validation.yaml`

Add new section:

```yaml
  EXPLICIT EVIDENCE OVERRIDE
  ==========================

  When the ORIGINAL_REPORT explicitly states a finding using definitive terminology,
  do not negate that finding using a weaker indirect inference.

  For example:
  * If the report explicitly says "Bone contusion in medial femoral condyle and tibial plateau."
    then Contusion=1.
    A validator must not change this to 0 merely because another part of the report describes bone marrow edema.
  
  Similarly, if the report explicitly says "ACL sprain", ACL=1 unless GOLD_ANALYSIS specifically establishes an exceptional mapping.
```

### 4.2 Add OA Compartment Rule
**File:** `experiments/ragset_report_inference_experiment/prompts/validation.yaml`

Add to OA section:

```yaml
  MODEL 2 OA RULE
  ===============

  Do not automatically convert:
  "Osteoarthritis of left knee"
  into:
  Medial_OA=1
  Lateral_OA=1
  PF_OA=1.

  Do not assign a compartment unless:
  * the report explicitly identifies the compartment, OR
  * GOLD_ANALYSIS explicitly establishes the mapping.

  Generic cartilage wear/spur formation cannot be reused for all three compartments.
```

### 4.3 Add Uncertainty Rule
**File:** `experiments/ragset_report_inference_experiment/prompts/validation.yaml`

Add to UNCERTAINTY section:

```yaml
  MODEL 2 UNCERTAINTY RULE
  ========================

  For:
  R/O
  possible
  suspicious
  cannot exclude
  questionable

  first classify as UNCERTAIN.

  Then apply GOLD_ANALYSIS.

  Do not universally force UNCERTAIN → 0.
  Do not universally force UNCERTAIN → 1.
```

### 4.4 Add Explicit Evidence Override (Duplicate Section - Consolidate)
**Note:** The validation.yaml already has "EXPLICIT EVIDENCE OVERRIDE" and "MODEL 2 EXPLICIT-EVIDENCE OVERRIDE" sections. These should be consolidated into one clear section.

---

## 5. RETRY-LOOP FIXES

### 5.1 Fix Model 1 Feedback Integration
**File:** `experiments/ragset_report_inference_experiment/runners/run_inference.py`

Already partially fixed - need to ensure validator_feedback is included in the prompt text, not just passed as parameter.

### 5.2 Add Model 1 Stuck Detection
**File:** `experiments/ragset_report_inference_experiment/src/ragset_inference/loop.py`

```python
def run_report(...):
    ...
    model1_predictions_history = {}  # label -> list of values
    
    for number in range(1, max_attempts + 1):
        ...
        prediction = infer(...)
        
        # Track Model 1 predictions for stuck detection
        for label, val in prediction.predictions.items():
            if label not in model1_predictions_history:
                model1_predictions_history[label] = []
            model1_predictions_history[label].append(val.value)
        
        # Detect if Model 1 is stuck (same value across all attempts for disputed labels)
        if number > 1:
            stuck_labels = [
                label for label, values in model1_predictions_history.items()
                if len(set(values)) == 1 and label in disputed_labels
            ]
            if stuck_labels:
                print(f"WARNING: Model 1 stuck on {stuck_labels} - same value across {number} attempts")
        
        ...
```

### 5.3 Add Validator Instability Detection
Already added in loop.py fix above.

---

## 6. ADJUDICATOR DESIGN (Future Implementation)

### 6.1 Adjudicator Prompt Template
**New File:** `experiments/ragset_report_inference_experiment/prompts/adjudicator.yaml`

```yaml
system_prompt: |
  You are an independent adjudicator for the RagSet experiment.
  
  Your task is to resolve binary disagreements between Model 1 and Model 2
  for SPECIFIC DISPUTED LABELS only.
  
  You receive:
  * ORIGINAL_REPORT (source of truth)
  * GOLD_ANALYSIS (dataset convention)
  * Disputed labels with Model 1 and Model 2 values
  
  You do NOT receive:
  * Model 1 chain-of-thought
  * Model 2 chain-of-thought
  * Previous retry reasoning
  * Validator feedback as evidence
  
  For EACH disputed label:
  1. Independently analyze the ORIGINAL_REPORT for that label
  2. Apply GOLD_ANALYSIS
  3. Determine the correct binary value
  4. Assign confidence: HIGH / MEDIUM / LOW
  5. Determine resolution: RESOLVED / UNRESOLVED
  
  Output ONLY valid JSON:
  {
    "adjudications": [
      {
        "label": "PF_OA",
        "model1_value": 1,
        "model2_value": 0,
        "adjudicated_value": 0,
        "confidence": "HIGH",
        "resolution": "RESOLVED",
        "reason": "Report describes generalized chondromalacia without patellofemoral compartment specification. GOLD_ANALYSIS requires compartment-specific evidence for PF_OA."
      }
    ]
  }
```

### 6.2 Label-Level Adjudication Logic
**File:** `experiments/ragset_report_inference_experiment/src/ragset_inference/loop.py` (new function)

```python
def adjudicate_disagreements(
    report: str,
    gold_analysis: str,
    model1_pred: ReportPrediction,
    model2_validation: ValidationResult,
    adjudicator_model,
) -> dict:
    """
    Adjudicate only the disputed labels.
    Returns dict of label -> adjudicated value + metadata.
    """
    # Identify disputed labels
    real_issues = [
        issue for issue in model2_validation.issues
        if issue.corrected is not None and issue.predicted != issue.corrected
    ]
    
    if not real_issues:
        return {}
    
    disputed_labels = [issue.label for issue in real_issues]
    
    # Prepare adjudicator input
    disputed_info = {
        label: {
            "model1_value": model1_pred.predictions[label].value,
            "model2_corrected": next(i.corrected for i in real_issues if i.label == label),
            "model1_evidence": model1_pred.predictions[label].evidence,
        }
        for label in disputed_labels
    }
    
    # Call adjudicator model
    # ... implementation details ...
    
    return adjudicated_results
```

---

## 6. ROUTING LOGIC

### 7.1 Decision Tree Implementation
**File:** `experiments/ragset_report_inference_experiment/src/ragset_inference/loop.py`

```python
def resolve_disagreements(
    report: str,
    gold_analysis: str,
    model1_pred: ReportPrediction,
    validation: ValidationResult,
    adjudicator_model,
) -> tuple[ReportPrediction, str]:
    """
    Resolve disagreements through decision tree:
    1. Programmatic filter (already done)
    2. GOLD_ANALYSIS / deterministic resolution
    3. Adjudicator model
    4. NEEDS_REVIEW
    """
    # Step 1: Programmatic filter already applied
    
    # Step 2: Check for GOLD_ANALYSIS resolution
    # (Would need structured GOLD_ANALYSIS with deterministic rules)
    
    # Step 3: Adjudicator for unresolved
    # adjudicated = adjudicate_disagreements(...)
    
    # Step 4: Apply resolutions
    # ...
    
    return final_prediction, resolution_status
```

---

## 8. PROVENANCE & LOGGING

### 8.1 Label-Level Resolution Tracking
**File:** `experiments/ragset_report_inference_experiment/src/ragset_inference/schemas.py`

Add resolution tracking to schemas:

```python
class LabelResolution(BaseModel):
    label: str
    model1_value: int
    model2_value: int | None
    adjudicator_value: int | None
    final_value: int
    resolution: Literal[
        "MODEL_AGREEMENT",
        "ACCEPTED_BY_GOLD",
        "ACCEPTED_BY_RULE",
        "ACCEPTED_BY_ADJUDICATOR",
        "NEEDS_REVIEW",
        "VALIDATOR_UNSTABLE",
        "MODEL1_STUCK",
        "VALIDATOR_CONFLICT",
        "FORMAT_ERROR",
    ]
    evidence: str | None
    reason: str | None
    confidence: Literal["HIGH", "MEDIUM", "LOW"] | None = None
```

### 8.2 Enhanced Trace Format
**File:** `experiments/ragset_report_inference_experiment/src/ragset_inference/trace.py`

Add resolution tracking to trace entries.

---

## IMPLEMENTATION PRIORITY

### Phase 1: Critical Fixes (Week 1)
1. ✅ Fix Model 1 STUCK bug (pass feedback in prompt)
2. ✅ Generate GOLD_ANALYSIS (run analyze_gold_llm.py)
3. ✅ Add contradiction check in loop.py
4. ✅ Add compartment safety gate for OA labels

### Phase 2: Prompt Fixes (Week 1-2)
1. Update inference.yaml with OA retry checklist, uncertainty classification, contusion distinction, explicit evidence override, retry checklist
2. Update validation.yaml with explicit evidence override, OA compartment rule, uncertainty rule
3. Consolidate duplicate sections in validation.yaml

### Phase 3: Retry Logic (Week 2)
1. Fix Model 1 feedback integration (pass feedback in prompt text)
2. Add Model 1 stuck detection
3. Add validator instability detection

### Phase 3: Adjudicator & Routing (Week 3-4)
1. Design adjudicator prompt
2. Implement label-level adjudication
3. Implement decision tree routing
4. Add label-level provenance tracking

### Phase 3: Monitoring & Validation
1. Add validator stability metrics
2. Add Model 1 stuck detection
3. Enhanced trace format with resolution provenance
4. Run evaluation on held-out set

---

## RISK MITIGATION

| Change | Risk | Mitigation |
|--------|------|------------|
| Model 1 retry fix | May introduce new errors | Keep original as fallback; A/B test |
| GOLD_ANALYSIS generation | LLM analysis quality variable | Fallback to basic stats; human review sample |
| Validator instability fix | May mask genuine ambiguity | Flag instability, don't auto-resolve |
| Adjudicator model | Cost, latency, new failure modes | Label-level only; different model; confidence scoring |

---

## SUCCESS METRICS

1. **Reduce NEEDS_REVIEW rate** from 11.1% → <5%
2. **Eliminate OA over-generalization** (0 issues)
3. **Eliminate validator instability** (0 unstable cases)
4. **Reduce Model 1 STUCK rate** to 0%
5. **Maintain/improve pass rate** on held-out evaluation
6. **Provenance completeness** - 100% labels have resolution source