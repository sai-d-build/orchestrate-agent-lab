from datetime import datetime, timezone
import json
import yaml
from pathlib import Path
from typing import Dict, Any
from langgraph.config import RunnableConfig
from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_state import RagSetState
from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import (
    ReportPrediction,
    CritiqueResult,
    CritiqueStatus,
    CritiqueIssueType,
    JudgeResult,
    TraceRecord,
    JudgeAction,
    JudgeReasonCode,
    LABEL_KEYS,
)
from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import (
    run_model1_attempt,
    _detect_validator_oscillation,
    _detect_model1_stuck,
    _validate_critic_output,
)
from experiments.ragset_report_inference_experiment.src.ragset_inference.policy_loader import get_canonical_policy_text, get_policy_hash, get_policy_version
from experiments.ragset_report_inference_experiment.src.ragset_inference.retrieval import GoldRetriever
from experiments.ragset_report_inference_experiment.src.ragset_inference.models import InferenceModel, CriticModel, JudgeModel
from experiments.ragset_report_inference_experiment.src.ragset_inference.trace import compute_prompt_hash, compute_response_hash


# Load prompt configurations
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

with open(PROMPTS_DIR / "inference.yaml", "r", encoding="utf-8") as f:
    INF_CFG = yaml.safe_load(f)

with open(PROMPTS_DIR / "validation_critic.yaml", "r", encoding="utf-8") as f:
    VAL_CFG = yaml.safe_load(f)

with open(PROMPTS_DIR / "judge.yaml", "r", encoding="utf-8") as f:
    JUDGE_CFG = yaml.safe_load(f)


def initialize_node(state: RagSetState, config: RunnableConfig) -> RagSetState:
    """
    Initialize state: load canonical policy, retrieve gold context, set up initial values.
    """
    # Extract models from config
    retriever = config["configurable"]["retriever"]
    inference_model = config["configurable"]["inference_model"]
    critic_model = config["configurable"]["critic_model"]
    judge_model = config["configurable"]["judge_model"]
    
    report = state["report"]
    study_uid = state["study_instance_uid"]
    max_attempts = state.get("max_attempts", 3)

    # Load canonical policy
    policy_text = get_canonical_policy_text()
    policy_hash = get_policy_hash()
    policy_version = get_policy_version()

    # Retrieve gold context
    retrieved = retriever.retrieve(report)
    gold_context = "\n---\n".join(
        f"STUDY {x['study_id']}\nLABELS: {x['labels']}\nREPORT:\n{x['report']}"
        for x in retrieved
    )

    # Initialize state
    new_state = dict(state)
    new_state.update({
        "canonical_policy": policy_text,
        "retrieved_gold_context": gold_context,
        "attempt": 1,
        "max_attempts": max_attempts,
        "current_prediction": None,
        "current_critique": None,
        "current_judgment": None,
        "prediction_history": [],
        "critique_history": [],
        "judgment_history": [],
        "oscillation_detected": False,
        "stuck_detected": False,
        "oscillating_labels": [],
        "stuck_labels": [],
        "final_status": None,
        "final_prediction": None,
        "review_reason": None,
        "trace_records": [],
        "safety_gate_failure": None,
        "policy_hash": policy_hash,
        "policy_version": policy_version,
    })

    # Write initialize trace
    trace = TraceRecord(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        study_instance_uid=study_uid,
        graph_node="initialize",
        stage="inference",
        attempt=0,
        requested_model=inference_model.model,
        actual_model=None,
        provider=inference_model.provider,
        prompt_hash="",
        response_hash=None,
        latency_seconds=None,
        input_tokens=None,
        output_tokens=None,
        status="success",
        error=None,
        judge_action=None,
        policy_hash=policy_hash,
        policy_version=policy_version,
    )
    new_state["trace_records"].append(trace)

    return new_state


def inference_node(state: RagSetState, config: RunnableConfig) -> RagSetState:
    """
    Invoke Model 1 (InferenceModel) via run_model1_attempt.
    """
    # Extract model from config
    inference_model = config["configurable"]["inference_model"]
    
    study_uid = state["study_instance_uid"]
    report = state["report"]
    attempt = state["attempt"]
    max_attempts = state["max_attempts"]
    policy = state["canonical_policy"]
    gold_context = state["retrieved_gold_context"]
    previous_prediction = state["current_prediction"]
    validator_feedback = state["current_critique"].model_dump() if state["current_critique"] else None

    start = datetime.now(timezone.utc)

    # Run single Model 1 attempt
    prediction = run_model1_attempt(
        report=report,
        gold_context=gold_context,
        policy=policy,
        labels=LABEL_KEYS,
        infer=inference_model.run,
        previous_prediction=previous_prediction,
        validator_feedback=validator_feedback,
        attempt=attempt,
    )

    end = datetime.now(timezone.utc)
    latency = (end - start).total_seconds()

    # Build prompt for trace
    template_vars = {
        "original_report": report,
        "gold_analysis": policy,
        "retrieved_gold_examples": gold_context,
    }
    if validator_feedback:
        feedback_json = json.dumps(validator_feedback, ensure_ascii=False, indent=2)
        template_vars["validator_feedback_section"] = (
            "VALIDATOR FEEDBACK (for reconsideration, NOT as evidence):\n"
            f"{feedback_json}\n\n"
            "Re-read the ORIGINAL_REPORT from scratch. Validator feedback identifies a disputed interpretation; "
            "it is not automatically ground truth. Recompute the affected labels using the ORIGINAL_REPORT "
            "and GOLD ANNOTATION POLICY. Do not blindly copy the validator correction."
        )
    else:
        template_vars["validator_feedback_section"] = ""

    user_prompt = INF_CFG["user_prompt_template"].format(**template_vars)
    full_prompt = INF_CFG["system_prompt"] + "\n" + user_prompt

    trace = TraceRecord(
        timestamp_utc=end.isoformat(),
        study_instance_uid=study_uid,
        graph_node="inference",
        stage="inference",
        attempt=attempt,
        requested_model=inference_model.model,
        actual_model=getattr(inference_model, 'last_actual_model', inference_model.model),
        provider=inference_model.provider,
        prompt_hash=compute_prompt_hash(full_prompt),
        response_hash=compute_response_hash(prediction.model_dump_json()),
        latency_seconds=latency,
        input_tokens=getattr(inference_model, 'last_token_usage', {}).get('input_tokens'),
        output_tokens=getattr(inference_model, 'last_token_usage', {}).get('output_tokens'),
        status="success",
        error=None,
        judge_action=None,
        policy_hash=state.get("policy_hash"),
        policy_version=state.get("policy_version"),
    )

    new_state = dict(state)
    new_state["current_prediction"] = prediction
    # Append to prediction history for use by critic/judge/finalize
    new_state["prediction_history"] = state["prediction_history"] + [prediction]
    new_state["trace_records"] = state["trace_records"] + [trace]

    return new_state


def critic_node(state: RagSetState, config: RunnableConfig) -> RagSetState:
    """
    Invoke Model 2 (CriticModel) with Model 1 output.
    Update oscillation/stuck detection flags.
    """
    # Extract model from config
    critic_model = config["configurable"]["critic_model"]
    
    study_uid = state["study_instance_uid"]
    report = state["report"]
    attempt = state["attempt"]
    prediction = state["current_prediction"]
    policy = state["canonical_policy"]
    gold_context = state["retrieved_gold_context"]
    prediction_history = state["prediction_history"]
    critique_history = state["critique_history"]

    start = datetime.now(timezone.utc)

    # Prepare Model 1 evidence for critique prompt
    model_1_evidence = {}
    for label, pred in prediction.predictions.items():
        model_1_evidence[label] = pred.evidence

    # Call critic model
    critique = critic_model.run(
        system=VAL_CFG["system_prompt"],
        user=VAL_CFG["user_prompt_template"].format(
            original_report=state["report"],
            gold_analysis=state["canonical_policy"],
            model_1_prediction=prediction.model_dump_json(),
            model_1_evidence=json.dumps({label: pred.evidence for label, pred in prediction.predictions.items()}, ensure_ascii=False, indent=2),
            retrieved_gold_examples=state["retrieved_gold_context"],
        ),
    )

    # Deterministic safety gate: validate Critic output integrity
    safety_gate_failure = _validate_critic_output(critique, report, prediction)
    if safety_gate_failure:
        # Log safety gate failure but don't alter Model 2's clinical conclusion
        # Pass to Model 3 for workflow decision
        safety_gate_failure["attempt"] = attempt
        safety_gate_failure["stage"] = "validation"
        # Create trace for safety gate failure
        trace = TraceRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            study_instance_uid=study_uid,
            graph_node="critic",
            stage="validation",
            attempt=attempt,
            requested_model=critic_model.model,
            actual_model=getattr(critic_model, 'last_actual_model', critic_model.model),
            provider=critic_model.provider,
            prompt_hash=compute_prompt_hash(VAL_CFG["system_prompt"]),
            response_hash=compute_response_hash(critique.model_dump_json()),
            latency_seconds=0.0,
            input_tokens=None,
            output_tokens=None,
            status="safety_gate_failure",
            error=json.dumps(safety_gate_failure, ensure_ascii=False),
            judge_action=None,
            policy_hash=state.get("policy_hash"),
            policy_version=state.get("policy_version"),
        )
        new_state = dict(state)
        new_state["current_critique"] = critique
        new_state["safety_gate_failure"] = safety_gate_failure
        new_state["trace_records"] = state["trace_records"] + [trace]
        return new_state

    end = datetime.now(timezone.utc)
    latency = (end - start).total_seconds()

    # Update oscillation/stuck detection
    # Build validator history from ALL critique issues (including history)
    validator_history = {label: [] for label in LABEL_KEYS}
    # Aggregate proposed values from critique history + current critique
    all_critiques = state["critique_history"] + [critique]
    for c in all_critiques:
        for issue in c.issues:
            if issue.proposed_value is not None:
                validator_history[issue.label].append(issue.proposed_value)

    oscillating_labels = _detect_validator_oscillation(validator_history)
    oscillation_detected = len(oscillating_labels) > 0

    # Build model1 history for stuck detection
    # Use prediction_history directly (current prediction already appended by inference_node)
    model1_history = {label: [] for label in LABEL_KEYS}
    for pred in state["prediction_history"]:
        for label, val in pred.predictions.items():
            model1_history[label].append(val.value)

    disputed_labels = set()
    for issue in critique.issues:
        if issue.proposed_value is not None:
            disputed_labels.add(issue.label)

    stuck_labels = _detect_model1_stuck(model1_history, disputed_labels)
    stuck_detected = len(stuck_labels) > 0

    # Build prompt for trace
    user_prompt = VAL_CFG["user_prompt_template"].format(
        original_report=state["report"],
        gold_analysis=state["canonical_policy"],
        model_1_prediction=prediction.model_dump_json(),
        model_1_evidence=json.dumps({label: pred.evidence for label, pred in prediction.predictions.items()}, ensure_ascii=False, indent=2),
        retrieved_gold_examples=state["retrieved_gold_context"],
    )
    full_prompt = VAL_CFG["system_prompt"] + "\n" + user_prompt

    trace = TraceRecord(
        timestamp_utc=end.isoformat(),
        study_instance_uid=study_uid,
        graph_node="critic",
        stage="validation",
        attempt=state["attempt"],
        requested_model=critic_model.model,
        actual_model=getattr(critic_model, 'last_actual_model', critic_model.model),
        provider=critic_model.provider,
        prompt_hash=compute_prompt_hash(full_prompt),
        response_hash=compute_response_hash(critique.model_dump_json()),
        latency_seconds=latency,
        input_tokens=getattr(critic_model, 'last_token_usage', {}).get('input_tokens'),
        output_tokens=getattr(critic_model, 'last_token_usage', {}).get('output_tokens'),
        status="success",
        error=None,
        judge_action=None,
        policy_hash=state.get("policy_hash"),
        policy_version=state.get("policy_version"),
    )

    new_state = dict(state)
    new_state["current_critique"] = critique
    new_state["oscillation_detected"] = oscillation_detected
    new_state["stuck_detected"] = stuck_detected
    new_state["oscillating_labels"] = oscillating_labels
    new_state["stuck_labels"] = stuck_labels
    # Append to critique history for use by judge/finalize
    new_state["critique_history"] = state["critique_history"] + [critique]
    new_state["trace_records"] = state["trace_records"] + [trace]

    return new_state


def judge_node(state: RagSetState, config: RunnableConfig) -> RagSetState:
    """
    Invoke Model 3 (JudgeModel) with Model 1 + Model 2 + history.
    """
    # Extract model from config
    judge_model = config["configurable"]["judge_model"]
    
    study_uid = state["study_instance_uid"]
    attempt = state["attempt"]
    max_attempts = state["max_attempts"]

    start = datetime.now(timezone.utc)

    # Build user prompt for judge
    # Use histories directly (current items already appended by their respective nodes)
    safety_gate_failure = state.get("safety_gate_failure")
    safety_gate_actionability = safety_gate_failure.get("actionability") if safety_gate_failure else None
    
    judgment = judge_model.run(
        system=JUDGE_CFG["system_prompt"],
        user=JUDGE_CFG["user_prompt_template"].format(
            original_report=state["report"],
            canonical_policy=state["canonical_policy"],
            retrieved_gold_examples=state["retrieved_gold_context"],
            model_1_prediction=state["current_prediction"].model_dump_json(),
            model_2_critique=state["current_critique"].model_dump_json(),
            prediction_history=json.dumps([p.model_dump() for p in state["prediction_history"]], ensure_ascii=False),
            critique_history=json.dumps([c.model_dump() for c in state["critique_history"]], ensure_ascii=False),
            judgment_history=json.dumps([j.model_dump() for j in state["judgment_history"]], ensure_ascii=False),
            attempt_number=state["attempt"],
            max_attempts=state["max_attempts"],
            oscillation_detected=state["oscillation_detected"],
            oscillating_labels=state["oscillating_labels"],
            stuck_detected=state["stuck_detected"],
            stuck_labels=state["stuck_labels"],
            safety_gate_failure=json.dumps(safety_gate_failure, ensure_ascii=False) if safety_gate_failure else "null",
            safety_gate_actionability=safety_gate_actionability if safety_gate_actionability else "null",
        ),
    )

    end = datetime.now(timezone.utc)
    latency = (end - start).total_seconds()

    # Build prompt for trace
    safety_gate_failure = state.get("safety_gate_failure")
    safety_gate_actionability = safety_gate_failure.get("actionability") if safety_gate_failure else None
    
    user_prompt = JUDGE_CFG["user_prompt_template"].format(
        original_report=state["report"],
        canonical_policy=state["canonical_policy"],
        retrieved_gold_examples=state["retrieved_gold_context"],
        model_1_prediction=state["current_prediction"].model_dump_json(),
        model_2_critique=state["current_critique"].model_dump_json(),
        prediction_history=json.dumps([p.model_dump() for p in state["prediction_history"]], ensure_ascii=False),
        critique_history=json.dumps([c.model_dump() for c in state["critique_history"]], ensure_ascii=False),
        judgment_history=json.dumps([j.model_dump() for j in state["judgment_history"]], ensure_ascii=False),
        attempt_number=state["attempt"],
        max_attempts=state["max_attempts"],
        oscillation_detected=state["oscillation_detected"],
        oscillating_labels=state["oscillating_labels"],
        stuck_detected=state["stuck_detected"],
        stuck_labels=state["stuck_labels"],
        safety_gate_failure=json.dumps(safety_gate_failure, ensure_ascii=False) if safety_gate_failure else "null",
        safety_gate_actionability=safety_gate_actionability if safety_gate_actionability else "null",
    )
    full_prompt = JUDGE_CFG["system_prompt"] + "\n" + user_prompt

    trace = TraceRecord(
        timestamp_utc=end.isoformat(),
        study_instance_uid=study_uid,
        graph_node="judge",
        stage="judgment",
        attempt=attempt,
        requested_model=judge_model.model,
        actual_model=getattr(judge_model, 'last_actual_model', judge_model.model),
        provider=judge_model.provider,
        prompt_hash=compute_prompt_hash(full_prompt),
        response_hash=compute_response_hash(judgment.model_dump_json()),
        latency_seconds=latency,
        input_tokens=getattr(judge_model, 'last_token_usage', {}).get('input_tokens'),
        output_tokens=getattr(judge_model, 'last_token_usage', {}).get('output_tokens'),
        status="success",
        error=None,
        judge_action=judgment.action,
        policy_hash=state.get("policy_hash"),
        policy_version=state.get("policy_version"),
    )

    new_state = dict(state)
    new_state["current_judgment"] = judgment
    # Append to judgment history for use by finalize
    new_state["judgment_history"] = state["judgment_history"] + [judgment]
    new_state["trace_records"] = state["trace_records"] + [trace]

    return new_state


def _determine_finalization_metadata(state: RagSetState, action: JudgeAction) -> tuple[str, str]:
    """
    Determine the conflict state and terminal reason code for finalization.
    
    Returns:
        (conflict_state, terminal_reason_code)
    
    CRITICAL: The terminal_reason_code MUST be consistent with the action per the
    action/reason-code consistency rules in the Judge prompt. Invalid combinations:
    - PASS + INTEGRITY_FAILURE (PASS only allows NO_ACTIONABLE_ISSUE, MODEL2_UNSUPPORTED_CORRECTION)
    - PASS + OSCILLATION/STUCK/MAX_ATTEMPTS/etc.
    """
    judgment = state.get("current_judgment")
    current_critique = state.get("current_critique")
    current_prediction = state.get("current_prediction")
    oscillation_detected = state.get("oscillation_detected", False)
    stuck_detected = state.get("stuck_detected", False)
    safety_gate_failure = state.get("safety_gate_failure")
    attempt = state.get("attempt", 1)
    max_attempts = state.get("max_attempts", 3)
    
    # Determine conflict state based on terminal action and state
    if action == JudgeAction.PASS:
        # PASS with safety_gate_failure is INVALID per action/reason consistency rules.
        # If safety_gate_failure exists, the Judge should have returned NEEDS_REVIEW or STOP.
        # We correct this by treating it as NEEDS_REVIEW with INTEGRITY_FAILURE.
        if safety_gate_failure:
            return "INTEGRITY_FAILURE", JudgeReasonCode.INTEGRITY_FAILURE.value
        # Check if judge explicitly indicated MODEL2_UNSUPPORTED_CORRECTION
        judgment = state.get("current_judgment")
        if judgment and judgment.reason_code == JudgeReasonCode.MODEL2_UNSUPPORTED_CORRECTION:
            return "MODEL2_UNSUPPORTED_CORRECTION", JudgeReasonCode.MODEL2_UNSUPPORTED_CORRECTION.value
        return "NONE", JudgeReasonCode.NO_ACTIONABLE_ISSUE.value
    
    elif action == JudgeAction.AMBIGUOUS:
        # Check if it's no-evidence ambiguity or policy unresolved
        if current_critique:
            has_unresolved = any(
                issue.issue_type in (CritiqueIssueType.UNRESOLVED_POLICY, CritiqueIssueType.REPORT_AMBIGUITY)
                for issue in current_critique.issues
            )
            has_insufficient = any(
                issue.issue_type == CritiqueIssueType.INSUFFICIENT_EVIDENCE
                for issue in current_critique.issues
            )
            if has_insufficient and not has_unresolved:
                return "NO_EVIDENCE", JudgeReasonCode.NO_EVIDENCE_AMBIGUOUS.value
            if has_unresolved:
                return "POLICY_UNRESOLVED", JudgeReasonCode.POLICY_UNRESOLVED.value
        return "AMBIGUOUS", JudgeReasonCode.REPORT_AMBIGUITY.value
    
    elif action == JudgeAction.NEEDS_REVIEW:
        # Check if judge explicitly indicated a specific reason
        judgment = state.get("current_judgment")
        if judgment and judgment.reason_code in (
            JudgeReasonCode.MODEL2_UNSUPPORTED_CORRECTION,
            JudgeReasonCode.MODEL1_STUCK_RETAINED,
            JudgeReasonCode.OSCILLATION,
            JudgeReasonCode.MAX_ATTEMPTS,
            JudgeReasonCode.REPEATED_CRITIQUE,
            JudgeReasonCode.INTEGRITY_FAILURE,
            JudgeReasonCode.MODEL2_CONFLICT,
        ):
            return judgment.reason_code.value, judgment.reason_code.value
        
        if attempt >= max_attempts:
            return "MAX_ATTEMPTS", JudgeReasonCode.MAX_ATTEMPTS.value
        if oscillation_detected:
            return "OSCILLATION", JudgeReasonCode.OSCILLATION.value
        
        # Distinguish Model1 stuck (ignoring valid feedback) from Model2 unsupported corrections
        if stuck_detected:
            current_critique = state.get("current_critique")
            if current_critique:
                # Check if Model2's critique has unsupported CLEAR_* issues
                has_unsupported_clear = any(
                    issue.issue_type in (CritiqueIssueType.CLEAR_POLICY_CONFLICT, CritiqueIssueType.CLEAR_REPORT_CONFLICT)
                    and (issue.evidence is None or issue.evidence == "" or issue.policy_rule is None or issue.policy_rule == "")
                    for issue in current_critique.issues
                )
                if has_unsupported_clear:
                    return "MODEL2_UNSUPPORTED_CORRECTION", JudgeReasonCode.MODEL2_UNSUPPORTED_CORRECTION.value
            return "MODEL1_STUCK", JudgeReasonCode.MODEL1_STUCK_RETAINED.value
        
        if safety_gate_failure:
            return "INTEGRITY_FAILURE", JudgeReasonCode.INTEGRITY_FAILURE.value
        # Check for repeated critique
        critique_history = state.get("critique_history", [])
        current_critique = state.get("current_critique")
        all_critiques = critique_history + ([current_critique] if current_critique else [])
        if len(all_critiques) >= 2:
            # Check for repeated UNRESOLVED/AMBIGUOUS critiques
            unresolved_count = sum(
                1 for c in all_critiques
                if c and any(i.issue_type in (CritiqueIssueType.UNRESOLVED_POLICY, CritiqueIssueType.REPORT_AMBIGUITY) for i in c.issues)
            )
            if unresolved_count >= 2:
                return "REPEATED_CRITIQUE", JudgeReasonCode.REPEATED_CRITIQUE.value
        return "MODEL2_CONFLICT", JudgeReasonCode.MODEL2_CONFLICT.value
    
    elif action == JudgeAction.STOP:
        return "SYSTEM_ERROR", JudgeReasonCode.SYSTEM_ERROR.value
    
    else:
        return "UNKNOWN", JudgeReasonCode.SYSTEM_ERROR.value


def _select_best_prediction_from_history(state: RagSetState) -> tuple[ReportPrediction | None, str, int | None]:
    """
    Select the best defensible prediction from the complete attempt history.
    
    Precedence (per requirements):
    1. Explicit ORIGINAL_REPORT evidence + Canonical policy (Model 2 PASS)
    2. Policy-supported prediction (only SUPPORTED issues, no conflicts/ambiguity)
    3. MODEL2_UNSUPPORTED_CORRECTION - trust Model 1 when Judge rejects Model 2's correction
    4. Terminal status fallback - RESPECT the Judge's terminal action
    
    CRITICAL: When Judge returns NEEDS_REVIEW (OSCILLATION, STUCK, MAX_ATTEMPTS,
    REPEATED_CRITIQUE, INTEGRITY_FAILURE) or AMBIGUOUS, we MUST NOT claim
    "resolved conflicts" or "history-based resolution". The workflow FAILED to converge.
    
    Returns:
        (selected_prediction, selection_reason, selected_attempt_number)
    """
    report = state["report"]
    policy = state["canonical_policy"]
    prediction_history = state["prediction_history"]
    critique_history = state["critique_history"]
    judgment_history = state["judgment_history"]
    current_prediction = state["current_prediction"]
    current_critique = state["current_critique"]
    current_judgment = state["current_judgment"]
    oscillation_detected = state.get("oscillation_detected", False)
    stuck_detected = state.get("stuck_detected", False)
    oscillating_labels = state.get("oscillating_labels", [])
    stuck_labels = state.get("stuck_labels", [])
    safety_gate_failure = state.get("safety_gate_failure")
    attempt = state["attempt"]
    max_attempts = state["max_attempts"]
    
    # Build complete attempt list (history already includes current)
    all_predictions = prediction_history
    all_critiques = critique_history
    all_judgments = judgment_history
    
    if not all_predictions:
        return None, "No predictions available", None
    
    # Terminal action from current judgment
    terminal_action = current_judgment.action if current_judgment else JudgeAction.NEEDS_REVIEW
    terminal_reason = current_judgment.reason_code if current_judgment else JudgeReasonCode.SYSTEM_ERROR
    
    # ============================================================
    # PRIORITY 0: TERMINAL WORKFLOW STATES - RESPECT THE JUDGE
    # ============================================================
    # If Judge ruled NEEDS_REVIEW for convergence failure, do NOT claim resolution.
    # The workflow FAILED to establish safe convergence (per Judge prompt Section 10).
    if terminal_action == JudgeAction.NEEDS_REVIEW:
        if terminal_reason == JudgeReasonCode.OSCILLATION:
            reason = f"Workflow failed to converge (oscillation on {oscillating_labels}); prediction retained for review, NOT validated as correct"
            return current_prediction, reason, attempt
        if terminal_reason in (JudgeReasonCode.MODEL1_STUCK_RETAINED, JudgeReasonCode.MODEL1_STUCK):
            reason = f"Model 1 stuck on {stuck_labels} despite feedback; prediction retained for review, NOT validated as correct"
            return current_prediction, reason, attempt
        if terminal_reason == JudgeReasonCode.MAX_ATTEMPTS:
            reason = f"Maximum attempts ({max_attempts}) reached without convergence; prediction retained for review"
            return current_prediction, reason, attempt
        if terminal_reason == JudgeReasonCode.REPEATED_CRITIQUE:
            reason = "Repeated critique without resolution; prediction retained for review"
            return current_prediction, reason, attempt
        if terminal_reason == JudgeReasonCode.INTEGRITY_FAILURE:
            reason = "Safety gate failure prevents safe evaluation; prediction retained for review"
            return current_prediction, reason, attempt
        if terminal_reason == JudgeReasonCode.MODEL2_UNSUPPORTED_CORRECTION:
            reason = "Model 2 correction unsupported by policy/report; retaining Model 1 prediction"
            return current_prediction, reason, attempt
        # Generic NEEDS_REVIEW (MODEL2_CONFLICT or other)
        reason = "Workflow requires human review; prediction retained for review, NOT validated as correct"
        return current_prediction, reason, attempt
    
    # If Judge ruled AMBIGUOUS, acknowledge genuine ambiguity - do NOT pick "most stable"
    if terminal_action == JudgeAction.AMBIGUOUS:
        if terminal_reason == JudgeReasonCode.POLICY_UNRESOLVED:
            reason = "Genuine policy ambiguity (UNRESOLVED convention); prediction reflects ambiguity, NOT resolution"
        elif terminal_reason == JudgeReasonCode.NO_EVIDENCE_AMBIGUOUS:
            reason = "No usable imaging evidence for annotation; prediction reflects ambiguity, NOT resolution"
        else:
            reason = "Genuine report/policy ambiguity; prediction reflects ambiguity, NOT resolution"
        return current_prediction, reason, attempt
    
    # ============================================================
    # PRIORITY 1: MODEL2_UNSUPPORTED_CORRECTION - Trust Model 1
    # ============================================================
    if terminal_reason == JudgeReasonCode.MODEL2_UNSUPPORTED_CORRECTION:
        reason = "Model 2 correction unsupported by policy/report; retaining Model 1 prediction"
        return current_prediction, reason, attempt
    
    # ============================================================
    # PRIORITY 2: Model 2 PASS - Explicit validation
    # ============================================================
    pass_attempts = []
    for i, (pred, crit) in enumerate(zip(all_predictions, all_critiques)):
        if crit and crit.status == CritiqueStatus.PASS:
            pass_attempts.append((i, pred, crit))
    
    if pass_attempts:
        latest_pass_idx, latest_pass_pred, latest_pass_crit = pass_attempts[-1]
        reason = f"Model 2 PASS at attempt {latest_pass_idx + 1}"
        return latest_pass_pred, reason, latest_pass_idx + 1
    
    # ============================================================
    # PRIORITY 3: Policy-supported prediction (only SUPPORTED issues)
    # ============================================================
    policy_supported_attempts = []
    for i, (pred, crit) in enumerate(zip(all_predictions, all_critiques)):
        if not crit:
            continue
        has_clear_conflict = any(
            issue.issue_type in (CritiqueIssueType.CLEAR_POLICY_CONFLICT, CritiqueIssueType.CLEAR_REPORT_CONFLICT)
            for issue in crit.issues
        )
        has_unresolved = any(
            issue.issue_type in (CritiqueIssueType.UNRESOLVED_POLICY, CritiqueIssueType.REPORT_AMBIGUITY)
            for issue in crit.issues
        )
        has_insufficient_evidence = any(
            issue.issue_type == CritiqueIssueType.INSUFFICIENT_EVIDENCE
            for issue in crit.issues
        )
        if not has_clear_conflict and not has_unresolved and not has_insufficient_evidence:
            policy_supported_attempts.append((i, pred, crit))
    
    if policy_supported_attempts:
        latest_idx, latest_pred, latest_crit = policy_supported_attempts[-1]
        reason = f"Policy-supported (only SUPPORTED issues) at attempt {latest_idx + 1}"
        return latest_pred, reason, latest_idx + 1
    
    # ============================================================
    # PRIORITY 4: Resolved CLEAR_* conflicts (only for PASS terminal action)
    # ============================================================
    # Only look for resolved conflicts if terminal action is PASS (workflow converged)
    if terminal_action == JudgeAction.PASS:
        resolved_attempts = []
        for i in range(len(all_predictions)):
            if i == 0:
                crit = all_critiques[0] if all_critiques else None
                if crit and not any(
                    issue.issue_type in (CritiqueIssueType.CLEAR_POLICY_CONFLICT, CritiqueIssueType.CLEAR_REPORT_CONFLICT)
                    for issue in crit.issues
                ):
                    has_non_resolvable = any(
                        issue.issue_type in (CritiqueIssueType.INSUFFICIENT_EVIDENCE, CritiqueIssueType.UNRESOLVED_POLICY, CritiqueIssueType.REPORT_AMBIGUITY)
                        for issue in crit.issues
                    )
                    if not has_non_resolvable:
                        resolved_attempts.append((i, all_predictions[i]))
            else:
                prev_crit = all_critiques[i - 1] if i - 1 < len(all_critiques) else None
                curr_pred = all_predictions[i]
                if prev_crit:
                    clear_issues = [
                        issue for issue in prev_crit.issues
                        if issue.issue_type in (CritiqueIssueType.CLEAR_POLICY_CONFLICT, CritiqueIssueType.CLEAR_REPORT_CONFLICT)
                    ]
                    if clear_issues:
                        all_resolved = True
                        for issue in clear_issues:
                            curr_val = curr_pred.predictions[issue.label].value
                            if curr_val != issue.proposed_value:
                                all_resolved = False
                                break
                        if all_resolved:
                            resolved_attempts.append((i, curr_pred))
        
        if resolved_attempts:
            latest_idx, latest_pred = resolved_attempts[-1]
            reason = f"Resolved clear conflicts at attempt {latest_idx + 1}"
            return latest_pred, reason, latest_idx + 1
    
    # ============================================================
    # PRIORITY 5: Terminal PASS fallback
    # ============================================================
    if terminal_action == JudgeAction.PASS:
        reason = "Terminal PASS - using latest prediction"
        return current_prediction, reason, attempt
    
    # ============================================================
    # PRIORITY 6: STOP / Unknown
    # ============================================================
    if terminal_action == JudgeAction.STOP:
        reason = "Terminal STOP (system error) - using latest prediction"
        return current_prediction, reason, attempt
    
    # Fallback (should not reach here)
    reason = "Unknown terminal action - using latest prediction"
    return current_prediction, reason, attempt


def finalize_node(state: RagSetState) -> RagSetState:
    """
    Produce final result by evaluating complete inference history against
    original report and canonical policy. Every terminal route must pass here.
    """
    judgment = state["current_judgment"]
    action = judgment.action if judgment else JudgeAction.NEEDS_REVIEW

    # Determine conflict state and terminal reason code FIRST
    conflict_state, terminal_reason_code = _determine_finalization_metadata(state, action)
    
    # Determine final status based on judge action AND conflict state
    # PASS with integrity failure is NOT a pass - it's a needs_review
    # CRITICAL: If action=PASS but conflict_state=INTEGRITY_FAILURE, we must
    # override the terminal_action to NEEDS_REVIEW for action/reason consistency
    effective_action = action
    if action == JudgeAction.PASS and conflict_state == "INTEGRITY_FAILURE":
        effective_action = JudgeAction.NEEDS_REVIEW
        final_status = "needs_review"
    elif action == JudgeAction.PASS:
        final_status = "passed"
    elif action in (JudgeAction.AMBIGUOUS, JudgeAction.NEEDS_REVIEW, JudgeAction.STOP):
        final_status = "needs_review"
    else:
        # RETRY_MODEL1 should not reach finalize (handled by graph routing)
        final_status = "needs_review"

    # Select best prediction from complete history
    final_prediction, selection_reason, selected_attempt = _select_best_prediction_from_history(state)
    
    # Build comprehensive review reason from finalization metadata
    terminal_reason = judgment.rationale if judgment else "Workflow terminated"
    review_reason = f"{terminal_reason} | Finalization: {selection_reason} (selected attempt {selected_attempt})"

    # NOTE: Trace writing is now handled by the worker/main process to avoid duplication.
    # The trace_records in state are passed back and written once by the runner.

    new_state = dict(state)
    new_state["final_status"] = final_status
    new_state["final_prediction"] = final_prediction
    new_state["review_reason"] = review_reason
    # Add finalization metadata for traceability (authoritative)
    new_state["finalization_selected_attempt"] = selected_attempt
    new_state["finalization_reason"] = selection_reason
    # Use effective_action (corrected for consistency) not raw judge action
    new_state["finalization_terminal_action"] = effective_action.value if effective_action else "UNKNOWN"
    new_state["finalization_terminal_reason_code"] = terminal_reason_code
    new_state["finalization_conflict_state"] = conflict_state

    return new_state