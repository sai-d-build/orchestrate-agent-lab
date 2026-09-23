from datetime import datetime, timezone
import hashlib
import json
import yaml
from pathlib import Path
from typing import Dict, Any
from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_state import RagSetState
from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import (
    ReportPrediction,
    CritiqueResult,
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
)
from experiments.ragset_report_inference_experiment.src.ragset_inference.policy_loader import get_canonical_policy_text
from experiments.ragset_report_inference_experiment.src.ragset_inference.retrieval import GoldRetriever
from experiments.ragset_report_inference_experiment.src.ragset_inference.models import InferenceModel, CriticModel, JudgeModel
from experiments.ragset_report_inference_experiment.src.ragset_inference.trace import write_trace


# Load prompt configurations
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

with open(PROMPTS_DIR / "inference.yaml", "r", encoding="utf-8") as f:
    INF_CFG = yaml.safe_load(f)

with open(PROMPTS_DIR / "validation_critic.yaml", "r", encoding="utf-8") as f:
    VAL_CFG = yaml.safe_load(f)

with open(PROMPTS_DIR / "judge.yaml", "r", encoding="utf-8") as f:
    JUDGE_CFG = yaml.safe_load(f)


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _response_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def initialize_node(state: RagSetState, *, retriever: GoldRetriever, inference_model: InferenceModel, critic_model: CriticModel, judge_model: JudgeModel) -> RagSetState:
    """
    Initialize state: load canonical policy, retrieve gold context, set up initial values.
    """
    report = state["report"]
    study_uid = state["study_instance_uid"]
    max_attempts = state.get("max_attempts", 3)

    # Load canonical policy
    policy_text = get_canonical_policy_text()

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
    )
    new_state["trace_records"].append(trace)

    return new_state


def inference_node(state: RagSetState, *, inference_model: InferenceModel) -> RagSetState:
    """
    Invoke Model 1 (InferenceModel) via run_model1_attempt.
    """
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
        prompt_hash=_prompt_hash(full_prompt),
        response_hash=_response_hash(prediction.model_dump_json()),
        latency_seconds=latency,
        input_tokens=None,
        output_tokens=None,
        status="success",
        error=None,
        judge_action=None,
    )

    new_state = dict(state)
    new_state["current_prediction"] = prediction
    new_state["trace_records"] = state["trace_records"] + [trace]

    return new_state


def critic_node(state: RagSetState, *, critic_model: CriticModel) -> RagSetState:
    """
    Invoke Model 2 (CriticModel) with Model 1 output.
    Update oscillation/stuck detection flags.
    """
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

    end = datetime.now(timezone.utc)
    latency = (end - start).total_seconds()

    # Update oscillation/stuck detection
    # Build validator history from critique issues
    validator_history = {label: [] for label in LABEL_KEYS}
    for issue in critique.issues:
        if issue.proposed_value is not None:
            validator_history[issue.label].append(issue.proposed_value)

    oscillating_labels = _detect_validator_oscillation(validator_history)
    oscillation_detected = len(oscillating_labels) > 0

    # Build model1 history for stuck detection
    model1_history = {label: [] for label in LABEL_KEYS}
    for pred in state["prediction_history"] + [prediction]:
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
        prompt_hash=_prompt_hash(full_prompt),
        response_hash=_response_hash(critique.model_dump_json()),
        latency_seconds=latency,
        input_tokens=None,
        output_tokens=None,
        status="success",
        error=None,
        judge_action=None,
    )

    new_state = dict(state)
    new_state["current_critique"] = critique
    new_state["oscillation_detected"] = oscillation_detected
    new_state["stuck_detected"] = stuck_detected
    new_state["oscillating_labels"] = oscillating_labels
    new_state["stuck_labels"] = stuck_labels
    new_state["trace_records"] = state["trace_records"] + [trace]

    return new_state


def judge_node(state: RagSetState, *, judge_model: JudgeModel) -> RagSetState:
    """
    Invoke Model 3 (JudgeModel) with Model 1 + Model 2 + history.
    """
    study_uid = state["study_instance_uid"]
    attempt = state["attempt"]
    max_attempts = state["max_attempts"]

    start = datetime.now(timezone.utc)

    # Build user prompt for judge
    judgment = judge_model.run(
        system=JUDGE_CFG["system_prompt"],
        user=JUDGE_CFG["user_prompt_template"].format(
            original_report=state["report"],
            canonical_policy=state["canonical_policy"],
            retrieved_gold_examples=state["retrieved_gold_context"],
            model_1_prediction=state["current_prediction"].model_dump_json(),
            model_2_critique=state["current_critique"].model_dump_json(),
            prediction_history=json.dumps([p.model_dump() for p in state["prediction_history"] + [state["current_prediction"]]], ensure_ascii=False),
            critique_history=json.dumps([c.model_dump() for c in state["critique_history"] + [state["current_critique"]]], ensure_ascii=False),
            judgment_history=json.dumps([j.model_dump() for j in state["judgment_history"]], ensure_ascii=False),
            attempt_number=state["attempt"],
            max_attempts=state["max_attempts"],
            oscillation_detected=state["oscillation_detected"],
            oscillating_labels=state["oscillating_labels"],
            stuck_detected=state["stuck_detected"],
            stuck_labels=state["stuck_labels"],
        ),
    )

    end = datetime.now(timezone.utc)
    latency = (end - start).total_seconds()

    # Build prompt for trace
    user_prompt = JUDGE_CFG["user_prompt_template"].format(
        original_report=state["report"],
        canonical_policy=state["canonical_policy"],
        retrieved_gold_examples=state["retrieved_gold_context"],
        model_1_prediction=state["current_prediction"].model_dump_json(),
        model_2_critique=state["current_critique"].model_dump_json(),
        prediction_history=json.dumps([p.model_dump() for p in state["prediction_history"] + [state["current_prediction"]]], ensure_ascii=False),
        critique_history=json.dumps([c.model_dump() for c in state["critique_history"] + [state["current_critique"]]], ensure_ascii=False),
        judgment_history=json.dumps([j.model_dump() for j in state["judgment_history"]], ensure_ascii=False),
        attempt_number=state["attempt"],
        max_attempts=state["max_attempts"],
        oscillation_detected=state["oscillation_detected"],
        oscillating_labels=state["oscillating_labels"],
        stuck_detected=state["stuck_detected"],
        stuck_labels=state["stuck_labels"],
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
        prompt_hash=_prompt_hash(full_prompt),
        response_hash=_response_hash(judgment.model_dump_json()),
        latency_seconds=latency,
        input_tokens=None,
        output_tokens=None,
        status="success",
        error=None,
        judge_action=judgment.action,
    )

    new_state = dict(state)
    new_state["current_judgment"] = judgment
    new_state["trace_records"] = state["trace_records"] + [trace]

    return new_state


def finalize_node(state: RagSetState) -> RagSetState:
    """
    Produce final result and write accumulated traces.
    """
    judgment = state["current_judgment"]
    action = judgment.action if judgment else JudgeAction.NEEDS_REVIEW

    # Determine final status based on judge action
    if action == JudgeAction.PASS:
        final_status = "passed"
        review_reason = None
    elif action in (JudgeAction.AMBIGUOUS, JudgeAction.NEEDS_REVIEW, JudgeAction.STOP):
        final_status = "needs_review"
        review_reason = state["current_judgment"].rationale if state["current_judgment"] else "Workflow terminated"
    else:
        # RETRY_MODEL1 should not reach finalize (handled by graph routing)
        final_status = "needs_review"
        review_reason = "Unexpected RETRY_MODEL1 in finalize"

    # Write all accumulated traces
    for trace in state["trace_records"]:
        write_trace(
            path="experiments/ragset_report_inference_experiment/results/inference/model_trace.jsonl",
            report_id=trace.study_instance_uid,
            stage=trace.stage,
            model=trace.requested_model,
            attempt=trace.attempt,
            prompt="",  # Prompt not stored in TraceRecord
            status=trace.status,
            latency_seconds=trace.latency_seconds,
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            error=trace.error,
            provider=trace.provider,
            actual_model=trace.actual_model,
            graph_node=trace.graph_node,
            judge_action=trace.judge_action,
            response_hash=trace.response_hash,
        )

    new_state = dict(state)
    new_state["final_status"] = final_status
    new_state["final_prediction"] = state["current_prediction"]
    new_state["review_reason"] = review_reason

    return new_state