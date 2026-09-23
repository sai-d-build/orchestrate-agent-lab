from langgraph.graph import StateGraph, START, END
from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_state import RagSetState
from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import (
    initialize_node,
    inference_node,
    critic_node,
    judge_node,
    finalize_node,
)
from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import JudgeAction


def route_judgment(state: RagSetState) -> str:
    """
    Conditional routing from judge node based on JudgeAction.
    """
    judgment = state.get("current_judgment")
    if not judgment:
        return "finalize"

    action = judgment.action

    # Max-attempts guard: if attempt >= max_attempts, force NEEDS_REVIEW
    if state["attempt"] >= state["max_attempts"]:
        return "finalize"

    if action == JudgeAction.PASS:
        return "finalize"
    elif action == JudgeAction.AMBIGUOUS:
        return "finalize"
    elif action == JudgeAction.NEEDS_REVIEW:
        return "finalize"
    elif action == JudgeAction.STOP:
        return "finalize"
    elif action == JudgeAction.RETRY_MODEL1:
        return "inference"
    else:
        return "finalize"


def build_graph() -> StateGraph:
    """
    Build the LangGraph state graph for RagSet inference workflow.
    """
    graph = StateGraph(RagSetState)

    # Add nodes
    graph.add_node("initialize", initialize_node)
    graph.add_node("inference", inference_node)
    graph.add_node("critic", critic_node)
    graph.add_node("judge", judge_node)
    graph.add_node("finalize", finalize_node)

    # Add edges
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "inference")
    graph.add_edge("inference", "critic")
    graph.add_edge("critic", "judge")

    # Conditional routing from judge
    graph.add_conditional_edges(
        "judge",
        route_judgment,
        {
            "finalize": "finalize",
            "inference": "inference",
        }
    )

    graph.add_edge("finalize", END)

    return graph.compile()


def create_initial_state(
    study_instance_uid: str,
    report: str,
    max_attempts: int = 3,
) -> RagSetState:
    """
    Create initial state for a new report.
    """
    from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import LABEL_KEYS

    return {
        "study_instance_uid": study_instance_uid,
        "report": report,
        "canonical_policy": "",
        "retrieved_gold_context": "",
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
    }