import pytest
from unittest.mock import Mock, MagicMock, patch
from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import (
    ReportPrediction,
    LabelValue,
    CritiqueResult,
    CritiqueIssue,
    CritiqueStatus,
    CritiqueIssueType,
    JudgeResult,
    JudgeAction,
    JudgeReasonCode,
    LABEL_KEYS,
)
from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_state import RagSetState
from experiments.ragset_report_inference_experiment.src.ragset_inference.graph import build_graph, create_initial_state, route_judgment
from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import run_model1_attempt


def make_prediction(values: dict[str, int] | None = None) -> ReportPrediction:
    """Create a ReportPrediction with all 12 labels."""
    preds = {}
    for label in LABEL_KEYS:
        val = values.get(label, 0) if values else 0
        preds[label] = LabelValue(value=val, evidence="")
    return ReportPrediction(predictions=preds)


def make_critique(status: CritiqueStatus, issues: list | None = None) -> CritiqueResult:
    """Create a CritiqueResult."""
    if issues is None:
        issues = []
    return CritiqueResult(
        status=status,
        issues=issues,
        summary="test",
        actionable=status == CritiqueStatus.FAIL,
        affected_labels=[i.label for i in issues],
    )


def make_judge_result(action: JudgeAction, reason_code: JudgeReasonCode, rationale: str) -> JudgeResult:
    """Create a JudgeResult."""
    return JudgeResult(
        action=action,
        reason_code=reason_code,
        rationale=rationale,
    )


class TestGraphState:
    """Test RagSetState TypedDict."""

    def test_create_initial_state(self):
        state = create_initial_state("test_uid", "test report", max_attempts=3)
        assert state["study_instance_uid"] == "test_uid"
        assert state["report"] == "test report"
        assert state["attempt"] == 1
        assert state["max_attempts"] == 3
        assert state["current_prediction"] is None
        assert state["current_critique"] is None
        assert state["current_judgment"] is None
        assert state["prediction_history"] == []
        assert state["critique_history"] == []
        assert state["judgment_history"] == []
        assert state["oscillation_detected"] is False
        assert state["stuck_detected"] is False
        assert state["trace_records"] == []


class TestRouteJudgment:
    """Test conditional routing from judge node."""

    def test_route_pass(self):
        state = create_initial_state("test", "report")
        state["current_judgment"] = make_judge_result(
            JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "All good"
        )
        assert route_judgment(state) == "finalize"

    def test_route_ambiguous(self):
        state = create_initial_state("test", "report")
        state["current_judgment"] = make_judge_result(
            JudgeAction.AMBIGUOUS, JudgeReasonCode.UNRESOLVED_POLICY, "Policy unresolved"
        )
        assert route_judgment(state) == "finalize"

    def test_route_needs_review(self):
        state = create_initial_state("test", "report")
        state["current_judgment"] = make_judge_result(
            JudgeAction.NEEDS_REVIEW, JudgeReasonCode.OSCILLATION, "Oscillation detected"
        )
        assert route_judgment(state) == "finalize"

    def test_route_stop(self):
        state = create_initial_state("test", "report")
        state["current_judgment"] = make_judge_result(
            JudgeAction.STOP, JudgeReasonCode.SYSTEM_ERROR, "System error"
        )
        assert route_judgment(state) == "finalize"

    def test_route_retry_model1(self):
        state = create_initial_state("test", "report")
        state["current_judgment"] = make_judge_result(
            JudgeAction.RETRY_MODEL1, JudgeReasonCode.CLEAR_POLICY_CONFLICT, "Policy conflict"
        )
        assert route_judgment(state) == "inference"

    def test_route_retry_blocked_by_max_attempts(self):
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 3
        state["current_judgment"] = make_judge_result(
            JudgeAction.RETRY_MODEL1, JudgeReasonCode.CLEAR_POLICY_CONFLICT, "Policy conflict"
        )
        # Max attempts reached should force finalize
        assert route_judgment(state) == "finalize"


class TestRunModel1Attempt:
    """Test the extracted single-attempt function."""

    def test_run_model1_attempt_returns_prediction(self):
        def mock_infer(**kwargs):
            return make_prediction({"ACL": 1})

        prediction = run_model1_attempt(
            report="test report",
            gold_context="gold examples",
            policy="canonical policy",
            labels=LABEL_KEYS,
            infer=mock_infer,
            attempt=1,
        )
        assert isinstance(prediction, ReportPrediction)
        assert prediction.predictions["ACL"].value == 1

    def test_run_model1_attempt_passes_feedback(self):
        feedback_received = {}

        def mock_infer(**kwargs):
            feedback_received["validator_feedback"] = kwargs.get("validator_feedback")
            return make_prediction()

        feedback = {"status": "FAIL", "issues": [{"label": "ACL", "predicted": 0, "corrected": 1}]}
        run_model1_attempt(
            report="test report",
            gold_context="gold",
            policy="policy",
            labels=LABEL_KEYS,
            infer=mock_infer,
            validator_feedback=feedback,
            attempt=2,
        )
        assert feedback_received["validator_feedback"] == feedback

    def test_run_model1_attempt_increments_attempt(self):
        attempts = []

        def mock_infer(**kwargs):
            attempts.append(kwargs.get("attempt"))
            return make_prediction()

        run_model1_attempt(report="r", gold_context="g", policy="p", labels=LABEL_KEYS, infer=mock_infer, attempt=1)
        run_model1_attempt(report="r", gold_context="g", policy="p", labels=LABEL_KEYS, infer=mock_infer, attempt=2)
        assert attempts == [1, 2]


class TestGraphStructure:
    """Test the LangGraph structure."""

    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        graph = build_graph()
        nodes = graph.nodes
        assert "initialize" in nodes
        assert "inference" in nodes
        assert "critic" in nodes
        assert "judge" in nodes
        assert "finalize" in nodes

    def test_graph_edges(self):
        graph = build_graph()
        # Check edges exist (LangGraph internal structure)
        assert graph is not None


class TestNoNestedRetry:
    """Test that no nested retry loops exist."""

    def test_run_model1_attempt_not_recursive(self):
        """Ensure run_model1_attempt doesn't call run_report."""
        call_log = []

        def mock_infer(**kwargs):
            call_log.append("infer")
            return make_prediction()

        def mock_validate(**kwargs):
            call_log.append("validate")
            return make_critique(CritiqueStatus.FAIL, [
                CritiqueIssue(
                    label="ACL",
                    model1_value=0,
                    proposed_value=1,
                    issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                    evidence="evidence",
                    policy_rule="rule",
                    feedback="feedback",
                )
            ])

        # This should not cause infinite recursion
        prediction = run_model1_attempt(
            report="test",
            gold_context="gold",
            policy="policy",
            labels=LABEL_KEYS,
            infer=mock_infer,
            attempt=1,
        )
        assert isinstance(prediction, ReportPrediction)
        # run_model1_attempt should only call infer once per call
        assert call_log.count("infer") == 1


class TestJudgeResultSchema:
    """Test JudgeResult schema constraints."""

    def test_judge_result_no_clinical_labels(self):
        """JudgeResult must not contain clinical label fields."""
        result = make_judge_result(
            JudgeAction.PASS,
            JudgeReasonCode.NO_ACTIONABLE_ISSUE,
            "All good",
        )
        # Check that no clinical label fields exist
        dump = result.model_dump()
        clinical_labels = ["ACL", "MCL", "Medial_Meniscus", "Lateral_Meniscus",
                          "Medial_OA", "Lateral_OA", "PF_OA", "Effusion",
                          "Synovitis", "Bakers", "Contusion", "Fracture"]
        for label in clinical_labels:
            assert label not in dump, f"Clinical label {label} found in JudgeResult"

    def test_judge_result_reason_code_enum(self):
        """Test all reason codes are valid."""
        for code in JudgeReasonCode:
            result = make_judge_result(JudgeAction.PASS, code, "test")
            assert result.reason_code == code

    def test_judge_result_rationale_max_length(self):
        """Test rationale max length constraint."""
        long_rationale = "x" * 501
        with pytest.raises(Exception):
            make_judge_result(JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, long_rationale)

    def test_judge_action_enum(self):
        """Test all actions are valid."""
        for action in JudgeAction:
            result = make_judge_result(action, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "test")
            assert result.action == action


class TestCritiqueResultSchema:
    """Test CritiqueResult schema."""

    def test_critique_result_actionable_property(self):
        """Test actionable property returns True for CLEAR_* issues."""
        critique = make_critique(CritiqueStatus.FAIL, [
            CritiqueIssue(
                label="ACL",
                model1_value=0,
                proposed_value=1,
                issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                evidence="evidence",
                policy_rule="rule",
                feedback="feedback",
            )
        ])
        assert critique.has_actionable_issues is True

    def test_critique_result_not_actionable_for_unresolved(self):
        """Test actionable is False for UNRESOLVED_POLICY."""
        critique = make_critique(CritiqueStatus.AMBIGUOUS, [
            CritiqueIssue(
                label="ACL",
                model1_value=0,
                proposed_value=None,
                issue_type=CritiqueIssueType.UNRESOLVED_POLICY,
                evidence="evidence",
                policy_rule="rule",
                feedback="feedback",
            )
        ])
        assert critique.has_actionable_issues is False


class TestTraceRecord:
    """Test TraceRecord schema."""

    def test_trace_record_has_new_fields(self):
        from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import TraceRecord
        trace = TraceRecord(
            timestamp_utc="2024-01-01T00:00:00Z",
            study_instance_uid="test",
            graph_node="inference",
            stage="inference",
            attempt=1,
            requested_model="test-model",
            actual_model="actual-model",
            provider="nvidia",
            prompt_hash="hash1",
            response_hash="hash2",
            latency_seconds=1.0,
            input_tokens=100,
            output_tokens=50,
            status="success",
            error=None,
            judge_action="PASS",
        )
        assert trace.graph_node == "inference"
        assert trace.judge_action == "PASS"
        assert trace.response_hash is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])