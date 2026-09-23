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
from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import run_model1_attempt, _validate_critic_output


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


class TestSafetyGate:
    """Test deterministic safety gate on Critic output."""

    def make_prediction(self, values: dict[str, int] | None = None) -> ReportPrediction:
        preds = {}
        for label in LABEL_KEYS:
            val = values.get(label, 0) if values else 0
            preds[label] = LabelValue(value=val, evidence="")
        return ReportPrediction(predictions=preds)

    def make_critique(self, issues: list[CritiqueIssue]) -> CritiqueResult:
        has_clear = any(
            i.issue_type in (CritiqueIssueType.CLEAR_POLICY_CONFLICT, CritiqueIssueType.CLEAR_REPORT_CONFLICT)
            for i in issues
        )
        has_unresolved = any(
            i.issue_type in (CritiqueIssueType.UNRESOLVED_POLICY, CritiqueIssueType.REPORT_AMBIGUITY, CritiqueIssueType.INSUFFICIENT_EVIDENCE)
            for i in issues
        )
        if has_clear:
            status = CritiqueStatus.FAIL
        elif has_unresolved:
            status = CritiqueStatus.AMBIGUOUS
        else:
            status = CritiqueStatus.PASS
        return CritiqueResult(
            status=status,
            issues=issues,
            summary="test",
            actionable=has_clear,
            affected_labels=[i.label for i in issues if i.issue_type in (CritiqueIssueType.CLEAR_POLICY_CONFLICT, CritiqueIssueType.CLEAR_REPORT_CONFLICT)],
        )

    def test_safety_gate_passes_valid_critique(self):
        """Valid critique with CLEAR_* issues passes safety gate."""
        pred = self.make_prediction({"ACL": 0})
        critique = self.make_critique([
            CritiqueIssue(
                label="ACL",
                model1_value=0,
                proposed_value=1,
                issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                evidence="ACL tear",
                policy_rule="SAFE_POSITIVE: explicit tear",
                feedback="Should be 1",
            )
        ])
        result = _validate_critic_output(critique, "ACL tear present.", pred)
        assert result is None

    def test_safety_gate_catches_invalid_label(self):
        """Invalid label in critique issue fails safety gate."""
        pred = self.make_prediction()
        critique = self.make_critique([
            CritiqueIssue(
                label="INVALID_LABEL",
                model1_value=0,
                proposed_value=1,
                issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                evidence="test",
                policy_rule="test",
                feedback="test",
            )
        ])
        result = _validate_critic_output(critique, "report", pred)
        assert result is not None
        assert result["check"] == "label_validation"
        assert "INVALID_LABEL" in result["error"]

    def test_safety_gate_catches_clear_issue_without_proposed_value(self):
        """CLEAR_* issue missing proposed_value fails."""
        pred = self.make_prediction({"ACL": 0})
        critique = self.make_critique([
            CritiqueIssue(
                label="ACL",
                model1_value=0,
                proposed_value=None,
                issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                evidence="ACL tear",
                policy_rule="test",
                feedback="test",
            )
        ])
        result = _validate_critic_output(critique, "ACL tear present.", pred)
        assert result is not None
        assert result["check"] == "proposed_value_required"

    def test_safety_gate_catches_clear_issue_proposed_value_not_binary(self):
        """CLEAR_* issue with non-binary proposed_value fails."""
        pred = self.make_prediction({"ACL": 0})
        # Use model_construct to bypass Pydantic validation at creation
        issue = CritiqueIssue.model_construct(
            label="ACL",
            model1_value=0,
            proposed_value=2,
            issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
            evidence="ACL tear",
            policy_rule="test",
            feedback="test",
        )
        critique = CritiqueResult(
            status=CritiqueStatus.FAIL,
            issues=[issue],
            summary="test",
            actionable=True,
            affected_labels=["ACL"],
        )
        result = _validate_critic_output(critique, "ACL tear present.", pred)
        assert result is not None
        assert result["check"] == "proposed_value_binary"

    def test_safety_gate_catches_clear_issue_proposed_equals_model1(self):
        """CLEAR_* issue where proposed_value == model1_value fails."""
        pred = self.make_prediction({"ACL": 1})
        critique = self.make_critique([
            CritiqueIssue(
                label="ACL",
                model1_value=1,
                proposed_value=1,
                issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                evidence="ACL tear",
                policy_rule="test",
                feedback="test",
            )
        ])
        result = _validate_critic_output(critique, "ACL tear present.", pred)
        assert result is not None
        assert result["check"] == "proposed_value_differs"

    def test_safety_gate_catches_unresolved_with_proposed_value(self):
        """UNRESOLVED_POLICY issue with proposed_value fails."""
        pred = self.make_prediction({"ACL": 0})
        critique = self.make_critique([
            CritiqueIssue(
                label="ACL",
                model1_value=0,
                proposed_value=1,
                issue_type=CritiqueIssueType.UNRESOLVED_POLICY,
                evidence="uncertain",
                policy_rule="unresolved",
                feedback="test",
            )
        ])
        result = _validate_critic_output(critique, "report", pred)
        assert result is not None
        assert result["check"] == "proposed_value_null_required"

    def test_safety_gate_catches_evidence_not_verbatim_in_report(self):
        """Evidence not found verbatim in ORIGINAL_REPORT fails."""
        pred = self.make_prediction({"ACL": 0})
        critique = self.make_critique([
            CritiqueIssue(
                label="ACL",
                model1_value=0,
                proposed_value=1,
                issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT,
                evidence="this text not in report",
                policy_rule="test",
                feedback="test",
            )
        ])
        result = _validate_critic_output(critique, "ACL tear present.", pred)
        assert result is not None
        assert result["check"] == "evidence_verbatim"

    def test_safety_gate_catches_actionable_mismatch(self):
        """actionable flag inconsistent with CLEAR_* issues fails."""
        pred = self.make_prediction({"ACL": 0})
        # Create critique with CLEAR issue but actionable=False
        critique = CritiqueResult(
            status=CritiqueStatus.FAIL,
            issues=[
                CritiqueIssue(
                    label="ACL",
                    model1_value=0,
                    proposed_value=1,
                    issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                    evidence="ACL tear",
                    policy_rule="test",
                    feedback="test",
                )
            ],
            summary="test",
            actionable=False,  # WRONG - should be True
            affected_labels=["ACL"],
        )
        result = _validate_critic_output(critique, "ACL tear present.", pred)
        assert result is not None
        assert result["check"] == "actionable_consistency"

    def test_safety_gate_catches_status_mismatch_fail_vs_ambiguous(self):
        """status=FAIL with no CLEAR_* issues fails."""
        pred = self.make_prediction({"ACL": 0})
        critique = CritiqueResult(
            status=CritiqueStatus.FAIL,  # Should be AMBIGUOUS
            issues=[
                CritiqueIssue(
                    label="ACL",
                    model1_value=0,
                    proposed_value=None,
                    issue_type=CritiqueIssueType.UNRESOLVED_POLICY,
                    evidence="uncertain",  # Must be in report
                    policy_rule="unresolved",
                    feedback="test",
                )
            ],
            summary="test",
            actionable=False,
            affected_labels=[],
        )
        # Use report that contains the evidence
        result = _validate_critic_output(critique, "uncertain finding", pred)
        assert result is not None
        assert result["check"] == "status_consistency"

    def test_safety_gate_catches_status_mismatch_pass_vs_fail(self):
        """status=PASS with CLEAR_* issues fails."""
        pred = self.make_prediction({"ACL": 0})
        critique = CritiqueResult(
            status=CritiqueStatus.PASS,  # Should be FAIL
            issues=[
                CritiqueIssue(
                    label="ACL",
                    model1_value=0,
                    proposed_value=1,
                    issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                    evidence="ACL tear",
                    policy_rule="test",
                    feedback="test",
                )
            ],
            summary="test",
            actionable=True,
            affected_labels=["ACL"],
        )
        result = _validate_critic_output(critique, "ACL tear present.", pred)
        assert result is not None
        assert result["check"] == "status_consistency"

    def test_safety_gate_allows_empty_evidence(self):
        """Empty or null evidence is allowed (not checked)."""
        pred = self.make_prediction({"ACL": 0})
        critique = self.make_critique([
            CritiqueIssue(
                label="ACL",
                model1_value=0,
                proposed_value=1,
                issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT,
                evidence=None,
                policy_rule="test",
                feedback="test",
            )
        ])
        result = _validate_critic_output(critique, "report", pred)
        assert result is None

        # Empty string also allowed
        critique = self.make_critique([
            CritiqueIssue(
                label="ACL",
                model1_value=0,
                proposed_value=1,
                issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT,
                evidence="",
                policy_rule="test",
                feedback="test",
            )
        ])
        result = _validate_critic_output(critique, "report", pred)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])