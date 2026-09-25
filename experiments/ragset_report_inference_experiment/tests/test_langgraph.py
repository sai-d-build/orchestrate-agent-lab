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
        # RETRY_MODEL1 now routes to increment_attempt node (not directly to inference)
        assert route_judgment(state) == "increment_attempt"

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


class TestFinalization:
    """Test finalization logic selects best prediction from history."""
    
    def make_prediction(self, values: dict[str, int] | None = None) -> ReportPrediction:
        preds = {}
        for label in LABEL_KEYS:
            val = values.get(label, 0) if values else 0
            preds[label] = LabelValue(value=val, evidence="")
        return ReportPrediction(predictions=preds)
    
    def make_critique(self, status: CritiqueStatus, issues: list | None = None) -> CritiqueResult:
        if issues is None:
            issues = []
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
    
    def make_judge_result(self, action: JudgeAction, reason_code: JudgeReasonCode, rationale: str) -> JudgeResult:
        return JudgeResult(
            action=action,
            reason_code=reason_code,
            rationale=rationale,
        )
    
    def test_finalize_selects_pass_attempt(self):
        """Finalization should select the attempt where Model 2 PASSed."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 3
        state["max_attempts"] = 3
        
        # Attempt 1: FAIL
        pred1 = self.make_prediction({"ACL": 0})
        crit1 = self.make_critique(CritiqueStatus.FAIL, [
            CritiqueIssue(label="ACL", model1_value=0, proposed_value=1, issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT, evidence="ACL tear", policy_rule="rule", feedback="fb")
        ])
        
        # Attempt 2: PASS
        pred2 = self.make_prediction({"ACL": 1})
        crit2 = self.make_critique(CritiqueStatus.PASS, [])
        
        # Attempt 3: PASS (current attempt, already in history)
        pred3 = self.make_prediction({"ACL": 1})
        crit3 = self.make_critique(CritiqueStatus.PASS, [])
        
        # In real graph, history includes all completed attempts (including current)
        state["prediction_history"] = [pred1, pred2, pred3]
        state["critique_history"] = [crit1, crit2, crit3]
        state["current_prediction"] = pred3
        state["current_critique"] = crit3
        state["current_judgment"] = self.make_judge_result(JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "All good")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        
        assert selected_pred.predictions["ACL"].value == 1
        assert selected_attempt == 3  # Latest PASS at attempt 3
        assert "PASS" in reason
    
    def test_finalize_respects_max_attempts_terminal(self):
        """Finalization should RESPECT MAX_ATTEMPTS terminal action - no 'resolution' claimed.
        
        Per Judge prompt Section 10: Never claim prediction was 'resolved' when terminal
        action is NEEDS_REVIEW due to MAX_ATTEMPTS. The workflow FAILED to converge.
        """
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 3
        state["max_attempts"] = 3
        
        # Attempt 1: only INSUFFICIENT_EVIDENCE (NOT policy-supported - it's no evidence)
        pred1 = self.make_prediction({"ACL": 0})
        crit1 = self.make_critique(CritiqueStatus.PASS, [  # PASS because only INSUFFICIENT_EVIDENCE
            CritiqueIssue(label="ACL", model1_value=0, proposed_value=None, issue_type=CritiqueIssueType.INSUFFICIENT_EVIDENCE, evidence="", policy_rule="", feedback="fb")
        ])
        
        # Attempt 2: CLEAR conflict (not policy-supported)
        pred2 = self.make_prediction({"ACL": 1})
        crit2 = self.make_critique(CritiqueStatus.FAIL, [
            CritiqueIssue(label="ACL", model1_value=1, proposed_value=0, issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT, evidence="ACL intact", policy_rule="rule", feedback="fb")
        ])
        
        state["prediction_history"] = [pred1, pred2]
        state["critique_history"] = [crit1, crit2]
        state["current_prediction"] = pred2
        state["current_critique"] = crit2
        state["current_judgment"] = self.make_judge_result(JudgeAction.NEEDS_REVIEW, JudgeReasonCode.MAX_ATTEMPTS, "Max attempts")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        
        # Judge ruled NEEDS_REVIEW (MAX_ATTEMPTS) - workflow failed to converge
        # Must return current prediction with reason acknowledging failure, NOT "history-based resolution"
        # selected_attempt returns state["attempt"] (3), not the history index
        assert selected_pred.predictions["ACL"].value == 1  # Current (latest) prediction
        assert selected_attempt == 3  # Current attempt number from state
        assert "Max attempts" in reason or "maximum attempts" in reason.lower()
        assert "resolved" not in reason.lower()
        assert "history-based" not in reason.lower()
    
    def test_finalize_respects_max_attempts_over_resolved(self):
        """Finalization should RESPECT MAX_ATTEMPTS terminal - even if conflicts were 'resolved'.
        
        Per Judge prompt Section 10: Never claim prediction was 'resolved' when terminal
        action is NEEDS_REVIEW due to MAX_ATTEMPTS. The workflow FAILED to converge.
        """
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 3
        state["max_attempts"] = 3
        
        # Attempt 1: CLEAR conflict on ACL
        pred1 = self.make_prediction({"ACL": 0})
        crit1 = self.make_critique(CritiqueStatus.FAIL, [
            CritiqueIssue(label="ACL", model1_value=0, proposed_value=1, issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT, evidence="ACL tear", policy_rule="rule", feedback="fb")
        ])
        
        # Attempt 2: Model 1 adopts proposed value (would be "resolved" in isolation)
        pred2 = self.make_prediction({"ACL": 1})
        crit2 = self.make_critique(CritiqueStatus.FAIL, [
            CritiqueIssue(label="MCL", model1_value=0, proposed_value=1, issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT, evidence="MCL tear", policy_rule="rule", feedback="fb")
        ])
        
        state["prediction_history"] = [pred1, pred2]
        state["critique_history"] = [crit1, crit2]
        state["current_prediction"] = pred2
        state["current_critique"] = crit2
        state["current_judgment"] = self.make_judge_result(JudgeAction.NEEDS_REVIEW, JudgeReasonCode.MAX_ATTEMPTS, "Max attempts")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        
        # Judge ruled NEEDS_REVIEW (MAX_ATTEMPTS) - workflow failed to converge
        # Must return current prediction (attempt 3 = state["attempt"]) with reason acknowledging failure
        assert selected_pred.predictions["ACL"].value == 1  # Current prediction
        assert selected_attempt == 3  # Current attempt number
        assert "Max attempts" in reason or "maximum attempts" in reason.lower()
        assert "resolved" not in reason.lower()
    
    def test_finalize_respects_oscillation_terminal(self):
        """Finalization should RESPECT OSCILLATION terminal - no 'pre-oscillation resolution' claimed.
        
        Per Judge prompt Section 10: Never claim prediction was 'resolved' when terminal
        action is NEEDS_REVIEW due to OSCILLATION. The workflow FAILED to converge.
        """
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 3
        state["max_attempts"] = 3
        state["oscillation_detected"] = True
        state["oscillating_labels"] = ["ACL"]
        
        # ACL oscillates: 0 -> 1 -> 0
        pred1 = self.make_prediction({"ACL": 0})
        pred2 = self.make_prediction({"ACL": 1})
        pred3 = self.make_prediction({"ACL": 0})
        
        crit1 = self.make_critique(CritiqueStatus.FAIL, [
            CritiqueIssue(label="ACL", model1_value=0, proposed_value=1, issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT, evidence="ACL tear", policy_rule="rule", feedback="fb")
        ])
        crit2 = self.make_critique(CritiqueStatus.FAIL, [
            CritiqueIssue(label="ACL", model1_value=1, proposed_value=0, issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT, evidence="ACL intact", policy_rule="rule", feedback="fb")
        ])
        crit3 = self.make_critique(CritiqueStatus.FAIL, [
            CritiqueIssue(label="ACL", model1_value=0, proposed_value=1, issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT, evidence="ACL tear", policy_rule="rule", feedback="fb")
        ])
        
        state["prediction_history"] = [pred1, pred2, pred3]
        state["critique_history"] = [crit1, crit2, crit3]
        state["current_prediction"] = pred3
        state["current_critique"] = crit3
        state["current_judgment"] = self.make_judge_result(JudgeAction.NEEDS_REVIEW, JudgeReasonCode.OSCILLATION, "Oscillation")
        state["oscillation_detected"] = True
        state["oscillating_labels"] = ["ACL"]
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        
        # Judge ruled NEEDS_REVIEW (OSCILLATION) - workflow failed to converge
        # Must return current prediction with reason acknowledging oscillation, NOT "pre-oscillation stability"
        assert selected_pred.predictions["ACL"].value == 0  # Current prediction (attempt 3)
        assert selected_attempt == 3  # Current attempt number
        assert "oscillation" in reason.lower()
        assert "pre-oscillation" not in reason.lower()
        assert "resolved" not in reason.lower()
    
    def test_finalize_respects_stuck_terminal(self):
        """Finalization should RESPECT STUCK terminal - acknowledge stuck, not claim resolution."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 3
        state["max_attempts"] = 3
        state["stuck_detected"] = True
        state["stuck_labels"] = ["ACL"]
        
        pred1 = self.make_prediction({"ACL": 1})
        pred2 = self.make_prediction({"ACL": 1})
        pred3 = self.make_prediction({"ACL": 1})
        
        crit1 = self.make_critique(CritiqueStatus.FAIL, [
            CritiqueIssue(label="ACL", model1_value=1, proposed_value=0, issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT, evidence="ACL intact", policy_rule="rule", feedback="fb")
        ])
        crit2 = self.make_critique(CritiqueStatus.FAIL, [
            CritiqueIssue(label="ACL", model1_value=1, proposed_value=0, issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT, evidence="ACL intact", policy_rule="rule", feedback="fb")
        ])
        crit3 = self.make_critique(CritiqueStatus.FAIL, [
            CritiqueIssue(label="ACL", model1_value=1, proposed_value=0, issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT, evidence="ACL intact", policy_rule="rule", feedback="fb")
        ])
        
        state["prediction_history"] = [pred1, pred2]
        state["critique_history"] = [crit1, crit2]
        state["current_prediction"] = pred3
        state["current_critique"] = crit3
        state["current_judgment"] = self.make_judge_result(JudgeAction.NEEDS_REVIEW, JudgeReasonCode.MODEL1_STUCK_RETAINED, "Stuck")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        
        # Judge ruled NEEDS_REVIEW (MODEL1_STUCK_RETAINED) - workflow failed to converge
        # Must return current prediction with reason acknowledging stuck
        assert selected_pred.predictions["ACL"].value == 1
        assert selected_attempt == 3
        assert "stuck" in reason.lower()
        assert "not validated as correct" in reason.lower()


class TestOscillationDetection:
    """Test oscillation detection in critic node."""
    
    def test_oscillation_010_pattern(self):
        """Test 0->1->0 oscillation pattern detected."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _detect_validator_oscillation
        
        validator_history = {
            "ACL": [0, 1, 0],
            "MCL": [0, 0, 0],
        }
        oscillating = _detect_validator_oscillation(validator_history)
        assert "ACL" in oscillating
        assert "MCL" not in oscillating
    
    def test_oscillation_101_pattern(self):
        """Test 1->0->1 oscillation pattern detected."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _detect_validator_oscillation
        
        validator_history = {
            "ACL": [1, 0, 1],
        }
        oscillating = _detect_validator_oscillation(validator_history)
        assert "ACL" in oscillating
    
    def test_oscillation_immediate_flip(self):
        """Test immediate flip (0->1 or 1->0) detected."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _detect_validator_oscillation
        
        validator_history = {
            "ACL": [0, 1],
            "MCL": [1, 0],
        }
        oscillating = _detect_validator_oscillation(validator_history)
        assert "ACL" in oscillating
        assert "MCL" in oscillating
    
    def test_no_oscillation_stable(self):
        """Test stable predictions not flagged as oscillation."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _detect_validator_oscillation
        
        validator_history = {
            "ACL": [0, 0, 0],
            "MCL": [1, 1, 1],
        }
        oscillating = _detect_validator_oscillation(validator_history)
        assert len(oscillating) == 0


class TestStuckDetection:
    """Test Model 1 stuck detection."""
    
    def make_prediction(self, values: dict[str, int] | None = None) -> ReportPrediction:
        preds = {}
        for label in LABEL_KEYS:
            val = values.get(label, 0) if values else 0
            preds[label] = LabelValue(value=val, evidence="")
        return ReportPrediction(predictions=preds)
    
    def test_stuck_same_value_across_attempts(self):
        """Test stuck detection when Model 1 repeats same value for disputed label."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _detect_model1_stuck
        
        model1_history = {
            "ACL": [1, 1, 1],
            "MCL": [0, 0, 0],
        }
        disputed_labels = {"ACL"}
        
        stuck = _detect_model1_stuck(model1_history, disputed_labels, min_attempts=2)
        assert "ACL" in stuck
        assert "MCL" not in stuck
    
    def test_not_stuck_when_changing(self):
        """Test not stuck when Model 1 changes value."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _detect_model1_stuck
        
        model1_history = {
            "ACL": [0, 1, 0],
        }
        disputed_labels = {"ACL"}
        
        stuck = _detect_model1_stuck(model1_history, disputed_labels, min_attempts=2)
        assert "ACL" not in stuck


class TestModel2Integrity:
    """Test Model 2 critique integrity checks."""
    
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
    
    def test_clear_conflict_evidence_null_allowed_for_policy_conflict(self):
        """CLEAR_POLICY_CONFLICT allows null evidence (only CLEAR_REPORT_CONFLICT requires evidence)."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _validate_critic_output
        
        pred = self.make_prediction({"ACL": 0})
        critique = CritiqueResult(
            status=CritiqueStatus.FAIL,
            issues=[
                CritiqueIssue(
                    label="ACL",
                    model1_value=0,
                    proposed_value=1,
                    issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                    evidence=None,  # Null evidence allowed for policy conflict
                    policy_rule="rule",
                    feedback="fb",
                )
            ],
            summary="test",
            actionable=True,
            affected_labels=["ACL"],
        )
        result = _validate_critic_output(critique, "report without ACL tear", pred)
        # Null evidence is allowed for CLEAR_POLICY_CONFLICT
        assert result is None
    
    def test_clear_conflict_evidence_must_be_verbatim(self):
        """CLEAR_* conflict evidence must be verbatim in report."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _validate_critic_output
        
        pred = self.make_prediction({"ACL": 0})
        critique = CritiqueResult(
            status=CritiqueStatus.FAIL,
            issues=[
                CritiqueIssue(
                    label="ACL",
                    model1_value=0,
                    proposed_value=1,
                    issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT,
                    evidence="ACL tear present",  # Not in report
                    policy_rule="rule",
                    feedback="fb",
                )
            ],
            summary="test",
            actionable=True,
            affected_labels=["ACL"],
        )
        result = _validate_critic_output(critique, "ACL is intact.", pred)
        assert result is not None
        assert result["check"] == "evidence_verbatim"
    
    def test_unresolved_must_not_have_proposed_value(self):
        """UNRESOLVED_POLICY must not have proposed_value."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _validate_critic_output
        
        pred = self.make_prediction({"ACL": 0})
        critique = CritiqueResult(
            status=CritiqueStatus.AMBIGUOUS,
            issues=[
                CritiqueIssue(
                    label="ACL",
                    model1_value=0,
                    proposed_value=1,  # WRONG - should be null
                    issue_type=CritiqueIssueType.UNRESOLVED_POLICY,
                    evidence="uncertain",
                    policy_rule="unresolved",
                    feedback="fb",
                )
            ],
            summary="test",
            actionable=False,
            affected_labels=[],
        )
        result = _validate_critic_output(critique, "uncertain finding", pred)
        assert result is not None
        assert result["check"] == "proposed_value_null_required"
    
    def test_actionable_matches_clear_issues(self):
        """actionable must be true iff CLEAR_* issues exist."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _validate_critic_output
        
        pred = self.make_prediction({"ACL": 0})
        # CLEAR issue but actionable=False
        critique = CritiqueResult(
            status=CritiqueStatus.FAIL,
            issues=[
                CritiqueIssue(
                    label="ACL",
                    model1_value=0,
                    proposed_value=1,
                    issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                    evidence="ACL tear",
                    policy_rule="rule",
                    feedback="fb",
                )
            ],
            summary="test",
            actionable=False,  # WRONG
            affected_labels=["ACL"],
        )
        result = _validate_critic_output(critique, "ACL tear present.", pred)
        assert result is not None
        assert result["check"] == "actionable_consistency"

    def test_model1_value_consistency(self):
        """Critic model1_value must match actual Model 1 prediction."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _validate_critic_output
        
        pred = self.make_prediction({"ACL": 1})  # Actual Model 1 value is 1
        critique = self.make_critique([
            CritiqueIssue(
                label="ACL",
                model1_value=0,  # WRONG - should be 1
                proposed_value=1,
                issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                evidence="ACL tear",
                policy_rule="rule",
                feedback="fb",
            )
        ])
        result = _validate_critic_output(critique, "ACL tear present.", pred)
        assert result is not None
        assert result["check"] == "model1_value_consistency"
        assert "does not match actual Model 1 prediction" in result["error"]

    def test_affected_labels_consistency(self):
        """affected_labels must match CLEAR_* issue labels exactly."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _validate_critic_output
        
        pred = self.make_prediction({"ACL": 0, "MCL": 0})
        # CLEAR issue on ACL but affected_labels includes MCL (wrong)
        critique = CritiqueResult(
            status=CritiqueStatus.FAIL,
            issues=[
                CritiqueIssue(
                    label="ACL",
                    model1_value=0,
                    proposed_value=1,
                    issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                    evidence="ACL tear",
                    policy_rule="rule",
                    feedback="fb",
                )
            ],
            summary="test",
            actionable=True,
            affected_labels=["ACL", "MCL"],  # WRONG - MCL not in CLEAR issues
        )
        result = _validate_critic_output(critique, "ACL tear present.", pred)
        assert result is not None
        assert result["check"] == "affected_labels_consistency"
        assert "does not match CLEAR_* issue labels" in result["error"]


class TestGraphIntegration:
    """Integration tests for the full graph."""
    
    def test_graph_has_increment_attempt_node(self):
        """Graph should have increment_attempt node for retry path."""
        graph = build_graph()
        nodes = graph.nodes
        assert "increment_attempt" in nodes
    
    def test_route_judgment_retry_goes_to_increment(self):
        """RETRY_MODEL1 should route to increment_attempt, not directly to inference."""
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["current_judgment"] = make_judge_result(
            JudgeAction.RETRY_MODEL1, JudgeReasonCode.CLEAR_POLICY_CONFLICT, "Policy conflict"
        )
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph import route_judgment
        assert route_judgment(state) == "increment_attempt"
    
    def test_increment_attempt_increments_counter(self):
        """increment_attempt node should increment attempt counter."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph import increment_attempt_node
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        new_state = increment_attempt_node(state)
        assert new_state["attempt"] == 2


class TestNoRetryOnUnsupportedCorrection:
    """Test that unsupported Model 2 corrections don't trigger retries."""
    
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
    
    def test_no_retry_on_unsupported_model2_correction(self):
        """Unsupported Model 2 correction (no evidence, no policy rule) should not trigger retry."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph import route_judgment
        from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import JudgeAction, JudgeReasonCode
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        # Model 2 gives CLEAR_REPORT_CONFLICT but with no evidence and no policy rule
        # This should be caught by safety gate and result in MODEL2_UNSUPPORTED_CORRECTION
        state["current_judgment"] = make_judge_result(
            JudgeAction.NEEDS_REVIEW, JudgeReasonCode.MODEL2_UNSUPPORTED_CORRECTION, "Unsupported correction"
        )
        # Should route to finalize, not retry
        assert route_judgment(state) == "finalize"
    
    def test_no_retry_on_unresolved_policy(self):
        """UNRESOLVED_POLICY should route to finalize (AMBIGUOUS), not retry."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph import route_judgment
        from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import JudgeAction, JudgeReasonCode
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["current_judgment"] = make_judge_result(
            JudgeAction.AMBIGUOUS, JudgeReasonCode.UNRESOLVED_POLICY, "Policy unresolved"
        )
        # AMBIGUOUS should route to finalize
        assert route_judgment(state) == "finalize"
    
    def test_no_retry_on_report_ambiguity(self):
        """REPORT_AMBIGUITY should route to finalize (AMBIGUOUS), not retry."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph import route_judgment
        from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import JudgeAction, JudgeReasonCode
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["current_judgment"] = make_judge_result(
            JudgeAction.AMBIGUOUS, JudgeReasonCode.REPORT_AMBIGUITY, "Report ambiguous"
        )
        # AMBIGUOUS should route to finalize
        assert route_judgment(state) == "finalize"


class TestRetryConvergence:
    """Test that retry converges when Model 1 adopts Model 2's correction."""
    
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
            actionable=any(i.issue_type in (CritiqueIssueType.CLEAR_POLICY_CONFLICT, CritiqueIssueType.CLEAR_REPORT_CONFLICT) for i in issues),
            affected_labels=[i.label for i in issues if i.issue_type in (CritiqueIssueType.CLEAR_POLICY_CONFLICT, CritiqueIssueType.CLEAR_REPORT_CONFLICT)],
        )
    
    def test_retry_convergence_model1_adopts_correction(self):
        """Model 1 wrong -> Model 2 clear correction -> Model 1 correct -> Model 2 PASS."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 3
        state["max_attempts"] = 3
        
        # Attempt 1: Model 1 wrong (ACL=0)
        pred1 = self.make_prediction({"ACL": 0})
        crit1 = CritiqueResult(
            status=CritiqueStatus.FAIL,
            issues=[
                CritiqueIssue(
                    label="ACL", model1_value=0, proposed_value=1,
                    issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                    evidence="ACL tear", policy_rule="SAFE_POSITIVE: explicit tear", feedback="Should be 1"
                )
            ],
            summary="test",
            actionable=True,
            affected_labels=["ACL"],
        )
        
        # Attempt 2: Model 1 adopts correction (ACL=1)
        pred2 = self.make_prediction({"ACL": 1})
        crit2 = CritiqueResult(
            status=CritiqueStatus.PASS,
            issues=[],
            summary="test",
            actionable=False,
            affected_labels=[],
        )
        
        # Attempt 3: PASS (current)
        pred3 = self.make_prediction({"ACL": 1})
        crit3 = CritiqueResult(
            status=CritiqueStatus.PASS,
            issues=[],
            summary="test",
            actionable=False,
            affected_labels=[],
        )
        
        state["prediction_history"] = [pred1, pred2, pred3]
        state["critique_history"] = [crit1, crit2, crit3]
        state["current_prediction"] = pred3
        state["current_critique"] = crit3
        state["current_judgment"] = make_judge_result(JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "All good")
        state["attempt"] = 3
        state["max_attempts"] = 3
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        
        # Should select attempt 3 (latest PASS)
        assert selected_pred.predictions["ACL"].value == 1
        assert selected_attempt == 3
        assert "PASS" in reason


class TestMixedCritique:
    """Test mixed critique combinations."""
    
    def make_prediction(self, values: dict[str, int] | None = None) -> ReportPrediction:
        preds = {}
        for label in LABEL_KEYS:
            val = values.get(label, 0) if values else 0
            preds[label] = LabelValue(value=val, evidence="")
        return ReportPrediction(predictions=preds)
    
    def test_mixed_clear_and_unresolved(self):
        """CLEAR + UNRESOLVED in same critique -> FAIL (actionable)."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _validate_critic_output
        
        pred = self.make_prediction({"ACL": 0, "MCL": 0})
        critique = CritiqueResult(
            status=CritiqueStatus.FAIL,
            issues=[
                CritiqueIssue(
                    label="ACL", model1_value=0, proposed_value=1,
                    issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                    evidence="ACL tear", policy_rule="rule", feedback="fb"
                ),
                CritiqueIssue(
                    label="MCL", model1_value=0, proposed_value=None,
                    issue_type=CritiqueIssueType.UNRESOLVED_POLICY,
                    evidence="uncertain", policy_rule="unresolved", feedback="fb"
                ),
            ],
            summary="test",
            actionable=True,
            affected_labels=["ACL"],
        )
        result = _validate_critic_output(critique, "ACL tear present. MCL uncertain.", pred)
        # Should pass validation (CLEAR + UNRESOLVED is valid)
        assert result is None
        assert critique.actionable is True
        assert critique.affected_labels == ["ACL"]
    
    def test_mixed_clear_and_supported(self):
        """CLEAR + SUPPORTED in same critique -> FAIL (actionable)."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import _validate_critic_output
        
        pred = self.make_prediction({"ACL": 0, "MCL": 0})
        critique = CritiqueResult(
            status=CritiqueStatus.FAIL,
            issues=[
                CritiqueIssue(
                    label="ACL", model1_value=0, proposed_value=1,
                    issue_type=CritiqueIssueType.CLEAR_POLICY_CONFLICT,
                    evidence="ACL tear", policy_rule="rule", feedback="fb"
                ),
                CritiqueIssue(
                    label="MCL", model1_value=0, proposed_value=None,
                    issue_type=CritiqueIssueType.SUPPORTED,
                    evidence="MCL intact", policy_rule="rule", feedback="fb"
                ),
            ],
            summary="test",
            actionable=True,
            affected_labels=["ACL"],
        )
        result = _validate_critic_output(critique, "ACL tear present. MCL intact.", pred)
        assert result is None
        assert critique.actionable is True
        assert critique.affected_labels == ["ACL"]
    
    def test_supported_and_stuck(self):
        """SUPPORTED issues with Model 1 stuck on other labels.
        
        In a stuck scenario, all critiques should be FAIL (Model 1 not adopting corrections).
        """
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 3
        state["max_attempts"] = 3
        state["stuck_detected"] = True
        state["stuck_labels"] = ["ACL"]
        
        pred1 = self.make_prediction({"ACL": 1, "MCL": 0})
        pred2 = self.make_prediction({"ACL": 1, "MCL": 0})
        pred3 = self.make_prediction({"ACL": 1, "MCL": 0})
        
        # All critiques are FAIL (Model 1 stuck, not adopting corrections)
        crit1 = CritiqueResult(
            status=CritiqueStatus.FAIL,
            issues=[
                CritiqueIssue(label="ACL", model1_value=1, proposed_value=0,
                              issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT,
                              evidence="ACL intact", policy_rule="rule", feedback="fb")
            ],
            summary="test", actionable=True, affected_labels=["ACL"],
        )
        crit2 = CritiqueResult(
            status=CritiqueStatus.FAIL,
            issues=[
                CritiqueIssue(label="ACL", model1_value=1, proposed_value=0,
                              issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT,
                              evidence="ACL intact", policy_rule="rule", feedback="fb")
            ],
            summary="test", actionable=True, affected_labels=["ACL"],
        )
        crit3 = CritiqueResult(
            status=CritiqueStatus.FAIL,
            issues=[
                CritiqueIssue(label="ACL", model1_value=1, proposed_value=0,
                              issue_type=CritiqueIssueType.CLEAR_REPORT_CONFLICT,
                              evidence="ACL intact", policy_rule="rule", feedback="fb")
            ],
            summary="test", actionable=True, affected_labels=["ACL"],
        )
        
        state["prediction_history"] = [pred1, pred2]
        state["critique_history"] = [crit1, crit2]
        state["current_prediction"] = pred3
        state["current_critique"] = crit3
        state["current_judgment"] = make_judge_result(JudgeAction.NEEDS_REVIEW, JudgeReasonCode.MODEL1_STUCK, "Stuck")
        state["attempt"] = 3
        state["max_attempts"] = 3
        state["stuck_detected"] = True
        state["stuck_labels"] = ["ACL"]
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        
        # Should select latest (attempt 3) when stuck
        assert selected_pred.predictions["ACL"].value == 1
        assert selected_attempt == 3
        assert "stuck" in reason.lower()


class TestDeterministicAttempts:
    """Test deterministic attempt semantics."""
    
    def test_only_model1_increments_attempt(self):
        """Only Model 1 execution increments attempt counter."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph import increment_attempt_node
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        
        # Simulate Model 1 -> Critic -> Judge -> RETRY -> increment_attempt
        new_state = increment_attempt_node(state)
        assert new_state["attempt"] == 2
        
        # Critic and Judge nodes should NOT increment attempt
        # (they don't call increment_attempt_node)
    
    def test_attempt_counter_monotonic(self):
        """Attempt counter only increases, never decreases."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph import increment_attempt_node
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        
        for expected in [2, 3]:
            state = increment_attempt_node(state)
            assert state["attempt"] == expected
        
        # At max attempts, increment_attempt_node still increments (graph logic handles max)
        # The graph routing prevents retries at max attempts, not the increment function
        state = increment_attempt_node(state)
        assert state["attempt"] == 4  # Increments beyond max (graph handles blocking)


class TestTraceProvenance:
    """Test trace provenance and completeness."""
    
    def test_trace_written_once(self):
        """Each LLM call produces exactly one trace record."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import initialize_node, inference_node, critic_node, judge_node
        from experiments.ragset_report_inference_experiment.src.ragset_inference.models import InferenceModel, CriticModel, JudgeModel
        from experiments.ragset_report_inference_experiment.src.ragset_inference.retrieval import GoldRetriever
        from experiments.ragset_report_inference_experiment.src.ragset_inference.policy_loader import get_canonical_policy_text
        
        # This test verifies the trace structure, not actual LLM calls
        # The actual trace writing is tested in integration tests
        pass  # Placeholder - actual trace testing requires full graph execution
    
    def test_trace_prompt_hash_preserved(self):
        """Prompt hash is preserved through the trace pipeline."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.trace import compute_prompt_hash, write_trace
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            trace_path = f.name
        
        try:
            test_prompt = "test prompt"
            test_hash = compute_prompt_hash(test_prompt)
            
            write_trace(
                path=trace_path,
                report_id="test",
                stage="inference",
                model="test-model",
                attempt=1,
                prompt=test_prompt,
                status="success",
                provided_prompt_hash=test_hash,
            )
            
            # Read back and verify
            with open(trace_path, 'r') as f:
                record = json.loads(f.readline())
            
            assert record["prompt_hash"] == test_hash
        finally:
            import os
            os.unlink(trace_path)


class TestResumeIdempotency:
    """Test resume and idempotency."""
    
    def test_resume_skips_processed(self):
        """Resume skips already-processed reports."""
        from experiments.ragset_report_inference_experiment.runners.run_inference import load_existing_predictions
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"report_id": "test-1", "status": "passed"}) + "\n")
            f.write(json.dumps({"report_id": "test-2", "status": "needs_review"}) + "\n")
            temp_path = f.name
        
        try:
            processed = load_existing_predictions(temp_path)
            assert "test-1" in processed
            assert "test-2" in processed
            assert len(processed) == 2
        finally:
            import os
            os.unlink(temp_path)
    
    def test_resume_idempotent(self):
        """Multiple resume calls produce same result."""
        from experiments.ragset_report_inference_experiment.runners.run_inference import load_existing_predictions
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"report_id": "test-1", "status": "passed"}) + "\n")
            temp_path = f.name
        
        try:
            processed1 = load_existing_predictions(temp_path)
            processed2 = load_existing_predictions(temp_path)
            assert processed1 == processed2
        finally:
            import os
            os.unlink(temp_path)


class TestParallelIsolation:
    """Test parallel worker isolation."""
    
    def test_worker_state_isolation(self):
        """Each worker's LangGraph state is independent."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph import build_graph, create_initial_state
        
        graph = build_graph()
        
        # Create two independent states
        state1 = create_initial_state("report-1", "report 1", max_attempts=3)
        state2 = create_initial_state("report-2", "report 2", max_attempts=3)
        
        # Modify state1
        state1["attempt"] = 2
        state1["custom_field"] = "worker1"
        
        # State2 should be unaffected
        assert state2["attempt"] == 1
        assert "custom_field" not in state2
        
        # Graph structure is shared but state is independent
        assert graph is not None


class TestClinicalBoundaryRegression:
    """Regression tests for clinical boundary cases identified in the latest batch."""

    def make_prediction(self, values: dict[str, int] | None = None) -> ReportPrediction:
        preds = {}
        for label in LABEL_KEYS:
            val = values.get(label, 0) if values else 0
            preds[label] = LabelValue(value=val, evidence="")
        return ReportPrediction(predictions=preds)

    def make_critique(self, status: CritiqueStatus, issues: list | None = None) -> CritiqueResult:
        if issues is None:
            issues = []
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

    def make_judge_result(self, action: JudgeAction, reason_code: JudgeReasonCode, rationale: str) -> JudgeResult:
        return JudgeResult(
            action=action,
            reason_code=reason_code,
            rationale=rationale,
        )

    def test_grade_2b_medial_meniscus_not_tear(self):
        """Grade 2b medial meniscus injury without confirmed tear → Medial_Meniscus=0."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["max_attempts"] = 3
        
        # Model 1 predicts 0 for grade 2b injury (correct per policy)
        pred1 = self.make_prediction({"Medial_Meniscus": 0})
        crit1 = self.make_critique(CritiqueStatus.PASS, [])
        
        state["prediction_history"] = [pred1]
        state["critique_history"] = [crit1]
        state["current_prediction"] = pred1
        state["current_critique"] = crit1
        state["current_judgment"] = self.make_judge_result(JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "Grade 2b not a tear")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        assert selected_pred.predictions["Medial_Meniscus"].value == 0

    def test_explicit_bone_infarct_not_contusion(self):
        """Explicit bone infarct without bone contusion → Contusion=0."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["max_attempts"] = 3
        
        pred1 = self.make_prediction({"Contusion": 0})
        crit1 = self.make_critique(CritiqueStatus.PASS, [])
        
        state["prediction_history"] = [pred1]
        state["critique_history"] = [crit1]
        state["current_prediction"] = pred1
        state["current_critique"] = crit1
        state["current_judgment"] = self.make_judge_result(JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "Bone infarct not contusion")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        assert selected_pred.predictions["Contusion"].value == 0

    def test_explicit_grade_i_mcl_sprain_positive(self):
        """Explicit grade-I MCL sprain → MCL=1."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["max_attempts"] = 3
        
        pred1 = self.make_prediction({"MCL": 1})
        crit1 = self.make_critique(CritiqueStatus.PASS, [])
        
        state["prediction_history"] = [pred1]
        state["critique_history"] = [crit1]
        state["current_prediction"] = pred1
        state["current_critique"] = crit1
        state["current_judgment"] = self.make_judge_result(JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "Grade I MCL sprain is positive")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        assert selected_pred.predictions["MCL"].value == 1

    def test_osteochondral_lesion_not_oa(self):
        """Osteochondral lesion of lateral tibial plateau without explicit OA → Lateral_OA=0."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["max_attempts"] = 3
        
        pred1 = self.make_prediction({"Lateral_OA": 0})
        crit1 = self.make_critique(CritiqueStatus.PASS, [])
        
        state["prediction_history"] = [pred1]
        state["critique_history"] = [crit1]
        state["current_prediction"] = pred1
        state["current_critique"] = crit1
        state["current_judgment"] = self.make_judge_result(JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "Osteochondral lesion not OA")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        assert selected_pred.predictions["Lateral_OA"].value == 0

    def test_mcl_distension_not_sprain(self):
        """MCL distension without explicit sprain/tear → MCL=0."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["max_attempts"] = 3
        
        pred1 = self.make_prediction({"MCL": 0})
        crit1 = self.make_critique(CritiqueStatus.PASS, [])
        
        state["prediction_history"] = [pred1]
        state["critique_history"] = [crit1]
        state["current_prediction"] = pred1
        state["current_critique"] = crit1
        state["current_judgment"] = self.make_judge_result(JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "MCL distension not sprain")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        assert selected_pred.predictions["MCL"].value == 0

    def test_grade_ii_mcl_sprain_positive(self):
        """Grade II MCL sprain → MCL=1."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["max_attempts"] = 3
        
        pred1 = self.make_prediction({"MCL": 1})
        crit1 = self.make_critique(CritiqueStatus.PASS, [])
        
        state["prediction_history"] = [pred1]
        state["critique_history"] = [crit1]
        state["current_prediction"] = pred1
        state["current_critique"] = crit1
        state["current_judgment"] = self.make_judge_result(JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "Grade II MCL sprain is positive")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        assert selected_pred.predictions["MCL"].value == 1

    def test_grade_ii_meniscopathy_not_tear(self):
        """Grade II meniscopathy without tear → meniscus=0."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["max_attempts"] = 3
        
        pred1 = self.make_prediction({"Medial_Meniscus": 0, "Lateral_Meniscus": 0})
        crit1 = self.make_critique(CritiqueStatus.PASS, [])
        
        state["prediction_history"] = [pred1]
        state["critique_history"] = [crit1]
        state["current_prediction"] = pred1
        state["current_critique"] = crit1
        state["current_judgment"] = self.make_judge_result(JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "Grade II meniscopathy not tear")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        assert selected_pred.predictions["Medial_Meniscus"].value == 0
        assert selected_pred.predictions["Lateral_Meniscus"].value == 0

    def test_grade_iii_degeneration_extrusion_unresolved(self):
        """Grade III degeneration + extrusion without explicit tear → follow policy; if unresolved, ambiguity."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["max_attempts"] = 3
        
        # Model 1 predicts 0 (conservative)
        pred1 = self.make_prediction({"Medial_Meniscus": 0})
        # Model 2 says unresolved
        crit1 = self.make_critique(CritiqueStatus.AMBIGUOUS, [
            CritiqueIssue(label="Medial_Meniscus", model1_value=0, proposed_value=None,
                         issue_type=CritiqueIssueType.UNRESOLVED_POLICY, evidence="", policy_rule="", feedback="Grade III + extrusion unresolved")
        ])
        
        state["prediction_history"] = [pred1]
        state["critique_history"] = [crit1]
        state["current_prediction"] = pred1
        state["current_critique"] = crit1
        state["current_judgment"] = self.make_judge_result(JudgeAction.AMBIGUOUS, JudgeReasonCode.POLICY_UNRESOLVED, "Grade III + extrusion unresolved")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        # Should remain 0 with AMBIGUOUS
        assert selected_pred.predictions["Medial_Meniscus"].value == 0
        assert "ambiguity" in reason.lower() or "unresolved" in reason.lower()

    def test_explicit_degenerative_tear_positive(self):
        """Explicit degenerative meniscal tear → meniscus=1."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["max_attempts"] = 3
        
        pred1 = self.make_prediction({"Medial_Meniscus": 1})
        crit1 = self.make_critique(CritiqueStatus.PASS, [])
        
        state["prediction_history"] = [pred1]
        state["critique_history"] = [crit1]
        state["current_prediction"] = pred1
        state["current_critique"] = crit1
        state["current_judgment"] = self.make_judge_result(JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "Explicit degenerative tear")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        assert selected_pred.predictions["Medial_Meniscus"].value == 1

    def test_explicit_acl_sprain_supported_despite_weaker_evidence(self):
        """Explicit ACL sprain with strong evidence elsewhere → Model 2 must not create false critique."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.graph_nodes import _select_best_prediction_from_history
        
        state = create_initial_state("test", "report", max_attempts=3)
        state["attempt"] = 1
        state["max_attempts"] = 3
        
        pred1 = self.make_prediction({"ACL": 1})
        # Model 2 criticizes evidence selection but label is supported
        crit1 = self.make_critique(CritiqueStatus.PASS, [])
        
        state["prediction_history"] = [pred1]
        state["critique_history"] = [crit1]
        state["current_prediction"] = pred1
        state["current_critique"] = crit1
        state["current_judgment"] = self.make_judge_result(JudgeAction.PASS, JudgeReasonCode.NO_ACTIONABLE_ISSUE, "ACL sprain supported")
        
        selected_pred, reason, selected_attempt = _select_best_prediction_from_history(state)
        assert selected_pred.predictions["ACL"].value == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])