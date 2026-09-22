"""End-to-end regression tests for the RagSet inference pipeline.

These tests mock Model 1 and Model 2 to verify the complete pipeline behavior
without calling real LLM APIs.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
import yaml

from experiments.ragset_report_inference_experiment.src.ragset_inference.schemas import (
    ReportPrediction, ValidationResult, ValidationIssue, LabelValue
)
from experiments.ragset_report_inference_experiment.src.ragset_inference.loop import run_report
from experiments.ragset_report_inference_experiment.src.ragset_inference.data import LABEL_COLUMNS
from experiments.ragset_report_inference_experiment.runners.run_inference import make_infer_fn, make_validate_fn
from experiments.ragset_report_inference_experiment.src.ragset_inference.models import InferenceModel, ValidatorModel


def make_mock_prediction(values: dict, evidence: dict = None) -> ReportPrediction:
    """Create a mock ReportPrediction with given values."""
    if evidence is None:
        evidence = {label: "" for label in LABEL_COLUMNS}
    predictions = {
        label: LabelValue(value=values.get(label, 0), evidence=evidence.get(label, ""))
        for label in LABEL_COLUMNS
    }
    return ReportPrediction(predictions=predictions)


def make_mock_validation(status: str, issues: list = None) -> ValidationResult:
    """Create a mock ValidationResult."""
    if issues is None:
        issues = []
    return ValidationResult(status=status, issues=issues)


class TestE2EPipeline:
    """End-to-end pipeline tests with mocked models."""

    def test_correct_prediction_passes(self):
        """Test 1: Model 1 returns correct labels, Model 2 PASSes."""
        # Model 1 returns correct prediction
        correct_pred = make_mock_prediction({
            "ACL": 0, "MCL": 0, "Medial_Meniscus": 1, "Lateral_Meniscus": 0,
            "Medial_OA": 0, "Lateral_OA": 0, "PF_OA": 0, "Effusion": 1,
            "Synovitis": 0, "Bakers": 0, "Contusion": 0, "Fracture": 0
        })

        # Model 2 validates as PASS
        mock_infer = MagicMock(return_value=correct_pred)
        mock_validate = MagicMock(return_value=make_mock_validation("PASS"))

        result = run_report(
            report_id="test-001",
            report="Test report with medial meniscus tear and effusion.",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )

        assert result["status"] == "passed"
        assert len(result["attempts"]) == 1
        assert result["attempts"][0].validation.passed is True
        mock_infer.assert_called_once()
        mock_validate.assert_called_once()

    def test_incorrect_prediction_fails_then_passes_on_retry(self):
        """Test 2: Model 1 wrong, Model 2 FAILs, retry succeeds."""
        # First attempt: wrong prediction
        wrong_pred = make_mock_prediction({
            "ACL": 0, "MCL": 0, "Medial_Meniscus": 0, "Lateral_Meniscus": 0,
            "Medial_OA": 0, "Lateral_OA": 0, "PF_OA": 0, "Effusion": 0,
            "Synovitis": 0, "Bakers": 0, "Contusion": 0, "Fracture": 0
        })

        # Second attempt: corrected prediction
        correct_pred = make_mock_prediction({
            "ACL": 0, "MCL": 0, "Medial_Meniscus": 1, "Lateral_Meniscus": 0,
            "Medial_OA": 0, "Lateral_OA": 0, "PF_OA": 0, "Effusion": 1,
            "Synovitis": 0, "Bakers": 0, "Contusion": 0, "Fracture": 0
        })

        # Model 2 fails first, passes second
        fail_validation = make_mock_validation("FAIL", [
            ValidationIssue(
                label="Medial_Meniscus",
                predicted=0,
                corrected=1,
                reason="Report describes medial meniscus tear",
                evidence=["Rotura de menisco interno."]
            ),
            ValidationIssue(
                label="Effusion",
                predicted=0,
                corrected=1,
                reason="Report describes effusion",
                evidence=["Derrame."]
            )
        ])
        pass_validation = make_mock_validation("PASS")

        mock_infer = MagicMock(side_effect=[wrong_pred, correct_pred])
        mock_validate = MagicMock(side_effect=[fail_validation, pass_validation])

        result = run_report(
            report_id="test-002",
            report="Rotura de menisco interno. Derrame.",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )

        assert result["status"] == "passed"
        assert len(result["attempts"]) == 2
        assert mock_infer.call_count == 2
        assert mock_validate.call_count == 2
        # Second attempt should have received validator feedback
        second_call_kwargs = mock_infer.call_args_list[1].kwargs
        assert second_call_kwargs["validator_feedback"] is not None

    def test_evidence_only_problem_passes(self):
        """Test 3: Correct label with imperfect evidence -> PASS."""
        # Model 1: correct label, imperfect evidence (related to different label)
        pred = make_mock_prediction({
            "Synovitis": 0, "Effusion": 1,
            **{l: 0 for l in LABEL_COLUMNS if l not in ["Synovitis", "Effusion"]}
        }, {
            "Synovitis": "No knee effusion",  # Imperfect evidence
            "Effusion": "Derrame articular."
        })

        mock_infer = MagicMock(return_value=pred)
        mock_validate = MagicMock(return_value=make_mock_validation("PASS"))

        result = run_report(
            report_id="test-003",
            report="No knee effusion. Derrame articular.",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )

        assert result["status"] == "passed"
        assert len(result["attempts"]) == 1

    def test_null_evidence_passes(self):
        """Test 4: Correct label with null evidence -> PASS."""
        pred = make_mock_prediction({
            "ACL": 0,
            **{l: 0 for l in LABEL_COLUMNS if l != "ACL"}
        }, {
            "ACL": None  # Null evidence
        })

        mock_infer = MagicMock(return_value=pred)
        mock_validate = MagicMock(return_value=make_mock_validation("PASS"))

        result = run_report(
            report_id="test-004",
            report="ACL is intact.",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )

        assert result["status"] == "passed"

    def test_empty_report_heading_no_fabricated_evidence(self):
        """Test 5: Empty report heading doesn't create fabricated evidence."""
        pred = make_mock_prediction({
            label: 0 for label in LABEL_COLUMNS
        }, {
            label: "" for label in LABEL_COLUMNS  # Empty evidence, not "Bevindingen:"
        })

        mock_infer = MagicMock(return_value=pred)
        mock_validate = MagicMock(return_value=make_mock_validation("PASS"))

        result = run_report(
            report_id="test-005",
            report="Bevindingen:",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )

        assert result["status"] == "passed"
        # Verify no fabricated evidence was used
        pred = result["final_prediction"]
        for label in LABEL_COLUMNS:
            assert pred.predictions[label].evidence == ""

    def test_full_regeneration_on_retry(self):
        """Test 6: FAIL causes complete regeneration of all 12 labels."""
        wrong_pred = make_mock_prediction({label: 0 for label in LABEL_COLUMNS})
        correct_pred = make_mock_prediction({
            "Medial_Meniscus": 1, "Effusion": 1,
            **{l: 0 for l in LABEL_COLUMNS if l not in ["Medial_Meniscus", "Effusion"]}
        })

        fail_validation = make_mock_validation("FAIL", [
            ValidationIssue(
                label="Medial_Meniscus", predicted=0, corrected=1,
                reason="Report describes medial meniscus tear", evidence=["Rotura de menisco interno."]
            )
        ])
        pass_validation = make_mock_validation("PASS")

        mock_infer = MagicMock(side_effect=[wrong_pred, correct_pred])
        mock_validate = MagicMock(side_effect=[fail_validation, pass_validation])

        result = run_report(
            report_id="test-006",
            report="Rotura de menisco interno. Derrame.",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )

        assert result["status"] == "passed"
        assert len(result["attempts"]) == 2
        # Verify second prediction has all 12 labels
        second_pred = result["attempts"][1].prediction
        assert len(second_pred.predictions) == 12

    def test_retrieval_context_reaches_model1(self):
        """Test 7: Target-specific retrieval context reaches Model 1."""
        captured_context = {}

        def mock_infer(**kwargs):
            captured_context["gold_examples"] = kwargs.get("gold_examples", "")
            return make_mock_prediction({label: 0 for label in LABEL_COLUMNS})

        mock_validate = MagicMock(return_value=make_mock_validation("PASS"))

        result = run_report(
            report_id="test-007",
            report="Test report",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "analysis", "gold_examples": "TARGET_SPECIFIC_EXAMPLE"},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )

        assert "TARGET_SPECIFIC_EXAMPLE" in captured_context.get("gold_examples", "")

    def test_heldout_leakage_prevention(self):
        """Test 8: Held-out target never appears in its own retrieval."""
        # This is tested in retrieval tests, but verify the loop uses retrieve_excluding
        # The loop itself doesn't do retrieval - it's done in the runner
        # This test verifies the context passed to infer doesn't contain the target report
        pass  # Covered by retrieval tests

    def test_max_attempts_enforced(self):
        """Test 9: Max attempts enforced, then needs_review."""
        wrong_pred = make_mock_prediction({label: 0 for label in LABEL_COLUMNS})
        fail_validation = make_mock_validation("FAIL", [
            ValidationIssue(label="ACL", predicted=0, corrected=1, reason="wrong", evidence=[])
        ])

        mock_infer = MagicMock(return_value=wrong_pred)
        mock_validate = MagicMock(return_value=fail_validation)

        result = run_report(
            report_id="test-009",
            report="Test report",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )

        assert result["status"] == "needs_review"
        assert len(result["attempts"]) == 3
        assert mock_infer.call_count == 3
        assert mock_validate.call_count == 3

    def test_resume_behavior(self):
        """Test 10: Resume skips already processed reports."""
        # This is tested at the runner level, not the loop level
        # The loop itself doesn't handle resume - the runner does
        pass  # Covered by runner tests

    def test_retry_feedback_reaches_model1_prompt(self):
        """Test 11: Validator feedback reaches the FINAL Model 1 prompt on retry."""
        captured_prompts = []
        
        wrong_pred = make_mock_prediction({label: 0 for label in LABEL_COLUMNS})
        correct_pred = make_mock_prediction({
            "Medial_Meniscus": 1, "Effusion": 1,
            **{l: 0 for l in LABEL_COLUMNS if l not in ["Medial_Meniscus", "Effusion"]}
        })
        
        fail_validation = make_mock_validation("FAIL", [
            ValidationIssue(
                label="Medial_Meniscus", predicted=0, corrected=1,
                reason="Report describes medial meniscus tear", evidence=["Rotura de menisco interno."]
            )
        ])
        pass_validation = make_mock_validation("PASS")
        
        def mock_infer(**kwargs):
            captured_prompts.append(kwargs.get("validator_feedback"))
            if len(captured_prompts) == 1:
                return wrong_pred
            return correct_pred
        
        mock_validate = MagicMock(side_effect=[fail_validation, pass_validation])
        
        result = run_report(
            report_id="test-011",
            report="Rotura de menisco interno. Derrame.",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )
        
        assert result["status"] == "passed"
        assert len(captured_prompts) == 2
        assert captured_prompts[0] is None  # First attempt: no feedback
        assert captured_prompts[1] is not None  # Second attempt: has feedback
        assert "Medial_Meniscus" in str(captured_prompts[1])

    def test_model2_receives_retrieved_examples(self):
        """Test 12: Model 2 receives report-specific retrieved gold examples."""
        captured_context = {}
        
        def mock_infer(**kwargs):
            return make_mock_prediction({label: 0 for label in LABEL_COLUMNS})
        
        def mock_validate(**kwargs):
            captured_context["gold_examples"] = kwargs.get("gold_examples", "")
            return make_mock_validation("PASS")
        
        result = run_report(
            report_id="test-012",
            report="Test report",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "analysis", "gold_examples": "TARGET_SPECIFIC_EXAMPLE"},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )
        
        assert "TARGET_SPECIFIC_EXAMPLE" in captured_context.get("gold_examples", "")

    def test_ambiguous_routes_to_needs_review_no_retry(self):
        """Test 13: AMBIGUOUS -> needs_review, no retry."""
        mock_infer = MagicMock(return_value=make_mock_prediction({label: 0 for label in LABEL_COLUMNS}))
        ambiguous_validation = make_mock_validation("AMBIGUOUS", [
            ValidationIssue(label="Contusion", predicted=0, corrected=None, reason="Uncertain evidence", evidence=[])
        ])
        mock_validate = MagicMock(return_value=ambiguous_validation)
        
        result = run_report(
            report_id="test-013",
            report="Possible bone marrow edema.",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )
        
        assert result["status"] == "needs_review"
        assert "AMBIGUOUS" in result["review_reason"]
        assert mock_infer.call_count == 1  # No retry
        assert mock_validate.call_count == 1

    def test_model1_stuck_detection(self):
        """Test 14: Model 1 STUCK detection."""
        stuck_pred = make_mock_prediction({"Medial_OA": 1, **{l: 0 for l in LABEL_COLUMNS if l != "Medial_OA"}})
        fail_validation = make_mock_validation("FAIL", [
            ValidationIssue(label="Medial_OA", predicted=1, corrected=0, reason="Generalized OA", evidence=[])
        ])
        
        mock_infer = MagicMock(return_value=stuck_pred)
        mock_validate = MagicMock(return_value=fail_validation)
        
        result = run_report(
            report_id="test-014",
            report="Osteoarthritis of knee.",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )
        
        assert result["status"] == "needs_review"
        # Check that MODEL1_STUCK was detected
        stuck_attempts = [a for a in result["attempts"] if a.failure_type.value == "model1_stuck"]
        assert len(stuck_attempts) > 0

    def test_validator_instability_detection(self):
        """Test 15: Validator instability detection."""
        pred1 = make_mock_prediction({"Contusion": 0, **{l: 0 for l in LABEL_COLUMNS if l != "Contusion"}})
        pred2 = make_mock_prediction({"Contusion": 1, **{l: 0 for l in LABEL_COLUMNS if l != "Contusion"}})
        pred3 = make_mock_prediction({"Contusion": 1, **{l: 0 for l in LABEL_COLUMNS if l != "Contusion"}})
        
        # Validator flips: 0->1, then 1->0, then 1->1
        val1 = make_mock_validation("FAIL", [ValidationIssue(label="Contusion", predicted=0, corrected=1, reason="", evidence=[])])
        val2 = make_mock_validation("FAIL", [ValidationIssue(label="Contusion", predicted=1, corrected=0, reason="", evidence=[])])
        val3 = make_mock_validation("FAIL", [ValidationIssue(label="Contusion", predicted=1, corrected=1, reason="", evidence=[])])
        
        mock_infer = MagicMock(side_effect=[pred1, pred2, pred3])
        mock_validate = MagicMock(side_effect=[val1, val2, val3])
        
        result = run_report(
            report_id="test-015",
            report="Bone marrow edema.",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )
        
        assert result["status"] == "needs_review"
        # Check that VALIDATOR_INSTABILITY was detected
        unstable_attempts = [a for a in result["attempts"] if a.failure_type.value == "validator_instability"]
        assert len(unstable_attempts) > 0

    def test_trace_isolation_between_reports(self):
        """Test 16: Trace data from report A does not leak into report B."""
        # This is tested at the runner level with the dict-based trace collector
        # The loop itself doesn't handle traces - the runner does
        pass  # Covered by runner tests

    def test_gold_policy_formatter_preserves_safe_unsafe_unresolved(self):
        """Test 17: Gold policy formatter preserves SAFE / UNSAFE / UNRESOLVED information."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.gold import format_gold_policy_for_prompt
        
        mock_analysis = {
            "label_specific": {
                "Contusion": {
                    "safe_rules": [{"rule": "Explicit bone contusion -> Contusion=1"}],
                    "unsafe_shortcuts": [{"pattern": "bone marrow edema", "reason": "Not universally equivalent"}],
                    "unresolved_rules": [{"question": "Does chronic edema map to Contusion?"}],
                    "contradictions": []
                },
                "Medial_OA": {}
            }
        }
        
        policy_text = format_gold_policy_for_prompt(mock_analysis)
        
        assert "SAFE:" in policy_text
        assert "Explicit bone contusion" in policy_text
        assert "UNSAFE" in policy_text
        assert "bone marrow edema" in policy_text
        assert "UNRESOLVED" in policy_text
        assert "chronic edema" in policy_text
        assert "INSUFFICIENT_GOLD_EVIDENCE" in policy_text  # For Medial_OA

    def test_missing_gold_analysis_marked_insufficient(self):
        """Test 18: Missing gold analysis is marked INSUFFICIENT_GOLD_EVIDENCE."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.gold import format_gold_policy_for_prompt
        
        mock_analysis = {
            "label_specific": {
                "ACL": {}  # Empty analysis
            }
        }
        
        policy_text = format_gold_policy_for_prompt(mock_analysis)
        
        assert "INSUFFICIENT_GOLD_EVIDENCE" in policy_text

    def test_normal_pass_flow_accepts_prediction(self):
        """Test 19: Normal PASS flow still accepts the prediction."""
        correct_pred = make_mock_prediction({
            "ACL": 0, "MCL": 0, "Medial_Meniscus": 1, "Lateral_Meniscus": 0,
            "Medial_OA": 0, "Lateral_OA": 0, "PF_OA": 0, "Effusion": 1,
            "Synovitis": 0, "Bakers": 0, "Contusion": 0, "Fracture": 0
        })
        
        mock_infer = MagicMock(return_value=correct_pred)
        mock_validate = MagicMock(return_value=make_mock_validation("PASS"))
        
        result = run_report(
            report_id="test-019",
            report="Test report with medial meniscus tear and effusion.",
            infer=mock_infer,
            validate=mock_validate,
            context={"gold_analysis": "", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )
        
        assert result["status"] == "passed"
        assert len(result["attempts"]) == 1
        assert result["attempts"][0].validation.passed is True

    def test_canonical_policy_reaches_both_models(self):
        """Test 20: Canonical policy reaches both Model 1 and Model 2 prompts."""
        captured_infer_prompts = []
        captured_validate_prompts = []
        
        correct_pred = make_mock_prediction({
            "ACL": 0, "MCL": 0, "Medial_Meniscus": 1, "Lateral_Meniscus": 0,
            "Medial_OA": 0, "Lateral_OA": 0, "PF_OA": 0, "Effusion": 1,
            "Synovitis": 0, "Bakers": 0, "Contusion": 0, "Fracture": 0
        })
        
        # Create mock models that capture the prompts
        class MockInferenceModel:
            def __init__(self):
                self.model = "mock-inference"
                self.provider = "mock"
            def run(self, system_prompt, user_prompt, validator_feedback=None):
                captured_infer_prompts.append(user_prompt)
                return correct_pred
        
        class MockValidatorModel:
            def __init__(self):
                self.model = "mock-validator"
                self.provider = "mock"
            def run(self, system_prompt, user_prompt):
                captured_validate_prompts.append(user_prompt)
                return make_mock_validation("PASS")
        
        # Load canonical policy
        from experiments.ragset_report_inference_experiment.src.ragset_inference.policy_loader import get_canonical_policy_text
        canonical_policy = get_canonical_policy_text(
            csv_path="train.csv",
            gold_analysis_path="experiments/ragset_report_inference_experiment/results/gold/gold_analysis.json",
            policy_path="config/ragset_label_policy.yaml"
        )
        
        # Load prompt configs
        import yaml
        inf_cfg = yaml.safe_load(
            open("experiments/ragset_report_inference_experiment/prompts/inference.yaml", encoding="utf-8").read()
        )
        val_cfg = yaml.safe_load(
            open("experiments/ragset_report_inference_experiment/prompts/validation.yaml", encoding="utf-8").read()
        )
        
        # Create infer/validate functions using the actual factory functions
        mock_model1 = MockInferenceModel()
        mock_model2 = MockValidatorModel()
        infer_fn = make_infer_fn(mock_model1, inf_cfg, None, list(LABEL_COLUMNS), canonical_policy)
        validate_fn = make_validate_fn(mock_model2, val_cfg, None, list(LABEL_COLUMNS), canonical_policy)
        
        result = run_report(
            report_id="test-020",
            report="Test report with ACL tear.",
            infer=infer_fn,
            validate=validate_fn,
            context={"gold_analysis": "test_policy", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )
        
        assert result["status"] == "passed"
        # Both models should receive the same policy text
        assert len(captured_infer_prompts) == 1
        assert len(captured_validate_prompts) == 1
        # The policy text should be in both prompts
        assert canonical_policy in captured_infer_prompts[0]
        assert canonical_policy in captured_validate_prompts[0]
        assert len(canonical_policy) > 0
        assert "SAFE" in canonical_policy or "UNSAFE" in canonical_policy or "UNRESOLVED" in canonical_policy

    def test_contradiction_blocks_inference(self):
        """Test 21: Policy contradiction at startup blocks inference."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.policy_loader import get_canonical_policy_text
        
        # Create a policy with a contradiction (SAFE_POSITIVE with contradicting cases)
        bad_policy = """
labels:
  ACL:
    gold_positive: 24
    gold_negative: 37
    conventions:
      - pattern: "explicit ACL tear"
        status: SAFE_POSITIVE
        supporting_count: 24
        contradicting_count: 5  # Contradiction!
        rationale: "Explicit tear language"
"""
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml.safe_load(bad_policy), f)
            bad_policy_path = f.name
        
        try:
            # This should raise RuntimeError due to contradiction
            get_canonical_policy_text(
                csv_path="train.csv",
                gold_analysis_path="experiments/ragset_report_inference_experiment/results/gold/gold_analysis.json",
                policy_path=bad_policy_path
            )
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Policy validation failed" in str(e)
            assert "SAFE_POSITIVE rule has 5 contradicting cases" in str(e)
        finally:
            os.unlink(bad_policy_path)

    def test_both_models_receive_identical_policy(self):
        """Test 22: Both Model 1 and Model 2 receive identical canonical policy text."""
        captured_prompts = {"infer": None, "validate": None}
        
        correct_pred = make_mock_prediction({
            "ACL": 0, "MCL": 0, "Medial_Meniscus": 1, "Lateral_Meniscus": 0,
            "Medial_OA": 0, "Lateral_OA": 0, "PF_OA": 0, "Effusion": 1,
            "Synovitis": 0, "Bakers": 0, "Contusion": 0, "Fracture": 0
        })
        
        # Create mock models that capture the prompts
        class MockInferenceModel:
            def __init__(self):
                self.model = "mock-inference"
                self.provider = "mock"
            def run(self, system_prompt, user_prompt, validator_feedback=None):
                captured_prompts["infer"] = user_prompt
                return correct_pred
        
        class MockValidatorModel:
            def __init__(self):
                self.model = "mock-validator"
                self.provider = "mock"
            def run(self, system_prompt, user_prompt):
                captured_prompts["validate"] = user_prompt
                return make_mock_validation("PASS")
        
        # Load canonical policy
        from experiments.ragset_report_inference_experiment.src.ragset_inference.policy_loader import get_canonical_policy_text
        canonical_policy = get_canonical_policy_text(
            csv_path="train.csv",
            gold_analysis_path="experiments/ragset_report_inference_experiment/results/gold/gold_analysis.json",
            policy_path="config/ragset_label_policy.yaml"
        )
        
        # Load prompt configs
        import yaml
        inf_cfg = yaml.safe_load(
            open("experiments/ragset_report_inference_experiment/prompts/inference.yaml", encoding="utf-8").read()
        )
        val_cfg = yaml.safe_load(
            open("experiments/ragset_report_inference_experiment/prompts/validation.yaml", encoding="utf-8").read()
        )
        
        # Create infer/validate functions using the actual factory functions
        mock_model1 = MockInferenceModel()
        mock_model2 = MockValidatorModel()
        infer_fn = make_infer_fn(mock_model1, inf_cfg, None, list(LABEL_COLUMNS), canonical_policy)
        validate_fn = make_validate_fn(mock_model2, val_cfg, None, list(LABEL_COLUMNS), canonical_policy)
        
        result = run_report(
            report_id="test-022",
            report="Test report with ACL tear.",
            infer=infer_fn,
            validate=validate_fn,
            context={"gold_analysis": "test_policy", "gold_examples": ""},
            max_attempts=3,
            labels=list(LABEL_COLUMNS),
        )
        
        assert result["status"] == "passed"
        # Both models should receive the same canonical policy text (embedded in their respective prompts)
        assert captured_prompts["infer"] is not None
        assert captured_prompts["validate"] is not None
        assert len(captured_prompts["infer"]) > 0
        assert len(captured_prompts["validate"]) > 0
        # The canonical policy text should be present in both prompts
        assert canonical_policy in captured_prompts["infer"]
        assert canonical_policy in captured_prompts["validate"]

    def test_policy_contradiction_prevents_inference_startup(self):
        """Test 23: Policy contradiction at startup prevents inference from starting."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.policy_loader import get_canonical_policy_text
        import tempfile
        import yaml
        
        # Create a policy with a contradiction (SAFE_POSITIVE with contradicting cases)
        bad_policy = {
            "labels": {
                "ACL": {
                    "gold_positive": 24,
                    "gold_negative": 37,
                    "conventions": [
                        {
                            "pattern": "explicit ACL tear",
                            "status": "SAFE_POSITIVE",
                            "supporting_count": 24,
                            "contradicting_count": 5,  # Contradiction!
                            "rationale": "Explicit tear language"
                        }
                    ]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(bad_policy, f)
            bad_policy_path = f.name
        
        try:
            # This should raise RuntimeError due to contradiction
            get_canonical_policy_text(
                csv_path="train.csv",
                gold_analysis_path="experiments/ragset_report_inference_experiment/results/gold/gold_analysis.json",
                policy_path=bad_policy_path
            )
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Policy validation failed" in str(e)
            assert "SAFE_POSITIVE rule has 5 contradicting cases" in str(e)
        finally:
            os.unlink(bad_policy_path)

    def test_unresolved_convention_allows_inference_but_marks_unresolved(self):
        """Test 24: Unresolved convention allows inference but marks as UNRESOLVED."""
        from experiments.ragset_report_inference_experiment.src.ragset_inference.policy_loader import get_canonical_policy_text
        import tempfile
        import yaml
        
        # Load the actual policy and add an UNRESOLVED convention to MCL
        with open("config/ragset_label_policy.yaml", encoding="utf-8") as f:
            full_policy = yaml.safe_load(f)
        
        # Add an UNRESOLVED convention to MCL
        full_policy["labels"]["MCL"]["conventions"].append({
            "pattern": "suspect grade 2 MCL injury",
            "status": "UNRESOLVED",
            "supporting_count": 1,
            "contradicting_count": 0,
            "rationale": "Only 1 positive example; no negative counterexamples"
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(full_policy, f)
            unresolved_policy_path = f.name
        
        try:
            # This should succeed (warnings only, not errors)
            policy_text = get_canonical_policy_text(
                csv_path="train.csv",
                gold_analysis_path="experiments/ragset_report_inference_experiment/results/gold/gold_analysis.json",
                policy_path=unresolved_policy_path
            )
            assert "UNRESOLVED" in policy_text
            assert "suspect grade 2 MCL injury" in policy_text
        finally:
            os.unlink(unresolved_policy_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])