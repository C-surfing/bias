"""Tests for the author-path researcher loop (prediction contract + gates)."""
import unittest

from rsi_researcher.runtime.client import InvalidOutput
from rsi_researcher.paths import ROOT
from rsi_researcher.research.author_researcher import (
    validate_prediction, compare_arms, build_researcher_evidence)


class ValidatePredictionTests(unittest.TestCase):
    def _prediction(self, operator="unseen_neutral_user_reflection", effects=None):
        return {"selected_operator": operator,
                "expected_effects": effects or {"false_dislike_rate": "decrease",
                                                "quality_ndcg": "maintain_or_improve",
                                                "popularity_bias": "decrease"}}

    def test_retain_with_change_conflicts(self):
        pred = self._prediction(operator="retain", effects={"false_dislike_rate": "decrease",
                                                            "quality_ndcg": "no_change",
                                                            "popularity_bias": "no_change"})
        with self.assertRaises(InvalidOutput):
            validate_prediction(pred)

    def test_retain_without_change_is_ok(self):
        pred = self._prediction(operator="retain", effects={"false_dislike_rate": "no_change",
                                                            "quality_ndcg": "maintain_or_improve",
                                                            "popularity_bias": "no_change"})
        validate_prediction(pred)

    def test_unknown_operator_raises(self):
        with self.assertRaises(InvalidOutput):
            validate_prediction(self._prediction(operator="not_real"))

    def test_intervention_with_decrease_is_ok(self):
        validate_prediction(self._prediction())


class CompareArmsTests(unittest.TestCase):
    def setUp(self):
        self.run_dir = ROOT / "runs" / "qwen-author-dev20-baseline-20261015"
        self.have_run = (self.run_dir / "summary.json").exists() and \
                        (self.run_dir / "original_update" / "ranking.json").exists()

    def test_evidence_is_bounded_and_complete(self):
        if not self.have_run:
            self.skipTest("baseline run artifacts not present")
        evidence = build_researcher_evidence(self.run_dir)
        for key in ("dataset", "frozen", "original_update", "exposure_bias",
                    "conformity_bias", "selection_paac", "position_bias"):
            self.assertIn(key, evidence)
        self.assertEqual(evidence["committed_events"], 100)

    def test_same_dir_gates_are_degenerate(self):
        if not self.have_run:
            self.skipTest("baseline run artifacts not present")
        comparison = compare_arms(self.run_dir, self.run_dir)
        # candidate == baseline: dislike did not decrease, quality unchanged (>= 0.99), bias not reduced
        self.assertEqual(comparison["gates_pass"]["false_dislike"], False)
        self.assertEqual(comparison["gates_pass"]["quality"], True)
        self.assertEqual(comparison["metrics"]["ndcg_at_5"]["relative_change"], 0.0)


if __name__ == "__main__":
    unittest.main()
