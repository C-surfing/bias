"""Real feedback stays sparse and candidate comparisons stay paired."""

from copy import deepcopy
import math
import unittest

from rsi_researcher.evaluation.observed import evaluate_observed, paired_comparison


class ObservedFixture:
    def setUp(self):
        self.items = ["head", "tail1", "tail2", "tail3", "tail4"]
        self.counts = {"head": 100, "tail1": 8, "tail2": 6, "tail3": 2, "tail4": 0}
        self.rows = [{"user": "u", "item": item, "rating": rating}
                     for item, rating in zip(self.items, [1, 5, 3, 1, 1])]

    def evaluate(self, ranking, rows=None, **kwargs):
        return evaluate_observed(self.rows if rows is None else rows, ranking,
                                 self.counts, catalog_items=self.items, k=2, **kwargs)


class ObservedEvaluationTests(ObservedFixture, unittest.TestCase):
    def test_observed_panel_quality_matches_hand_calculation(self):
        result = self.evaluate({"u": self.items})
        discount = 1 / math.log2(3)
        self.assertAlmostEqual(result["ndcg"], 4 * discount / (4 + 2 * discount))
        self.assertAlmostEqual(result["head_share"], 1 / (1 + discount))
        self.assertAlmostEqual(result["panel_excess_head_share"], 1 / (1 + discount))
        self.assertEqual(result["tail_relevant_recall"], 1)
        self.assertEqual(result["quality_users"], 1)
        ideal = self.evaluate({"u": ["tail1", "tail2", "head", "tail3", "tail4"]})
        self.assertEqual(ideal["ndcg"], 1)
        self.assertEqual(ideal["panel_excess_head_share"], 0)

    def test_unobserved_item_is_not_a_negative_or_extra_candidate(self):
        counts = self.counts | {"unknown": 60}
        result = evaluate_observed(self.rows, {"u": self.items}, counts,
                                   catalog_items=self.items + ["unknown"], k=2, head_fraction=.1)
        self.assertEqual(result["observed_ratings"], 5)
        self.assertEqual(result["per_user"][0]["observed_items"], 5)
        invalid = evaluate_observed(self.rows, {"u": self.items + ["unknown"]}, counts,
                                    catalog_items=self.items + ["unknown"], k=2)
        self.assertEqual(invalid["invalid_rankings"], 1)
        self.assertEqual(invalid["ndcg"], 0)
        self.assertEqual(invalid["panel_excess_head_share"], 1)

    def test_invalid_outputs_receive_quality_and_bias_penalties(self):
        for order in (None, [], self.items[:-1], self.items[:-1] + ["head"], [1] * 5):
            with self.subTest(order=order):
                result = self.evaluate({"u": order})
                self.assertEqual(result["evaluated_users"], 1)
                self.assertEqual(result["quality_users"], 1)
                self.assertEqual(result["invalid_rankings"], 1)
                self.assertEqual((result["ndcg"], result["head_share"], result["panel_excess_head_share"]), (0, 1, 1))
        self.assertEqual(self.evaluate({})["invalid_rankings"], 1)

    def test_zero_ideal_users_keep_fixed_explicit_support(self):
        zeros = [{"user": "zero", "item": item, "rating": 1} for item in self.items]
        result = self.evaluate({"u": self.items, "zero": self.items}, rows=self.rows + zeros)
        self.assertEqual((result["evaluated_users"], result["quality_users"], result["zero_ideal_users"]), (2, 1, 1))
        zero_user = next(row for row in result["per_user"] if row["user"] == "zero")
        self.assertIsNone(zero_user["ndcg"])
        self.assertIsNone(zero_user["panel_excess_head_share"])
        self.assertIsNone(zero_user["tail_relevant_recall"])
        zero_only = self.evaluate({"zero": self.items}, rows=zeros)
        comparison = paired_comparison(zero_only, zero_only, bootstrap_samples=10)
        self.assertIsNone(zero_only["ndcg"])
        self.assertEqual(comparison["status"], "inconclusive")
        self.assertIsNone(comparison["uncertainty"]["metrics"]["ndcg"]["ci95"])

    def test_no_tail_positives_are_not_scored_as_perfect_recall(self):
        rows = [{"user": "u", "item": item, "rating": 5 if item == "head" else 1} for item in self.items]
        result = self.evaluate({"u": self.items}, rows=rows)
        self.assertIsNone(result["tail_relevant_recall"])
        self.assertEqual(result["tail_supported_users"], 0)
        self.assertEqual(result["no_tail_positive_users"], 1)

    def test_binary_feedback_uses_observed_zero_not_missing(self):
        rows = [{"user": "u", "item": "head", "rating": 0},
                {"user": "u", "item": "tail1", "rating": 1}]
        result = self.evaluate({"u": ["tail1", "head"]}, rows=rows, gain_floor=0, positive_threshold=1)
        self.assertEqual(result["ndcg"], 1)
        self.assertEqual(result["tail_relevant_recall"], 1)
        self.assertEqual(result["observed_ratings"], 2)

    def test_ideal_head_reference_averages_relevance_ties(self):
        rows = [{"user": "u", "item": "head", "rating": 5},
                {"user": "u", "item": "tail1", "rating": 5}]
        result = evaluate_observed(rows, {"u": ["head", "tail1"]}, self.counts,
                                   catalog_items=self.items, k=1)
        self.assertEqual(result["ndcg"], 1)
        self.assertEqual(result["per_user"][0]["panel_ideal_head_share"], .5)

    def test_fixed_catalog_keeps_head_membership_across_splits(self):
        result = self.evaluate({"u": self.items})
        other = self.evaluate({"v": ["tail1", "tail2"]}, rows=[
            {"user": "v", "item": "tail1", "rating": 5},
            {"user": "v", "item": "tail2", "rating": 4}])
        self.assertEqual(result["protocol"]["head_items"], other["protocol"]["head_items"])
        self.assertEqual(result["protocol"]["training_counts_fingerprint"], other["protocol"]["training_counts_fingerprint"])

    def test_invalid_truth_and_duplicate_observations_are_rejected(self):
        for rating in (float("nan"), float("inf"), True):
            with self.subTest(rating=rating), self.assertRaises(ValueError):
                self.evaluate({"u": self.items}, rows=self.rows + [{"user": "v", "item": "head", "rating": rating}])
        with self.assertRaises(ValueError):
            self.evaluate({"u": self.items}, rows=self.rows + self.rows[:1])
        with self.assertRaises(ValueError):
            self.evaluate({"unexpected": self.items})
        with self.assertRaises(ValueError):
            evaluate_observed(self.rows, {"u": self.items}, self.counts, catalog_items=self.items[:-1])


class PairedComparisonTests(ObservedFixture, unittest.TestCase):
    def test_point_gates_and_uncertainty_have_separate_meanings(self):
        baseline = self.evaluate({"u": self.items})
        candidate = self.evaluate({"u": ["tail1", "tail2", "head", "tail3", "tail4"]})
        result = paired_comparison(baseline, candidate, bootstrap_samples=20)
        self.assertEqual(result["status"], "point_gates_passed")
        self.assertEqual(result["point_gates"], {"quality": True, "bias": True, "valid_rankings": True})
        self.assertFalse(result["statistical_promotion"])
        self.assertFalse(result["causal_claim"])
        self.assertEqual(result["paired_users"], 1)
        self.assertAlmostEqual(result["uncertainty"]["metrics"]["ndcg"]["mean_difference"], result["quality_change"])
        self.assertEqual(result, paired_comparison(baseline, candidate, bootstrap_samples=20))

    def test_bias_reduction_does_not_override_quality_loss(self):
        baseline = self.evaluate({"u": self.items})
        candidate = self.evaluate({"u": ["tail2", "tail3", "head", "tail1", "tail4"]})
        result = paired_comparison(baseline, candidate, bootstrap_samples=20)
        self.assertTrue(result["point_gates"]["bias"])
        self.assertFalse(result["point_gates"]["quality"])
        self.assertEqual(result["status"], "rejected")

    def test_no_reference_bias_is_unknown_not_a_pass(self):
        ideal = self.evaluate({"u": ["tail1", "tail2", "head", "tail3", "tail4"]})
        result = paired_comparison(ideal, ideal, bootstrap_samples=10)
        self.assertIsNone(result["point_gates"]["bias"])
        self.assertEqual(result["status"], "inconclusive")

    def test_comparison_refuses_different_users_labels_panels_and_popularity(self):
        baseline = self.evaluate({"u": self.items})
        changed_labels = deepcopy(self.rows)
        changed_labels[0]["rating"] = 2
        bad_label = self.evaluate({"u": self.items}, rows=changed_labels)
        bad_count = evaluate_observed(self.rows, {"u": self.items}, self.counts | {"head": 101},
                                      catalog_items=self.items, k=2)
        bad_users = self.evaluate({"v": self.items}, rows=[row | {"user": "v"} for row in self.rows])
        for other in (bad_label, bad_count, bad_users):
            with self.subTest(other=other), self.assertRaises(ValueError):
                paired_comparison(baseline, other, bootstrap_samples=10)

    def test_all_ranked_users_remain_paired_even_with_failed_candidate(self):
        rows = self.rows + [row | {"user": "v"} for row in self.rows]
        baseline = self.evaluate({"u": self.items, "v": self.items}, rows=rows)
        candidate = self.evaluate({"u": ["tail1", "tail2", "head", "tail3", "tail4"]}, rows=rows)
        result = paired_comparison(baseline, candidate, seed=7, bootstrap_samples=40)
        self.assertEqual(result["paired_users"], 2)
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["point_gates"]["valid_rankings"])
        quality = result["uncertainty"]["metrics"]["ndcg"]
        self.assertLess(quality["ci95"][0], quality["ci95"][1])


if __name__ == "__main__":
    unittest.main()
