import math
import unittest

from rsi_researcher.evaluation.metrics import evaluate, point_decision


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.panel = [f"i{i:02d}" for i in range(12)]
        self.queries = [{"query_id": "q", "user": "u", "candidates": self.panel}]
        self.truth = {"u": {i: 12 - n for n, i in enumerate(self.panel)}}
        self.guards = dict.fromkeys(("position", "memory_coverage", "learning", "budget"), True)

    def score(self, ranking, popular):
        return evaluate(self.queries, {"q": ranking}, self.truth, popular)

    def test_ideal_and_legitimate_popularity_have_no_excess(self):
        score = self.score(self.panel, set(self.panel[:3]))
        self.assertEqual(score["utility"], 1)
        self.assertEqual(score["bias"], 0)

    def test_extra_head_exposure_matches_hand_calculation(self):
        ranking = [self.panel[-1]] + self.panel[:-1]
        score = self.score(ranking, {self.panel[-1]})
        self.assertAlmostEqual(score["bias"], 1 / sum(1 / math.log2(j + 2) for j in range(10)))
        self.assertLess(score["utility"], 1)

    def test_over_suppression_remains_visible(self):
        score = self.score(list(reversed(self.panel)), {self.panel[0]})
        self.assertEqual(score["bias"], 0)
        self.assertLess(score["signed_excess"], 0)

    def test_invalid_and_missing_outputs_stay_in_denominator(self):
        for bad in (None, [], self.panel[:-1], self.panel[:-1] + [self.panel[0]], [{}] * 12):
            with self.subTest(bad=bad):
                score = self.score(bad, set())
                self.assertEqual((score["utility"], score["bias"], score["invalid_rankings"]), (0, 1, 1))
        score = evaluate(self.queries, {}, self.truth, set())
        self.assertEqual(score["query_count"], 1)

    def test_bad_truth_is_not_a_zero_score(self):
        for bad in (float("nan"), float("inf"), -1, True):
            self.truth["u"][self.panel[0]] = bad
            with self.assertRaises(ValueError):
                self.score(self.panel, set())

    def test_point_gates_are_independent_and_do_not_claim_support(self):
        baseline = {"utility": 1.0, "bias": 0.2, "query_count": 2, "invalid_rankings": 0}
        candidate = baseline | {"utility": 0.99, "bias": 0.18}
        decision = point_decision(baseline, candidate, self.guards, 0.001)
        self.assertEqual(decision["status"], "eligible")
        self.assertEqual(decision["statistical_support"], "not_tested")
        for change in ({"utility": 0.989}, {"bias": 0.181}, {"invalid_rankings": 1}):
            self.assertEqual(point_decision(baseline, candidate | change, self.guards, 0.001)["status"], "rejected")
        for guard in self.guards:
            self.assertEqual(point_decision(baseline, candidate, self.guards | {guard: False}, 0.001)["status"], "rejected")
            self.assertEqual(point_decision(baseline, candidate, self.guards | {guard: None}, 0.001)["status"], "inconclusive")
        self.assertEqual(point_decision(baseline | {"bias": 0}, candidate, self.guards, 0.001)["status"], "inconclusive")
        self.assertEqual(point_decision(baseline | {"bias": 0}, candidate, self.guards | {"position": False}, 0.001)["status"], "rejected")
        with self.assertRaises(ValueError):
            point_decision(baseline, candidate | {"query_count": 1}, self.guards, 0.001)


if __name__ == "__main__":
    unittest.main()
