from copy import deepcopy
import unittest

from rsi_researcher.datasets.controlled import build_world
from rsi_researcher.evaluation.learning import audit_learning


class LearningAuditTests(unittest.TestCase):
    def setUp(self):
        self.world = build_world(400, users=1, events_per_user=2)
        self.query = self.world["public"]["queries"][0]
        truth = self.world["private"]["utilities"][self.query["user"]]
        self.ideal = sorted(self.query["candidates"], key=lambda i: (-truth[i], i))
        self.pair = [{self.query["query_id"]: self.ideal}, {self.query["query_id"]: self.ideal}]
        self.events = [{"user": self.query["user"], "positive": self.ideal[0]},
                       {"user": self.query["user"], "positive": self.ideal[-1]}]

    def test_truth_order_ties_and_observed_denominator(self):
        result = audit_learning(self.world, self.events, self.pair)
        self.assertEqual(result["mean_pair_accuracy"], 1)
        self.assertEqual(result["mean_observed_pair_accuracy"], 1)
        self.assertEqual(result["queries"][0]["observed_non_tied_pairs"], 1)
        self.assertLess(result["queries"][0]["non_tied_pairs"], 66)
        self.assertIsNone(result["memory_coverage_guard"])

    def test_one_observation_and_tied_observations_do_not_fake_coverage(self):
        for events in (self.events[:1], [self.events[0], self.events[0]]):
            result = audit_learning(self.world, events, self.pair)
            self.assertIsNone(result["mean_observed_pair_accuracy"])
            self.assertEqual(result["queries_with_observed_comparison"], 0)

    def test_reverse_and_invalid_rankings_keep_fixed_denominator(self):
        result = audit_learning(self.world, self.events, [{self.query["query_id"]: list(reversed(self.ideal))}] * 2)
        self.assertEqual(result["mean_pair_accuracy"], 0)
        pair = deepcopy(self.pair)
        pair[1][self.query["query_id"]] = None
        result = audit_learning(self.world, self.events, pair)
        self.assertEqual(result["mean_pair_accuracy"], 0.5)
        self.assertEqual(result["mean_observed_pair_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
