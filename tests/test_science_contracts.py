"""Scientific action contracts must be causal-question-specific, not generic deltas."""
import unittest
from rsi_researcher.research import crossbias_discovery as baseline
from rsi_researcher.research import scientific_experiments as experiments
from rsi_researcher.research.scientific_loop import review_plan


class ScientificContractTests(unittest.TestCase):
    def test_coat_each_action_uses_its_own_numeric_counterfactual(self):
        expected = {
            "gamma_stress": "gamma_reference_margin",
            "propensity_shuffle": "propensity_placebo_margin",
            "split_utility": "brier_gain",
        }
        for action, metric in expected.items():
            with self.subTest(action=action):
                self.assertIn(metric, experiments.ACTION_METRICS[action])
                data = baseline.coat_observation(action)
                panels = experiments.panel_measurements(
                    "coat_propensity" if action != "split_utility" else "coat_utility",
                    action, data)
                self.assertEqual(len(panels), 2)
                self.assertTrue(all(metric in p and "quality_gate" in p for p in panels))
                self.assertTrue(all(isinstance(p[metric], float) for p in panels))
        self.assertEqual(experiments.ACTION_METRICS["gamma_stress"],
                         ("gamma_reference_margin",))
        self.assertEqual(experiments.ACTION_METRICS["propensity_shuffle"],
                         ("propensity_placebo_margin",))

    def test_mismatched_plan_rejected_even_if_critic_approved(self):
        plan = {"action": "propensity_shuffle", "metric": "brier_gain"}
        review = {"decision": "approve"}
        self.assertEqual(review_plan("coat_propensity", plan, review),
                         "REJECTED_ACTION_METRIC_MISMATCH")
        plan["metric"] = "propensity_placebo_margin"
        self.assertEqual(review_plan("coat_propensity", plan, review),
                         "ACCEPTED_DESIGN_ONLY")
        plan["action"] = "split_utility"
        self.assertEqual(review_plan("coat_propensity", plan, review),
                         "REJECTED_UNREGISTERED_ACTION")

    def test_popularity_falsifier_rejects_reversed_forecast_direction(self):
        plan = {"action": "popularity_prompt", "metric": "head_excess_delta",
                "direction": "increase"}
        self.assertEqual(review_plan("agentcf_popularity", plan, {"decision": "approve"}),
                         "REJECTED_PREDECLARED_DIRECTION_MISMATCH")
        plan["direction"] = "decrease"
        self.assertEqual(review_plan("agentcf_popularity", plan, {"decision": "approve"}),
                         "ACCEPTED_DESIGN_ONLY")

    def test_real_propensity_negative_control_verdict(self):
        data = baseline.coat_observation("propensity_shuffle")
        plan = {"action":"propensity_shuffle", "metric":"propensity_placebo_margin",
                "minimum_abs_effect":0.0001,"direction":"increase"}
        verdict = experiments.independent_verdict(
            "coat_propensity","propensity_shuffle",plan,data)
        self.assertEqual(len(verdict["panel_measurements"]),2)
        self.assertEqual(verdict["independently_confirmed_discoveries"],0)
        with self.assertRaises(ValueError):
            experiments.independent_verdict(
                "coat_propensity","propensity_shuffle",
                {**plan,"metric":"brier_gain"}, data)

    def test_coat_gamma_no_placebo_substitution(self):
        data = baseline.coat_observation("gamma_stress")
        plan = {"action":"gamma_stress", "metric":"gamma_reference_margin",
                "minimum_abs_effect":0.0001,"direction":"increase"}
        observed = experiments.independent_verdict("coat_propensity",
                                                    "gamma_stress",plan,data)
        self.assertEqual(len(observed["panel_measurements"]),2)
        self.assertEqual(observed["forecast_metric"],"gamma_reference_margin")
        self.assertIn("mechanism_development_check",observed)

    def test_only_registered_task_action_metrics_pass(self):
        with self.assertRaises(ValueError):
            experiments.action_metrics("agentcf_memory","propensity_shuffle")
        with self.assertRaises(ValueError):
            experiments.action_metrics("coat_utility","popularity_prompt")
        self.assertIn("head_excess_delta",
                      experiments.action_metrics("agentcf_popularity","popularity_prompt"))


if __name__ == "__main__":
    unittest.main()
