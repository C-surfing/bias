"""Scientific action contracts must be causal-question-specific, not generic deltas."""
import unittest
from rsi_researcher.research import crossbias_discovery as baseline
from rsi_researcher.research import scientific_experiments as experiments
from rsi_researcher.research.scientific_loop import review_plan, public_evidence


class ScientificContractTests(unittest.TestCase):
    def test_agent_research_question_uses_frozen_five_bias_targets(self):
        memory=public_evidence("agentcf_memory",[],"qwen_no_history")
        self.assertIsNone(memory["formal_five_bias_target"])
        self.assertTrue(memory["auxiliary_mechanism_not_debias_goal"])
        self.assertEqual(len(memory["five_bias_registry_scope"]),5)
        selection=public_evidence("coat_utility",[],"qwen_no_history")
        self.assertEqual(selection["formal_five_bias_target"],"selection")
        self.assertFalse(selection["auxiliary_mechanism_not_debias_goal"])

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

    def test_five_bias_coverage_is_not_four_scientific_experiment_names(self):
        rows=experiments.five_bias_coverage()
        self.assertEqual({r["bias_id"] for r in rows},{
            "popularity","exposure","selection","filter_bubble","fairness"})
        self.assertEqual(len(rows),5)
        self.assertEqual(next(r for r in rows if r["bias_id"]=="selection")
                         ["existing_experimental_tasks"],
                         ["coat_propensity","coat_utility"])
        self.assertEqual(experiments.EXPERIMENT_BIAS_TARGET["agentcf_memory"],None)
        self.assertEqual([r["existing_experimental_tasks"] for r in rows
                          if r["bias_id"] in ("fairness","exposure","filter_bubble")],
                         [[],[],[]])

    def test_causally_unidentified_popularity_proxy_cannot_trigger_debias_acceptance(self):
        proxy={"task":"agentcf_popularity","action":"popularity_prompt",
               "forecast_metric":"head_excess_delta",
               "mechanism_forecast_passes_both_panels":True,
               "known_target_quality_guard_passes_both_panels":True,
               "registered_action_falsifier_passes":None,
               "debias_utility_state":"PROXY_GAIN_WITH_QUALITY_GUARD_DEVELOPMENT_ONLY",
               "evidence_grade":"new_unconfirmed_development"}
        marked=experiments.classify_debias_evidence(proxy)
        self.assertEqual(marked["bias_task_id"],"popularity")
        self.assertFalse(marked["debias_candidate_eligible"])
        self.assertEqual(marked["registered_falsifier_status"],"NOT_IDENTIFIED")
        self.assertEqual(marked["debias_evidence_stage"],
                         "POPULARITY_EXPOSURE_PROXY_ONLY")

    def test_coat_falsifier_control_not_debias_and_only_real_method_can_qualify(self):
        good={"task":"coat_utility","action":"split_utility",
              "forecast_metric":"brier_gain",
              "predeclared_direction":"increase",
              "mechanism_forecast_passes_both_panels":True,
              "known_target_quality_guard_passes_both_panels":True,
              "registered_action_falsifier_passes":True,
              "evidence_grade":"new_unconfirmed_development"}
        self.assertTrue(experiments.classify_debias_evidence(good)
                        ["debias_candidate_eligible"])
        for change in ({"action":"gamma_stress"},
                       {"predeclared_direction":"decrease"},
                       {"registered_action_falsifier_passes":False},
                       {"known_target_quality_guard_passes_both_panels":False},
                       {"evidence_grade":"previous_development_only"}):
            self.assertFalse(experiments.classify_debias_evidence(
                {**good,**change})["debias_candidate_eligible"])

    def test_only_registered_task_action_metrics_pass(self):
        with self.assertRaises(ValueError):
            experiments.action_metrics("agentcf_memory","propensity_shuffle")
        with self.assertRaises(ValueError):
            experiments.action_metrics("coat_utility","popularity_prompt")
        self.assertIn("head_excess_delta",
                      experiments.action_metrics("agentcf_popularity","popularity_prompt"))


if __name__ == "__main__":
    unittest.main()
