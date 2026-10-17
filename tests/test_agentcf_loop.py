"""Tests for the autonomous AgentCF bias-discovery loop (pure-Python parts)."""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from rsi_researcher.research.agentcf_loop import compare_bias, _select, _next_allowed_operators, _plan_schema, PROPOSABLE_OPERATORS, run_arm, OPERATOR_TARGETS


def _summary(ndcg, excess, dislike, conf, tau, head=0.9, tail=0.3):
    return {"quality_popularity": {"ndcg_at_5": ndcg, "hit_at_5": 0.7,
                                   "top5_head_share": 0.3, "candidate_head_share": 0.245,
                                   "excess_popularity_exposure": excess},
            "exposure_bias": {"false_dislike_memory_rate": dislike},
            "conformity_bias": {"select_reason_popularity_rate": conf,
                                "user_memory_popularity_rate": 0.75},
            "position_bias": {"input_order_following_tau": tau},
            "selection_paac": {"head": {"mean_update_count": head, "mean_growth_ratio": 1.8},
                               "mid": {"mean_update_count": 0.5, "mean_growth_ratio": 1.5},
                               "tail": {"mean_update_count": tail, "mean_growth_ratio": 1.3},
                               "unseen": {"mean_update_count": 0.0, "mean_growth_ratio": 1.0}}}


class CompareBiasTests(unittest.TestCase):
    def setUp(self):
        self.baseline = _summary(0.404, 0.035, 0.75, 0.78, 0.23)

    def test_win_on_target_bias_despite_collateral(self):
        # exposure target: dislike down + quality up; excess (collateral) up is
        # reported but NOT a hard gate.
        candidate = _summary(0.42, 0.06, 0.20, 0.5, 0.1)
        result = compare_bias(self.baseline, candidate, "exposure")
        self.assertTrue(result["debiasing_win"])
        self.assertTrue(result["target_gate"])
        self.assertEqual(result["target"]["bias"], "exposure")
        self.assertTrue(result["target"]["improved"])
        self.assertFalse(result["metrics"]["excess_popularity_exposure"]["improved"])

    def test_quality_loss_blocks_win(self):
        candidate = _summary(0.30, 0.02, 0.2, 0.5, 0.1)
        result = compare_bias(self.baseline, candidate, "exposure")
        self.assertFalse(result["debiasing_win"])
        self.assertFalse(result["quality_gate"])
        self.assertTrue(result["target_gate"])  # target improved but quality failed

    def test_target_not_improved_blocks_win(self):
        candidate = _summary(0.42, 0.02, 0.80, 0.5, 0.1)
        result = compare_bias(self.baseline, candidate, "exposure")
        self.assertFalse(result["debiasing_win"])
        self.assertFalse(result["target_gate"])

    def test_popularity_target_uses_excess(self):
        candidate = _summary(0.40, 0.02, 0.75, 0.78, 0.23)
        result = compare_bias(self.baseline, candidate, "popularity")
        self.assertTrue(result["target"]["improved"])  # excess 0.035 -> 0.02
        self.assertEqual(result["target"]["label"], "excess_popularity_exposure")

    def test_selection_paac_target_uses_head_tail_gap(self):
        candidate = _summary(0.40, 0.03, 0.75, 0.78, 0.23, head=0.9, tail=0.6)
        result = compare_bias(self.baseline, candidate, "selection_paac")
        self.assertEqual(result["target"]["label"], "head_tail_update_gap")
        self.assertTrue(result["target"]["improved"])  # gap 0.6 -> 0.3

    def test_none_target_cannot_count_as_an_improvement(self):
        candidate = _summary(0.40, 0.03, 0.75, 0.78, 0.23)
        result = compare_bias(self.baseline, candidate, "none")
        self.assertIsNone(result["target"])
        self.assertFalse(result["target_gate"])
        self.assertFalse(result["debiasing_win"])

    def test_reverse_order_following_is_not_a_position_win(self):
        candidate = _summary(0.42, 0.03, 0.75, 0.78, -0.8)
        result = compare_bias(self.baseline, candidate, "position")
        self.assertFalse(result["target_gate"])
        self.assertFalse(result["metrics"]["position_tau"]["improved"])

    def test_a_tiny_decrease_does_not_pass_the_registered_target(self):
        candidate = _summary(0.42, 0.03, 0.74, 0.78, 0.23)
        result = compare_bias(self.baseline, candidate, "exposure")
        self.assertTrue(result["target"]["improved"])
        self.assertFalse(result["proxy_goal_pass"])

    def test_nonpositive_excess_is_not_an_unbounded_improvement(self):
        baseline = _summary(0.404, -0.02, 0.75, 0.78, 0.23)
        candidate = _summary(0.42, -0.30, 0.75, 0.78, 0.23)
        result = compare_bias(baseline, candidate, "popularity")
        self.assertIsNone(result["target"]["relative_reduction"])
        self.assertFalse(result["proxy_goal_pass"])


class SelectTests(unittest.TestCase):
    def test_no_win_selects_retain(self):
        trials = [{"round": 1, "operator": "x", "comparison": {"debiasing_win": False, "target": None}}]
        result = _select(trials)
        self.assertFalse(result["selected"])
        self.assertEqual(result["best_by_bias"], {})

    def test_builds_per_bias_map(self):
        def trial(round_, op, bias, change):
            return {"round": round_, "operator": op,
                    "comparison": {"debiasing_win": True,
                                   "target": {"bias": bias, "change": change, "candidate": 0.1},
                                   "metrics": {"ndcg_at_5": {"change": 0.01}}}}
        result = _select([trial(1, "a", "exposure", -0.5),
                          trial(2, "b", "popularity", -0.1),
                          trial(3, "c", "exposure", -0.3)])
        self.assertTrue(result["selected"])
        self.assertEqual(set(result["best_by_bias"]), {"exposure", "popularity"})
        self.assertEqual(result["best_by_bias"]["exposure"]["operator"], "a")  # -0.5 beats -0.3
        self.assertEqual(result["best_by_bias"]["popularity"]["operator"], "b")

    def test_skips_noop_target(self):
        def trial(round_, op, target):
            return {"round": round_, "operator": op,
                    "comparison": {"debiasing_win": True, "target": target,
                                   "metrics": {"ndcg_at_5": {"change": 0.0}}}}
        result = _select([trial(1, "retain", None)])
        self.assertFalse(result["selected"])


class ExplorationGuardTests(unittest.TestCase):
    def test_registered_subset_never_expands_to_other_mechanisms(self):
        registered=["balanced_observed_replay","neutral_candidate_order"]
        self.assertEqual(_next_allowed_operators(set(),registered),registered)
        self.assertEqual(_next_allowed_operators({registered[0]},registered),[registered[1]])
        self.assertEqual(_next_allowed_operators(set(registered),registered),[])

    def test_excludes_failed_operators(self):
        allowed = _next_allowed_operators({"unseen_neutral_user_reflection"})
        self.assertNotIn("unseen_neutral_user_reflection", allowed)
        self.assertIn("neutralize_rank_popularity", allowed)
        self.assertNotIn("retain", allowed)  # retain is a fallback, not proposable
        self.assertEqual(len(allowed), len(PROPOSABLE_OPERATORS) - 1)

    def test_stops_when_registered_actions_are_exhausted(self):
        allowed = _next_allowed_operators(set(PROPOSABLE_OPERATORS))
        self.assertEqual(allowed, [])

    def test_all_five_biases_have_a_real_registered_action(self):
        self.assertEqual({OPERATOR_TARGETS[op] for op in PROPOSABLE_OPERATORS},
                         {"exposure", "selection_paac", "conformity", "popularity", "position"})

    def test_plan_schema_enum_matches_allowed(self):
        allowed = ["retain", "neutralize_rank_popularity"]
        schema = _plan_schema(allowed)
        self.assertEqual(sorted(schema["properties"]["operator"]["enum"]), sorted(allowed))

    def test_plan_schema_forbids_failed_operator(self):
        schema = _plan_schema(["retain"])
        self.assertEqual(schema["properties"]["operator"]["enum"], ["retain"])

    def test_ranking_intervention_reuses_training_source(self):
        with TemporaryDirectory() as temporary:
            with patch("rsi_researcher.research.agentcf_loop.subprocess.run") as process:
                process.return_value.returncode = 0
                run_arm("config.json", Path(temporary) / "trial", "neutral_candidate_order",
                        baseline_dir="frozen-baseline")
            command = process.call_args.args[0]
            self.assertIn("rank-only", command)
            self.assertNotIn("author-check", command)
            self.assertIn("frozen-baseline", command)


if __name__ == "__main__":
    unittest.main()
