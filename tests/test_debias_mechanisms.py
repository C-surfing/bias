"""Five-bias scientific controls; no peeking at Coat holdout in tests."""
from pathlib import Path
import json, tempfile, unittest
from unittest.mock import patch

from rsi_researcher.research.exposure_synthetic import evaluate_dr
from rsi_researcher.research.feedback_pair_control import evaluate, paired_user
from rsi_researcher.research.selection_confirmation import check_protocol, verdict
from rsi_researcher.research.five_bias_development import (
    evaluate_debias_action, evaluate_joint_coat_design,
    fairness_coat, popularity_coat,
)
from rsi_researcher.research.scientific_director import conduct_five_bias_debias
from tests.test_five_debias_loop import ScientificFake
from rsi_researcher.research.coat_selection import DATA,SEED
from rsi_researcher.datasets.coat import load_coat

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/("runs/five-bias-unified-development-20261020-04-normalized-dr/"
              "five_bias_development.json")
PROTOCOL=ROOT/"configs/selection_confirm_protocol_20261020.json"
DEV=ROOT/("runs/coat-fivebias-selection-debias-20261020-01/"
          "coat_selection_development.json")


class FalsifiableBiasMechanismTests(unittest.TestCase):
    def test_selection_frozen_before_holdout_once(self):
        p=json.loads(PROTOCOL.read_text())
        dev=check_protocol(p,DEV)
        self.assertEqual(dev["selected_development_policy"],
                         "estimated_propensity_clip_100")
        self.assertEqual(p["max_hit_loss"],.02)
        # No load_coat() with test labels here: test the judgement on mocks.
        b={"Brier_random_development":.15,"NDCG5_random_development":.7,
           "Hit5_random_development":.95}
        t={"Brier_random_development":.145,"NDCG5_random_development":.68,
           "Hit5_random_development":.945}
        a=verdict(b,t,{"illustrative_user_bootstrap_95":[-.01,-.001]},p)
        self.assertFalse(a["passes_all_frozen_requirements"])
        self.assertFalse(a["requirements"]["ndcg_at_least_99pct_baseline"])

    def test_dr_oracle_is_probability_and_model_uses_independent_train(self):
        result=evaluate_dr(reps=6,events=500)
        self.assertTrue(result["reward_model_training_separate_from_evaluation"])
        self.assertFalse(result["target_policy_uses_hidden_true_reward"])
        for policy,item in result["methods"].items():
            self.assertLessEqual(item["oracle_value"],1)
            self.assertGreaterEqual(item["oracle_value"],0)
            self.assertTrue(all(0<=val["rmse"]<=1
                                for val in item["estimators"].values()))

    def test_paired_sequential_simulator_uses_common_noise_and_novel_seeds(self):
        a=paired_user(20261020,5,.22)
        self.assertEqual(a,paired_user(20261020,5,.22))
        self.assertGreater(a["treated"]["exposure_entropy"],0)
        out=evaluate(seeds=4,users=4)
        self.assertTrue(out["same_initial_state"])
        self.assertTrue(out["same_exogenous_uniforms_across_policies"])
        self.assertTrue(out["no_real_world_causal_claim"])
        self.assertTrue(out["panels"]["unseen_simulation"]["users"]==16)

    def test_observed_head_share_vs_known_relevant_opportunity(self):
        ds=load_coat(DATA,split_seed=SEED)
        result=popularity_coat(ds)
        self.assertGreater(result["known_positive_reference_users"],0)
        baseline=result["policies"]["0.0"]
        self.assertIn("head_exposure_excess_over_known_relevance",baseline)
        self.assertLess(baseline["head_exposure_excess_over_known_relevance"],0)
        self.assertFalse(result["identifies_unjustified_popularity_causally"])

    def test_fairness_new_train_only_method_has_groupwise_guards(self):
        ds=load_coat(DATA,split_seed=SEED)
        result=fairness_coat(ds)
        c=result["global_model_blend_train_only_candidates"][
            "global_blend_unweighted_0.25"]
        self.assertGreater(c["gap_reduction"],.0005)
        self.assertTrue(c["no_group_error_loss_over_0_001"])
        self.assertTrue(c["accepted_development_candidate"])
        self.assertFalse(result["independent_confirmation"])

    def test_new_compositional_operator_is_recomputed_on_real_coat_dev(self):
        a=evaluate_joint_coat_design(.75,.05)
        self.assertFalse(a["independently_confirmed"])
        self.assertTrue(a["holdout_is_not_used"] if "holdout_is_not_used" in a else
                        a["source"].endswith("no holdout"))
        z=evaluate_joint_coat_design(0,0)
        self.assertAlmostEqual(z["brier_gain"],0,places=12)
        with self.assertRaises(ValueError):
            evaluate_joint_coat_design(-.1,.03)
        with self.assertRaises(ValueError):
            evaluate_joint_coat_design(.8,.3)

    def test_designer_can_propose_coefficients_unseen_by_rule(self):
        class ComposingAgent(ScientificFake):
            def ask(self,prompt,schema,context):
                if context["stage"]=="design" and "action" not in schema["properties"]:
                    self.requests+=1
                    self.roles.append("design")
                    return {"alpha":.82,"beta":.014,
                            "mechanism":"Mix propensity calibration with modest frequency regularization.",
                            "falsifier":"Fail the unchanged random dev quality guard."}
                return super().ask(prompt,schema,context)
        with tempfile.TemporaryDirectory() as tmp:
            m=ComposingAgent()
            result=conduct_five_bias_debias(
                Path(tmp)/"new_design",evidence=EVIDENCE,model=m,
                policy="qwen",joint_design_only=True,
                max_rounds=5,max_requests=20)
            self.assertEqual(len(result["rounds"]),1)
            r=result["rounds"][0]
            self.assertEqual(r["action"],"joint_propensity_popularity_design")
            self.assertEqual(r["joint_parameters"],{"alpha":.82,"beta":.014})
            self.assertIn("numeric_composition",r["numeric_audit"]["constraints"])
            self.assertEqual(result["independently_confirmed_debias"],0)

    def test_scientific_director_does_not_promote_simulator_or_proxy(self):
        frozen=json.loads(EVIDENCE.read_text())
        a=evaluate_debias_action(frozen,"filter_bubble",
                                 "controlled_uniform_exploration_0.22")
        self.assertEqual(a["evidence_class"],
                         "SYNTHETIC_SEQUENTIAL_FILTER_BUBBLE_CONTROL_ONLY")
        self.assertTrue(a["development_effect_passes"])
        self.assertFalse(a["development_debias_candidate"])
        b=evaluate_debias_action(frozen,"fairness",
                                 "global_blend_unweighted_0.25")
        self.assertGreater(b["measured_improvement"],0)
        self.assertTrue(b["development_effect_passes"])
        self.assertTrue(b["development_debias_candidate"])
        self.assertIn("predictive-error parity",b["candidate_scope"])


if __name__=="__main__":
    unittest.main()
