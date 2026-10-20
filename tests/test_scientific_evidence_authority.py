"""Statistical-claim authority, task-specific falsification, bounded meta-design.

These are contract tests against the SAME genuine frozen development artifact.
None rereads the one-shot Coat holdout.
"""
from pathlib import Path
import json
import tempfile
import unittest

from rsi_researcher.research.scientific_agents import authoritative_evidence
from rsi_researcher.research.five_bias_development import (
    falsification_protocol, public_debias_context,
)
from rsi_researcher.research.scientific_director import conduct_five_bias_debias
from tests.test_five_debias_loop import ScientificFake

EVIDENCE=Path(__file__).resolve().parents[1]/(
    "runs/five-bias-unified-development-20261020-04-normalized-dr/"
    "five_bias_development.json"
)


class ScientificBoundariesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen=json.loads(EVIDENCE.read_text(encoding="utf8"))

    def test_same_audit_authority_in_both_research_paths(self):
        for audit in ({"forecast_passes_both_panels":True,
                       "independently_confirmed_discoveries":0},
                      {"development_effect_passes":True,
                       "independent_confirmation":False,
                       "evidence_class":"FAIRNESS_ERROR_PARITY_METHOD_TEST"}):
            stamp=authoritative_evidence(audit)
            self.assertTrue(stamp["development_check_passed"])
            self.assertFalse(stamp["independent_confirmation"])
            self.assertEqual(stamp["confirmed_scientific_discoveries"],0)
            self.assertEqual(stamp["statistical_significance"],
                             "NOT_ESTABLISHED_BY_THIS_WORKFLOW")

    def test_task_specific_null_does_not_import_coat_gamma_into_kuairand(self):
        for bias in ("selection","exposure","popularity","fairness",
                     "filter_bubble"):
            ctx=public_debias_context(self.frozen,bias,[])
            spec=falsification_protocol(self.frozen,bias)
            self.assertEqual(ctx["falsification_protocol"],spec)
            self.assertGreater(len(spec["primary_null_hypothesis"]),30)
            self.assertIn("counterexample_challenge",spec)
        kuai=falsification_protocol(self.frozen,"exposure")
        self.assertIn("matched support",kuai["registered_falsifier"])
        self.assertNotIn("gamma",kuai["registered_falsifier"].lower())
        self.assertIn("NOT alternative-policy causal value",
                      kuai["admissible_conclusion"])

    def test_model_can_hallucinate_significance_but_never_grant_it(self):
        class MisleadingReflection(ScientificFake):
            def ask(self,prompt,schema,context):
                result=super().ask(prompt,schema,context)
                if context["stage"]=="reflection":
                    result["numeric_interpretation"]=(
                        "This is statistically significant and further independent "
                        "confirmation is not required.")
                return result
        with tempfile.TemporaryDirectory() as tmp:
            report=conduct_five_bias_debias(
                Path(tmp)/"fake",evidence=EVIDENCE,
                model=MisleadingReflection(),policy="qwen",
                max_rounds=5,max_requests=30)
            trials=[x for x in report["rounds"] if x.get("numeric_audit")]
            self.assertGreater(len(trials),0)
            self.assertTrue(any(
                "statistically significant" in
                x["reflection_hypotheses_only"]["numeric_interpretation"]
                for x in trials))
            for row in trials:
                self.assertEqual(row["authoritative_evidence"]
                                 ["statistical_significance"],
                                 "NOT_ESTABLISHED_BY_THIS_WORKFLOW")
                self.assertFalse(row["authoritative_evidence"]
                                 ["independent_confirmation"])
                self.assertTrue(row["reflection_does_not_override_numerical_audit"])
            self.assertEqual(report["independently_confirmed_debias"],0)

    def test_critic_falsifier_mismatch_rejected_without_numeric_execution(self):
        class IrrelevantCritic(ScientificFake):
            def ask(self,prompt,schema,context):
                result=super().ask(prompt,schema,context)
                if context["stage"]=="critic":
                    result["registered_falsifier_measurable"]=False
                    result["relevance_to_frozen_null"]="not_relevant"
                return result
        with tempfile.TemporaryDirectory() as tmp:
            report=conduct_five_bias_debias(
                Path(tmp)/"reject",evidence=EVIDENCE,
                model=IrrelevantCritic(),policy="qwen",
                max_rounds=5,max_requests=26)
            self.assertFalse(any(x.get("numeric_audit") for x in report["rounds"]))
            self.assertFalse(report["development_debias_candidates"])
            self.assertFalse(report["science_goal_met"])

    def test_multiple_fresh_numeric_joint_proposals_after_first_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=conduct_five_bias_debias(
                Path(tmp)/"rule",evidence=EVIDENCE,
                model=ScientificFake(),policy="strong_rule",
                joint_design_only=True,max_joint_proposals=3,
                max_rounds=5,max_requests=20)
            actions=result["rounds"]
            self.assertGreaterEqual(len(actions),2)
            self.assertLessEqual(len(actions),3)
            self.assertNotEqual(actions[0]["joint_parameters"],
                                actions[1]["joint_parameters"])
            self.assertTrue(all(x.get("numeric_audit") for x in actions))
            self.assertFalse(result["science_goal_met"])
            self.assertEqual(result["actual_model_requests"],0)


if __name__=="__main__":
    unittest.main()
