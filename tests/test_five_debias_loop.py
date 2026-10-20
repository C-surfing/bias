"""Constrained, repeated five-bias Debias search; real frozen data, not fake effects."""
from pathlib import Path
import tempfile
import unittest

from rsi_researcher.research.five_bias_development import (
    DEBIAS_ACTIONS,evaluate_debias_action,public_debias_context,
)
from rsi_researcher.research.scientific_director import conduct_five_bias_debias

SOURCE=Path(__file__).resolve().parents[1]/(
 "runs/five-bias-unified-development-20261020-04-normalized-dr/five_bias_development.json")


class ScientificFake:
    """Replays only ROLE contracts; cannot choose/evaluate numeric outcomes."""
    def __init__(self):
        self.requests=0
        self.scope={}
        self.roles=[]
        self.prompts=[]
    def ask(self,prompt,schema,context):
        self.requests+=1
        stage=context["stage"]
        self.roles.append(stage)
        self.prompts.append((stage,prompt))
        if stage=="diagnosis":
            return {"hypothesis":"Logged exposure sampling can alter measured user choices.",
                    "competing_explanation":"Candidate support and relevance can explain difference.",
                    "missing_evidence":"A paired randomized feedback panel with support controls.",
                    "why_this_matters":"Separates measured correction from an unrelated ranking change."}
        if stage=="design":
            return {"action":next(a for a in schema["properties"]["action"]["enum"]
                                   if a!="abstain"),
                    "hypothesis":"A registered intervention may reduce systematic error.",
                    "competing_explanation":"Data coverage may explain the apparent gain.",
                    "falsifier":"Fail a frozen minimum-effect or relevance-preservation guard.",
                    "metric":schema["properties"]["metric"]["enum"][0],
                    "direction":"increase",
                    "minimum_abs_effect":.05,
                    "budget_justification":"One measured, frozen action within a bounded budget."}
        if stage=="critic":
            return {"decision":"approve",
                    "reason":"Accept one registered development experiment, not confirmation.",
                    "confounding_risk":"Observational logs may violate causal identification.",
                    "better_discriminating_test":"Confirm using a new source or independent cohort.",
                    **({
                        "counterexample_if_null_true":"Observational item coverage may alter predictions even with no identified debias mechanism.",
                        "unexcluded_competing_mechanism":"Catalog support and user selection effects are not eliminated.",
                        "registered_falsifier_measurable":True,
                        "relevance_to_frozen_null":"relevant_but_not_identifying",
                    } if "relevance_to_frozen_null" in schema["properties"] else {})}
        if stage=="reflection":
            return {"evidence_state":"inconclusive",
                    "numeric_interpretation":"This development result is not an independent discovery.",
                    "remaining_alternative":"Exposure support or logging policies remain alternatives.",
                    "next_question":"Try another predeclared legal Debias method in the inventory.",
                    "candidate_research_skill":"Only preserve candidate methods with measured quality guards.",
                    "suggested_next_task":"stop",
                    "next_task_reason":"External scheduler owns continuation after independent audit."}
        raise ValueError(stage)


class FiveBiasSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.frozen=json.loads(SOURCE.read_text(encoding="utf8"))

    def test_selection_outcome_is_frozen_and_stable(self):
        c=evaluate_debias_action(self.frozen,"selection",
                                 "estimated_propensity_clip_100")
        self.assertGreater(c["measured_improvement"],.003)
        self.assertTrue(c["development_debias_candidate"])
        self.assertEqual(c["independently_confirmed_discoveries"],0)
        self.assertFalse(c["independent_confirmation"])

    def test_exposure_is_proxy_even_with_good_prediction(self):
        c=evaluate_debias_action(self.frozen,"exposure","random_equal_budget")
        self.assertGreater(c["measured_improvement"],.07)
        self.assertTrue(c["development_effect_passes"])
        self.assertFalse(c["mechanism_identification_sufficient"])
        self.assertFalse(c["development_debias_candidate"])

    def test_popularity_tradeoff_and_failed_group_fairness(self):
        good=evaluate_debias_action(self.frozen,"popularity","penalty_0.1")
        bad=evaluate_debias_action(self.frozen,"popularity","penalty_0.2")
        fair=evaluate_debias_action(self.frozen,"fairness","group_calibrated")
        self.assertTrue(good["development_effect_passes"])
        self.assertFalse(good["development_debias_candidate"])
        self.assertFalse(bad["development_effect_passes"])
        self.assertFalse(fair["development_effect_passes"])
        with self.assertRaises(ValueError):
            evaluate_debias_action(self.frozen,"filter_bubble","made_up")
        with self.assertRaises(ValueError):
            evaluate_debias_action(self.frozen,"fairness","arbitrary_test")

    def test_prompt_does_not_leak_untested_action_outcomes(self):
        selection=public_debias_context(self.frozen,"selection",[])
        text=str(selection)
        self.assertNotIn("0.149545",text)
        self.assertNotIn("estimated_propensity_clip_100",text)
        self.assertIn("baseline_only",selection)
        self.assertIn("BEFORE minus AFTER",
                      selection["metric_direction_semantics"])

    def test_fixed_rule_tries_multiple_registered_methods_not_one_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=conduct_five_bias_debias(Path(tmp)/"rule",evidence=SOURCE,
               model=ScientificFake(),policy="strong_rule",max_rounds=14)
            self.assertGreaterEqual(len(r["rounds"]),5)
            self.assertEqual(r["actual_model_requests"],0)
            self.assertTrue(any(x["bias_id"]=="fairness" for x in r["rounds"]))
            self.assertIn(r["task_status"]["filter_bubble"],(
                "DEVELOPMENT_EFFECT_WITH_QUALITY_GUARD",
                "ABSTAIN_BY_DESIGNER",
                "REJECTED_PREDECLARED_CONTRACT",
                "REJECTED_BY_CRITIC",
                "EXHAUSTED_METHODS_NO_DEBIAS"))
            self.assertFalse(r["science_goal_met"])
            self.assertEqual(r["independently_confirmed_debias"],0)
            self.assertIn(r["status"],(
                "STOP_DEVELOPMENT_DEBIAS_CANDIDATE_PENDING_CONFIRMATION",
                "STOP_NO_FURTHER_REGISTERED_METHODS"))

    def test_four_actual_agent_roles_keep_searching_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=ScientificFake()
            r=conduct_five_bias_debias(Path(tmp)/"qwen-fake",evidence=SOURCE,
                model=model,policy="qwen",max_rounds=14,max_requests=85)
            self.assertGreaterEqual(len(r["rounds"]),4)
            self.assertIn(r["task_status"]["filter_bubble"],(
                "DEVELOPMENT_EFFECT_WITH_QUALITY_GUARD",
                "ABSTAIN_BY_DESIGNER",
                "REJECTED_PREDECLARED_CONTRACT",
                "REJECTED_BY_CRITIC",
                "EXHAUSTED_METHODS_NO_DEBIAS"))
            self.assertIn("diagnosis",model.roles)
            self.assertIn("design",model.roles)
            self.assertIn("critic",model.roles)
            self.assertIn("reflection",model.roles)
            self.assertFalse(r["science_goal_met"])
            self.assertGreaterEqual(len([x for x in r["rounds"] if
                                           x["bias_id"]=="selection"]),1)
            self.assertGreaterEqual(len([x for x in r["rounds"] if
                                           x["bias_id"]=="fairness"]),1)
            self.assertTrue(all(x["numeric_audit"]["independent_confirmation"] is False
                                for x in r["rounds"] if x.get("numeric_audit")))
            self.assertTrue(all("estimated_propensity_clip_100" not in text
                                for stage,text in model.prompts if stage=="diagnosis"))

    def test_bounded_abort_never_development_success_from_critique(self):
        class Rejector(ScientificFake):
            def ask(self,prompt,schema,context):
                answer=super().ask(prompt,schema,context)
                if context["stage"]=="critic":
                    answer["decision"]="reject"
                return answer
        with tempfile.TemporaryDirectory() as tmp:
            r=conduct_five_bias_debias(Path(tmp)/"reject",evidence=SOURCE,
                model=Rejector(),policy="qwen",max_rounds=5,max_requests=60)
            self.assertEqual(len(r["development_debias_candidates"]),0)
            self.assertIn(r["status"],(
                "STOP_ROUND_CEILING","STOP_NO_FURTHER_REGISTERED_METHODS"))
            self.assertEqual(r["independently_confirmed_debias"],0)

if __name__=="__main__":
    unittest.main()
