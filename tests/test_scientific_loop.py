"""Offline tests drive the EXACT production orchestrator and real experiment adapter.

FakeModel is a deterministic model stub, not Qwen and not evidence of RSI success.
"""
import json
from pathlib import Path
import tempfile
import unittest

from rsi_researcher.research import scientific_experiments as ex
from rsi_researcher.research.scientific_agents import (
    DiagnosisAgent, DesignerAgent, CriticAgent, ReflectionAgent,
)
from rsi_researcher.research.scientific_loop import campaign, review_plan

class FakeModel:
    def __init__(self, reject=False, next_task="stop"):
        self.requests = 0
        self.scope = {}
        self.reject = reject
        self.next_task = next_task
        self.roles = []

    def ask(self, prompt, schema, context):
        self.requests += 1
        stage = context["stage"]
        self.roles.append(stage)
        properties = schema["properties"]
        if stage == "diagnosis":
            return {"hypothesis": "Learned popularity cues alter candidate ranking.",
                    "competing_explanation": "The user-item candidate mix causes the difference.",
                    "missing_evidence": "Paired candidate panels under repeated no-change rankings.",
                    "why_this_matters": "Isolates a proposed prompt intervention from ranking noise."}
        if stage == "design":
            actions = properties["action"]["enum"]
            action = "popularity_prompt" if "popularity_prompt" in actions else actions[0]
            return {"action": action,
                    "hypothesis": "The registered intervention changes ranking behavior.",
                    "competing_explanation": "Model sampling noise explains observed rank changes.",
                    "falsifier": "No paired effect beyond repeat noise in both cohorts.",
                    "metric": properties["metric"]["enum"][0],
                    "direction": "decrease" if action == "popularity_prompt" else "increase",
                    "minimum_abs_effect": ex.DEVELOPMENT_EFFECT_FLOOR[
                        properties["metric"]["enum"][0]],
                    "budget_justification": "One bounded developmental intervention using fixed cohorts."}
        if stage == "critic":
            return {"decision": "reject" if self.reject else "approve",
                    "reason": "Repeat controls are required to distinguish model noise.",
                    "confounding_risk": "Only one known validation positive item per user.",
                    "better_discriminating_test": "Repeat unchanged ranking before applying intervention."}
        if stage == "reflection":
            return {"evidence_state": "inconclusive",
                    "numeric_interpretation": "Development evidence does not prove confirmation.",
                    "remaining_alternative": "The candidate mixture and model noise can explain results.",
                    "next_question": "Can the effect replicate on another independent cohort?",
                    "candidate_research_skill": "Require paired repeat controls and a numeric quality guard.",
                    "suggested_next_task": self.next_task if self.next_task in
                        properties["suggested_next_task"]["enum"] else "stop",
                    "next_task_reason": "The next study should test a distinct cohort."}
        if stage == "select":
            return {"selected_id":properties["selected_id"]["enum"][0],
                    "reason":"The specified candidate matches my modeled interests."}
        if stage == "user_update":
            return {"memory":"Observed feedback produced an updated preference summary."}
        if stage == "item_update":
            return {"first_memory":"Updated first item memory from feedback.",
                    "second_memory":"Updated second item memory from feedback."}
        if stage == "rank":
            return {"ranking":properties["ranking"]["items"]["enum"]}
        raise ValueError("unexpected model stage: " + str(stage))


class MultiAgentTests(unittest.TestCase):
    def test_roles_have_distinct_contracts_and_tools_are_not_model_controlled(self):
        model = FakeModel()
        self.assertNotEqual(DiagnosisAgent.schema, CriticAgent.schema)
        self.assertNotEqual(ReflectionAgent.schema, CriticAgent.schema)
        proposal = DesignerAgent(model).propose({"data":"development only"},
            "episode-01", ["popularity_prompt"], ["ndcg_delta"])
        self.assertEqual(proposal["action"], "popularity_prompt")
        self.assertEqual(model.roles, ["design"])
        self.assertEqual(model.scope["scientific_role"], "design")

    def test_archived_e2e_is_explicitly_replay_and_not_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"archive"
            model = FakeModel()
            result = campaign(out, ["agentcf_memory"], "archive", model, cohort_size=4)
            self.assertEqual(result["status"], "COMPLETED_DEVELOPMENT")
            self.assertEqual(model.roles[:3], ["diagnosis", "design", "critic"])
            self.assertEqual(model.roles[-1], "reflection")
            self.assertEqual(result["independently_confirmed_discoveries"], 0)
            audit = json.loads((out/"episode-001/audit.json").read_text())
            self.assertEqual(audit["evidence_grade"], "previous_development_only")
            self.assertFalse(json.loads((out/"episode-001/candidate_skill.json").read_text())["validated_skill"])
            with self.assertRaises(FileExistsError):
                campaign(out, ["agentcf_memory"], "archive", FakeModel())

    def test_real_fresh_agentcf_code_path_with_fake_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"fresh"
            model = FakeModel()
            result = campaign(out, ["agentcf_popularity"], "fresh", model,
                              cohort_size=4, seed=20261020)
            self.assertEqual(result["status"], "COMPLETED_DEVELOPMENT")
            self.assertEqual(result["independently_confirmed_discoveries"], 0)
            run = json.loads((out/"episode-001/execution.json").read_text())
            self.assertEqual(run["status"], "FRESH_AGENTCF_DEVELOPMENT")
            self.assertEqual(run["training_events"], 8)
            self.assertEqual(set(run["panels"]), {"A","B"})
            self.assertIn("select", model.roles)
            self.assertIn("rank", model.roles)
            self.assertTrue((out/"episode-001/cohort_frozen.json").exists())
            self.assertTrue((out/"episode-001/audit.json").exists())
            facts=json.loads((out/"episode-001/evidence.json").read_text())
            observation=facts["new_agentcf_input_observations"]
            self.assertEqual(observation["training_events"],8)
            self.assertIn("trained_items_in_query_candidates",observation)
            self.assertNotIn("target", json.dumps(observation).lower())
            training=json.loads((out/"episode-001/agentcf_training.json").read_text())
            self.assertEqual(len(training["trained_state_sha256"]),64)

    def test_rejected_design_never_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"rejected"
            result = campaign(out, ["agentcf_popularity"], "fresh",
                              FakeModel(reject=True), cohort_size=4)
            self.assertEqual(result["episodes"][0]["status"], "DESIGN_REJECTED")
            self.assertEqual(result["status"], "COMPLETED_NO_EXECUTION")
            self.assertEqual(result["executed_episodes"], 0)
            self.assertFalse((out/"episode-001/execution.json").exists())
            review = json.loads((out/"episode-001/design_review.json").read_text())
            self.assertEqual(len(review["revisions"]), 1)

    def test_rejected_agent_can_still_run_explicit_fixed_rule_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"control"
            result = campaign(out, ["agentcf_popularity"], "fresh",
                              FakeModel(reject=True), cohort_size=4, rule_fallback=True)
            self.assertEqual(result["episodes"][0]["status"], "COMPLETED_RULE_CONTROL")
            self.assertEqual(result["executed_episodes"], 1)
            frozen = json.loads((out/"episode-001/frozen_plan.json").read_text())
            self.assertEqual(frozen["decision"], "REJECTED_BY_CRITIC")
            audit = json.loads((out/"episode-001/audit.json").read_text())
            self.assertEqual(audit["proposal_origin"], "fixed_rule_control_after_qwen_rejection")
            self.assertTrue(audit["critic_disagreed"])
            self.assertEqual(audit["independently_confirmed_discoveries"], 0)

    def test_auto_scheduling_reuses_task_with_disjoint_new_users(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"auto"
            model = FakeModel(next_task="agentcf_popularity")
            result = campaign(out, ["agentcf_popularity"], "fresh", model,
                              max_episodes=2, cohort_size=4, auto=True)
            self.assertEqual(len(result["episodes"]), 2)
            a = json.loads((out/"episode-001/cohort_frozen.json").read_text())
            b = json.loads((out/"episode-002/cohort_frozen.json").read_text())
            self.assertFalse(set(a["users"]) & set(b["users"]))
            history = json.loads((out/"episode-002/evidence.json").read_text())
            self.assertTrue(history["relevant_prior_numeric_evidence"])
            self.assertEqual(result["independently_confirmed_discoveries"], 0)

    def test_real_coat_cpu_development_is_not_independent_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"coat"
            result = campaign(out, ["coat_propensity"], "fresh", FakeModel())
            self.assertEqual(result["status"], "COMPLETED_DEVELOPMENT")
            run = json.loads((out/"episode-001/execution.json").read_text())
            self.assertEqual(run["execution_kind"],
                             "NEW_CPU_CALCULATION_OLD_DEVELOPMENT_PANEL")
            audit = json.loads((out/"episode-001/audit.json").read_text())
            self.assertEqual(audit["independently_confirmed_discoveries"], 0)

    def test_missing_memory_support_is_inconclusive_not_discovery(self):
        plan = {"action":"item_memory_ablation", "metric":"rank_change_excess",
                "direction":"increase", "minimum_abs_effect":0.1}
        verdict = ex.independent_verdict("agentcf_memory", "item_memory_ablation",
             plan, {"status":"ABSTAIN_NO_UPDATED_CANDIDATE_MEMORY",
                    "reason":"No updated candidate memory", "panels":{}})
        self.assertEqual(verdict["evidence_grade"],"inconclusive")
        self.assertEqual(verdict["independently_confirmed_discoveries"],0)

    def test_unregistered_tasks_and_repeat_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                campaign(Path(tmp)/"bad", ["unknown_task"],"archive",FakeModel())
            with self.assertRaises(ValueError):
                campaign(Path(tmp)/"bad2", ["agentcf_memory","agentcf_memory"],
                         "archive",FakeModel())

    def test_cohort_seed_samples_distinct_training_only_users(self):
        a=ex.prepare_cohort(set(),20261019,4)
        b=ex.prepare_cohort(set(),20261019,4)
        c=ex.prepare_cohort(set(),20261020,4)
        first, repeat, another=(set(cohort[0]["users"]) for cohort in (a,b,c))
        self.assertEqual(first,repeat)
        self.assertNotEqual(first,another)
        self.assertFalse(first & ex.known_prior_users())

    def test_historical_inverted_popularity_prediction_is_rejected(self):
        plan={"action":"popularity_prompt","metric":"head_excess_delta",
              "direction":"increase", "minimum_abs_effect":.05}
        self.assertEqual(review_plan("agentcf_popularity",plan,{"decision":"approve"}),
                         "REJECTED_PREDECLARED_DIRECTION_MISMATCH")

    def test_proposal_below_fixed_metric_floor_is_rejected(self):
        plan={"action":"item_memory_ablation","metric":"rank_change_excess",
              "direction":"increase","minimum_abs_effect":.001}
        self.assertEqual(review_plan("agentcf_memory",plan,{"decision":"approve"}),
                         "REJECTED_BELOW_PREDECLARED_EFFECT_FLOOR")

    def test_null_point_estimate_cannot_establish_equivalence(self):
        row={"query_id":"1","ranking":["a","b","c","d","e"],
             "ndcg_at_5":1.0,"excess_popularity_exposure":0.0}
        result={"status":"FRESH_AGENTCF_DEVELOPMENT","panels":{
            name:{"original":[row], "repeat":[row], "popularity_prompt":[row]}
            for name in ("A","B")}}
        plan={"action":"popularity_prompt","metric":"head_excess_delta",
              "direction":"approximately_zero", "minimum_abs_effect":.02}
        audit=ex.independent_verdict("agentcf_popularity","popularity_prompt",
                                     plan,result)
        self.assertFalse(audit["forecast_passes_both_panels"])
        self.assertEqual(audit["independently_confirmed_discoveries"],0)

    def test_qwen_cannot_make_tiny_change_pass_with_zero_threshold(self):
        base={"query_id":"q1","ranking":["a","b","c","d","e"],
              "ndcg_at_5":1.0,"excess_popularity_exposure":0.0}
        shifted={**base,"excess_popularity_exposure":0.01}
        panels={name:{"original":[base],"repeat":[base],
                      "item_memory_ablation":[shifted]} for name in ("A","B")}
        proposal={"action":"item_memory_ablation","metric":"head_excess_delta",
                  "direction":"increase","minimum_abs_effect":0.0}
        verdict=ex.independent_verdict("agentcf_memory","item_memory_ablation",
                  proposal,{"status":"FRESH_AGENTCF_DEVELOPMENT","panels":panels})
        self.assertFalse(verdict["forecast_passes_both_panels"])
        self.assertEqual(verdict["model_predicted_minimum_effect"],0.0)
        self.assertEqual(verdict["evaluation_minimum_effect"],.02)

    def test_independent_python_audit_rejects_malformed_numeric_plan(self):
        with self.assertRaises(ValueError):
            ex.independent_verdict("agentcf_popularity","popularity_prompt",
                {"action":"wrong_action","metric":"ndcg_delta",
                 "direction":"increase","minimum_abs_effect":0.1}, {})


if __name__ == "__main__":
    unittest.main()
