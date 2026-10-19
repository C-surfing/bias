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
from rsi_researcher.research.scientific_loop import campaign, review_plan, fixed_rule_proposal

class FakeModel:
    def __init__(self, reject=False, next_task="stop", abstain=False):
        self.requests = 0
        self.scope = {}
        self.reject = reject
        self.next_task = next_task
        self.abstain = abstain
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
            action = ("abstain" if self.abstain else
                      "popularity_prompt" if "popularity_prompt" in actions else
                      next(x for x in actions if x != "abstain"))
            valid_metrics = properties["metric"]["enum"]
            selected_metric = next(
                (metric for metric in ex.ACTION_METRICS.get(action,())
                 if metric in valid_metrics),valid_metrics[0])
            return {"action": action,
                    "hypothesis": "The registered intervention changes ranking behavior.",
                    "competing_explanation": "Model sampling noise explains observed rank changes.",
                    "falsifier": "No paired effect beyond repeat noise in both cohorts.",
                    "metric": selected_metric,
                    "direction": "decrease" if action == "popularity_prompt" else "increase",
                    "minimum_abs_effect": ex.DEVELOPMENT_EFFECT_FLOOR[selected_metric],
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

    def test_designer_may_abstain_before_expensive_intervention(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=FakeModel(abstain=True)
            result=campaign(Path(tmp)/"abstained", ["coat_propensity"],
                            "fresh", model)
            self.assertEqual(result["status"], "COMPLETED_NO_EXECUTION")
            self.assertEqual(result["episodes"][0]["status"],
                             "ABSTAIN_NO_DISCRIMINATING_EXPERIMENT")
            self.assertNotIn("critic", model.roles)
            self.assertNotIn("rank", model.roles)
            self.assertFalse((Path(tmp)/"abstained/episode-001/execution.json").exists())

    def test_zero_baseline_target_ndcg_does_not_block_utility_improvement(self):
        panels={}
        for group in ("A","B"):
            original={"query_id":group,"ranking":["a","b","c"],
                      "ndcg_at_5":0.0,"excess_popularity_exposure":.4}
            changed={**original,"ndcg_at_5":.3,"excess_popularity_exposure":.1}
            panels[group]={"original":[original],"repeat":[original],
                           "popularity_prompt":[changed]}
        audit=ex.independent_verdict("agentcf_popularity","popularity_prompt",
            {"action":"popularity_prompt","metric":"head_excess_delta",
             "direction":"decrease","minimum_abs_effect":.02},
            {"status":"FRESH_AGENTCF_DEVELOPMENT","panels":panels})
        self.assertTrue(audit["known_target_quality_guard_passes_both_panels"])
        self.assertEqual(audit["debias_utility_state"],
                         "PROXY_GAIN_WITH_QUALITY_GUARD_DEVELOPMENT_ONLY")

    def test_mechanism_effect_remains_visible_with_known_target_quality_harm(self):
        rows={}
        for group in ("A","B"):
            original={"query_id":group,"ranking":["a","b","c","d","e"],
                      "ndcg_at_5":1.0,"excess_popularity_exposure":.4}
            repeat=dict(original)
            changed={**original,"ranking":["e","d","c","b","a"],
                     "ndcg_at_5":0.4,"excess_popularity_exposure":.1}
            rows[group]={"original":[original],"repeat":[repeat],
                         "popularity_prompt":[changed]}
        plan={"action":"popularity_prompt","metric":"head_excess_delta",
              "direction":"decrease","minimum_abs_effect":.02}
        audit=ex.independent_verdict("agentcf_popularity","popularity_prompt",
                plan,{"status":"FRESH_AGENTCF_DEVELOPMENT","panels":rows})
        self.assertTrue(audit["mechanism_forecast_passes_both_panels"])
        self.assertFalse(audit["known_target_quality_guard_passes_both_panels"])
        self.assertFalse(audit["forecast_passes_both_panels"])
        self.assertEqual(audit["debias_utility_state"],"PROXY_GAIN_WITH_UTILITY_COST")
        self.assertEqual(audit["independently_confirmed_discoveries"],0)

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

    def test_fixed_rule_uses_same_metric_floor_as_qwen(self):
        for task, expected in (("coat_propensity",.0001),
                               ("agentcf_memory",.2),
                               ("agentcf_popularity",.02)):
            rule=fixed_rule_proposal(task)
            self.assertEqual(rule["minimum_abs_effect"],expected)
            self.assertEqual(review_plan(task,rule,{"decision":"approve"}),
                             "ACCEPTED_DESIGN_ONLY")

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

    def test_cross_user_only_ablation_exposes_traceable_reader_specific_changes(self):
        # This is execution of the real author-CDs training and rank adapters
        # with deterministic inference; it does not claim Qwen discovery.
        with tempfile.TemporaryDirectory() as tmp:
            model=FakeModel()
            cohort=ex.prepare_cohort(set(),20270001,12)
            prepared=ex.train_and_observe_agentcf(cohort,model,Path(tmp))
            support=prepared["training"]["cross_user_candidate_links_by_panel"]
            self.assertGreater(support["A"],0)
            self.assertGreater(support["B"],0)
            output=Path(tmp)/"targeted"
            output.mkdir()
            result=ex.run_fresh_agentcf("agentcf_memory",
                "cross_user_memory_ablation",model,output,cohort,prepared)
            self.assertEqual(result["status"],"FRESH_AGENTCF_DEVELOPMENT")
            readers={q["query_id"]:q for q in cohort[2]}
            writers={k:set(v) for k,v in
                     prepared["training"]["item_memory_writer_users"].items()}
            restored=result["targeted_cross_user_memory_removals"]
            self.assertTrue(any(restored.values()))
            for qid,items in restored.items():
                user=readers[qid]["user"]
                self.assertTrue(set(items)<=set(readers[qid]["candidates"]))
                for item in items:
                    self.assertTrue(writers[item])
                    self.assertNotIn(user,writers[item])
            audit=ex.independent_verdict("agentcf_memory",
                "cross_user_memory_ablation",
                {"action":"cross_user_memory_ablation","metric":"linked_rank_change_excess",
                 "direction":"increase","minimum_abs_effect":.2},result)
            self.assertEqual(audit["independently_confirmed_discoveries"],0)
            self.assertTrue(all(p["linked_query_count"]>0
                                for p in audit["panel_measurements"]))
            self.assertIn("mechanism_forecast_state",audit)
            self.assertIn("debias_utility_state",audit)

    def test_designer_can_select_cross_user_specific_falsifier(self):
        class FocusedModel(FakeModel):
            def ask(self,prompt,schema,context):
                result=super().ask(prompt,schema,context)
                if context["stage"]=="design":
                    result.update(action="cross_user_memory_ablation",
                                  metric="linked_rank_change_excess",
                                  minimum_abs_effect=.2,direction="increase")
                return result
        with tempfile.TemporaryDirectory() as tmp:
            model=FocusedModel()
            result=campaign(Path(tmp)/"targeted",["agentcf_memory"],"fresh",
                            model,cohort_size=12,seed=20270001)
            self.assertEqual(result["executed_episodes"],1)
            episode=Path(tmp)/"targeted/episode-001"
            self.assertEqual(json.loads((episode/"frozen_plan.json").read_text())
                             ["proposal"]["action"],"cross_user_memory_ablation")
            self.assertIn("cross_user_memory_ablation",
                json.loads((episode/"evidence.json").read_text())
                ["action_metric_contracts"])

    def test_feedback_semantics_prompt_changes_only_memory_writing(self):
        prompts=json.loads((ex.LOCAL_DATA/"author_prompts.json").read_text())
        neutral=ex.neutral_feedback_prompts(prompts)
        changed={k for k in prompts if neutral[k]!=prompts[k]}
        self.assertEqual(changed,{
            "user_prompt_template","user_prompt_template_true",
            "item_prompt_template","item_prompt_template_true"})
        self.assertEqual(
            neutral["system_prompt_template_evaluation_basic"],
            prompts["system_prompt_template_evaluation_basic"])
        self.assertIn("unknown exposure",neutral["item_prompt_template_true"])
        self.assertIn("no dislike",neutral["user_prompt_template_true"].lower())

    def test_real_feedback_semantics_path_uses_original_training_repeat_control(self):
        class PromptSensitiveModel(FakeModel):
            def ask(self,prompt,schema,context):
                answer=super().ask(prompt,schema,context)
                if context["stage"]=="item_update":
                    # Deterministic oracle for prompt routing, NOT a real
                    # measurement of how Qwen interprets false negatives.
                    if "sampled comparison CD" in prompt:
                        answer={"first_memory":"Neutral semantic Item Memory.",
                                "second_memory":"Neutral semantic Item Memory."}
                if context["stage"]=="rank":
                    candidates=schema["properties"]["ranking"]["items"]["enum"]
                    answer={"ranking":list(reversed(candidates))
                            if "Neutral semantic Item Memory." in prompt
                            else candidates}
                return answer

        with tempfile.TemporaryDirectory() as tmp:
            model=PromptSensitiveModel()
            cohort=ex.prepare_cohort(set(),20270001,12)
            prepared=ex.train_and_observe_agentcf(cohort,model,Path(tmp))
            out=Path(tmp)/"semantic"
            out.mkdir()
            run=ex.run_fresh_agentcf("agentcf_memory",
                "neutral_feedback_memory_propagation",model,out,cohort,prepared)
            self.assertEqual(run["status"],"FRESH_AGENTCF_DEVELOPMENT")
            self.assertTrue(run["training_repeat_control_used"])
            protocol=json.loads((out/"feedback_semantics_protocol.json").read_text())
            self.assertEqual(len(protocol["intervened_memory_writing_template_keys"]),4)
            self.assertGreater(sum(protocol["changed_readers_by_panel"].values()),0)
            self.assertEqual(set(protocol["linked_readers_by_panel"]),{"A","B"})
            self.assertEqual(protocol["original_training_state_sha256"],
                             prepared["training"]["trained_state_sha256"])
            self.assertTrue(all("writer_user" in event and
                                "observed_positive_item" in event
                                for event in protocol["neutral_feedback_training_calls"]))
            self.assertTrue(all("sampling" not in key for key in protocol))
            for panel in run["panels"].values():
                self.assertIn("training_repeat",panel)
                self.assertEqual(len(panel["original"]),len(panel["training_repeat"]))
            plan={"action":"neutral_feedback_memory_propagation",
                  "metric":"feedback_semantic_excess_over_training_repeat",
                  "direction":"increase","minimum_abs_effect":.2}
            verdict=ex.independent_verdict("agentcf_memory",plan["action"],plan,run)
            self.assertEqual(len(verdict["panel_measurements"]),2)
            for panel in verdict["panel_measurements"]:
                self.assertGreater(panel["linked_query_count"],0)
                self.assertEqual(panel["unchanged_training_repeat_change_rate"],0)
                self.assertGreaterEqual(panel["feedback_semantic_excess_over_training_repeat"],.2)
            self.assertTrue(verdict["mechanism_forecast_passes_both_panels"])
            self.assertEqual(verdict["independently_confirmed_discoveries"],0)

    def test_training_noise_can_explain_apparent_feedback_semantic_rank_change(self):
        original={"query_id":"q1","ranking":["a","b","c"],
                  "ndcg_at_5":1.0,"excess_popularity_exposure":0.0}
        reversed_row={**original,"ranking":["c","b","a"]}
        result={"action":"neutral_feedback_memory_propagation",
                "status":"FRESH_AGENTCF_DEVELOPMENT",
                "eligible_readers_by_query":{"q1":["a"]},
                "panels":{group:{
                    "original":[original],"repeat":[original],
                    "training_repeat":[reversed_row],
                    "neutral_feedback_memory_propagation":[reversed_row]}
                    for group in ("A","B")}}
        plan={"action":"neutral_feedback_memory_propagation",
              "metric":"feedback_semantic_excess_over_training_repeat",
              "direction":"increase","minimum_abs_effect":.2}
        audit=ex.independent_verdict("agentcf_memory",plan["action"],plan,result)
        self.assertEqual([x["feedback_semantic_excess_over_training_repeat"]
                          for x in audit["panel_measurements"]],[0,0])
        self.assertFalse(audit["mechanism_forecast_passes_both_panels"])
        self.assertEqual(audit["independently_confirmed_discoveries"],0)

    def test_feedback_semantics_without_memory_effect_abstains(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=FakeModel()
            cohort=ex.prepare_cohort(set(),20270001,12)
            prepared=ex.train_and_observe_agentcf(cohort,model,Path(tmp))
            out=Path(tmp)/"no_effect"
            out.mkdir()
            result=ex.run_fresh_agentcf("agentcf_memory",
                "neutral_feedback_memory_propagation",model,out,cohort,prepared)
            self.assertEqual(result["status"],
                             "ABSTAIN_NO_FEEDBACK_SEMANTIC_MEMORY_CHANGE")
            self.assertEqual(ex.independent_verdict("agentcf_memory",
                    "neutral_feedback_memory_propagation",
                    {"action":"neutral_feedback_memory_propagation",
                     "metric":"feedback_semantic_excess_over_training_repeat",
                     "direction":"increase","minimum_abs_effect":.2},result)
                             ["independently_confirmed_discoveries"],0)

    def test_shared_memory_without_cross_user_read_abstains_not_finds_bias(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=FakeModel()
            cohort=ex.prepare_cohort(set(),20261025,4)
            prepared=ex.train_and_observe_agentcf(cohort,model,Path(tmp))
            prepared["training"]["cross_user_candidate_memory_links"]=0
            result=ex.run_fresh_agentcf("agentcf_memory","item_memory_ablation",
                                       model,Path(tmp),cohort,prepared)
            self.assertEqual(result["status"],"ABSTAIN_NO_CROSS_USER_MEMORY_PATH")
            proposal={"action":"item_memory_ablation","metric":"rank_change_excess",
                      "direction":"increase","minimum_abs_effect":.2}
            audit=ex.independent_verdict("agentcf_memory","item_memory_ablation",
                                         proposal,result)
            self.assertEqual(audit["evidence_grade"],"inconclusive")
            self.assertEqual(audit["independently_confirmed_discoveries"],0)

    def test_memory_replication_requires_cross_user_support_in_both_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=FakeModel()
            cohort=ex.prepare_cohort(set(),20261025,4)
            prepared=ex.train_and_observe_agentcf(cohort,model,Path(tmp))
            prepared["training"]["cross_user_candidate_memory_links"]=1
            prepared["training"]["cross_user_candidate_links_by_panel"]={"A":1,"B":0}
            result=ex.run_fresh_agentcf("agentcf_memory","item_memory_ablation",
                                       model,Path(tmp),cohort,prepared)
            self.assertEqual(result["status"],
                             "ABSTAIN_UNSUPPORTED_CROSS_USER_REPLICATION")

    def test_scientific_abstention_does_not_count_as_executed(self):
        from unittest.mock import patch
        simulated={"action":"popularity_prompt",
                   "status":"ABSTAIN_NO_CROSS_USER_MEMORY_PATH",
                   "reason":"no cross-user memory path","panels":{}}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "rsi_researcher.research.scientific_loop.experiments.execute",
                return_value=simulated):
            result=campaign(Path(tmp)/"abstain",["agentcf_popularity"],
                            "fresh",FakeModel(),cohort_size=4)
            self.assertEqual(result["status"],"COMPLETED_NO_EXECUTION")
            self.assertEqual(result["executed_episodes"],0)
            self.assertEqual(result["episodes"][0]["status"],
                             "ABSTAIN_INSUFFICIENT_MECHANISM_SUPPORT")

    def test_independent_python_audit_rejects_malformed_numeric_plan(self):
        with self.assertRaises(ValueError):
            ex.independent_verdict("agentcf_popularity","popularity_prompt",
                {"action":"wrong_action","metric":"ndcg_delta",
                 "direction":"increase","minimum_abs_effect":0.1}, {})


if __name__ == "__main__":
    unittest.main()
