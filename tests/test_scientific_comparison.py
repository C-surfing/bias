"""Fair comparisons use separate budgets and identical author cohorts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_researcher.research.scientific_comparison import compare_development
from tests.test_scientific_loop import FakeModel


def config():
    return {"llm":{"provider":"vllm","model":"qwen3-8b",
                   "base_url":"http://127.0.0.1:18351/v1"},
            "budget":{"max_requests":80,"max_tokens":800000,"max_seconds":1200}}


class ComparisonTests(unittest.TestCase):
    def test_same_cohort_and_equal_budget_for_all_arms(self):
        called=[]
        def fake_campaign(directory,tasks,mode,**kwargs):
            directory=Path(directory)
            directory.mkdir()
            (directory/"episode-001").mkdir()
            ep=directory/"episode-001"
            (ep/"cohort_frozen.json").write_text(json.dumps({
                "users":["a","b","c","d"],"scientific_input_sha256":"a"*64}))
            (ep/"execution.json").write_text(json.dumps({
                "action":"popularity_prompt","status":"FRESH_AGENTCF_DEVELOPMENT"}))
            (ep/"audit.json").write_text(json.dumps({
                "task":"agentcf_popularity","action":"popularity_prompt",
                "independently_confirmed_discoveries":0}))
            (ep/"agentcf_training.json").write_text(json.dumps({
                "trained_state_sha256":"c"*64}))
            called.append((kwargs["arm"],kwargs["seed"],kwargs["config"]["budget"]))
            return {"status":"COMPLETED_DEVELOPMENT","executed_episodes":1,
                    "independently_confirmed_discoveries":0}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "rsi_researcher.research.scientific_comparison.campaign",fake_campaign), patch(
                "rsi_researcher.research.scientific_comparison.prepare_shared_states",
                return_value=({},{"requests":0,"usage_measured":False})):
            out=Path(tmp)/"compare"
            result=compare_development(out,["agentcf_popularity"],config(),
                arms=("strong_rule","qwen_no_history","qwen_audited_history","qwen_shuffled_history"),
                cohort_size=4,seed=77)
            self.assertEqual(len(called),4)
            self.assertEqual({x[1] for x in called},{77})
            self.assertEqual(len({str(x[2]) for x in called}),1)
            self.assertTrue(result["cohort_pairing"]["agentcf_popularity"]
                            ["identical_frozen_user_cohorts"])
            self.assertFalse(result["memory_comparison_interpretable"])
            self.assertTrue(result["cohort_pairing"]["agentcf_popularity"]
                            ["identical_frozen_experiment_inputs"])
            self.assertEqual(result["status"],"COMPLETED_DEVELOPMENT")
            self.assertTrue(result["policy_arms_share_trained_state"])
            self.assertEqual(result["independently_confirmed_scientific_discoveries"],0)
            with self.assertRaises(FileExistsError):
                compare_development(out,["agentcf_popularity"],config(),cohort_size=4)
            # A prior-campaign PATH does not prove the Qwen history arm saw a record.
            unused=compare_development(Path(tmp)/"empty_history",["agentcf_popularity"],
                config(),arms=("qwen_no_history","qwen_audited_history"),
                cohort_size=4,prior_campaign="dummy-previous-run")
            self.assertFalse(unused["memory_comparison_interpretable"])
            self.assertEqual(unused["cohort_pairing"]["agentcf_popularity"]
                             ["history_records_seen_by_arm"]["qwen_audited_history"],0)

    def test_real_orchestrator_reuses_one_trained_state_across_policy_arms(self):
        """FakeModel exercises the production trainer and both real campaigns."""
        created={}
        def fresh_model(directory, *args, **kwargs):
            model=FakeModel()
            created[str(directory)]=model
            return model
        with tempfile.TemporaryDirectory() as tmp, patch(
                "rsi_researcher.research.scientific_comparison.create_model",
                side_effect=fresh_model), patch(
                "rsi_researcher.research.scientific_loop.create_model",
                side_effect=fresh_model):
            result=compare_development(Path(tmp)/"paired",
                ["agentcf_popularity"],config(),
                arms=("strong_rule","qwen_no_history"),cohort_size=4,seed=20261025)
            pairing=result["cohort_pairing"]["agentcf_popularity"]
            self.assertEqual(result["status"],"COMPLETED_DEVELOPMENT")
            self.assertTrue(pairing["identical_trained_agentcf_states"])
            self.assertTrue(result["policy_arms_share_trained_state"])
            self.assertGreater(result["shared_training_model_usage"]["requests"],0)
            shared=Path(tmp)/"paired/shared_training"
            self.assertEqual(len(list(shared.glob("task-*/prepared_agentcf.json"))),1)
            self.assertFalse(list((Path(tmp)/"paired/strong_rule").rglob("prepared_agentcf.json")))
            for arm in ("strong_rule","qwen_no_history"):
                episode=Path(tmp)/"paired"/arm/"episode-001"
                evidence=json.loads((episode/"evidence.json").read_text())
                self.assertTrue(evidence["shared_preparation_cost_accounted_separately"])
                facts=evidence["new_agentcf_input_observations"]
                self.assertTrue(facts["observed_training_memory_changes"])
                self.assertIn("mean_baseline_head_excess_vs_candidate_mix",facts)
                self.assertNotIn('"target":',json.dumps(facts))
                ranks=json.loads((episode/"agentcf_ranking_panels.json").read_text())
                self.assertEqual(ranks["A"]["original"][0]["origin"],
                                 "predecision_frozen_baseline")
                self.assertEqual(ranks["A"]["original"][0]["calls"],0)
            self.assertNotIn("diagnosis",created[str(Path(tmp)/"paired/strong_rule")].roles)
            self.assertIn("diagnosis",created[str(Path(tmp)/"paired/qwen_no_history")].roles)

    def test_invalid_comparison_fails_before_executing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"bad"
            with self.assertRaises(ValueError):
                compare_development(out,["agentcf_popularity","agentcf_popularity"],
                                    config(),cohort_size=4)
            self.assertFalse(out.exists())
            with self.assertRaises(ValueError):
                compare_development(out,["unregistered_bias"],config(),cohort_size=4)
            with self.assertRaises(ValueError):
                compare_development(out,["agentcf_popularity"],
                                    {**config(),"llm":{"provider":"deepseek"}},cohort_size=4)

    def test_rejected_arm_does_not_get_fabricated_numeric_outcome(self):
        def rejected(directory,tasks,mode,**kwargs):
            directory=Path(directory)
            directory.mkdir()
            return {"status":"COMPLETED_NO_EXECUTION","executed_episodes":0}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "rsi_researcher.research.scientific_comparison.campaign",rejected), patch(
                "rsi_researcher.research.scientific_comparison.prepare_shared_states",
                return_value=({},{"requests":0,"usage_measured":False})):
            result=compare_development(Path(tmp)/"test",["agentcf_memory"],
                                       config(),arms=("qwen_no_history",),cohort_size=4)
            paired=result["cohort_pairing"]["agentcf_memory"]
            self.assertFalse(paired["all_arms_ran_experiment"])
            self.assertFalse(paired["identical_frozen_user_cohorts"])
            self.assertEqual(result["status"],"INCOMPLETE_DEVELOPMENT_COMPARISON")

    def test_equal_inputs_but_different_trained_memories_only_support_end_to_end(self):
        def varying_memory(directory,tasks,mode,**kwargs):
            ep=Path(directory)/"episode-001"
            ep.mkdir(parents=True)
            (ep/"cohort_frozen.json").write_text(json.dumps({
                "users":["a","b","c","d"], "scientific_input_sha256":"b"*64}))
            (ep/"execution.json").write_text(json.dumps({
                "action":"popularity_prompt", "status":"FRESH_AGENTCF_DEVELOPMENT"}))
            (ep/"audit.json").write_text(json.dumps({
                "task":"agentcf_popularity", "action":"popularity_prompt",
                "independently_confirmed_discoveries":0}))
            (ep/"agentcf_training.json").write_text(json.dumps({
                "trained_state_sha256":kwargs["arm"]}))
            return {"status":"COMPLETED_DEVELOPMENT", "executed_episodes":1}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "rsi_researcher.research.scientific_comparison.campaign", varying_memory), patch(
                "rsi_researcher.research.scientific_comparison.prepare_shared_states",
                return_value=({},{"requests":0,"usage_measured":False})):
            result=compare_development(Path(tmp)/"comp",["agentcf_popularity"],
                config(),arms=("strong_rule","qwen_no_history"),cohort_size=4)
            self.assertEqual(result["status"],"COMPLETED_DEVELOPMENT")
            self.assertFalse(result["policy_arms_share_trained_state"])
            pair=result["cohort_pairing"]["agentcf_popularity"]
            self.assertTrue(pair["identical_frozen_experiment_inputs"])
            self.assertFalse(pair["identical_trained_agentcf_states"])

    def test_same_users_but_different_inputs_is_not_a_paired_comparison(self):
        def mismatch(directory,tasks,mode,**kwargs):
            ep=Path(directory)/"episode-001"
            ep.mkdir(parents=True)
            (ep/"cohort_frozen.json").write_text(json.dumps({
                "users":["a","b","c","d"],
                "scientific_input_sha256":kwargs["arm"]*4}))
            (ep/"execution.json").write_text(json.dumps({
                "action":"popularity_prompt","status":"FRESH_AGENTCF_DEVELOPMENT"}))
            (ep/"audit.json").write_text(json.dumps({
                "task":"agentcf_popularity","action":"popularity_prompt",
                "independently_confirmed_discoveries":0}))
            return {"status":"COMPLETED_DEVELOPMENT","executed_episodes":1}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "rsi_researcher.research.scientific_comparison.campaign",mismatch), patch(
                "rsi_researcher.research.scientific_comparison.prepare_shared_states",
                return_value=({},{"requests":0,"usage_measured":False})):
            with self.assertRaisesRegex(ValueError,"inputs differ"):
                compare_development(Path(tmp)/"mismatch",["agentcf_popularity"],
                                    config(),arms=("strong_rule","qwen_no_history"),
                                    cohort_size=4)
            saved=json.loads((Path(tmp)/"mismatch/comparison.json").read_text())
            self.assertEqual(saved["status"],"FAILED_CLOSED")

if __name__=="__main__":
    unittest.main()
