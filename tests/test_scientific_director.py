"""Tests for the OUTER RSI research controller, not a finding of RSI progress."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_researcher.research.scientific_director import (
    conduct, decide_acceptance, eligibility,
)
from rsi_researcher.research.scientific_loop import verified_experience
from rsi_researcher.research import scientific_experiments as ex
from tests.test_scientific_loop import FakeModel


class ContinuousDirectorTests(unittest.TestCase):
    def test_multiple_real_data_development_cycles_and_cross_cycle_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"director"
            model=FakeModel()
            result=conduct(output,["coat_utility","agentcf_popularity",
                                   "agentcf_memory"],
                           model=model,seed=20270001,cohort_size=8,
                           max_cycles=3,max_nodes_per_cycle=2,
                           max_total_requests=260,max_stagnant_cycles=3)
            self.assertEqual(len(result["cycles"]),3)
            self.assertIn(result["state"],(
                "STOP_NO_IDENTIFIABLE_PROGRESS","STOP_SCHEDULE_EXHAUSTED"))
            self.assertTrue(all(c["status"] != "FAILED_CLOSED" for c in result["cycles"]))
            self.assertTrue(all(entry["tree_ref"] for entry in result["cycles"]))
            self.assertFalse(result["scientific_goal_accepted"])
            self.assertEqual(result["total_model_requests"],model.requests)
            self.assertEqual(result["independently_confirmed_discoveries"],0)
            popularity=result["cycles"][1]
            memory=result["cycles"][2]
            self.assertEqual(memory["seed"],popularity["seed"])
            self.assertFalse(popularity["shared_development_state_reused"])
            self.assertTrue(memory["shared_development_state_reused"])
            t1=json.loads((output/memory["tree_ref"]/"tree_index.json").read_text())
            t2=json.loads((output/popularity["tree_ref"]/"tree_index.json").read_text())
            self.assertEqual(t1["training_sha256"],t2["training_sha256"])
            self.assertEqual(t1["state_ref"],t2["state_ref"])
            self.assertTrue((output/"director_state.json").is_file())
            self.assertTrue((output/"protocol.sha256.json").is_file())
            history=verified_experience(Path(tmp)/"another",output)
            self.assertTrue(history)
            self.assertTrue(all("source_sha256" in entry for entry in history))

    def test_no_writer_reader_overlap_abstains_before_any_qwen_training(self):
        from rsi_researcher.research.scientific_tree import search
        synthetic=(
            {"users":{"u1":"","u2":""}},
            [{"event_id":"e1","user":"u1","positive":"i1","negative":"i2"}],
            [{"query_id":"a","user":"u1","candidates":["i1","i2"]},
             {"query_id":"b","user":"u2","candidates":["i2","i3"]}])
        model=FakeModel()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("rsi_researcher.research.scientific_tree."
                       "experiments.prepare_cohort",return_value=synthetic):
                tree=search(Path(tmp)/"no_support","agentcf_memory",
                            model=model,cohort_size=4)
            self.assertEqual(tree["status"],
                             "ABSTAIN_PREDECISION_MECHANISM_SUPPORT")
            self.assertEqual(tree["identifiability_gate"],
                             "ABSTAIN_UNIDENTIFIABLE_CROSS_USER_MEMORY")
            self.assertEqual(model.requests,0)
            self.assertEqual(tree["attempted_nodes"],0)
            self.assertIsNone(tree["state_ref"])
            self.assertEqual(json.loads((Path(tmp)/
                "no_support/identifiability_gate.json").read_text())["phase"],
                "PRETRAINING_STRUCTURAL_SUPPORT")

    def test_preflight_blocked_memory_prevents_all_diagnostic_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            from rsi_researcher.research.scientific_tree import search
            model=FakeModel()
            actual=ex.train_and_observe_agentcf
            def lacking_support(cohort, model, output):
                prepared=actual(cohort,model,output)
                prepared["training"]["cross_user_candidate_links_by_panel"]={
                    "A":0,"B":0}
                return prepared
            with patch("rsi_researcher.research.scientific_tree."
                       "experiments.train_and_observe_agentcf",
                       side_effect=lacking_support):
                tree=search(Path(tmp)/"preflight","agentcf_memory",model=model,
                            seed=20270001,cohort_size=8)
            self.assertEqual(tree["status"],
                             "ABSTAIN_PREDECISION_MECHANISM_SUPPORT")
            self.assertEqual(tree["attempted_nodes"],0)
            self.assertEqual(len(tree["nodes"]),0)
            self.assertFalse(any(role in ("diagnosis","design","critic","reflection")
                                 for role in model.roles))
            evidence=json.loads((Path(tmp)/"preflight/identifiability_gate.json")
                                 .read_text())
            self.assertFalse(evidence["passes"])
            self.assertFalse(evidence["observes_validation_labels"])
            self.assertEqual(tree["independently_confirmed_discoveries"],0)

    def test_readiness_stops_early_but_never_auto_promotes_science(self):
        # Test the control flow, not a true independently confirmed result.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"qualified"
            model=FakeModel()
            def stub_tree(output,task,**kwargs):
                path=Path(output)
                (path/"nodes"/"N001").mkdir(parents=True)
                audit={"task":task,"action":"split_utility",
                       "forecast_metric":"brier_gain",
                       "predeclared_direction":"increase",
                       "execution_kind":"FRESH_DEVELOPMENT",
                       "mechanism_forecast_passes_both_panels":True,
                       "registered_action_falsifier_passes":True,
                       "known_target_quality_guard_passes_both_panels":True,
                       "falsification_state":"DEVELOPMENT_PATTERN",
                       "evidence_grade":"new_unconfirmed_development",
                       "panel_measurements":[],"reason":"Synthetic control test",
                       "independently_confirmed_discoveries":0}
                (path/"nodes/N001/audit.json").write_text(json.dumps(audit))
                model.requests+=1
                return {"status":"COMPLETED_DEVELOPMENT_SEARCH",
                        "state_ref":None,"attempted_nodes":1,
                        "executed_development_nodes":1,
                        "nodes":{"N001":{"audit_ref":"nodes/N001/audit.json"}}}
            with patch("rsi_researcher.research.scientific_director.search",
                       side_effect=stub_tree):
                result=conduct(root,["coat_utility","agentcf_popularity",
                                     "agentcf_memory"],
                               model=model,max_cycles=3,
                               max_stagnant_cycles=3,min_ready=1,min_families=1)
            self.assertEqual(len(result["cycles"]),1)
            self.assertEqual(result["state"],
                "STOP_DEVELOPMENT_READY_FOR_INDEPENDENT_CONFIRMATION")
            self.assertTrue(result["acceptance"]["development_readiness_met"])
            self.assertFalse(result["acceptance"]["scientific_success"])
            self.assertFalse(result["scientific_goal_accepted"])
            self.assertEqual(result["independently_confirmed_discoveries"],0)

    def test_two_agent_task_names_do_not_count_as_two_biases(self):
        result=decide_acceptance([{"ready_candidates":[
            {"task":"coat_utility","bias_task_id":"selection"},
            {"task":"coat_propensity","bias_task_id":"selection"}]}],
            min_ready=2,min_families=2)
        self.assertFalse(result["development_readiness_met"])
        self.assertEqual(result["distinct_ready_families"],["selection"])
        self.assertTrue(result[
            "default_two_bias_acceptance_currently_blocked_by_missing_identified_methods"])

    def test_no_fake_readiness_from_unconfirmed_scalar_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            audit={"task":"coat_utility","action":"gamma_stress",
                   "execution_kind":"NEW_CPU_CALCULATION_OLD_DEVELOPMENT_PANEL",
                   "mechanism_forecast_passes_both_panels":True,
                   "registered_action_falsifier_passes":False,
                   "known_target_quality_guard_passes_both_panels":True,
                   "falsification_state":"REGISTERED_ACTION_FALSIFIER_FAILED"}
            (run/"audit.json").write_text(json.dumps(audit))
            self.assertEqual(eligibility(run,{"nodes":{
                "N001":{"audit_ref":"audit.json"}}}),[])
            self.assertFalse(decide_acceptance([],min_ready=1,
                                               min_families=1)
                             ["development_readiness_met"])

    def test_support_failure_redirects_to_contrasting_same_cohort_task(self):
        from rsi_researcher.research.scientific_director import choose_next
        pending=[{"task":"coat_utility","seed":20270001},
                 {"task":"agentcf_popularity","seed":20270001}]
        last={"task":"agentcf_memory","seed":20270001,
              "status":"ABSTAIN_PREDECISION_MECHANISM_SUPPORT"}
        i,reason=choose_next(pending,last)
        self.assertEqual(pending[i]["task"],"agentcf_popularity")
        self.assertEqual(reason,
            "memory_support_failure_switch_to_same_cohort_popularity")
        self.assertEqual(choose_next(pending,None)[0],0)

    def test_input_validation_and_repeated_coat_never_claims_new_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):
                conduct(root/"duplicate",["coat_utility","coat_utility"],
                        model=FakeModel())
            with self.assertRaises(ValueError):
                conduct(root/"infinite",["coat_utility"],model=FakeModel(),
                        max_cycles=100000)
            result=conduct(root/"coat",["coat_utility"],model=FakeModel(),
                           max_visits_per_task=3,max_cycles=3,
                           max_stagnant_cycles=3)
            self.assertEqual(len(result["cycles"]),1)
            self.assertEqual(result["state"],"STOP_SCHEDULE_EXHAUSTED")


if __name__ == "__main__":
    unittest.main()
