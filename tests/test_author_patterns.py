"""Research evidence checks: noise controls, source histories and paired inference."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsi_researcher.evaluation.author import score_author_ranking
from rsi_researcher.evaluation.author_patterns import paired_quality, order_control_effect, describe_run, pattern_brief
from rsi_researcher.research.author_experiment import controlled_order_rankings, rank_only
from rsi_researcher.research.agentcf_loop import compare_bias, _render_report
from rsi_researcher.research.author_evidence import record_author_trial, compact_author_history
from rsi_researcher.research.journal import ExperimentJournal
from rsi_researcher.paths import ROOT
from tests.test_agentcf_loop import _summary


PROFILE={"training_item_counts":{"a":3,"b":2,"c":1},"head_min_count":3,"tail_max_count":1}


def row(order=("a","b","c"),ranking=("a","b","c"),user="u"):
    query={"query_id":"q-"+user,"user":user,"target":"a","candidates":list(order)}
    return {"query":query,"ranking":list(ranking),"valid":True,**score_author_ranking(list(ranking),query,PROFILE)}


class PairedQualityTests(unittest.TestCase):
    def test_identical_positive_quality_supports_the_floor_conditionally(self):
        rows=[row(user=str(i)) for i in range(4)]
        result=paired_quality(rows,rows,samples=200)
        self.assertEqual(result["ndcg_change_95"],[0,0])
        self.assertEqual(result["quality_evidence"],"SUPPORTED_CONDITIONALLY")
        self.assertEqual(result["target_groups"]["head"]["users"],4)

    def test_changed_target_or_candidate_pool_is_not_a_paired_experiment(self):
        candidate=row()
        candidate["query"]["target"]="b"
        with self.assertRaises(ValueError):
            paired_quality([row()],[candidate])

    def test_failed_rankings_stay_in_the_sample_but_cannot_be_scientific_support(self):
        failed=row(ranking=())
        failed["valid"]=False
        result=paired_quality([row()],[failed],samples=50)
        self.assertEqual(result["users"],1)
        self.assertEqual(result["candidate_ndcg"],0)
        self.assertEqual(result["quality_evidence"],"INVALID_EXECUTION")

    def test_observed_quality_loss_is_distinct_from_no_reference_quality(self):
        worst=row(ranking=())
        self.assertEqual(paired_quality([row()],[worst],samples=50)["quality_evidence"],"QUALITY_LOSS_SUPPORTED_CONDITIONALLY")
        self.assertEqual(paired_quality([worst],[worst],samples=50)["quality_evidence"],"NO_REFERENCE_QUALITY")


class OrderControlTests(unittest.TestCase):
    def test_copied_order_has_flip_effect_without_sampling_noise(self):
        result=order_control_effect([row()],[row()],[row(order=("c","b","a"),ranking=("c","b","a"))])
        self.assertEqual(result["same_order_distance"],0)
        self.assertEqual(result["excess_order_instability"],1)

    def test_sampling_noise_is_not_attributed_to_the_input_order(self):
        result=order_control_effect([row()],[row(ranking=("c","b","a"))],
                                    [row(order=("c","b","a"),ranking=("c","b","a"))])
        self.assertEqual(result["excess_order_instability"],0)

    def test_changed_feedback_is_rejected_even_when_query_ids_match(self):
        wrong=row(order=("c","b","a"))
        wrong["query"]["user"]="another"
        with self.assertRaises(ValueError):
            order_control_effect([row()],[row()],[wrong])

    def test_six_panels_share_one_state_and_only_call_ranking(self):
        prompts=json.loads((ROOT/"data/author_prompts.json").read_text(encoding="utf-8"))
        state={"users":{"u":"I like music."},"items":{i:{"title":i,"memory":i} for i in "abc"},"completed":[]}
        before=deepcopy(state)
        class Model:
            scope={}
            def __init__(self): self.calls=[]
            def ask(self,prompt,schema,context):
                self.calls.append(context)
                return {"ranking":list("abc")}
        model=Model()
        with TemporaryDirectory() as temporary:
            directory=Path(temporary)
            primary=rank_only(directory,model,state,[row()["query"]],prompts,PROFILE,"neutral_candidate_order")
            result=controlled_order_rankings(directory,model,state,[row()["query"]],prompts,PROFILE,"neutral_candidate_order",primary)
            self.assertTrue((directory/"order_controls.json").exists())
        self.assertEqual(len(model.calls),6)
        self.assertTrue(all(c["stage"]=="rank" for c in model.calls))
        self.assertEqual(state,before)
        self.assertEqual(result["baseline"]["order_effect"]["excess_order_instability"],0)

    def test_position_success_requires_controls_and_cannot_be_bought_with_noise(self):
        baseline=_summary(.4,.03,.7,.7,.3)
        candidate=_summary(.4,.03,.7,.7,.01)
        self.assertFalse(compare_bias(baseline,candidate,"position")["target_gate"])
        quality={"mean_ndcg_at_5":.4,"mean_hit_at_5":.7,"mean_top5_head_share":.3,"mean_candidate_head_share":.2}
        effect=lambda noise,flip:{"queries":20,"valid_queries":20,"same_order_distance":noise,
                                  "reversed_order_distance":flip,"excess_order_instability":flip-noise}
        candidate["position_bias"]["controls"]={"baseline":{"quality":quality,"order_effect":effect(0,1)},
                                               "candidate":{"quality":quality,"order_effect":effect(0,0)}}
        good=compare_bias(baseline,candidate,"position")
        self.assertTrue(good["proxy_goal_pass"])
        self.assertEqual(good["target"]["label"],"noise_adjusted_order_instability")
        candidate["position_bias"]["controls"]["candidate"]["order_effect"]=effect(.5,.5)
        self.assertFalse(compare_bias(baseline,candidate,"position")["proxy_goal_pass"])

    def test_missing_order_measurements_remain_a_recorded_failure(self):
        baseline=_summary(.4,.03,.7,.7,.3)
        candidate=_summary(.4,.03,.7,.7,.01)
        quality={"mean_ndcg_at_5":.4,"mean_hit_at_5":.7,"mean_top5_head_share":.3,"mean_candidate_head_share":.2}
        missing={"queries":20,"valid_queries":0,"same_order_distance":None,
                 "reversed_order_distance":None,"excess_order_instability":None}
        candidate["position_bias"]["controls"]={label:{"quality":quality,"order_effect":missing} for label in ("baseline","candidate")}
        comparison=compare_bias(baseline,candidate,"position")
        self.assertFalse(comparison["target_gate"])
        archive={"status":"completed","rounds":1,"baseline_probes":baseline,
                 "trials":[{"round":1,"operator":"neutral_candidate_order","plan":{"targeted_bias":"position"},
                            "comparison":comparison,"learn":{}}],"selection":{"selected":False,"reason":"measurement incomplete"}}
        with TemporaryDirectory() as temporary:
            _render_report(Path(temporary),archive)
            self.assertIn("未测得",(Path(temporary)/"report.md").read_text(encoding="utf-8"))


class HistoryTests(unittest.TestCase):
    def test_new_claims_persisted_claims_and_shared_memory_paths_are_distinct(self):
        with TemporaryDirectory() as temporary:
            path=Path(temporary);arm=path/"original_update";arm.mkdir()
            items={k:{"title":"Album "+k.upper(),"memory":"Initial "+k} for k in "xyzwabcdef"}
            initial={"users":{"u":"I like music.","v":"I like music."},"items":items,"completed":[]}
            events=[{"event_id":"e1","user":"u","positive":"x","negative":"y","candidates":["y","x"]},
                    {"event_id":"e2","user":"v","positive":"x","negative":"z","candidates":["z","x"]},
                    {"event_id":"e3","user":"u","positive":"w","negative":"y","candidates":["y","w"]}]
            state=deepcopy(initial)
            for event in events:
                before=deepcopy(state)
                if event["user"]=="u": state["users"]["u"]="I dislike Album Y."
                state["items"][event["positive"]]["memory"]="Updated "+event["event_id"]
                state["completed"].append(event["event_id"])
                (arm/(event["event_id"]+".json")).write_text(json.dumps({"event":event,"before":before,"after":state}),encoding="utf-8")
            query={"query_id":"q-u","user":"u","target":"x","candidates":list(items)}
            profile={"dataset":"fixture","training_item_counts":{"x":3,"w":1},"head_min_count":3,"tail_max_count":1}
            for filename,data in [(path/"input.json",{"initial":initial,"events":events}),
                                  (path/"data_profile.json",profile),(arm/"state.json",state),
                                  (arm/"ranking.json",[{"query":query,"ranking":list(items),"valid":True}])]:
                filename.write_text(json.dumps(data),encoding="utf-8")
            result=describe_run(path)
            self.assertEqual(result["patterns"]["exposure"]["new_explicit_claim_events"],1)
            self.assertEqual(result["patterns"]["exposure"]["persisted_explicit_claim_events"],1)
            self.assertEqual(result["shared_memory"]["observed_cross_user_reads"],1)
            self.assertEqual(result["shared_memory"]["reads_after_flagged_writer_events"],1)
            self.assertEqual(result["shared_memory"]["negative_item_memory_writes"],0)
            self.assertIn("I dislike Album Y",pattern_brief(result)["exposure"]["examples"][0]["current_memory_quote"])

    def test_history_survives_reload_without_copying_old_scores_or_user_ids(self):
        with TemporaryDirectory() as temporary:
            journal=ExperimentJournal(Path(temporary)/"journal.json")
            comparison=compare_bias(_summary(.4,.03,.7,.7,.3),_summary(.42,.03,.2,.7,.3),"exposure")
            comparison["quality_uncertainty"]={"relative_change":.05,"quality_evidence":"INCONCLUSIVE"}
            plan={"operator":"unseen_neutral_user_reflection","hypothesis":"secret-user-42 has old score 0.123456"}
            conditions={"dataset":"CDs-100-user-dense","seed":1}
            candidate=record_author_trial(journal,Path(temporary)/"trial",plan,comparison,{"lesson":"secret-user-42"},conditions)
            journal.record_validation(candidate,{"status":"failed","reason":"different condition"})
            history=compact_author_history(ExperimentJournal(journal.path))
            self.assertEqual(len(history),1)
            self.assertEqual(history[0]["quality_change_label"],"+5.0%")
            self.assertNotIn("secret-user-42",json.dumps(history))
            self.assertNotIn("0.123456",json.dumps(history))
            self.assertEqual(history[0]["quality_evidence"],"INCONCLUSIVE")
            self.assertEqual(history[0]["recent_validation_feedback"][0]["status"],"failed")

    def test_real_saved_run_produces_five_pattern_cards_without_a_model(self):
        path=ROOT/"runs/qwen-author-dev20-baseline-20261015"
        if not (path/"summary.json").exists():
            self.skipTest("saved real run not included in source bundle")
        result=describe_run(path)
        self.assertEqual(set(result["patterns"]),{"exposure","selection","conformity","position","popularity"})
        self.assertEqual(result["shared_memory"]["negative_item_memory_writes"],0)
        self.assertGreater(result["shared_memory"]["observed_cross_user_reads"],0)
        brief=pattern_brief(result)
        self.assertNotIn("before",json.dumps(brief))
        self.assertEqual(sum(v["queries"] for v in result["target_group_quality"].values()),len(result["per_query"]))


if __name__=="__main__":
    unittest.main()
