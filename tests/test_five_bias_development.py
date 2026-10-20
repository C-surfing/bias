"""Scientific controls for all five frozen recommendation-bias goals.

Tests verify split isolation, matched-support falsifiers, group fairness
accounting and explicit non-identifiability, not independent scientific proof.
"""
from csv import DictWriter
from pathlib import Path
import tempfile
import unittest

from rsi_researcher.datasets.coat import load_coat
from rsi_researcher.research.bias_registry import TASKS, verify as registry_verify
from rsi_researcher.research.coat_selection import DATA, SEED
from rsi_researcher.research.five_bias_development import (
    _click_model, _hash_bucket, exposure_real, popularity_coat,
    fairness_coat, filter_bubble_kuairand,
)


def csv_write(path,rows,fields):
    with path.open("w",encoding="utf8",newline="") as stream:
        writer=DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FiveBiasEvidenceTests(unittest.TestCase):
    def test_frozen_five_bias_semantics_and_mechanism_not_sixth_bias(self):
        self.assertTrue(registry_verify())
        self.assertEqual({t.id for t in TASKS},{
            "selection","exposure","popularity","filter_bubble","fairness"})

    def test_exposure_random_split_equal_budget_and_common_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            fields=["user_id","video_id","time_ms","is_click","is_rand","tab"]
            standard=[{"user_id":str(i%11),"video_id":str(i%7),
                       "time_ms":str(i),"is_click":str(int(i%3==0)),
                       "is_rand":"0","tab":str(i%2)} for i in range(400)]
            random=[{"user_id":str(i%11),"video_id":str(i%7),
                       "time_ms":str(i+50000),"is_click":str(int(i%5<2)),
                       "is_rand":"1","tab":str(i%2)} for i in range(2300)]
            csv_write(p/"std.csv",standard,fields)
            csv_write(p/"random.csv",random,fields)
            result=exposure_real(p/"random.csv",p/"std.csv")
            self.assertTrue(result["same_training_impressions_control"])
            self.assertEqual(result["policies"]["standard_only"]
                             ["training_impressions"],400)
            self.assertEqual(result["policies"]["random_equal_budget"]
                             ["training_impressions"],400)
            self.assertGreater(result["random_reserved_impressions_unscored"],0)
            self.assertGreater(result["shared_item_tab_keys"],0)
            self.assertGreater(result["policies"]["random_equal_budget"]
                               ["common_support_development_impressions"],0)
            self.assertFalse(result["causal_OPE_identified"])
            # Future dev/heldout outcomes never affect training keys or priors.
            model, prior, count=_click_model([(("x","1"),0),(("x","1"),1)])
            self.assertEqual(model[("x","1")],.5)
            self.assertEqual(prior,.5)
            self.assertEqual(count[("x","1")],2)

    def test_exposure_rejects_false_randomization_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            fields=["user_id","video_id","time_ms","is_click","is_rand","tab"]
            bad=[dict(user_id="1",video_id="1",time_ms="1",
                      is_click="1",is_rand="0",tab="0")]
            csv_write(p/"std.csv",bad,fields)
            csv_write(p/"random.csv",bad,fields)
            with self.assertRaises(ValueError):
                exposure_real(p/"random.csv",p/"std.csv")

    def test_fixed_popularity_reranker_shows_utility_tradeoff_on_real_coat(self):
        ds=load_coat(DATA,split_seed=SEED)
        r=popularity_coat(ds)
        self.assertEqual(r["bias_id"],"popularity")
        self.assertEqual(set(r["policies"]),{"0.0","0.05","0.1","0.2"})
        self.assertTrue(r["policies"]["0.0"]["quality_guard_99pct_ndcg"])
        self.assertLess(r["policies"]["0.2"]["head_top_20pct_exposure_share"],
                        r["policies"]["0.0"]["head_top_20pct_exposure_share"])
        self.assertFalse(r["identifies_unjustified_popularity_causally"])
        self.assertFalse(r["independent_confirmation"])

    def test_explicit_group_fairness_intervention_can_fail(self):
        ds=load_coat(DATA,split_seed=SEED)
        r=fairness_coat(ds)
        self.assertEqual(r["bias_id"],"fairness")
        self.assertEqual(r["status"],"GROUP_PREDICTION_CALIBRATION_DEVELOPMENT")
        self.assertEqual(set(r["unadjusted"]),{"men","women"})
        self.assertGreater(r["absolute_brier_gap_after"],
                           r["absolute_brier_gap_before"])
        self.assertFalse(r["recommendation_rank_change"])
        self.assertFalse(r["independent_confirmation"])

    def test_topic_longitudinal_inspection_must_abstain_causal_bubble(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            csv_write(p/"features.csv",[
                 {"video_id":str(i),"tag":str(i%3)} for i in range(6)],
                 ["video_id","tag"])
            fields=["user_id","video_id","is_click"]
            early=[{"user_id":"u","video_id":str(i%6),"is_click":"0"}
                   for i in range(20)]
            late=[{"user_id":"u","video_id":str(i%2),"is_click":"1"}
                  for i in range(6)]
            csv_write(p/"early.csv",early,fields)
            csv_write(p/"late.csv",late,fields)
            r=filter_bubble_kuairand(p/"features.csv",p/"early.csv",p/"late.csv")
            self.assertEqual(r["status"],
                             "ABSTAIN_NO_MATCHED_INTERVENTION_TRAJECTORY")
            self.assertEqual(r["users_with_at_least_5_tagged_exposures_each_period"],1)
            self.assertTrue(r["matched_same_number_of_exposures_per_user"])
            self.assertFalse(r["identified_filter_bubble_effect"])
            self.assertFalse(r["independent_confirmation"])


if __name__=="__main__":
    unittest.main()
