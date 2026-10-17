"""Fresh AgentCF cohort / formal forecast contract regression tests."""
import unittest
from copy import deepcopy
from rsi_researcher.research import fresh_predictions as f


class FreshForecastTests(unittest.TestCase):
    def test_fresh_cohort_is_user_disjoint(self):
        original,events,queries=f.cohort_data()
        self.assertEqual(len(original["users"]),6)
        self.assertEqual(len(queries),6)
        self.assertEqual(len(events),12)
        self.assertTrue(set(x["query_id"] for x in queries)==
                        set("valid-"+u for u in original["users"]))

    def test_predeclared_contract_accept_and_reject(self):
        for t in f.TASKS:
            good=f.rule_forecast(t)
            self.assertTrue(f.validate(t,good)[0])
            bad=deepcopy(good);bad["action"]="shell"
            self.assertFalse(f.validate(t,bad)[0])
            bad=deepcopy(good);bad["metric"]="oracle_accuracy"
            self.assertFalse(f.validate(t,bad)[0])
            bad=deepcopy(good);bad["minimum_abs_effect"]=-0.1
            self.assertFalse(f.validate(t,bad)[0])
            bad=deepcopy(good);bad["minimum_abs_effect"]=True
            self.assertFalse(f.validate(t,bad)[0])
            bad=deepcopy(good);bad["outcome"]="yes"
            self.assertFalse(f.validate(t,bad)[0])

    def test_sign_magnitude_two_panel_and_quality_gate(self):
        good=f.rule_forecast("item_memory")
        p={"quality_pass":True,"excess_rank_change":0.25,
            "head_excess_delta":-0.05,"ndcg_delta":0.0}
        val=f.score_contract(good,{"A":p,"B":p})
        self.assertTrue(val["both_panels_pass"])
        damaged=dict(p,quality_pass=False)
        self.assertFalse(f.score_contract(good,{"A":p,"B":damaged})["both_panels_pass"])
        reversed_sign=dict(p,excess_rank_change=-0.5)
        self.assertFalse(f.score_contract(good,{"A":p,"B":reversed_sign})["both_panels_pass"])
        with self.assertRaises(ValueError):
            f.score_contract(good,{"A":p})

    def test_repeat_noise_adjusted_numeric_effect(self):
        def r(name,order,ndcg=.4,head=.15):
            return {"query_id":name,"ranking":order,"ndcg_at_5":ndcg,
                    "excess_popularity_exposure":head,"valid":True}
        orig=[r("q1",[str(i) for i in range(10)])]
        same=deepcopy(orig)
        alter=[r("q1",[str(i) for i in reversed(range(10))])]
        result=f.score_panel(orig,same,alter)
        self.assertEqual(result["excess_rank_change"],1.0)
        self.assertEqual(result["repeat_change_rate"],0.0)
        self.assertTrue(result["quality_pass"])

    def test_protocol_has_fixed_disjoint_panels(self):
        p=f.protocol()
        self.assertEqual(p["training_events"],12)
        self.assertEqual(len(p["split"]["A"]),3)
        self.assertEqual(len(p["split"]["B"]),3)
        self.assertFalse(set(p["split"]["A"]) & set(p["split"]["B"]))
        self.assertFalse(p["predeclared_numeric_contract"]["independent_confirmation"])


if __name__=="__main__":
    unittest.main()
