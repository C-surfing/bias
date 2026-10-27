"""Target-bias outcome must be identified independently of utility and proxies."""
import unittest
from rsi_researcher.llmrank.bias_objectives import OBJECTIVES, CONTRACT, evaluate_target
from rsi_researcher.llmrank.science import audit, expansion_priority


def panel(**extras):
    return {"5":{"users":50,"valid_rankings":50,"invalid_rankings":0,
                 "NDCG@5":0.40,"HR@5":0.50,
                 "ARP@5":100.0,"APLT@5":0.30,
                 "PopREO@5":0.50,"popreo_group_positive_counts":[10]*5,
                 "CandidatePositionBias_NDCG@5":0.25,**extras}}


class BiasTargetTests(unittest.TestCase):
    def test_popularity_proxy_improvement_is_not_debias(self):
        before=panel()
        after=panel(**{"ARP@5":55.0,"APLT@5":0.6})
        a=audit(before,after)
        self.assertFalse(a["eligible_development_only"])
        self.assertTrue(a["exploratory_proxy_only"])
        self.assertEqual(a["failure_mode"],"target_bias_unchanged")
        self.assertEqual(a["target_bias_assessment"]["metric"],"PopREO@5")

    def test_popreo_improvement_and_utility_guard_must_both_pass(self):
        baseline=panel()
        better=panel(**{"PopREO@5":0.30,"NDCG@5":0.394,"ARP@5":105.0})
        self.assertTrue(audit(baseline,better)["eligible_development_only"])
        worse=panel(**{"PopREO@5":0.30,"NDCG@5":0.30})
        value=audit(baseline,worse)
        self.assertFalse(value["eligible_development_only"])
        self.assertEqual(value["failure_mode"],"debias_quality_tradeoff")

    def test_missing_or_sparse_popreo_is_inconclusive_not_zero(self):
        baseline=panel()
        candidate=panel(**{"PopREO@5":0.10,"ARP@5":70.})
        baseline["5"]["popreo_group_positive_counts"]=[2,10,10,10,18]
        candidate["5"]["popreo_group_positive_counts"]=[2,10,10,10,18]
        result=audit(baseline,candidate)
        self.assertFalse(result["eligible_development_only"])
        self.assertEqual(result["failure_mode"],"insufficient_target_bias_support")
        self.assertIsNone(audit(panel(**{"PopREO@5":None}),candidate)
                          ["target_bias_assessment"]["reduction"])

    def test_position_target_ignores_arp_and_uses_absolute_bias(self):
        baseline=panel()
        proxy=panel(**{"ARP@5":20.0})
        self.assertFalse(audit(baseline,proxy,bias_target="candidate_position")
                         ["eligible_development_only"])
        bias_improved=panel(**{"CandidatePositionBias_NDCG@5":-0.06})
        self.assertTrue(audit(baseline,bias_improved,bias_target="candidate_position")
                        ["eligible_development_only"])
        bias_worse=panel(**{"CandidatePositionBias_NDCG@5":-0.39})
        self.assertFalse(audit(baseline,bias_worse,bias_target="candidate_position")
                         ["eligible_development_only"])

    def test_position_target_tree_priority_is_not_arp_priority(self):
        baseline=panel()
        candidate=panel(**{"ARP@5":20., "CandidatePositionBias_NDCG@5":0.30})
        self.assertLess(expansion_priority(baseline,candidate,"candidate_position"),0)

    def test_paired_target_bootstrap_preserves_zero_difference(self):
        from rsi_researcher.llmrank.bias_objectives import bootstrap_target_bias
        items=[f"i{n}" for n in range(50)]
        pop={item:index+1 for index,item in enumerate(items)}
        titles={item:item for item in items}
        rows=[]
        for index,target in enumerate(items):
            # Exactly ten observed positives per popularity quintile.
            rank=[target]+[item for item in items if item!=target][:19]
            rows.append({"user":f"u{index}","target":target,"ranking":rank,
                         "observed_votes":1,"repaired_votes":0})
        result={"random":rows,"first":rows}
        evidence=bootstrap_target_bias(result,result,pop,titles,"popularity",
                                       draws=200)
        self.assertEqual(evidence["paired_users"],50)
        self.assertGreaterEqual(evidence["valid_resamples"],160)
        self.assertEqual(evidence["ci95"],[0.,0.])

    def test_target_not_supported_fails_closed(self):
        with self.assertRaises(ValueError):
            evaluate_target(panel()["5"],panel()["5"],"selection")

    def test_target_contract_is_explicit(self):
        self.assertIn("popularity",OBJECTIVES)
        self.assertIn("candidate_position",OBJECTIVES)
        self.assertTrue(CONTRACT.startswith("llmrank_target_bias_"))


if __name__=="__main__":
    unittest.main()
