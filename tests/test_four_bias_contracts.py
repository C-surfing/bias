"""Scientific boundary tests for the four-bias workflow."""
import unittest
from rsi_researcher.research.four_bias_contracts import CONTRACTS,assess

class FourBiasContractTests(unittest.TestCase):
    def test_popularity_proxy_only_cannot_be_promoted(self):
        b={"PopREO@5":.4,"NDCG@5":.1,"HR@5":.2,"ARP@5":80.}
        a={"PopREO@5":.4,"NDCG@5":.1,"HR@5":.2,"ARP@5":5.}
        result=assess("popularity",b,a,identification=CONTRACTS["popularity"].identification,
                     evidence_grade="development_only")
        self.assertFalse(result["development_eligible"])
    def test_position_absolute_and_utility_guard(self):
        b={"CandidatePositionBias_NDCG@5":-.20,"NDCG@5":.2,"HR@5":.2}
        a={"CandidatePositionBias_NDCG@5":+.02,"NDCG@5":.16,"HR@5":.2}
        out=assess("candidate_position",b,a,
             identification=CONTRACTS["candidate_position"].identification,
             evidence_grade="development_only")
        self.assertGreater(out["bias_reduction"],0)
        self.assertFalse(out["development_eligible"])
    def test_selection_preserves_quality(self):
        b={"Brier":.2,"NDCG@50":.6}
        a={"Brier":.18,"NDCG@50":.599}
        out=assess("selection",b,a,identification=CONTRACTS["selection"].identification,
            evidence_grade="semi_synthetic_extra_mnar")
        self.assertTrue(out["development_eligible"])
        self.assertEqual(out["confirmed_discoveries"],0)
    def test_unidentified_exposure_fails_closed(self):
        b={"SNIPS":.001};a={"SNIPS":.02}
        out=assess("exposure",b,a,identification="unknown",
          evidence_grade="not_identified",ess=10000)
        self.assertEqual(out["status"],"not_identified")
        self.assertFalse(out["development_eligible"])
    def test_low_effective_sample_blocks_exposure(self):
        b={"SNIPS":.001};a={"SNIPS":.02}
        out=assess("exposure",b,a,
                identification=CONTRACTS["exposure"].identification,
                evidence_grade="development_only",ess=22)
        self.assertFalse(out["development_eligible"])
    def test_mainline_contracts_agree_on_primary_metrics(self):
        from rsi_researcher.llmrank.bias_objectives import OBJECTIVES
        self.assertEqual(OBJECTIVES["popularity"]["primary_metric"],CONTRACTS["popularity"].metric)
        self.assertIn(OBJECTIVES["candidate_position"]["primary_metric"],
                      CONTRACTS["candidate_position"].metric)

if __name__=="__main__": unittest.main()
