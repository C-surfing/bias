"""Mainline task adapter contracts: only tests that protect scientific conclusions."""
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from rsi_researcher.llmrank.science import audit,select_case_panels,expansion_priority
from rsi_researcher.llmrank.tools import candidate_snapshot
from rsi_researcher.llmrank.strategy import BASELINE,canonical
from rsi_researcher.llmrank.formula import validate,score
from rsi_researcher.llmrank.agents import PROPOSAL, STRATEGY, RankGateway
from jsonschema import Draft202012Validator

SOURCE=Path(__file__).resolve().parent.parent / "datasets" / "llmrank"

class LLMRankMainlineTests(unittest.TestCase):
    def test_pareto_candidate_can_be_expanded_without_being_accepted(self):
        b={"5":{"NDCG@5":0.278351,"HR@5":0.571429,
                 "ARP@5":491.085714,"APLT@5":0.657143,
                 "valid_rankings":7,"invalid_rankings":0,"users":7,
                 "CandidatePositionBias_NDCG@5":-0.09758}}
        n={"5":{**b["5"],"NDCG@5":0.214286,"HR@5":0.285714,
                "ARP@5":354.666667,"APLT@5":0.833333}}
        self.assertFalse(audit(b,n)["eligible_development_only"])
        self.assertGreater(expansion_priority(b,n),0)
    def test_frozen_candidate_diagnostics_do_not_expose_targets(self):
        c=[{"history":["A"],"target":"B","candidates":["A","B"]}]
        v=candidate_snapshot(c,{"A":12,"B":0})
        self.assertFalse(v["target_labels_sent_to_agent"])
        self.assertNotIn("B",str(v))
    def test_formal_strategy_contract(self):
        Draft202012Validator.check_schema(PROPOSAL)
        proposal={
          "hypothesis":"Popularity correction can reduce average output popularity.",
          "alternative":"The apparent bias could come from candidate retrieval.",
          "falsifier":"Fail if ARP does not fall at the utility floor.",
          "predicted_metric":"ARP@5",
          "strategy":BASELINE}
        Draft202012Validator(PROPOSAL).validate(proposal)
        self.assertEqual(canonical(BASELINE)["score_expression"],"base")
        valid=validate("base - 0.2*pop + 0.1*match")
        self.assertAlmostEqual(score(valid,{"base":0.5,"pop":0.4,"match":0.3}),.45)
        with self.assertRaises(ValueError):
            validate("__import__('os').system('id')")
    def test_independent_utility_plus_bias_gate(self):
        m={"5":{"NDCG@5":0.4,"HR@5":0.5,"ARP@5":200.0,
          "APLT@5":0.3,"invalid_rankings":0,"valid_rankings":20,"users":20,
          "CandidatePositionBias_NDCG@5":0.2}}
        better={"5":{**m["5"],"NDCG@5":0.395,"ARP@5":175.0}}
        worse={"5":{**m["5"],"NDCG@5":0.30,"ARP@5":175.0}}
        self.assertTrue(audit(m,better)["eligible_development_only"])
        self.assertFalse(audit(m,worse)["eligible_development_only"])
    @unittest.skipUnless(SOURCE.joinpath("ml-1m.random").exists(),"official data absent")
    def test_candidate_and_holdout_disjoint(self):
        dev,test,titles,pop,protocol=select_case_panels(
             "ml-1m",SOURCE,"random",18,20261021)
        self.assertTrue(set(protocol["dev_users"]).isdisjoint(protocol["holdout_users"]))
        self.assertGreater(len(dev),2)
        self.assertGreater(len(test),2)
        for case in dev+test:
            self.assertEqual(len(case["candidates"]),20)
            self.assertEqual(case["candidates"].count(case["target"]),1)
            self.assertTrue(all(i in titles for i in case["candidates"]))

if __name__=="__main__":
    unittest.main()
