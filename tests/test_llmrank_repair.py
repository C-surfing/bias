from unittest import TestCase
from rsi_researcher.llmrank.strategy import (
    BASELINE,stable_aliases,repair_alias_permutation,rank,evaluate_strategy
)

class DuplicatingModel:
    calls=0
    def __init__(self,output):
        self.output=output
        self.calls=0
    def ask(self,*args):
        self.calls+=1
        return {"ranking":self.output}

class RepairContractTests(TestCase):
    def setUp(self):
        items=[f"I{i:02d}" for i in range(20)]
        self.case={"user":"u1","history":items[:5],
                   "target":items[10],"candidates":items}
        self.titles={x:f"Example title {x}" for x in items}
        self.pop={x:i+1 for i,x in enumerate(items)}
        self.aliases=stable_aliases(self.case)
        self.raw=list(self.aliases.values())
    def test_duplicate_omission_repairs_without_using_target(self):
        malformed=self.raw[:-1]+[self.raw[0]]
        repaired,was_repaired=repair_alias_permutation(malformed,
                  {a:1 for a in self.raw})
        self.assertTrue(was_repaired)
        self.assertEqual(len(set(repaired)),20)
        self.assertEqual(repaired[:19],self.raw[:19])
        model=DuplicatingModel(malformed)
        prediction,meta=rank(model,self.case,self.case["candidates"],
                             self.titles,BASELINE,self.pop,123,"repair",
                             return_details=True)
        self.assertEqual(len(set(prediction)),20)
        self.assertEqual(meta["repaired_votes"],1)
        self.assertEqual(meta["observed_votes"],1)
        self.assertEqual(model.calls,1)
    def test_unrecognizable_result_stays_invalid(self):
        model=DuplicatingModel(["unknown"]*20)
        ranking,meta=rank(model,self.case,self.case["candidates"],self.titles,
                          BASELINE,self.pop,123,"invalid",return_details=True)
        self.assertIsNone(ranking)
        self.assertEqual(meta["repaired_votes"],0)
        self.assertEqual(meta["observed_votes"],1)
    def test_evaluator_reports_repair_fraction_separately(self):
        malformed=self.raw[:-1]+[self.raw[0]]
        model=DuplicatingModel(malformed)
        result=evaluate_strategy(model,[self.case],self.titles,self.pop,
                                 BASELINE,123,"controlled")
        m=result["metrics"]["5"]
        self.assertEqual(m["valid_rankings"],1)
        self.assertEqual(m["invalid_rankings"],0)
        self.assertEqual(m["repaired_votes"],1)
        self.assertEqual(m["ranking_repair_rate"],1.0)
