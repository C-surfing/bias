"""Relevance-preserving debias primitives, no label access or Qwen cost."""
from unittest import TestCase
from rsi_researcher.llmrank.strategy import (
    BASELINE,canonical,stable_aliases,rank,FIXED
)

class FrozenRanker:
    def __init__(self,case):
        self.calls=0
        aliases=stable_aliases(case)
        self.prediction=[aliases[i] for i in case["candidates"]]
    def ask(self,*_args):
        self.calls+=1
        return {"ranking":self.prediction}

class ConditionalDebiasTests(TestCase):
    def setUp(self):
        self.case={"user":"u1","history":["i1","i2","i3"],
                   "target":"i4",
                   "candidates":[f"i{i}" for i in range(1,21)]}
        self.titles={f"i{i}":f"title item {i}" for i in range(1,21)}
        self.pop={f"i{i}":1000//i if i<=3 else 1 for i in range(1,21)}
    def run_strategy(self,mode):
        model=FrozenRanker(self.case)
        strategy=canonical({**BASELINE,"popularity_lambda":.4,"rerank_policy":mode})
        prediction=rank(model,self.case,self.case["candidates"],
                        self.titles,strategy,self.pop,29,"test")
        self.assertEqual(model.calls,1)
        self.assertEqual(set(prediction),set(self.case["candidates"]))
        self.assertEqual(len(prediction),20)
        return prediction
    def test_protected_top3_limits_utility_damage(self):
        global_rank=self.run_strategy("global")
        guarded=self.run_strategy("protect_top3")
        self.assertEqual(guarded[:3],self.case["candidates"][:3])
        self.assertNotEqual(global_rank[:3],guarded[:3])
    def test_personalized_gate_changes_penalty_by_history(self):
        global_rank=self.run_strategy("global")
        personalized=self.run_strategy("personalized_gate")
        self.assertEqual(len(set(personalized)),20)
        self.assertNotEqual(global_rank,personalized)
    def test_only_registered_policies_execute(self):
        with self.assertRaises(ValueError):
            canonical({**BASELINE,"rerank_policy":"run_code"})
        for strategy in FIXED.values():
            self.assertIsInstance(canonical(strategy),dict)
