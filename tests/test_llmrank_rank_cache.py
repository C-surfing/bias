"""Immutable Qwen ranking caches across development rounds."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from rsi_researcher.llmrank.agents import RankGateway

class DummyModel:
    def __init__(self,revision="A"):
        self.requests=0
        self.metadata={"model":"dummy","revision":revision}
    def ask(self,prompt,schema,context):
        self.requests+=1
        return {"ranking":schema["properties"]["ranking"]["items"]["enum"]}

class CrossRoundFreezeTests(TestCase):
    def test_same_rank_input_reused_and_changed_input_recomputed(self):
        with TemporaryDirectory() as folder:
            prior=Path(folder)/"round-01.jsonl"
            a=DummyModel()
            gateway=RankGateway(a,prior)
            answer=gateway.ask("Rank all.","C01: Item A\nC02: Item B","first")
            self.assertEqual(a.requests,1)
            self.assertTrue(prior.is_file())
            b=DummyModel()
            reused=RankGateway(b,Path(folder)/"round-02.jsonl",
                               preload_paths=[prior])
            self.assertEqual(reused.ask("Rank all.","C01: Item A\nC02: Item B","second"),answer)
            self.assertEqual(b.requests,0)
            self.assertEqual(reused.cache_hits,1)
            self.assertEqual(reused.preloaded_rankings,1)
            reused.ask("Rank all.","C01: Item A\nC02: Changed B","changed")
            self.assertEqual(b.requests,1)
            c=DummyModel(revision="B")
            newmodel=RankGateway(c,preload_paths=[prior])
            newmodel.ask("Rank all.","C01: Item A\nC02: Item B","new-revision")
            self.assertEqual(c.requests,1)
