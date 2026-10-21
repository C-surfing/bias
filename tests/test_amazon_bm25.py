"""Frozen Amazon title BM25 vs random candidate contracts."""
from unittest import TestCase
from pathlib import Path

from rsi_researcher.llmrank.retrieval import TitleBM25
from rsi_researcher.llmrank.dataset import cases_for
from rsi_researcher.llmrank.science import select_case_panels

class AmazonRetrievalTests(TestCase):
    def test_bm25_is_deterministic_and_never_exposes_other_known_items(self):
        titles={
            "a":"moisturizing lavender shampoo",
            "b":"organic lavender conditioner",
            "c":"lavender hair wash repair",
            "d":"organic shampoo for dry hair",
            "e":"sports running socks",
            "f":"wooden game board",
            "g":"shampoo and conditioner lavender",
        }
        titles.update({f"z{i}":f"unrelated item unique{i}" for i in range(35)})
        index=TitleBM25(titles)
        history=["a","b"]
        blocked={"a","b","g"}
        first=index.select(history,blocked,titles,20261028,19)
        second=index.select(history,blocked,titles,20261028,19)
        self.assertEqual(first,second)
        self.assertEqual(len(set(first)),19)
        self.assertTrue(set(first).isdisjoint(blocked))
        self.assertIn("c",first[:5])
        train={"u":history}
        positives={"u":{"validation":"g","test":"g"}}
        known={"u":blocked}
        cases=cases_for("validation",["u"],train,positives,known,titles,
                        20261028,20,"bm25")
        self.assertEqual(len(cases),1)
        self.assertEqual(len(set(cases[0]["candidates"])),20)
        self.assertEqual(cases[0]["candidates"].count("g"),1)
    def test_ranking_with_bm25_does_not_change_source_of_gt(self):
        root=Path(__file__).resolve().parent.parent/"datasets"/"amazon"/"processed"/"Beauty"
        if not (root/"Beauty.inter.zip").exists():
            self.skipTest("Beauty not ready")
        dev,holdout,titles,pop,protocol=select_case_panels(
            "Beauty",root,"bm25",24,20261030)
        self.assertEqual(protocol["candidate_generation"],"train_title_bm25")
        self.assertEqual(len(dev),14)
        self.assertEqual(len(holdout),10)
        self.assertEqual(len(set(protocol["dev_users"]) & set(protocol["holdout_users"])),0)
        self.assertTrue(all(len(set(c["candidates"]))==20 and
                            c["target"] in c["candidates"] for c in dev+holdout))
