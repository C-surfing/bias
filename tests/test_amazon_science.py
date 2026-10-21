"""Amazon 2014 5-core scientific data contract and frozen candidate checks."""
import json
from pathlib import Path
import unittest

from rsi_researcher.llmrank.dataset import load_dataset
from rsi_researcher.llmrank.science import select_case_panels
from rsi_researcher.llmrank.amazon import NAMES

ROOT=Path(__file__).resolve().parent.parent/"datasets"/"amazon"/"processed"

class AmazonScienceTests(unittest.TestCase):
    def test_three_categories_are_registered(self):
        self.assertEqual(set(NAMES),{"Beauty","Sports","CDs"})

    def test_all_three_prepared_datasets_and_frozen_protocol(self):
        expected={
            "Beauty":(22334,12094,198255),
            "Sports":(35359,18267,294151),
            "CDs":(52808,49866,814068),
        }
        for name,(users,items,events) in expected.items():
            with self.subTest(dataset=name):
                directory=ROOT/name
                record=json.loads((directory/"provenance.json").read_text(encoding="utf-8"))
                self.assertEqual((record["users_retained"],record["items_retained"],
                                  record["interactions_retained"]),(users,items,events))
                self.assertTrue(all(len(v)==64 for v in record["source_sha256"].values()))
                dev,heldout,titles,pop,protocol=select_case_panels(
                    name,directory,"random",24,20261028)
                self.assertEqual(len(dev),14)
                self.assertEqual(len(heldout),10)
                self.assertEqual(len(set(protocol["dev_users"]) &
                                     set(protocol["holdout_users"])),0)
                self.assertEqual(len(titles),items)
                self.assertTrue(all(len(set(case["candidates"]))==20 for case in dev+heldout))
                self.assertEqual(protocol["candidate_generation"],"frozen_random")

    @unittest.skipUnless((ROOT/"Beauty"/"provenance.json").is_file(),
                         "Beauty has not yet been converted")
    def test_beauty_end_to_end_data_contract(self):
        source=ROOT/"Beauty"
        provenance=json.loads((source/"provenance.json").read_text())
        self.assertEqual(provenance["source_release"],
                         "McAuley Amazon Product Data 2014, 5-core")
        self.assertEqual(provenance["reviews_before_filter"],198502)
        self.assertEqual(provenance["users_retained"],22334)
        train,positives,known,titles,pop=load_dataset(source,"Beauty")
        self.assertEqual(len(train),22334)
        self.assertEqual(len(titles),12094)
        self.assertGreater(sum(pop.values()),100000)
        dev,holdout,item_titles,train_pop,contract=select_case_panels(
            "Beauty",source,"random",24,20261028)
        self.assertEqual(contract["candidate_source"],"frozen_random")
        self.assertTrue(set(contract["dev_users"]).isdisjoint(contract["holdout_users"]))
        self.assertGreater(len(dev),5)
        self.assertGreater(len(holdout),5)
        for case in [*dev,*holdout]:
            self.assertEqual(len(case["candidates"]),20)
            self.assertEqual(len(set(case["candidates"])),20)
            self.assertIn(case["target"],case["candidates"])
            self.assertTrue(all(item in titles for item in case["candidates"]))
            self.assertTrue(len(case["history"])<=50)
            self.assertFalse(set(case["candidates"])-{case["target"]} & known[case["user"]])
