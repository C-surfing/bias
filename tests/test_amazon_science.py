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
