"""Science tree invariants: no proxy promotion, no accidental goal switching."""
import unittest
from rsi_researcher.research.four_bias_tree import canon,choose_parent,random_strategy,SCHEMA
from rsi_researcher.research.four_bias_benchmarks import user_bucket,ci95,SEEDS
from rsi_researcher.research.four_bias_contracts import CONTRACTS

class FourBiasTreeTests(unittest.TestCase):
    def test_selection_parameter_schema_and_bounds(self):
        self.assertEqual(canon("selection",{"blend":.34567,"clip":3.21567}),
                         {"blend":.346,"clip":3.216})
        for params in ({"blend":1.5,"clip":3},{"blend":.4,"clip":0}):
            with self.assertRaises(ValueError):canon("selection",params)
    def test_exposure_parameter_schema_and_uniform_reference(self):
        self.assertEqual(canon("exposure",{"top":20,"mass":.25}),
                         {"top":20,"mass":.25})
        with self.assertRaises(ValueError):canon("exposure",{"top":41,"mass":.6})
        self.assertAlmostEqual(.25/20,(1-.25)/(80-20))
    def test_split_is_stable_user_disjoint_across_seeds(self):
        ids=range(1000)
        buckets=[user_bucket(i) for i in ids]
        self.assertTrue(all(user_bucket(i)==buckets[i] for i in ids))
        self.assertGreater(buckets.count("dev"),100)
        self.assertGreater(buckets.count("confirm"),50)
        self.assertTrue(set(buckets)=={"train","dev","confirm"})
        self.assertEqual(len(SEEDS),3)
    def test_fixed_failure_priority_prefers_falsifiable_parent(self):
        root={"id":"ROOT","parent":None,"depth":0,"expansions":0,
              "status":"BASELINE","audit":{"failure_mode":"baseline"}}
        failed={"id":"N001","parent":"ROOT","depth":1,"expansions":0,
                "status":"MEASURED","audit":{"failure_mode":"unstable_or_unresolved_effect",
                "mean_primary_gain":-.1}}
        self.assertEqual(choose_parent([root,failed])["id"],"N001")
        self.assertEqual(choose_parent([root])["id"],"ROOT")
    def test_equal_no_effect_never_bootstraps_to_significance(self):
        self.assertEqual(ci95([0.]*200),[0.,0.])
    def test_bias_estimands_not_misidentified(self):
        self.assertEqual(CONTRACTS["selection"].metric,"Brier")
        self.assertEqual(CONTRACTS["exposure"].metric,"SNIPS")
        self.assertNotEqual(CONTRACTS["exposure"].identification,
                            CONTRACTS["selection"].identification)

if __name__=="__main__":
    unittest.main()
