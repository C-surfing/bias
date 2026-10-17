"""Synthetic counterfactual estimator and identification-gate unit tests."""
from pathlib import Path
import math
import tempfile
import unittest

from rsi_researcher.research import exposure_synthetic as e


class ControlledExposureTests(unittest.TestCase):
    def test_logging_positivity_and_policy_normalization(self):
        p=e.logging_distribution()
        self.assertEqual(len(p),e.ITEMS)
        self.assertTrue(all(x>0 for x in p))
        self.assertAlmostEqual(sum(p),1.0)
        for u in [0,7,21,49]:
            for policy in e.TARGETS:
                self.assertAlmostEqual(sum(e.target_distribution(u,policy)),1.0)
        self.assertGreater(e.true_value("personalized_oracle"),e.true_value("uniform"))

    def test_missing_propensity_must_abstain(self):
        result=e.simulate(e.SEED,100,propensities_visible=False)
        self.assertEqual(result["status"],"ABSTAIN_NO_PROPENSITY")
        self.assertIsNone(result["ips"])
        self.assertIsNone(result["snips"])

    def test_estimators_and_ess_finite(self):
        result=e.simulate(e.SEED,1000)
        self.assertEqual(result["status"],"IDENTIFIED_SYNTHETIC_OPE")
        for entry in result["targets"].values():
            self.assertGreater(entry["ESS"],0)
            self.assertLessEqual(entry["ESS"],1000+1e-8)
            self.assertTrue(math.isfinite(entry["IPS"]))
            self.assertTrue(math.isfinite(entry["SNIPS"]))

    def test_small_protocol_result_and_output_contract(self):
        with tempfile.TemporaryDirectory(dir=e.ROOT/"runs") as temp:
            output=e.run(Path(temp)/"small",reps=4,events=250)
            self.assertEqual(output["status"],"SYNTHETIC_KNOWN_PROPENSITY_BENCHMARK_COMPLETE")
            self.assertFalse(output["independent_real_data_confirmation"])
            self.assertFalse(output["test_labels_read"])
            self.assertEqual(output["missing_propensity_control"]["status"],"ABSTAIN_NO_PROPENSITY")
        with self.assertRaises(ValueError):
            e.run(Path("/tmp/outside-rsi"),reps=2,events=100)


if __name__=="__main__":
    unittest.main()
