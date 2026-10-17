"""CPU-only Coat propensity sensitivity; no holdout or Qwen."""
from pathlib import Path
import tempfile
import unittest

from rsi_researcher.research.coat_sensitivity import run, GAMMAS


class CoatSensitivityTests(unittest.TestCase):
    def test_fixed_development_and_perturbation_contract(self):
        with tempfile.TemporaryDirectory(dir="/data/cjh/rsi/project/runs") as parent:
            out=run(Path(parent)/"sensitivity")
            self.assertEqual(out["status"], "DEV_PROPENSITY_SENSITIVITY_ONLY")
            self.assertEqual(out["gammas"], list(GAMMAS))
            self.assertEqual(out["train_rows"], 6960)
            self.assertEqual(out["development_rows"],2065)
            self.assertEqual(out["withheld_random_holdout_rows"],2209)
            self.assertFalse(out["holdout_evaluated"])
            self.assertEqual(out["model_requests"],0)
            self.assertEqual(len(out["sweep"]),3)
            for item in out["sweep"]:
                self.assertEqual([x["cap"] for x in item["cases"]],[5,20,100])
                for c in item["cases"]:
                    self.assertTrue(0<=c["fraction_capped"]<=1)
                    self.assertGreater(c["ESS"],0)
                    self.assertIn(c["NDCG_safeguard_pass"],(True,False))

    def test_path_outside_project_rejected(self):
        with self.assertRaises(ValueError):
            run(Path("/tmp/not-our-output"))


if __name__=="__main__":
    unittest.main()
