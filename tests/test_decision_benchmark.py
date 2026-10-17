"""Frozen development/evaluation synthetic research decision benchmark tests."""
from pathlib import Path
import json
import tempfile
import unittest
from rsi_researcher.research import decision_benchmark as b


class DecisionBenchmarkTests(unittest.TestCase):
    def test_disjoint_and_balanced(self):
        dev,ev=b.tasks()
        self.assertEqual(len(dev),7)
        self.assertEqual(len(ev),14)
        self.assertEqual({x["mode"] for x in dev},set(b.MODES))
        self.assertEqual({x["mode"] for x in ev},set(b.MODES))
        self.assertFalse(set(x["task_id"] for x in dev)&set(x["task_id"] for x in ev))

    def test_public_tasks_contain_no_oracle_values(self):
        protocol=b.frozen_payload()
        self.assertEqual(len(protocol["evaluation_tasks"]),14)
        for task in protocol["evaluation_tasks"]:
            self.assertFalse(any(k in task for k in ("oracle","true_value","best_action","mode")))
            self.assertEqual(task["allowed_actions"],list(b.ACTIONS))
        self.assertNotIn("evaluation_feedback",protocol)

    def test_unknown_propensity_and_support_abort(self):
        for mode in ("missing_propensity","unsupported"):
            sc=b.spec(mode,14,"evaluation")
            risk=b.risk_table(sc,repetitions=3)
            self.assertFalse(risk["identified"])
            self.assertEqual(risk["best_action"],"abstain")
            public=b.diagnostic(sc)
            self.assertEqual(b.rule(public),"abstain")

    def test_known_propensity_makes_estimators_finite(self):
        sc=b.spec("tail_shift",40,"evaluation")
        obs=b.estimator(sc,b.sample(sc,840))
        self.assertTrue(obs["supported"])
        self.assertGreater(obs["ess"],0)
        self.assertGreater(obs["max_weight"],0)

    def test_frozen_file_detects_tampering(self):
        with tempfile.TemporaryDirectory(dir=b.ROOT/"runs") as parent:
            output=Path(parent)/"bench"
            result=b.freeze(output)
            self.assertEqual(result["status"],"PROTOCOL_FROZEN_RULE_BASELINE_COMPLETE")
            self.assertEqual(result["model_calls"],0)
            self.assertEqual(len(result["rule"]["rows"]),14)
            self.assertEqual(len(b.read_frozen(output)["development_tasks"]),7)
            path=output/"frozen_protocol.json"
            path.write_text(path.read_text()+" ",encoding="utf8")
            with self.assertRaises(ValueError):
                b.read_frozen(output)

    def test_outside_directory_rejected(self):
        with self.assertRaises(ValueError):
            b.freeze(Path("/tmp/unknown-bias-benchmark"))


if __name__=="__main__":unittest.main()
