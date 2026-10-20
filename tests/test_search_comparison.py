"""Matched search policies on real-data executors, without Qwen claims."""
import json
from pathlib import Path
import tempfile
import unittest

from rsi_researcher.research.scientific_comparison import compare_tree_linear
from rsi_researcher.research.scientific_loop import verified_experience
from tests.test_scientific_loop import FakeModel


CFG={"llm":{"provider":"vllm","model":"qwen3-8b",
            "base_url":"http://127.0.0.1:18419/v1"},
     "budget":{"max_requests":240,"max_tokens":1000000,"max_seconds":1800}}


class MatchedSearchTests(unittest.TestCase):
    def test_real_coat_development_tree_vs_linear_equal_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"compare"
            report=compare_tree_linear(
                out,"coat_utility",CFG,seed=20270001,
                policies=("linear","tree"),max_nodes=2,max_requests=80,
                models={"linear":FakeModel(),"tree":FakeModel()})
            self.assertEqual(report["status"],"COMPLETED_DEVELOPMENT_COMPARISON")
            self.assertEqual(set(report["policies"]),{"linear","tree"})
            for policy in ("linear","tree"):
                tree=json.loads((out/policy/"tree_index.json").read_text())
                self.assertEqual(tree["attempted_nodes"],2)
                self.assertEqual(tree["independently_confirmed_discoveries"],0)
                self.assertEqual(tree["model_usage"]["usage_measured"],False)
                self.assertEqual(json.loads((out/policy/"protocol.json").read_text())
                                 ["effective_hard_model_budget"]["max_requests"],80)
            self.assertEqual(report["independently_confirmed_discoveries"],0)
            self.assertFalse(report["matched_trained_state"])
            self.assertEqual(report["shared_training_model_usage"]["requests"],0)

    def test_reuse_frozen_author_state_for_popularity_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            prior=compare_tree_linear(
                root/"first","agentcf_memory",CFG,
                seed=20270001,cohort_size=8,max_nodes=1,max_requests=90,
                models={"shared_training":FakeModel(),
                        "linear":FakeModel(),"tree":FakeModel()})
            state=prior["frozen_state_ref"]
            second=compare_tree_linear(
                root/"second","agentcf_popularity",CFG,
                seed=20270001,cohort_size=8,max_nodes=1,max_requests=60,
                prepared_state=state,
                models={"linear":FakeModel(),"tree":FakeModel()})
            self.assertTrue(second["matched_trained_state"])
            self.assertEqual(second["shared_training_model_usage"]["requests"],0)
            self.assertTrue(second["shared_training_model_usage"]
                            ["previous_development_state_reused"])
            self.assertEqual(second["previous_shared_training_usage"]["requests"],
                             prior["shared_training_model_usage"]["requests"])
            self.assertEqual(second["frozen_state_ref"],state)
            self.assertEqual(second["independently_confirmed_discoveries"],0)
            # Prevent accidental state reuse with different scientific inputs.
            with self.assertRaises(ValueError):
                compare_tree_linear(root/"bad","agentcf_popularity",CFG,
                    seed=20270002,cohort_size=8,max_nodes=1,
                    prepared_state=state,models={"linear":FakeModel(),
                                                "tree":FakeModel()})

    def test_real_author_cd_cohort_one_trained_state_both_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"author_compare"
            models={"shared_training":FakeModel(),
                    "tree":FakeModel(),"linear":FakeModel()}
            report=compare_tree_linear(
                out,"agentcf_memory",CFG,seed=20270001,cohort_size=12,
                max_nodes=2,max_requests=200,models=models)
            self.assertTrue(report["matched_trained_state"])
            self.assertGreater(report["shared_training_model_usage"]["requests"],0)
            self.assertTrue(report["frozen_state_ref"])
            one=report["policies"]["linear"]
            two=report["policies"]["tree"]
            self.assertEqual(one["frozen_training_sha256"],two["frozen_training_sha256"])
            self.assertEqual(one["executed_development_nodes"],2)
            self.assertEqual(two["executed_development_nodes"],2)
            self.assertEqual(one["independently_confirmed_discoveries"],0)
            tree_run=out/"tree"
            from_tree=verified_experience(Path(tmp)/"fresh",tree_run)
            self.assertTrue(from_tree)
            self.assertTrue(all(row["grade"]=="new_unconfirmed_development"
                                for row in from_tree))
            self.assertNotIn("target",json.dumps(from_tree))

if __name__=="__main__":
    unittest.main()
