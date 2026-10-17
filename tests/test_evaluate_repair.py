from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from rsi_researcher.paths import ROOT
from rsi_researcher.runtime.client import ModelFailure
from rsi_researcher.evaluation.repair import evaluate_states, position_distance
from tests.test_agentcf_runner import ScriptedModel


class RepairEvaluationTests(unittest.TestCase):
    def test_position_distance_known_permutations(self):
        self.assertEqual(position_distance(list("abcd"), list("abcd")), 0)
        self.assertEqual(position_distance(list("abcd"), list("dcba")), 1)
        self.assertAlmostEqual(position_distance(list("abcd"), list("bacd")), 1/6)
        self.assertEqual(position_distance([], list("abcd")), 1)
        with self.assertRaises(ValueError):
            position_distance(list("abcd"), list("abce"))

    def test_shared_panels_failed_output_penalty_state_unchanged_no_target_sent(self):
        prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))
        state = {"users": {"u": "user"}, "items": {i: {"memory": i, "title": i} for i in "abc"}, "completed": []}
        states = {"initial": deepcopy(state), "repaired": deepcopy(state)}
        before = deepcopy(states)
        queries = [{"query_id": "q", "user": "u", "candidates": list("abc"), "target": "a"}]
        model = ScriptedModel([{"ranking": list("abc")}, {"ranking": list("cba")},
                               {"ranking": list("abc")}, {"ranking": list("aaa")}, {"insertions": []}])
        with tempfile.TemporaryDirectory() as tmp:
            summary = evaluate_states(model, prompts, states, queries, tmp)
            self.assertEqual(summary["initial"]["utility"], 0.75)
            self.assertEqual(summary["repaired"]["utility"], 0.5)
            self.assertEqual(summary["repaired"]["invalid_rankings"], 1)
            self.assertEqual(summary["repaired"]["rank_distance"], 1)
        self.assertEqual(states, before)
        self.assertTrue(all('"target"' not in prompt for prompt, _, _ in model.calls))
        for status in (401, 402, 403):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                denied = ScriptedModel([ModelFailure("test-only service rejection", http_status=status)])
                with self.assertRaises(ModelFailure):
                    evaluate_states(denied, prompts, states, queries, tmp)
                self.assertEqual(len(denied.calls), 1)
                self.assertEqual(states, before)


if __name__ == "__main__":
    unittest.main()
