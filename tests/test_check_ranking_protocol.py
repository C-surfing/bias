from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from rsi_researcher.paths import ROOT
from rsi_researcher.diagnostics.ranking_protocol import run_check
from rsi_researcher.datasets.controlled import build_world
from tests.test_improvement_workflow import ProtocolModel


class RankingCheckTests(unittest.TestCase):
    def test_frozen_states_are_only_ranked_not_retrained(self):
        world = build_world(817, users=2, events_per_user=2)
        states = {"initial": deepcopy(world["public"]["initial"]), "original_update": deepcopy(world["public"]["initial"])}
        states["original_update"]["users"]["u000"] = "saved learned state"
        before = deepcopy(states)
        prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))
        model = ProtocolModel()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_check(Path(tmp), model, world, states, prompts)
            resumed = ProtocolModel(budget=1)
            self.assertEqual(run_check(Path(tmp), resumed, world, states, prompts, resume=True), result)
            self.assertEqual(resumed.requests, 0)
        self.assertEqual(states, before)
        self.assertEqual(model.requests, 8)
        self.assertTrue(all(context["stage"] == "rank" for _, context in model.calls))
        self.assertTrue(all("utilities" not in prompt for prompt, _ in model.calls))
        self.assertEqual(result["initial"]["query_count"], 4)


if __name__ == "__main__":
    unittest.main()
