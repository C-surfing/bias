from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from rsi_researcher.runtime.client import BudgetExceeded
from rsi_researcher.recommenders.replay import repair


class ReadingKernel:
    def __init__(self, rule="normal", fail=False):
        self.rule, self.fail, self.calls = rule, fail, []

    def event(self, state, event, arm):
        self.calls.append(event["event_id"])
        if self.fail:
            state["users"][event["user"]] = "partial write"
            raise BudgetExceeded("test failure")
        state = deepcopy(state)
        value = self.rule + ":" + state["users"][event["user"]] + ":" + ":".join(state["items"][i] for i in event["candidates"])
        state["users"][event["user"]] = value
        state["items"][event["positive"]] = value
        state["completed"].append(event["event_id"])
        return state


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.initial = {"users": dict.fromkeys("abc", "initial"), "items": dict.fromkeys("pqrs", "initial"), "completed": []}
        self.events = [{"event_id": eid, "user": u, "positive": p, "negative": n, "candidates": [n, p]}
                       for eid, u, p, n in (("e0", "a", "p", "q"), ("e1", "c", "r", "s"),
                                           ("e2", "b", "q", "p"), ("e3", "b", "s", "r"))]
        self.records = []
        state = deepcopy(self.initial)
        for event in self.events:
            after = ReadingKernel().event(state, event, "original_update")
            self.records.append({"event": event, "before": state, "after": after})
            state = after

    def test_selective_repair_matches_full_recomputation(self):
        original, corrected = ReadingKernel(), ReadingKernel("corrected")
        expected = deepcopy(self.initial)
        for event in self.events:
            kernel = ReadingKernel("corrected" if event["event_id"] == "e0" else "normal")
            expected = kernel.event(expected, event, "original_update")
        before = deepcopy(self.records)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "repair"
            result = repair(self.records, ["e0"], original, corrected, output)
            self.assertEqual(result, expected)
            self.assertEqual(original.calls, ["e2", "e3"])
            self.assertEqual(corrected.calls, ["e0"])
            self.assertEqual(json.loads((output / "e1.json").read_text())["action"], "reuse_independent")
        self.assertEqual(self.records, before)

    def test_failure_does_not_publish_final_state_or_mutate_source(self):
        before = deepcopy(self.records)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "repair"
            with self.assertRaises(BudgetExceeded):
                repair(self.records, ["e0"], ReadingKernel(fail=True), ReadingKernel("corrected"), output)
            self.assertFalse((output / "state.json").exists())
            self.assertEqual(json.loads((output / "result.json").read_text())["status"], "failed")
        self.assertEqual(self.records, before)

    def test_invalid_source_fails_before_creating_branch(self):
        self.records[1]["before"] = self.initial
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "repair"
            with self.assertRaises(ValueError):
                repair(self.records, ["e0"], ReadingKernel(), ReadingKernel(), output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
