from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
import unittest

from rsi_researcher.runtime.client import BudgetExceeded, InvalidOutput
from rsi_researcher.paths import ROOT
from rsi_researcher.datasets.controlled import build_world
from rsi_researcher.recommenders.execution import execute_arm, intervention, run_attempt, validate_proposal


class ProtocolModel:
    """Deterministic controller test only, never research evidence."""
    def __init__(self, budget=100, operator="uncertain_feedback"):
        self.requests, self.max_requests, self.operator = 0, budget, operator
        self.calls = []

    def ask(self, prompt, schema, context):
        if self.requests >= self.max_requests:
            raise BudgetExceeded("test budget")
        self.requests += 1
        self.calls.append((prompt, context))
        stage = context["stage"]
        if stage == "diagnose_and_propose":
            return dict(parent="B0", operator=self.operator, mechanism="test hypothesis",
                        evidence="test evidence", expected_direction="test direction", falsifier="test counterexample")
        if stage == "select":
            return {"selected_id": re.findall(r"ID=(\w+);", prompt)[0], "reason": "test"}
        if stage == "user_update":
            return {"memory": "Test user memory"}
        if stage == "item_update":
            return {"first_memory": "Test first memory", "second_memory": "Test second memory"}
        return {"ranking": re.findall(r"ID=(\w+);", prompt)}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.world = build_world(712, users=2, events_per_user=1)
        self.prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))

    def test_complete_attempt_isolated_archived_and_truth_not_sent(self):
        model = ProtocolModel()
        with tempfile.TemporaryDirectory() as tmp:
            decision = run_attempt(tmp, model, self.world, self.prompts)
            self.assertIn(decision["status"], ("rejected", "inconclusive"))
            directory = Path(tmp)
            self.assertEqual(json.loads((directory / "selection.json").read_text())["selected"], "B0")
            for arm in ("B0", "candidate"):
                before = json.loads((directory / arm / "train-0000.json").read_text())["before"]
                self.assertEqual(before, self.world["public"]["initial"])
            self.assertTrue((directory / "proposal.json").exists())
            for prompt, context in model.calls:
                self.assertNotIn('"utilities"', prompt)
                self.assertNotIn('"preperiod_counts"', prompt)
                if context["stage"] != "diagnose_and_propose":
                    self.assertNotIn("development_metrics", prompt)

    def test_unknown_operator_cannot_execute_or_replace_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            decision = run_attempt(tmp, ProtocolModel(operator="execute_code"), self.world, self.prompts)
            self.assertEqual(decision["status"], "failed")
            self.assertFalse((Path(tmp) / "candidate").exists())
            self.assertEqual(json.loads((Path(tmp) / "selection.json").read_text())["selected"], "B0")

    def test_budget_failure_is_archived_and_leaves_no_partial_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = ProtocolModel(budget=2)
            decision = run_attempt(tmp, model, self.world, self.prompts)
            self.assertEqual(decision["error"], "BudgetExceeded")
            self.assertEqual(model.requests, 2)
            self.assertFalse((Path(tmp) / "B0" / "train-0000.json").exists())

    def test_whitelist_changes_only_feedback_and_does_not_mutate_parent(self):
        before = dict(self.prompts)
        changed = intervention(self.prompts, "uncertain_feedback")
        expected = {"user_prompt_template", "user_prompt_template_true", "item_prompt_template", "item_prompt_template_true"}
        self.assertEqual({k for k in changed if changed[k] != before[k]}, expected)
        self.assertEqual(self.prompts, before)
        self.assertEqual(intervention(self.prompts, "retain"), before)
        with self.assertRaises(InvalidOutput):
            intervention(self.prompts, "arbitrary_python")
        with self.assertRaises(InvalidOutput):
            validate_proposal(dict(parent="B0", operator=[], mechanism="x", evidence="x",
                                   expected_direction="x", falsifier="x"))

    def test_private_item_memory_reaches_owner_but_not_another_user(self):
        class NamedWrites(ProtocolModel):
            def ask(self, prompt, schema, context):
                result = super().ask(prompt, schema, context)
                if context["stage"] == "item_update":
                    result = dict.fromkeys(("first_memory", "second_memory"), "saved " + context["event"])
                return result

        public = deepcopy(self.world["public"])
        original = deepcopy(public)
        events = [{"event_id": eid, "user": user, "positive": pos, "negative": neg, "candidates": [neg, pos]}
                  for eid, user, pos, neg in (("e0", "u000", "i000", "i001"),
                                             ("e1", "u001", "i000", "i001"),
                                             ("e2", "u000", "i002", "i000"))]
        for policy in ("original_update", "no_sharing"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp:
                model = NamedWrites()
                directory = Path(tmp) / "arm"
                execute_arm(directory, model, self.prompts, public, events, memory_policy=policy)
                selections = {c["event"]: p for p, c in model.calls if c["stage"] == "select" and c["round"] == 0}
                self.assertEqual("saved e0" in selections["e1"], policy == "original_update")
                self.assertIn("saved e0" if policy == "no_sharing" else "saved e1", selections["e2"])
                state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
                if policy == "no_sharing":
                    self.assertEqual(state["items"], public["initial"]["items"])
                    self.assertEqual(state["private_items"]["u000"]["i000"]["memory"], "saved e0")
                    self.assertEqual(state["private_items"]["u001"]["i000"]["memory"], "saved e1")
                    for prompt, context in model.calls:
                        if context["stage"] == "rank" and context["query"].startswith("dev-u001"):
                            self.assertNotIn("saved e2", prompt)
                    with self.assertRaises(ValueError):
                        execute_arm(directory, NamedWrites(), self.prompts, public, events, resume=True)
                resumed = NamedWrites(budget=0)
                execute_arm(directory, resumed, self.prompts, public, events, resume=True, memory_policy=policy)
                self.assertEqual(resumed.requests, 0)
        self.assertEqual(public, original)


if __name__ == "__main__":
    unittest.main()
