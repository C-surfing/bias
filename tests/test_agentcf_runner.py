from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from rsi_researcher.recommenders.agentcf import AgentCF
from rsi_researcher.runtime.client import AUTHOR_COMMIT, BudgetExceeded, DeepSeek, InvalidOutput, ModelFailure, atomic_json, parse_json, usage_summary
from rsi_researcher.paths import ROOT
from rsi_researcher.datasets.author import author_data
from rsi_researcher.diagnostics.agentcf_smoke import ndcg


class ScriptedModel:
    """Protocol test double; its outputs are never research results."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def ask(self, prompt, schema, context):
        self.calls.append((prompt, schema, context))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))
        self.state = {"users": {"u1": "initial", "u2": "initial"},
                      "items": {"p": {"title": "Same title", "memory": "positive"},
                                "n": {"title": "Same title", "memory": "negative"}}, "completed": []}
        self.event = {"event_id": "e1", "user": "u1", "positive": "p", "negative": "n", "candidates": ["n", "p"]}

    def successful_replies(self):
        return [{"selected_id": "p", "reason": "reason"}, {"memory": "updated user"},
                {"first_memory": "unused negative", "second_memory": "shared positive"},
                {"selected_id": "p", "reason": "reason"},
                {"first_memory": "unused negative", "second_memory": "shared positive 2"}]

    def test_rounds_write_user_and_only_positive_item(self):
        model = ScriptedModel(self.successful_replies())
        before = deepcopy(self.state)
        after = AgentCF(model, self.prompts).event(self.state, self.event, "original_update")
        self.assertEqual(self.state, before)
        self.assertEqual(after["users"], {"u1": "updated user", "u2": "initial"})
        self.assertEqual(after["items"]["n"], before["items"]["n"])
        self.assertEqual(after["items"]["p"]["memory"], "shared positive 2")
        self.assertEqual([call[2]["stage"] for call in model.calls], ["select", "user_update", "item_update", "select", "item_update"])

    def test_wrong_selections_update_user_on_both_rounds(self):
        replies = [{"selected_id": "n", "reason": "wrong"}, {"memory": "user1"},
                   {"first_memory": "n1", "second_memory": "p1"},
                   {"selected_id": "n", "reason": "wrong"}, {"memory": "user2"},
                   {"first_memory": "n2", "second_memory": "p2"}]
        after = AgentCF(ScriptedModel(replies), self.prompts).event(self.state, self.event, "original_update")
        self.assertEqual(after["users"]["u1"], "user2")
        self.assertEqual(after["items"]["p"]["memory"], "p2")

    def test_failure_leaves_no_partial_state(self):
        model = ScriptedModel(self.successful_replies()[:2] + [ModelFailure("failed")])
        before = deepcopy(self.state)
        with self.assertRaises(ModelFailure):
            AgentCF(model, self.prompts).event(self.state, self.event, "original_update")
        self.assertEqual(self.state, before)

    def test_frozen_and_trial_isolation(self):
        model = ScriptedModel(self.successful_replies())
        frozen = AgentCF(model, self.prompts).event(self.state, self.event, "frozen")
        updated = AgentCF(ScriptedModel(self.successful_replies()), self.prompts).event(self.state, self.event, "original_update")
        self.assertEqual(frozen["users"], self.state["users"])
        self.assertEqual(frozen["items"], self.state["items"])
        self.assertNotEqual(updated["items"], frozen["items"])
        self.assertEqual([call[2]["stage"] for call in model.calls], ["select", "user_update", "item_update", "select", "item_update"])
        self.assertIn("Same title", model.calls[1][0])
        self.assertNotIn("updated user", model.calls[2][0])
        self.assertNotIn("shared positive", model.calls[3][0])

    def test_memory_ablations_preserve_only_the_selected_write_kind(self):
        for arm, user_changed, item_changed in (("user_only", True, False), ("item_only", False, True)):
            with self.subTest(arm=arm):
                model = ScriptedModel(self.successful_replies())
                after = AgentCF(model, self.prompts).event(self.state, self.event, arm)
                self.assertEqual(after["users"] != self.state["users"], user_changed)
                self.assertEqual(after["items"] != self.state["items"], item_changed)
                self.assertEqual(after["completed"], ["e1"])
                self.assertEqual("updated user" in model.calls[2][0], user_changed)
                self.assertEqual("shared positive" in model.calls[3][0], item_changed)

    def test_shared_item_reaches_other_user_without_hidden_labels(self):
        after = AgentCF(ScriptedModel(self.successful_replies()), self.prompts).event(self.state, self.event, "original_update")
        model = ScriptedModel([{"ranking": ["p", "n"]}])
        query = {"query_id": "q2", "user": "u2", "candidates": ["n", "p"], "target": "p"}
        AgentCF(model, self.prompts).rank(after, query)
        self.assertIn("shared positive 2", model.calls[0][0])
        self.assertNotIn('"target"', model.calls[0][0])

    def test_reverse_order_still_updates_positive(self):
        event = self.event | {"candidates": ["p", "n"]}
        replies = self.successful_replies()
        replies[2] = {"first_memory": "p1", "second_memory": "n1"}
        replies[4] = {"first_memory": "p2", "second_memory": "n2"}
        after = AgentCF(ScriptedModel(replies), self.prompts).event(self.state, event, "original_update")
        self.assertEqual(after["items"]["p"]["memory"], "p2")
        self.assertEqual(after["items"]["n"]["memory"], "negative")

    def test_ranking_requires_exact_ids_despite_duplicate_titles(self):
        query = {"query_id": "q", "user": "u1", "candidates": ["p", "n"], "target": "p"}
        for ranking in [["p", "p"], ["p"], ["p", "foreign"], ["Same title", "Same title"]]:
            with self.subTest(ranking=ranking), self.assertRaises(InvalidOutput):
                AgentCF(ScriptedModel([{"ranking": ranking}, {"insertions": []}]), self.prompts).rank(self.state, query)

    def test_rank_applies_one_shared_correction_and_keeps_relative_order(self):
        query = {"query_id": "q", "user": "u1", "candidates": ["p", "n"], "target": "private"}
        model = ScriptedModel([{"ranking": ["p"]}, {"insertions": [{"item_id": "n", "before_id": None}]}])
        self.assertEqual(AgentCF(model, self.prompts).rank(self.state, query), ["p", "n"])
        self.assertEqual(len(model.calls), 2)
        self.assertNotIn("private", model.calls[1][0])
        schema = model.calls[0][1]
        self.assertEqual(schema["properties"]["ranking"]["items"]["enum"], ["n", "p"])
        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("private", json.dumps(schema))

    def test_illegal_selection_and_duplicate_json_are_rejected(self):
        with self.assertRaises(InvalidOutput):
            AgentCF(ScriptedModel([{"selected_id": "foreign", "reason": "x"}]), self.prompts).event(self.state, self.event, "original_update")
        with self.assertRaises(InvalidOutput):
            parse_json('{"ranking":[],"ranking":["p"]}')

    def test_ndcg_known_values(self):
        self.assertEqual(ndcg(["p", "n"], "p"), 1.0)
        self.assertAlmostEqual(ndcg(["n", "p"], "p"), 0.6309297535714575)
        self.assertEqual(ndcg(["n"], "p"), 0.0)

    def test_author_data_panels_and_order(self):
        initial, events, queries = author_data(ROOT / "data", 2, 2)
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0]["user"], events[2]["user"])
        self.assertNotEqual(events[0]["user"], events[1]["user"])
        self.assertEqual(len(queries), 2)
        self.assertTrue(all(len(set(q["candidates"])) == 10 for q in queries))
        self.assertFalse(any("target" in event for event in events))

    def test_retry_budget_and_secret_not_in_log(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "private-test-value"}):
            model = DeepSeek(tmp, max_requests=2)
            with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")), patch("time.sleep"):
                with self.assertRaises(BudgetExceeded):
                    model.ask("json test", '{"ok":true}', {"stage": "test"})
            summary = usage_summary(tmp)
            self.assertEqual(summary["requests"], 2)
            self.assertEqual(summary["requests_with_unknown_usage"], 2)
            self.assertNotIn("private-test-value", (Path(tmp) / "calls.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(DeepSeek(tmp, max_requests=2).requests, 2)

    def test_auth_failure_is_not_retried(self):
        for status in (401, 402, 403):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}):
                model = DeepSeek(tmp)
                failure = urllib.error.HTTPError("https://api.deepseek.com", status, "Rejected", {}, None)
                with patch("urllib.request.urlopen", side_effect=failure) as request:
                    with self.assertRaises(ModelFailure) as caught:
                        model.ask("json test", '{"ok":true}', {"stage": "test"})
                    self.assertEqual(caught.exception.http_status, status)
                    self.assertEqual(request.call_count, 1)

    def test_empty_json_retries_are_bounded_and_all_usage_is_counted(self):
        def packet(content):
            return json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
        for contents, succeeds in (([" ", " ", "{}"], True), ([" "] * 3, False)):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}):
                model = DeepSeek(tmp, max_requests=10)
                with patch("urllib.request.urlopen") as network, patch("time.sleep"):
                    network.return_value.__enter__.return_value.read.side_effect = [packet(c) for c in contents]
                    if succeeds:
                        self.assertEqual(model.ask("json", "{}", {}), {})
                    else:
                        with self.assertRaises(InvalidOutput):
                            model.ask("json", "{}", {})
                self.assertEqual(network.call_count, 3)
                self.assertEqual(usage_summary(tmp)["prompt_tokens"], 30)
                self.assertEqual(usage_summary(tmp)["completion_tokens"], 15)

    def test_atomic_snapshot_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            atomic_json(path, self.state)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), self.state)


if __name__ == "__main__":
    unittest.main()
