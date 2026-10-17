import json
import unittest

from rsi_researcher.recommenders.agentcf import AgentCF
from rsi_researcher.runtime.client import InvalidOutput
from rsi_researcher.paths import ROOT
from rsi_researcher.diagnostics.ranking_format import correct_format
from tests.test_agentcf_runner import ScriptedModel


class FormatProbeTests(unittest.TestCase):
    def test_corrected_membership_preserves_order_without_truth(self):
        model = ScriptedModel([{"insertions": [{"item_id": "c", "before_id": "b"}]}])
        prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))
        state = {"users": {"u": "preference"}, "items": {i: {"memory": i} for i in "abc"}}
        query = {"query_id": "q", "user": "u", "candidates": list("abc"), "target": "hidden-target"}
        result = correct_format(AgentCF(model, prompts), state, query, {"ranking": list("ab")})
        self.assertEqual(result, list("acb"))
        self.assertNotIn("hidden-target", model.calls[0][0])

    def test_bad_correction_is_not_silently_filled_or_retried(self):
        prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))
        state = {"users": {"u": "preference"}, "items": {i: {"memory": i} for i in "abc"}}
        query = {"query_id": "q", "user": "u", "candidates": list("abc")}
        for result in ([], [{"item_id": "a", "before_id": "b"}], [{"item_id": "c", "before_id": "c"}]):
            model = ScriptedModel([{"insertions": result}])
            with self.assertRaises(InvalidOutput):
                correct_format(AgentCF(model, prompts), state, query, {"ranking": list("ab")})
            self.assertEqual(len(model.calls), 1)

    def test_multiple_missing_items_and_duplicate_only_case(self):
        prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))
        state = {"users": {"u": "preference"}, "items": {i: {"memory": i} for i in "abcd"}}
        query = {"query_id": "q", "user": "u", "candidates": list("abcd")}
        model = ScriptedModel([{"insertions": [{"item_id": "c", "before_id": "b"}, {"item_id": "d", "before_id": "b"}]}])
        self.assertEqual(correct_format(AgentCF(model, prompts), state, query, {"ranking": list("aab")}), list("acdb"))
        no_call = ScriptedModel([])
        self.assertEqual(correct_format(AgentCF(no_call, prompts), state, query, {"ranking": list("aabcd")}), list("abcd"))
        self.assertEqual(no_call.calls, [])

    def test_empty_previous_ranking_requires_null_anchors(self):
        prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))
        state = {"users": {"u": "preference"}, "items": {i: {"memory": i} for i in "ab"}}
        query = {"query_id": "q", "user": "u", "candidates": list("ab")}
        model = ScriptedModel([{"insertions": [{"item_id": "b", "before_id": None}, {"item_id": "a", "before_id": None}]}])
        self.assertEqual(correct_format(AgentCF(model, prompts), state, query, {"ranking": []}), list("ba"))
        schema = model.calls[0][1]
        self.assertEqual(schema["properties"]["insertions"]["items"]["properties"]["before_id"], {"type": "null"})


if __name__ == "__main__":
    unittest.main()
