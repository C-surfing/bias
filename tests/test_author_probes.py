"""Unit tests for the model-free author bias probes (zero model calls)."""
import json
import unittest
from pathlib import Path

from rsi_researcher.datasets.author import author_data, author_profile, popularity_group
from rsi_researcher.evaluation.author import score_author_ranking
from rsi_researcher.evaluation.author_probes import (
    kendall_tau, probe_exposure_bias, probe_conformity_bias,
    probe_selection_paac, probe_position_bias, run_author_probes, _false_dislike)
from rsi_researcher.paths import ROOT


def _state(users=None, items=None, completed=None):
    return {"users": users or {}, "items": items or {}, "completed": completed or []}


class KendallTauTests(unittest.TestCase):
    def test_identical_order_is_one(self):
        self.assertEqual(kendall_tau(["a", "b", "c"], ["a", "b", "c"]), 1.0)

    def test_reversed_order_is_minus_one(self):
        self.assertEqual(kendall_tau(["a", "b", "c"], ["c", "b", "a"]), -1.0)

    def test_requires_same_items(self):
        with self.assertRaises(ValueError):
            kendall_tau(["a", "b"], ["a", "c"])


class ExposureBiasProbeTests(unittest.TestCase):
    def setUp(self):
        self.items = {
            "neg": {"title": "1967-1970 The Blue Album", "memory": "The CD is called '1967-1970 The Blue Album'."},
            "pos": {"title": "4", "memory": "The CD is called '4'."},
        }
        self.inputs = {"initial": {"items": self.items}}
        self.before = _state({"u1": "I enjoy listening to CDs very much."}, self.items)
        self.after = _state({"u1": "I dislike CDs with complex arrangements, like '1967-1970 The Blue Album'."},
                            self.items)

    def _record(self, after):
        return [{"event": {"event_id": "train-0000", "user": "u1", "positive": "pos",
                           "negative": "neg", "candidates": ["neg", "pos"]},
                 "before": self.before, "after": after, "memory_policy": "original_update"}]

    def test_detects_fabricated_dislike_of_unseen_item(self):
        result = probe_exposure_bias(self.after, self._record(self.after), self.inputs)
        self.assertEqual(result["false_dislike_events"], 1)
        self.assertEqual(result["false_dislike_memory_rate"], 1.0)
        self.assertEqual(result["representative_cases"][0]["negative_title"], "1967-1970 The Blue Album")

    def test_no_dislike_is_zero(self):
        neutral = _state({"u1": "I prefer melodic rock albums."}, self.items)
        result = probe_exposure_bias(neutral, self._record(neutral), self.inputs)
        self.assertEqual(result["false_dislike_events"], 0)

    def test_numeric_title_is_skipped(self):
        numeric_after = _state({"u1": "I dislike complex arrangements, like '4'."}, self.items)
        result = probe_exposure_bias(numeric_after, self._record(numeric_after), self.inputs)
        self.assertEqual(result["false_dislike_events"], 0)

    def test_positive_only_does_not_trigger(self):
        positive_after = _state({"u1": "I dislike '4' because it is too pop."}, self.items)
        result = probe_exposure_bias(positive_after, self._record(positive_after), self.inputs)
        self.assertEqual(result["false_dislike_events"], 0)

    def test_negated_dislike_is_not_a_dislike_error(self):
        for text in ["I do not have specific dislikes regarding '1967-1970 The Blue Album'.",
                     "I do not dislike '1967-1970 The Blue Album'.",
                     "I don't think I dislike '1967-1970 The Blue Album'.",
                     "I enjoy the intricate music of '1967-1970 The Blue Album'."]:
            with self.subTest(text=text):
                self.assertFalse(_false_dislike(text, "1967-1970 The Blue Album"))
        self.assertTrue(_false_dislike("I do not like '1967-1970 The Blue Album'.",
                                       "1967-1970 The Blue Album"))

    def test_raw_screen_is_retained_separately(self):
        neutral = _state({"u1": "I have no specific dislikes regarding '1967-1970 The Blue Album'."}, self.items)
        result = probe_exposure_bias(neutral, self._record(neutral), self.inputs)
        self.assertEqual(result["false_dislike_memory_rate"], 0)
        self.assertEqual(result["legacy_raw_lexicon_rate"], 1)


class ConformityBiasProbeTests(unittest.TestCase):
    def test_counts_popularity_terms_in_select_reasons(self):
        state = _state({"u1": "I enjoy music."}, {"i1": {"title": "X", "memory": "a classic hit album"}})
        calls = [
            {"status": "started", "call_id": 1, "context": {"stage": "select"}},
            {"status": "received", "call_id": 1,
             "content": "{\"selected_id\": \"i1\", \"reason\": \"It is a widely acclaimed classic hit.\"}"},
            {"status": "started", "call_id": 2, "context": {"stage": "select"}},
            {"status": "received", "call_id": 2,
             "content": "{\"selected_id\": \"i1\", \"reason\": \"Matches my taste for guitar music.\"}"},
        ]
        result = probe_conformity_bias(state, calls)
        self.assertEqual(result["popularity_prior_lexicon_rate"], 0.5)
        self.assertEqual(result["item_memory_popularity_rate"], 1.0)

    def test_empty_select_reasons_are_none(self):
        result = probe_conformity_bias(_state(), [])
        self.assertIsNone(result["popularity_prior_lexicon_rate"])

    def test_item_ids_and_word_fragments_do_not_count_as_popularity_reasoning(self):
        calls = [{"status": "started", "call_id": 1, "context": {"stage": "select"}},
                 {"status": "received", "call_id": 1,
                  "content": json.dumps({"selected_id": "popular-hit", "reason": "I like classical music and white covers."})}]
        self.assertEqual(probe_conformity_bias(_state(), calls)["select_reason_popularity_rate"], 0)


class SelectionPaacProbeTests(unittest.TestCase):
    def test_groups_and_update_counts(self):
        counts = {"head": 5, "mid": 2, "tail": 1}
        items = {"head": {"title": "h", "memory": "updated head memory"}, "mid": {"title": "m", "memory": "mid"},
                 "tail": {"title": "t", "memory": "tail"}, "unseen": {"title": "u", "memory": "unseen"}}
        inputs = {"initial": {"items": {k: {"title": v["title"], "memory": "The CD is called 'x'."}
                                        for k, v in items.items()}}}
        state = _state(items=items, completed=["train-0000", "train-0001", "train-0002"])
        records = [
            {"event": {"event_id": "train-0000", "user": "u1", "positive": "head", "negative": "unseen",
                       "candidates": ["unseen", "head"]}, "before": state, "after": state, "memory_policy": "original_update"},
            {"event": {"event_id": "train-0001", "user": "u2", "positive": "head", "negative": "unseen",
                       "candidates": ["unseen", "head"]}, "before": state, "after": state, "memory_policy": "original_update"},
            {"event": {"event_id": "train-0002", "user": "u3", "positive": "mid", "negative": "unseen",
                       "candidates": ["unseen", "mid"]}, "before": state, "after": state, "memory_policy": "original_update"},
        ]
        profile = {"training_item_counts": counts, "head_min_count": 3, "tail_max_count": 1,
                   "group_definition": "test"}
        result = probe_selection_paac(state, records, inputs, profile)
        self.assertEqual(result["groups"]["head"]["mean_update_count"], 2.0)
        self.assertEqual(result["groups"]["mid"]["mean_update_count"], 1.0)
        self.assertEqual(result["groups"]["tail"]["mean_update_count"], 0.0)
        self.assertEqual(result["groups"]["unseen"]["mean_update_count"], 0.0)
        self.assertEqual(result["groups"]["head"]["count"], 1)
        self.assertEqual(result["groups"]["unseen"]["count"], 1)


class PositionBiasProbeTests(unittest.TestCase):
    def test_identical_input_and_output_order(self):
        rows = [{"query": {"query_id": "q1", "candidates": ["a", "b", "c"]},
                 "ranking": ["a", "b", "c"], "valid": True,
                 "top5_head_share": 0.4, "candidate_head_share": 0.2}]
        result = probe_position_bias(rows)
        self.assertEqual(result["input_order_following_kendall_tau"], 1.0)

    def test_invalid_ranking_is_skipped(self):
        rows = [{"query": {"query_id": "q1", "candidates": ["a", "b", "c"]},
                 "ranking": [], "valid": False}]
        result = probe_position_bias(rows)
        self.assertIsNone(result["input_order_following_kendall_tau"])


class ExcessPopularityExposureTests(unittest.TestCase):
    def test_excess_is_head_share_minus_candidate_share(self):
        _, _, queries = author_data(ROOT / "data", 1, 1)
        profile = author_profile(ROOT / "data")
        query = queries[0]
        target = query["target"]
        others = [i for i in query["candidates"] if i != target]
        ranking = [target] + others
        score = score_author_ranking(ranking, query, profile)
        self.assertAlmostEqual(score["excess_popularity_exposure"],
                               score["top5_head_share"] - score["candidate_head_share"])


class SmokeRunProbesTests(unittest.TestCase):
    def test_runs_on_committed_local_run(self):
        run_dir = ROOT / "runs" / "qwen-author-day1-fixedtext-20261015"
        if not (run_dir / "original_update" / "state.json").exists():
            self.skipTest("local committed run directory not present")
        result = run_author_probes(run_dir)
        self.assertEqual(result["dataset"], "CDs-100-user-dense")
        self.assertGreaterEqual(result["exposure_bias"]["false_dislike_events"], 0)
        self.assertIn("head", result["selection_paac"]["groups"])


if __name__ == "__main__":
    unittest.main()
