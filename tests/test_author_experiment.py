"""Tests for author-data intervention operators."""
import json
import tempfile
import unittest
from pathlib import Path

from rsi_researcher.paths import ROOT
from rsi_researcher.research.author_experiment import author_intervention, rank_only, training_intervention


def _prompts():
    return json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))


class AuthorInterventionTests(unittest.TestCase):
    def test_retain_is_unchanged(self):
        prompts = _prompts()
        changed = author_intervention(prompts, "retain")
        self.assertEqual(changed, prompts)

    def test_h1_neutralizes_true_user_reflection(self):
        prompts = _prompts()
        changed = author_intervention(prompts, "unseen_neutral_user_reflection")
        true_text = changed["user_prompt_template_true"]
        self.assertNotIn("you indeed dislike the CD", true_text)
        self.assertIn("only sampled as a comparison", true_text)
        self.assertNotIn("determine your dislikes based on the features of CDs you really don't enjoy",
                         true_text)
        # item reflection is byte-identical (single-variable intervention)
        self.assertEqual(changed["item_prompt_template_true"], prompts["item_prompt_template_true"])
        self.assertEqual(changed["item_prompt_template"], prompts["item_prompt_template"])
        # ranking prompt unchanged
        self.assertEqual(changed["system_prompt_template_evaluation_basic"],
                         prompts["system_prompt_template_evaluation_basic"])

    def test_h1_neutralizes_wrong_user_reflection(self):
        prompts = _prompts()
        changed = author_intervention(prompts, "unseen_neutral_user_reflection")
        wrong_text = changed["user_prompt_template"]
        self.assertNotIn("you discovered that you don't like the CD that you initially chose", wrong_text)
        self.assertIn("only sampled; its exposure and listening history are unknown", wrong_text)
        self.assertNotIn("you have not listened to it", wrong_text)

    def test_h2_appends_neutral_rank_note_only(self):
        prompts = _prompts()
        changed = author_intervention(prompts, "neutralize_rank_popularity")
        self.assertIn("Ignore any popularity, commercial success, or mainstream reputation",
                      changed["system_prompt_template_evaluation_basic"])
        # user + item reflection unchanged
        self.assertEqual(changed["user_prompt_template"], prompts["user_prompt_template"])
        self.assertEqual(changed["item_prompt_template_true"], prompts["item_prompt_template_true"])

    def test_unknown_operator_raises(self):
        with self.assertRaises(ValueError):
            author_intervention(_prompts(), "not_an_operator")

    def test_select_popularity_neutralized(self):
        prompts = _prompts()
        changed = author_intervention(prompts, "neutralize_select_popularity")
        self.assertNotIn("use common knowledge to inform your decision", changed["system_prompt_template"])
        self.assertIn("without relying on popularity or mainstream reputation", changed["system_prompt_template"])
        # only the select prompt changes
        self.assertEqual(changed["user_prompt_template_true"], prompts["user_prompt_template_true"])
        self.assertEqual(changed["system_prompt_template_evaluation_basic"],
                         prompts["system_prompt_template_evaluation_basic"])

    def test_candidate_order_neutralized(self):
        prompts = _prompts()
        changed = author_intervention(prompts, "neutral_candidate_order")
        self.assertIn("order in which the candidates are listed is arbitrary",
                      changed["system_prompt_template_evaluation_basic"])
        self.assertEqual(changed["system_prompt_template"], prompts["system_prompt_template"])


class RankOnlyTests(unittest.TestCase):
    class FakeRankModel:
        scope = None
        def __init__(self):
            self.calls = []
        def ask(self, prompt, schema, context=None):
            self.calls.append(context)
            return {"ranking": sorted(schema["properties"]["ranking"]["items"]["enum"])}

    def _fixture(self):
        prompts = _prompts()
        items = {i: {"title": t, "memory": f"The CD is called '{t}'."}
                 for i, t in (("a", "A"), ("b", "B"), ("c", "C"))}
        state = {"users": {"u1": "I like A."}, "items": items, "completed": []}
        queries = [{"query_id": "q1", "user": "u1", "candidates": ["a", "b", "c"], "target": "a"}]
        profile = {"training_item_counts": {"a": 3, "b": 2, "c": 1}, "head_min_count": 3, "tail_max_count": 1}
        return state, queries, prompts, profile

    def test_rank_only_reranks_without_training(self):
        state, queries, prompts, profile = self._fixture()
        out = Path(tempfile.mkdtemp()) / "rank"
        out.mkdir()
        model = self.FakeRankModel()
        rows = rank_only(out, model, state, queries, prompts, profile)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(model.calls), 1)
        self.assertTrue((out / "ranking.json").is_file())
        self.assertTrue(rows[0]["valid"])

    def test_rank_only_intervenes_ranking_prompt(self):
        state, queries, prompts, profile = self._fixture()
        out = Path(tempfile.mkdtemp()) / "rank"
        out.mkdir()
        model = self.FakeRankModel()
        rank_only(out, model, state, queries, prompts, profile, operator="neutralize_rank_popularity")
        operator_record = json.loads((out / "ranking_operator.json").read_text(encoding="utf-8"))
        self.assertEqual(operator_record["operator"], "neutralize_rank_popularity")
        self.assertIn("Ignore any popularity", operator_record["ranking_prompt"])


class ObservedReplayTests(unittest.TestCase):
    def test_replay_preserves_feedback_panels_budget_and_user_coverage(self):
        from rsi_researcher.datasets.author import author_data
        _, source, _ = author_data(ROOT / "data", 20, 5, 20261015, "training_popularity_quartiles")
        original = json.loads(json.dumps(source))
        replay, plan = training_intervention(source, "balanced_observed_replay", 20261015)
        self.assertEqual(source, original)
        self.assertEqual(len(replay), len(source))
        self.assertEqual(len({e["event_id"] for e in replay}), len(replay))
        self.assertEqual({e["user"] for e in replay}, {e["user"] for e in source})
        by_id = {e["event_id"]: e for e in source}
        for event in replay:
            clean = {k: v for k, v in event.items() if k not in ("event_id", "source_event_id")}
            expected = {k: v for k, v in by_id[event["source_event_id"]].items() if k != "event_id"}
            self.assertEqual(clean, expected)
        self.assertEqual(replay, training_intervention(source, "balanced_observed_replay", 20261015)[0])
        self.assertLess(plan["unique_source_events"], len(replay))

    def test_replay_reduces_domination_by_repeated_observations(self):
        # All observations are real in this fixture, but 90% concern one item.
        # Weighted replay should give the ten minority items learning slots.
        source = [{"event_id": str(i), "user": "u", "positive": "head" if i < 90 else str(i),
                   "negative": "other", "candidates": ["other", "head" if i < 90 else str(i)]}
                  for i in range(100)]
        replay, _ = training_intervention(source, "balanced_observed_replay", 17)
        self.assertLess(sum(e["positive"] == "head" for e in replay), 30)
        self.assertEqual(author_intervention(_prompts(), "balanced_observed_replay"), _prompts())


if __name__ == "__main__":
    unittest.main()
