"""Author-data integration against a local HTTP fixture, never a model benchmark."""
from argparse import Namespace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from rsi_researcher.datasets.author import author_data, author_profile, popularity_group
from rsi_researcher.diagnostics.agentcf_smoke import run
from rsi_researcher.evaluation.author import score_author_ranking
from rsi_researcher.paths import ROOT
from tests import test_model_backends as backend_fixture


class AuthorProtocolTests(unittest.TestCase):
    def test_memory_pattern_allows_sentences_under_full_match_decoding(self):
        import re
        from jsonschema import Draft202012Validator
        from rsi_researcher.runtime.client import TEXT_SCHEMA
        sentence = "I enjoy piano CDs.\nOther preferences are unknown."
        self.assertIsNotNone(re.fullmatch(TEXT_SCHEMA["pattern"], sentence))
        Draft202012Validator(TEXT_SCHEMA).validate(sentence)
        self.assertIsNone(re.fullmatch(TEXT_SCHEMA["pattern"], " \n\t"))

    def test_candidate_pool_is_not_the_ranking_budget(self):
        initial, events, queries = author_data(ROOT / "data", 2, 5)
        self.assertEqual(len(events), 10)
        pools = dict(line.rstrip().split("\t") for line in (ROOT / "data/CDs-100-user-dense.random").read_text().splitlines())
        for query in queries:
            first_nine = [i for i in pools[query["user"]].split() if i != query["target"]][:9]
            self.assertEqual(set(query["candidates"]), set(first_nine + [query["target"]]))
        profile = author_profile(ROOT / "data")
        self.assertEqual(profile["training_events"], 500)
        self.assertEqual(profile["distinct_observed_items_per_user"], {"min": 6, "max": 6})
        self.assertEqual(popularity_group("absent", {}, 3, 1), "unseen")

    def test_metrics_distinguish_target_ranks_with_ten_candidates(self):
        _, _, queries = author_data(ROOT / "data", 1, 1)
        query, profile = queries[0], author_profile(ROOT / "data")
        others = [i for i in query["candidates"] if i != query["target"]]
        first = score_author_ranking([query["target"]] + others, query, profile)
        last = score_author_ranking(others + [query["target"]], query, profile)
        self.assertEqual(first["hit_at_5"], 1)
        self.assertEqual(last["hit_at_5"], 0)
        self.assertEqual(last["ndcg_at_5"], 0)
        self.assertGreater(last["ndcg_at_10"], 0)


class AuthorBackendTests(unittest.TestCase):
    setUp = backend_fixture.ModelBackendTests.setUp

    def test_balanced_replay_is_saved_and_resumes_without_new_requests(self):
        from rsi_researcher.research.author_experiment import training_intervention
        config = json.loads((ROOT / "configs/qwen_agentcf_author.json").read_text(encoding="utf-8"))
        config["llm"] = self.config["llm"]
        _, source_events, queries = author_data(ROOT / "data", 2, 5, config["seed"])
        replay, _ = training_intervention(source_events, "balanced_observed_replay", config["seed"])
        self.responses = [{"ranking": q["candidates"]} for q in queries]
        for event in replay:
            self.responses += [
                {"selected_id": event["positive"], "reason": "Observed preference matches."},
                {"memory": "Observed user memory."},
                {"first_memory": "Unobserved comparison.", "second_memory": "Observed item memory."},
                {"selected_id": event["positive"], "reason": "Observed preference matches."},
                {"first_memory": "Unobserved comparison.", "second_memory": "Observed item memory."}]
        self.responses += [{"ranking": q["candidates"]} for q in queries]
        config_path = self.root / "replay-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        args = Namespace(config=str(config_path), data=None, output=str(self.root / "replay"),
                         users=2, events_per_user=5, max_requests=70, resume=False,
                         operator="balanced_observed_replay")
        result = run(args)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["requests"], 54)
        saved = json.loads((self.root / "replay/input.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["events"], replay)
        self.assertEqual(saved["source_events"],source_events)
        self.assertEqual(saved["queries"], queries)
        self.assertEqual(saved["training_schedule"]["policy"], "balanced_observed_replay")
        args.resume = True
        run(args)
        self.assertEqual(len(self.requests), 54)

    def test_author_run_uses_vllm_skips_frozen_reflection_and_resumes_without_calls(self):
        _, events, queries = author_data(ROOT / "data", 2, 1)
        self.responses = [{"ranking": q["candidates"]} for q in queries]
        for event in events:
            self.responses += [
                {"selected_id": event["positive"], "reason": "Observed preference matches."},
                {"memory": "Observed user memory."},
                {"first_memory": "Unobserved item unchanged.", "second_memory": "Observed item memory."},
                {"selected_id": event["positive"], "reason": "Observed preference matches."},
                {"first_memory": "Unobserved item unchanged.", "second_memory": "Observed item memory."}]
        self.responses += [{"ranking": q["candidates"]} for q in queries]
        config = json.loads((ROOT / "configs/qwen_agentcf_author.json").read_text(encoding="utf-8"))
        config["llm"] = self.config["llm"]
        config_path = self.root / "author.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        args = Namespace(config=str(config_path), data=None, output=str(self.root / "run"),
                         users=2, events_per_user=1, max_requests=30, resume=False)
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "must-not-be-used"}):
            result = run(args)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.requests), 14)
        self.assertEqual(result["frozen_training_requests"], 0)
        self.assertTrue((self.root / "run/report.md").is_file())
        log = [json.loads(line) for line in (self.root / "run/calls.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["context"]["stage"] == "rank" for row in log if row["status"] == "started" and row["context"]["arm"] == "frozen"))
        self.assertTrue(all("Authorization" not in headers for _, _, headers in self.requests))
        args.resume = True
        repeated = run(args)
        self.assertEqual(repeated["requests"], 14)
        self.assertEqual(len(self.requests), 14)

    def test_transport_failure_does_not_become_zero_quality(self):
        from rsi_researcher.diagnostics.agentcf_smoke import execute
        from rsi_researcher.runtime.client import ModelFailure
        _, events, queries = author_data(ROOT / "data", 1, 1)
        initial, _, _ = author_data(ROOT / "data", 1, 1)
        class FailedModel:
            def ask(self, *_):
                raise ModelFailure("offline")
        output = self.root / "failed"
        output.mkdir()
        with self.assertRaises(ModelFailure):
            execute(output, FailedModel(), initial, events, queries,
                    json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8")), author_profile(ROOT / "data"))
        self.assertFalse((output / "summary.json").exists())
