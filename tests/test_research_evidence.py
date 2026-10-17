"""Evidence plumbing tests with authored rankings; not research-effect evidence."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from rsi_researcher.research.evidence import build_candidate_evidence, compact_candidate_evidence


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ResearchEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.items = [f"i{i:03d}" for i in range(12)]
        self.world = {"public": {"initial": {
            "users": {"A": "initial A", "B": "initial B"},
            "items": {item: {"title": item, "memory": "observed catalog note"} for item in self.items},
            "completed": []}, "queries": [
                {"query_id": "qA", "user": "A", "candidates": self.items},
                {"query_id": "qB", "user": "B", "candidates": self.items}]},
            "private": {"utilities": {u: {i: 4 if n < 6 else 1 for n, i in enumerate(self.items)}
                                       for u in ("A", "B")},
                        "popular": self.items[-3:], "sampling": "PRIVATE_TRACE_MARKER"}}
        self.events = [{"event_id": "e0", "user": "A", "positive": "i009", "negative": "i000",
                        "candidates": ["i000", "i009"]},
                       {"event_id": "e1", "user": "B", "positive": "i001", "negative": "i009",
                        "candidates": ["i009", "i001"]}]
        good, bad = self.items[:], list(reversed(self.items))
        self.baseline_pair = ({"qA": good, "qB": bad}, {"qA": good, "qB": bad})
        self.candidate_pair = ({"qA": bad, "qB": good}, {"qA": bad, "qB": good})

    def arm(self, path, label, private=False, reverse=False, actions=()):
        state = deepcopy(self.world["public"]["initial"])
        if private:
            state["private_items"] = {u: deepcopy(state["items"]) for u in state["users"]}
        for index, original in enumerate(self.events):
            event = deepcopy(original)
            if reverse:
                event["candidates"].reverse()
            before = deepcopy(state)
            user = event["user"]
            state["users"][user] = f"{label} {user} memory"
            notes = state["private_items"][user] if private else state["items"]
            notes[event["positive"]]["memory"] = f"{label} {event['event_id']} item note"
            state["completed"].append(event["event_id"])
            record = {"event": event, "before": before, "after": deepcopy(state)}
            if actions:
                record["action"] = actions[index]
            save(path / (event["event_id"] + ".json"), record)
        save(path / "state.json", state)
        return state

    def test_paired_rows_reveal_cancelled_mean_and_keep_invalid_penalties(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, c = Path(tmp) / "B0", Path(tmp) / "candidate"
            self.arm(b, "original")
            self.arm(c, "changed")
            original = deepcopy(self.world)
            result = build_candidate_evidence(self.world, b, c, self.baseline_pair, self.candidate_pair, "uncertain_feedback")
            self.assertEqual(self.world, original)
            self.assertAlmostEqual(result["summary"]["quality_change_mean"], 0)
            self.assertEqual(result["summary"]["quality_decreased"], 2)
            self.assertEqual(result["summary"]["quality_increased"], 2)
            self.assertEqual(result["summary"]["bias_increased"], 2)
            self.assertEqual(result["summary"]["bias_decreased"], 2)
            self.assertLess(result["paired_queries"][0]["quality_change"], 0)
            self.assertGreater(result["paired_queries"][0]["bias_change"], 0)
            self.assertNotIn("PRIVATE_TRACE_MARKER", json.dumps(result))
            self.assertNotIn('"utilities"', json.dumps(result))
            changed = deepcopy(self.candidate_pair)
            changed[0]["qA"] = None
            invalid = build_candidate_evidence(self.world, b, c, self.baseline_pair, changed, "uncertain_feedback")
            self.assertEqual(invalid["summary"]["candidate_invalid"], 1)
            self.assertEqual(invalid["paired_queries"][0]["quality_after"], 0)
            self.assertGreater(invalid["paired_queries"][0]["bias_after"], result["paired_queries"][0]["bias_after"])

    def test_private_notebooks_and_source_paths_show_actual_shared_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, c = Path(tmp) / "B0", Path(tmp) / "candidate"
            self.arm(b, "original")
            self.arm(c, "private", private=True)
            result = build_candidate_evidence(self.world, b, c, self.baseline_pair, self.candidate_pair, "no_sharing")
            execution = result["execution"]
            self.assertTrue(execution["private_item_notebooks"])
            self.assertTrue(execution["candidate_shared_catalog_equals_initial"])
            self.assertFalse(execution["candidate_shared_catalog_equals_baseline"])
            self.assertEqual(execution["baseline_cross_user_item_reads"], [
                {"event_id": "e1", "user": "B", "item": "i009", "source_event_id": "e0", "source_user": "A"}])
            self.assertEqual(execution["candidate_cross_user_item_reads"], [])
            changed = next(r for r in result["memory_changes"] if r["kind"] == "item" and r["id"] == "i009" and r["owner"] == "B")
            self.assertEqual(changed["baseline_source_event_ids"], ["e0"])
            self.assertEqual(changed["candidate_source_event_ids"], [])
            self.assertEqual(changed["candidate_note"], "observed catalog note")

    def test_replay_layout_and_swapped_event_facts_use_root_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, c = Path(tmp) / "B0", Path(tmp) / "candidate"
            self.arm(b, "original")
            state = self.arm(c, "replayed", reverse=True, actions=("repair_root", "replay_descendant"))
            save(c / "ranking/state.json", state)
            save(c / "plan.json", {"roots": ["e0"], "affected_events": ["e0", "e1"]})
            result = build_candidate_evidence(self.world, b, c, self.baseline_pair, self.candidate_pair, "dependency_replay")
            self.assertEqual(result["execution"]["replay"]["affected_events"], ["e0", "e1"])
            self.assertEqual(result["execution"]["actions"], {"repair_root": 1, "replay_descendant": 1})
            self.assertEqual(len(result["execution"]["training_candidate_order_changes"]), 2)
            self.assertEqual(result["memory_change_summary"]["changed_user_notes"], 2)
            self.assertEqual(result["memory_change_summary"]["changed_item_notes"], 2)

    def test_rerank_without_event_records_has_no_invented_memory_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, c = Path(tmp) / "B0", Path(tmp) / "candidate"
            state = self.arm(b, "original")
            save(c / "state.json", state)
            result = build_candidate_evidence(self.world, b, c, self.baseline_pair, self.candidate_pair, "popularity_rerank")
            self.assertEqual(result["memory_changes"], [])
            self.assertFalse(result["execution"]["candidate_training_executed"])
            self.assertEqual(result["execution"]["candidate_event_records_available"], 0)
            self.assertEqual(result["execution"]["read_paths_source"], "inherited_baseline")
            self.assertEqual(result["execution"]["candidate_cross_user_item_reads"],
                             result["execution"]["baseline_cross_user_item_reads"])
            self.assertNotEqual(result["summary"]["quality_change_range"], [0, 0])

    def test_compact_keeps_adverse_case_and_all_counts_marks_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, c = Path(tmp) / "B0", Path(tmp) / "candidate"
            self.arm(b, "original")
            self.arm(c, "candidate" * 100)
            full = build_candidate_evidence(self.world, b, c, self.baseline_pair, self.candidate_pair, "uncertain_feedback")
            original = deepcopy(full)
            compact = compact_candidate_evidence(full, max_queries=2, max_notes=1, max_events=1, text_limit=20)
            self.assertEqual(full, original)
            self.assertEqual(compact["summary"], full["summary"])
            self.assertTrue(any(r["quality_change"] < 0 for r in compact["paired_queries"]))
            self.assertTrue(any(r["bias_change"] > 0 for r in compact["paired_queries"]))
            self.assertEqual(compact["selection"]["paired_queries_total"], 4)
            self.assertEqual(compact["selection"]["memory_changes_total"], 4)
            self.assertEqual(compact["selection"]["memory_changes_shown"], 1)
            self.assertGreater(compact["selection"]["text_fields_truncated"], 0)
            self.assertIn("截取", compact["memory_changes"][0]["candidate_note"])


if __name__ == "__main__":
    unittest.main()
