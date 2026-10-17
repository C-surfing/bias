"""Shared research history must retain validation feedback and move with artifacts."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from rsi_researcher.research.journal import ExperimentJournal


class JournalTests(unittest.TestCase):
    def test_validation_feedback_is_visible_and_recording_a_trial_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = ExperimentJournal(root / "journal.json")
            archive = {"trial_id": "0001", "execution_status": "completed", "status": "probe_completed",
                       "proposal": {"operator": "no_sharing"}, "execution": {"experiment_id": "same-action"},
                       "research_update": {"next_question": "共享是否解释偏差？"}}
            candidate_id = journal.record_trial(root / "trials/0001", archive)
            journal.record_validation(candidate_id, {"status": "completed", "directory": str(root / "validation"),
                                     "environments": [{"seed": 202, "comparison": {"bias_change": .12}}],
                                     "evidence_scope": "development_validation_not_final_test"})
            journal.record_trial(root / "trials/0001", archive)
            history = journal.research_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["research_update"], archive["research_update"])
            self.assertEqual(history[0]["validation_feedback"][0]["environments"][0]["comparison"]["bias_change"], .12)
            self.assertEqual(history[0]["validation_feedback"][0]["directory"], "validation")

    def test_artifact_paths_remain_relative_after_directory_move(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "original"
            trial = root / "trials/0001"
            trial.mkdir(parents=True)
            journal = ExperimentJournal(root / "journal.json")
            archive = {"trial_id": "0001", "status": "failed", "baseline_source": str(trial / "B0"),
                       "world": {"private": {"utilities": {"hidden": 10}}}, "private": {"hidden": True}}
            journal.record_trial(trial, archive)
            moved = Path(temporary) / "moved"
            shutil.copytree(root, moved)
            history = ExperimentJournal(moved / "journal.json").research_history()
            self.assertTrue((moved / history[0]["trial_path"]).exists())
            self.assertEqual(history[0]["baseline_source"], "trials/0001/B0")
            self.assertNotIn("utilities", (moved / "journal.json").read_text(encoding="utf-8"))
            self.assertEqual(history[0]["status"], "failed")

    def test_reused_execution_shares_validation_but_is_not_a_new_repetition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = ExperimentJournal(root / "journal.json")
            first = {"trial_id": "0001", "execution": {"experiment_id": "fixed-result", "reused": False}}
            second = {"trial_id": "0002", "execution": {"experiment_id": "fixed-result", "reused": True}}
            candidate_id = journal.record_trial(root / "trials/0001", first)
            self.assertEqual(candidate_id, journal.record_trial(root / "trials/0002", second))
            journal.record_validation(candidate_id, {"status": "incomplete", "reason": "反证仍待解释"})
            histories = journal.research_history()
            self.assertEqual(len(histories), 2)
            self.assertTrue(histories[1]["execution"]["reused"])
            self.assertEqual(histories[0]["validation_feedback"], histories[1]["validation_feedback"])

    def test_unregistered_candidate_cannot_receive_orphan_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = ExperimentJournal(Path(temporary) / "journal.json")
            with self.assertRaisesRegex(ValueError, "absent"):
                journal.record_validation("missing", {"status": "completed"})
            self.assertEqual(journal.entries(), [])


if __name__ == "__main__":
    unittest.main()
