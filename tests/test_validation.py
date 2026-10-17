"""Independent validation freezes actions and does not rerun research personas."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_researcher.datasets.controlled import build_world
from rsi_researcher.paths import ROOT
from rsi_researcher.research.journal import ExperimentJournal
from rsi_researcher.research.validation import reconcile_validation, run_validation
from rsi_researcher.runtime.client import ModelFailure, atomic_json
from rsi_researcher.runtime.offline import OfflineResearchModel


def source_trial(root, operator="no_sharing", trial_id="0001"):
    """The source's state is deliberately unusable: validation must train anew."""
    batch = root / "discovery"
    trial = batch / "trials" / trial_id
    (trial / "candidate").mkdir(parents=True)
    world = build_world(seed=101, users=1, items=12, events_per_user=1)
    proposal = {"operator": operator, "root_event_ids": ["train-0000"] if operator == "dependency_replay" else [],
                "purpose": "mechanism_probe", "research_question": "共享是否解释偏差？",
                "prediction_if_hypothesis": "切断共享改变偏差。", "prediction_if_alternative": "倾斜仍保留。"}
    archive = {"trial_id": trial_id, "execution_status": "completed", "status": "probe_completed", "proposal": proposal,
               "execution": {"experiment_id": "source-" + trial_id},
               "research_update": {"next_question": "旧解释能否成立？"}}
    atomic_json(trial / "archive.json", archive)
    atomic_json(trial / "candidate/state.json", {"unusable_development_state": True})
    atomic_json(trial / "world.json", world)
    atomic_json(trial / "baseline_prompts.json", json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8")))
    atomic_json(trial / "strategy.json", {"world": {"seed": 101, "users": 1, "items": 12, "events_per_user": 1}, "rerank_strength": .1})
    atomic_json(trial / "protocol.json", {"source": "offline_test_fixture"})
    return batch, trial, archive


class ValidationTests(unittest.TestCase):
    def test_frozen_candidate_runs_new_baselines_and_feedback_returns_to_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch, source, archive = source_trial(root)
            journal = ExperimentJournal(root / "journal.json")
            model = OfflineResearchModel(budget=300)
            protocol = {"seeds": [202, 203], "strata": ["unrelated", "aligned"]}
            output = root / "validation"
            result = run_validation(output, model, batch, protocol, journal=journal)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["summary"]["planned_environments"], 4)
            self.assertEqual(len(result["summary"]["by_seed"]), 2)
            self.assertEqual(result["evidence_scope"], "development_validation_not_final_test")
            self.assertFalse(result["online_experiment"])
            self.assertEqual(result["promotion"], "none")
            self.assertTrue(all(context["stage"] not in ("diagnose", "design", "review", "learn") for _, context in model.calls))
            for row in result["environments"]:
                environment = output / "environments" / row["environment_id"]
                for arm in ("initial", "B0", "candidate"):
                    self.assertTrue((environment / arm / "state.json").exists())
                self.assertNotIn("unusable_development_state", (environment / "B0/state.json").read_text(encoding="utf-8"))
                self.assertTrue((environment / "candidate_evidence.json").exists())
                self.assertEqual(len(json.loads((environment / "B0/state.json").read_text(encoding="utf-8"))["completed"]), 1)
            frozen = json.loads((output / "frozen_candidate.json").read_text(encoding="utf-8"))
            self.assertEqual(frozen["proposal"], archive["proposal"])
            self.assertEqual(frozen["rerank_strength"], .1)
            histories = journal.research_history()
            self.assertEqual(histories[0]["validation_feedback"][0]["summary"]["planned_environments"], 4)
            self.assertNotIn("utilities", (root / "journal.json").read_text(encoding="utf-8"))

    def test_dependency_roots_are_not_silently_remapped_between_worlds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch, _, _ = source_trial(root, operator="dependency_replay")
            model = OfflineResearchModel()
            result = run_validation(root / "validation", model, batch, {"seeds": [202], "strata": ["aligned"]})
            self.assertEqual(result["status"], "needs_new_root_rule")
            self.assertEqual(model.requests, 0)
            self.assertEqual(result["environments"], [])

    def test_budget_failure_preserves_unrun_conditions_instead_of_averaging_survivors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch, _, _ = source_trial(root)
            result = run_validation(root / "validation", OfflineResearchModel(budget=2), batch,
                                    {"seeds": [202, 203], "strata": ["unrelated", "aligned"]})
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual([row["status"] for row in result["environments"]], ["budget_exhausted"] + ["not_run_budget"] * 3)
            self.assertEqual(result["summary"]["failed_or_unrun_environments"], 4)
            self.assertIsNone(result["summary"]["quality_change"])

    def test_one_failed_environment_is_retained_while_next_condition_executes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch, _, _ = source_trial(root)
            original_builder = build_world

            def fail_first(**arguments):
                if arguments["seed"] == 202:
                    raise RuntimeError("controlled test failure")
                return original_builder(**arguments)

            with patch("rsi_researcher.research.validation.build_world", side_effect=fail_first):
                result = run_validation(root / "validation", OfflineResearchModel(), batch,
                                        {"seeds": [202, 203], "strata": ["unrelated"]})
            self.assertEqual([row["status"] for row in result["environments"]], ["failed", "completed"])
            self.assertIsNone(result["summary"]["bias_change"])
            self.assertEqual(result["summary"]["by_seed"][0]["failed_conditions"], [result["environments"][0]["environment_id"]])

    def test_explicit_trial_selection_does_not_claim_best_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch, _, _ = source_trial(root, operator="no_sharing", trial_id="0001")
            source_trial(root, operator="swap_training_candidates", trial_id="0002")
            result = run_validation(root / "validation", OfflineResearchModel(), batch,
                                    {"seeds": [202], "strata": ["unrelated"]}, trial_id="0001")
            self.assertEqual(result["operator"], "no_sharing")
            self.assertIn("尚未宣称最优", result["selection_reason"])

    def test_development_seed_is_rejected_as_an_independent_validation_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch, _, _ = source_trial(root)
            model = OfflineResearchModel()
            with self.assertRaisesRegex(ValueError, "differ"):
                run_validation(root / "validation", model, batch, {"seeds": [101], "strata": ["aligned"]})
            self.assertEqual(model.requests, 0)

    def test_authentication_failure_stops_all_remaining_environments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch, _, _ = source_trial(root)
            model = OfflineResearchModel()
            with patch.object(model, "ask", side_effect=ModelFailure("controlled access failure", http_status=401)) as ask:
                result = run_validation(root / "validation", model, batch,
                                        {"seeds": [202, 203], "strata": ["unrelated", "aligned"]})
            self.assertEqual(ask.call_count, 1)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual([r["status"] for r in result["environments"]],
                             ["model_unavailable"] + ["not_run_model_unavailable"] * 3)
            self.assertEqual(result["environments"][0]["http_status"], 401)
            failure = json.loads((root / "validation/environments" / result["environments"][0]["environment_id"] / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["http_status"], 401)

    def test_parent_timeout_finishes_prewritten_conditions_and_returns_feedback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch, source, archive = source_trial(root)
            journal = ExperimentJournal(root / "journal.json")
            output = root / "validation"
            output.mkdir()
            candidate_id = journal.record_trial(source, archive)
            atomic_json(output / "result.json", {
                "status": "running", "candidate_id": candidate_id, "directory": str(output),
                "source_trial": str(source), "selection_reason": "离线测试中的冻结候选。",
                "evidence_scope": "development_validation_not_final_test", "promotion": "none",
                "environments": [
                    {"environment_id": "first", "seed": 202, "stratum": "unrelated", "status": "running"},
                    {"environment_id": "second", "seed": 203, "stratum": "aligned", "status": "pending"}]})
            result = reconcile_validation(output, {"status": "timed_out"}, journal)
            self.assertEqual(result["status"], "timed_out")
            self.assertEqual([row["status"] for row in result["environments"]], ["interrupted", "not_run"])
            self.assertEqual(result["summary"]["failed_or_unrun_environments"], 2)
            self.assertIsNone(result["summary"]["quality_change"])
            self.assertEqual(journal.research_history()[0]["validation_feedback"][0]["status"], "timed_out")
            self.assertIn("timed_out", (output / "report.md").read_text(encoding="utf-8"))
            # A second parent reconciliation must not append duplicate feedback.
            reconcile_validation(output, {"status": "timed_out"}, journal)
            self.assertEqual(len(journal.research_history()[0]["validation_feedback"]), 1)


if __name__ == "__main__":
    unittest.main()
