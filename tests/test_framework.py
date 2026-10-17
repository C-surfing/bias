from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from rsi_researcher.paths import ROOT
from rsi_researcher.runtime.client import BudgetExceeded
from rsi_researcher.datasets.controlled import build_world
from rsi_researcher.research.workflow import change_observation, inspect_existing, read, rerank, run_cycles, run_research, reconcile_execution
from rsi_researcher.runtime.offline import OfflineResearchModel
from rsi_researcher.research.operators import execution_key
from rsi_researcher.research.journal import ExperimentJournal
from rsi_researcher.research.validation import run_validation


class ResearchModel(OfflineResearchModel):
    """Test cases use the same explicitly offline adapter as the demo CLI."""


class ResearchFrameworkTests(unittest.TestCase):
    def setUp(self):
        self.world = build_world(71, users=2, items=20, events_per_user=2)
        self.prompts = read(ROOT / "data/author_prompts.json")
        self.config = read(ROOT / "configs/research_workflow.json")

    def test_research_chain_exchanges_hypotheses_and_keeps_outcomes_for_next_round(self):
        history = [{"status": "rejected", "lesson": "旧重排损害质量",
                    "research_update": {"next_question": "旧问题仍待区分。", "next_test": "旧档案只有文字，不猜测成执行指令。"}}]
        model = ResearchModel()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = deepcopy(self.world)
            archive = run_research(root, model, self.world, self.prompts, self.config, history)
            self.assertEqual(self.world, original)
            self.assertEqual(archive["selected"], "B0")
            self.assertIn(archive["status"], ("rejected", "inconclusive"))
            self.assertEqual([s["stage"] for s in read(root / "timeline.json")],
                             ["observe", "diagnose", "design", "review", "execute", "evaluate", "learn", "archive"])
            self.assertIn("alternative_explanation", archive["diagnosis"])
            self.assertIn("counterexample", archive["review"])
            self.assertEqual(read(root / "history_input.json"), history)
            self.assertIn("next_question", archive["research_update"])
            self.assertIsNone(read(root / "agenda.json")["suggested_experiment"])
            self.assertEqual(archive["proposal"]["operator"], "uncertain_feedback")
            self.assertEqual(read(root / "B0/train-0000.json")["before"], read(root / "candidate/train-0000.json")["before"])
            role_prompts = {c["stage"]: p for p, c in model.calls if c["stage"] in ("diagnose", "design", "review")}
            self.assertIn("旧重排损害质量", role_prompts["diagnose"])
            self.assertIn(archive["diagnosis"]["hypothesis"], role_prompts["design"])
            self.assertIn(archive["proposal"]["falsifier"], role_prompts["review"])
            for prompt, _ in model.calls:
                self.assertNotIn('"utilities"', prompt)
                self.assertNotIn('"positive_q"', prompt)
                self.assertNotIn('"sampling"', prompt)

    def test_reviewer_and_missing_exposure_data_change_the_research_plan(self):
        for operator, proceed, status in (("uncertain_feedback", False, "design_rejected"),
                                          ("exploration", True, "needs_new_exposure_data")):
            with self.subTest(operator=operator), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive = run_research(root, ResearchModel(operator, proceed), self.world, self.prompts, self.config)
                self.assertEqual(archive["status"], status)
                self.assertFalse((root / "candidate").exists())
                self.assertEqual(archive["selected"], "B0")

    def test_dependency_repair_is_executed_and_scored_in_the_same_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = run_research(root, ResearchModel("dependency_replay"), self.world, self.prompts, self.config)
            self.assertNotEqual(archive["status"], "failed")
            result = read(root / "candidate/result.json")
            self.assertEqual(result["status"], "completed")
            self.assertIn("train-0000", result["reexecuted"])
            self.assertEqual(read(root / "candidate/state.json"), read(root / "candidate/ranking/state.json"))
            self.assertEqual(read(root / "candidate/source_records.json")[0], read(root / "B0/train-0000.json"))
            self.assertEqual(archive["metrics"]["candidate"], read(root / "candidate/score.json"))

    def test_output_reranking_uses_visible_counts_and_keeps_membership(self):
        pair = ({"q": ["head", "middle", "tail"], "bad": None}, {"q": ["head", "middle", "tail"]})
        public = {"exposure": {"preperiod_counts": {"head": 100, "middle": 0, "tail": 0}}}
        original = deepcopy(pair)
        changed = rerank(pair, public, 1.5)
        self.assertEqual(changed[0]["q"], ["middle", "tail", "head"])
        self.assertIsNone(changed[0]["bad"])
        self.assertEqual(pair, original)
        self.assertEqual(rerank(pair, public, 0), pair)

    def test_existing_baseline_is_reused_without_training_or_initial_ranking_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "next"
            legacy_world = deepcopy(self.world)
            legacy_world["public"].pop("exposure")
            run_research(source, ResearchModel("retain"), legacy_world, self.prompts, self.config)
            before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
            model = ResearchModel("popularity_rerank", follow_agenda=False)
            archive = run_research(target, model, legacy_world, self.prompts, self.config,
                                   history=[read(source / "archive.json")], source_arm=source / "B0")
            self.assertEqual(model.requests, 4)
            self.assertEqual([c["stage"] for _, c in model.calls], ["diagnose", "design", "review", "learn"])
            self.assertEqual(archive["observation"], read(source / "archive.json")["observation"])
            self.assertIn(read(source / "archive.json")["research_update"]["next_question"], model.calls[0][0])
            self.assertEqual(archive["plan_handoff"]["planned_operator"], "no_sharing")
            self.assertFalse(archive["plan_handoff"]["action_followed"])
            self.assertTrue(archive["plan_handoff"]["reason"])
            self.assertEqual(before, {str(p): p.read_bytes() for p in source.rglob("*.json")})

    def test_two_rounds_reuse_b0_and_share_budget_while_reading_the_previous_lesson(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = ResearchModel()
            result = run_cycles(root, model, self.world, self.prompts, self.config)
            self.assertEqual(len(result["trials"]), 2)
            first, second = root / "trials/0001", root / "trials/0002"
            self.assertEqual(read(first / "B0/state.json"), read(second / "B0/state.json"))
            self.assertEqual(read(second / "history_input.json"), [read(first / "archive.json")])
            self.assertEqual(result["request_count"], model.requests)
            self.assertEqual([r["proposal"]["operator"] for r in result["trials"]], ["uncertain_feedback", "no_sharing"])
            self.assertEqual(read(second / "agenda.json")["suggested_experiment"], read(first / "archive.json")["research_update"]["next_experiment"])
            self.assertTrue(read(second / "plan_handoff.json")["action_followed"])
            self.assertEqual(read(second / "candidate/memory_control.json")["memory_policy"], "no_sharing")
            self.assertEqual(result["trials"][1]["status"], "probe_completed")
            self.assertEqual(result["research_update"]["next_experiment"]["operator"], "swap_training_candidates")
            self.assertIn("candidate_evidence", read(first / "archive.json"))
            role_calls = [(json.loads(p.split("\n", 1)[1]), c) for p, c in model.calls
                          if c["stage"] in ("learn", "diagnose")]
            self.assertTrue(role_calls[1][0]["candidate_evidence"])
            self.assertEqual(role_calls[2][0]["previous_research"][0]["candidate_evidence"],
                             read(first / "archive.json")["candidate_evidence"])

    def test_duplicate_physical_experiment_is_reused_without_becoming_a_repetition(self):
        class CountingModel(ResearchModel):
            def ask(self, prompt, schema, context):
                result = super().ask(prompt, schema, context)
                if context["stage"] == "learn" and not hasattr(self, "first_round_requests"):
                    self.first_round_requests = self.requests
                return result
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = CountingModel("uncertain_feedback", follow_agenda=False)
            result = run_cycles(root, model, self.world, self.prompts, self.config)
            first, second = result["trials"]
            self.assertFalse(first["execution"]["reused"])
            self.assertTrue(second["execution"]["reused"])
            self.assertFalse(second["execution"]["independent_repetition"])
            self.assertEqual(result["unique_candidate_experiments"], 1)
            self.assertEqual(first["metrics"], second["metrics"])
            self.assertEqual(model.requests - model.first_round_requests, 4)
            self.assertEqual(len(read(root / "experiment_index.json")), 1)
            self.assertEqual(result["execution_status"], "completed")
            self.assertEqual(result["research_status"], second["status"])

    def test_execution_identity_tracks_actual_intervention_not_question_wording(self):
        plan = {"operator": "dependency_replay", "root_event_ids": ["a", "b"], "purpose": "debias"}
        same = plan | {"root_event_ids": ["b", "a"], "purpose": "mechanism_probe", "research_question": "new question"}
        self.assertEqual(execution_key(plan, "baseline", .1), execution_key(same, "baseline", .9))
        self.assertNotEqual(execution_key(plan, "baseline", .1), execution_key(plan, "other baseline", .1))
        self.assertNotEqual(execution_key(plan, "baseline", .1), execution_key(plan | {"root_event_ids": ["a"]}, "baseline", .1))
        rerank_plan = plan | {"operator": "popularity_rerank", "root_event_ids": []}
        self.assertNotEqual(execution_key(rerank_plan, "baseline", .1), execution_key(rerank_plan, "baseline", .9))

    def test_unexpected_failure_preserves_last_scientific_result_but_fails_execution(self):
        class BrokenLearner(ResearchModel):
            def ask(self, prompt, schema, context):
                if context["stage"] == "learn":
                    raise RuntimeError("offline injected failure")
                return super().ask(prompt, schema, context)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                run_cycles(root, BrokenLearner("no_sharing", purpose="mechanism_probe"), self.world, self.prompts, self.config)
            archive = read(root / "archive.json")
            self.assertEqual(archive["execution_status"], "failed")
            self.assertEqual(archive["research_status"], "probe_completed")
            self.assertEqual(len(archive["trials"]), 1)
            self.assertTrue((root / "trials/0001/candidate_evidence.json").exists())

    def test_parent_timeout_updates_running_trial_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_cycles(root, ResearchModel(), self.world, self.prompts, self.config)
            trial = root / "trials/0002"
            record = read(trial / "archive.json")
            record["execution_status"] = "running"
            (trial / "archive.json").write_text(json.dumps(record), encoding="utf-8")
            archive = reconcile_execution(root, {"status": "timed_out"})
            self.assertEqual(archive["execution_status"], "timed_out")
            self.assertEqual(archive["status"], "failed")
            self.assertEqual(archive["trials"][-1]["execution_status"], "timed_out")
            self.assertIn("已到时间上限", (root / "report.md").read_text(encoding="utf-8"))

    def test_baseline_stage_timeout_and_missing_source_are_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "trials/0001"
            trial.mkdir(parents=True)
            (trial / "archive.json").write_text(json.dumps({"status": "running", "execution_status": "running",
                    "research_status": "not_evaluated", "mode_note": "offline timeout fixture", "trial_id": "0001"}), encoding="utf-8")
            (trial / "timeline.json").write_text("[]", encoding="utf-8")
            result = reconcile_execution(root, {"status": "timed_out"})
            self.assertEqual(result["trials"][0]["execution_status"], "timed_out")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = ResearchModel()
            with self.assertRaises(FileNotFoundError):
                run_cycles(root, model, self.world, self.prompts, self.config, source_arm=root / "missing/B0")
            self.assertEqual(read(root / "trials/0001/archive.json")["execution_status"], "failed")
            self.assertEqual(model.requests, 0)

    def test_independent_validation_feedback_reaches_next_diagnosis_and_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = ExperimentJournal(root / "journal.json")
            run_cycles(root / "discovery", ResearchModel(), self.world, self.prompts, self.config, journal=journal)
            result = run_validation(root / "validation", ResearchModel(), root / "discovery",
                        {"seeds": [991], "strata": ["unrelated"], "world": {"users": 2, "items": 20, "events_per_user": 1}}, journal=journal)
            self.assertEqual(result["status"], "completed")
            model = ResearchModel()
            run_cycles(root / "followup", model, self.world, self.prompts, self.config | {"candidate_limit": 1},
                       source_arm=root / "discovery/trials/0001/B0", journal=journal)
            payloads = {c["stage"]: json.loads(p.split("\n", 1)[1]) for p, c in model.calls if c["stage"] in ("diagnose", "learn")}
            for role in ("diagnose", "learn"):
                feedback = [r["validation_feedback"] for r in payloads[role]["previous_research"] if r.get("validation_feedback")]
                self.assertEqual(feedback[0][0]["summary"], result["summary"])
                self.assertNotIn('"utilities"', json.dumps(payloads[role]))

    def test_mechanism_probe_is_measured_without_applying_the_debias_promotion_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = run_research(root, ResearchModel("no_sharing", purpose="mechanism_probe"), self.world, self.prompts, self.config)
            self.assertEqual(archive["status"], "probe_completed")
            self.assertEqual(archive["selected"], "B0")
            self.assertNotIn("bias_margin", archive["decision"])
            self.assertEqual(read(root / "comparison.json")["purpose"], "mechanism_probe")
            self.assertEqual(read(root / "candidate/state.json")["items"], self.world["public"]["initial"]["items"])
            self.assertIn("private_items", read(root / "candidate/state.json"))

    def test_training_position_probe_preserves_events_feedback_and_final_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = run_research(root, ResearchModel("swap_training_candidates", purpose="mechanism_probe"), self.world, self.prompts, self.config)
            self.assertEqual(archive["status"], "probe_completed")
            baseline = read(root / "B0/state.json")
            candidate = read(root / "candidate/state.json")
            self.assertEqual(baseline["completed"], candidate["completed"])
            for eid in baseline["completed"]:
                a, b = read(root / "B0" / (eid + ".json"))["event"], read(root / "candidate" / (eid + ".json"))["event"]
                self.assertEqual(b, a | {"candidates": list(reversed(a["candidates"]))})
            before_pair, after_pair = read(root / "B0/rankings.json"), read(root / "candidate/rankings.json")
            for direction in before_pair:
                for q in before_pair[direction]:
                    self.assertEqual(set(before_pair[direction][q]), set(after_pair[direction][q]))

    def test_observation_detects_tradeoff_and_existing_result_import_is_read_only(self):
        self.assertTrue(change_observation({"utility": .5, "bias": .1}, {"utility": .6, "bias": .2})["quality_bias_tradeoff_observed"])
        self.assertFalse(change_observation({"utility": .5, "bias": .1}, {"utility": .6, "bias": .05})["quality_bias_tradeoff_observed"])
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            run_research(source, ResearchModel(), self.world, self.prompts, self.config)
            (source / "protocol.json").write_text(json.dumps({"purpose": "test fixture"}))
            (source / "decision.json").write_text(json.dumps({"status": "inconclusive"}))
            before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
            archive = inspect_existing(source, Path(tmp) / "inspection")
            self.assertEqual(archive["model_calls"], 0)
            self.assertEqual(before, {str(p): p.read_bytes() for p in source.rglob("*.json")})


if __name__ == "__main__":
    unittest.main()
