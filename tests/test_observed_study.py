"""Research may learn from development evidence, never held-out labels."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from jsonschema import ValidationError, validate

from rsi_researcher.research import benchmark


class StructuredResearchModel:
    """An explicit test double for the same checked plan/learn interface."""

    def __init__(self, policies=None):
        self.policies = policies or [("popularity_penalty", .4)] * 3
        self.calls = []
        self.scope = {}
        self.plan_count = 0
        self.learn_count = 0

    def ask(self, prompt, schema, context):
        payload = json.loads(prompt.split("\n", 1)[1])
        self.calls.append({"stage": context["stage"], "prompt": prompt,
                           "payload": deepcopy(payload), "schema": deepcopy(schema)})
        if context["stage"] == "observed_plan":
            operator, strength = self.policies[self.plan_count]
            self.plan_count += 1
            answer = {"hypothesis": f"训练观察频次可能产生热门倾斜，问题 {self.plan_count}",
                      "alternative_explanation": "热门物品可能确实更符合偏好",
                      "operator": operator, "strength": strength,
                      "prediction": "降低额外热门倾斜且维持质量",
                      "falsifier": "质量损失超过门槛会削弱解释",
                      "reason": "用同一开发面板做最小对照"}
        elif context["stage"] == "observed_learn":
            self.learn_count += 1
            answer = {"lesson": f"第 {self.learn_count} 轮需保留受损用户证据",
                      "next_question": f"下一轮问题 {self.learn_count}",
                      "next_operator": "user_calibrated",
                      "next_strength": payload["trial"]["policy"]["strength"],
                      "evidence_state": "inconclusive"}
        else:
            raise AssertionError(f"unexpected model stage: {context['stage']}")
        validate(answer, schema)
        return answer


def coat_fixture(*, heterogeneous_development=False, favorable_test=False):
    """Small sparse Coat-shaped observations: integers 1–5, absent = unknown."""
    items = list("abcdefghij")
    train = []
    # Single observed item per training user: no evaluation labels or item
    # co-occurrences can accidentally create the fallback score differences.
    for item_index, item in enumerate(items):
        count = 20 if item == "a" else 15 if item == "b" else 1
        for user_index in range(count):
            train.append({"user": f"train-{item_index}-{user_index}",
                          "item": item, "rating": 5 if item in "ab" else 4})
    patterns = [
        [5, 5, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 5, 5, 5, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 5, 5, 5],
        [1, 1, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 5, 5, 1, 1, 1, 1],
        [1, 1, 3, 3, 3, 3, 3, 3, 3, 3],
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    ]
    development = [
        {"user": f"development-{index}", "item": item,
         "rating": patterns[index][item_index] if heterogeneous_development
                   else 1 if item in "ab" else 5}
        for index in range(8) for item_index, item in enumerate(items)
    ]
    test = [
        {"user": f"heldout-private-{index}", "item": item,
         "rating": (1 if item in "ab" else 5) if favorable_test
                   else (5 if item in "ab" else 1)}
        for index in range(3) for item in items
    ]
    return {"dataset_id": "coat_checked_fixture", "fingerprint": "fixture-observations-v1",
            "item_ids": items, "user_ids": sorted({row["user"] for row in train + development + test}),
            "train": train, "development": development, "test": test,
            "positive_threshold": 4, "gain_floor": 1}


class ObservedStudyTests(unittest.TestCase):
    def setUp(self):
        # Some Windows environments deny access to the system temp directory.
        parent = Path(__file__).resolve().parent.parent / "outputs" / "test-temp"
        parent.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_study(self, name, model, dataset=None, **kwargs):
        return benchmark.run_study(self.root / name, model, dataset or coat_fixture(), **kwargs)

    def test_heldout_labels_are_first_read_after_every_research_round(self):
        model = StructuredResearchModel()
        dataset = coat_fixture()
        label_reads = []

        class HeldOutRow(dict):
            def __getitem__(row, key):
                if key == "rating":
                    self.assertEqual((model.plan_count, model.learn_count), (3, 3))
                    label_reads.append(row["user"])
                return super().__getitem__(key)

        dataset["test"] = [HeldOutRow(row) for row in dataset["test"]]
        result = self.run_study("delayed-test", model, dataset, rounds=3)
        self.assertTrue(label_reads)
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["validation"]["scope"], "held_out_observed_random_panels")
        for call in model.calls:
            self.assertNotIn("heldout-private-", call["prompt"])
            self.assertNotIn('"test"', call["prompt"])
            self.assertNotIn('"rating"', call["prompt"])

    def test_final_test_scores_only_baseline_and_frozen_candidate(self):
        model = StructuredResearchModel()
        dataset = coat_fixture()
        calls = []
        real_score = benchmark._score

        def capture(dataset_arg, recommender, rows, policy):
            if rows is dataset["test"]:
                frozen = json.loads((self.root / "frozen" / "frozen_candidate.json").read_text(encoding="utf-8"))
                self.assertEqual((model.plan_count, model.learn_count), (3, 3))
                self.assertEqual(frozen["selection_split"], "development")
                self.assertFalse(frozen["candidate_redesign"])
                calls.append(deepcopy(policy))
            return real_score(dataset_arg, recommender, rows, policy)

        with patch.object(benchmark, "_score", side_effect=capture):
            result = self.run_study("frozen", model, dataset, rounds=3)
        self.assertEqual(calls, [None, result["selected_policy"]])
        self.assertFalse(result["validation"]["test_candidate_selection"])
        self.assertFalse(result["validation"]["online_experiment"])
        self.assertFalse(result["validation"]["statistical_promotion"])

    def test_heldout_quality_loss_cannot_reselect_the_development_winner(self):
        policies = [("popularity_penalty", .05), ("popularity_penalty", .4)]
        favorable = self.run_study("favorable", StructuredResearchModel(policies),
                                   coat_fixture(favorable_test=True), rounds=2)
        adverse = self.run_study("adverse", StructuredResearchModel(policies), coat_fixture(), rounds=2)
        selected = {"operator": "popularity_penalty", "strength": .4}
        self.assertEqual(favorable["selected_policy"], selected)
        self.assertEqual(adverse["selected_policy"], selected)
        self.assertEqual(adverse["trials"][1]["comparison"]["status"], "point_gates_passed")
        self.assertLess(adverse["validation"]["metrics"]["candidate"]["ndcg"],
                        adverse["validation"]["metrics"]["B0"]["ndcg"])
        self.assertFalse(adverse["validation"]["comparison"]["point_gates"]["quality"])

    def test_history_ablation_hides_prior_and_new_lessons_from_every_round(self):
        prior = [{"research_update": {"lesson": "PRIVATE PRIOR LESSON"}}]
        model = StructuredResearchModel()
        result = self.run_study("no-history", model, rounds=3,
                                history_enabled=False, prior_history=prior, compute_test=False)
        self.assertEqual(len(result["trials"]), 3)
        self.assertTrue(all(call["payload"]["history"] == [] for call in model.calls))
        self.assertTrue(all("PRIVATE PRIOR LESSON" not in call["prompt"] for call in model.calls))
        self.assertEqual(result["trials"][0]["research_update"]["lesson"], "第 1 轮需保留受损用户证据")

    def test_prior_evidence_and_learned_counterevidence_reach_the_next_round(self):
        prior = [{"research_update": {"lesson": "上一个任务未解决的代价"}}]
        model = StructuredResearchModel()
        result = self.run_study("with-history", model, rounds=3,
                                prior_history=prior, compute_test=False)
        plans = [call for call in model.calls if call["stage"] == "observed_plan"]
        self.assertEqual(plans[0]["payload"]["history"], prior)
        self.assertEqual(len(plans[1]["payload"]["history"]), 2)
        learned = plans[1]["payload"]["history"][-1]
        self.assertEqual(learned["research_update"], result["trials"][0]["research_update"])
        self.assertIn("comparison", learned)
        self.assertIn("examples", learned)
        self.assertEqual(len(plans[2]["payload"]["history"]), 3)

    def test_same_action_is_reused_without_claiming_independent_repetition(self):
        model = StructuredResearchModel()
        dataset = coat_fixture()
        executions = []
        real_score = benchmark._score

        def capture(dataset_arg, recommender, rows, policy):
            if rows is dataset["development"] and policy == {"operator": "popularity_penalty", "strength": .4}:
                executions.append(policy.copy())
            return real_score(dataset_arg, recommender, rows, policy)

        with patch.object(benchmark, "_score", side_effect=capture):
            result = self.run_study("reuse", model, dataset, rounds=3, compute_test=False)
        records = result["trials"]
        self.assertEqual([trial["execution"]["reused"] for trial in records], [False, True, True])
        self.assertEqual(len({trial["execution"]["experiment_id"] for trial in records}), 1)
        self.assertTrue(all(not trial["execution"]["independent_repetition"] for trial in records))
        self.assertNotEqual(records[0]["proposal"]["hypothesis"], records[1]["proposal"]["hypothesis"])
        # One candidate execution plus the frozen development summary; not
        # one new physical experiment for each freshly worded hypothesis.
        self.assertEqual(len(executions), 2)
        for trial in records:
            self.assertEqual(trial["metrics"], records[0]["metrics"])

    def test_strategy_strength_grid_restricts_and_drives_real_ranking(self):
        strategy = deepcopy(benchmark.DEFAULT_STRATEGY)
        strategy.update(version="strategy-test", strength_grid=[0, .3], evidence_focus="tail_relevance")
        model = StructuredResearchModel([("user_calibrated", .3)])
        result = self.run_study("new-strength", model, rounds=1, strategy=strategy, compute_test=False)
        self.assertEqual(result["trials"][0]["policy"], {"operator": "user_calibrated", "strength": .3})
        self.assertEqual(model.calls[0]["schema"]["properties"]["strength"]["enum"], [0, .3])
        self.assertEqual(model.calls[1]["schema"]["properties"]["next_strength"]["enum"], [0, .3])
        self.assertEqual(model.calls[0]["payload"]["strategy"], strategy)
        self.assertEqual(model.calls[1]["payload"]["strategy"], strategy)
        self.assertIn("next_strength只能从strategy.strength_grid选择", model.calls[1]["prompt"])
        forbidden = StructuredResearchModel([("popularity_penalty", .4)])
        with self.assertRaises(ValidationError):
            self.run_study("forbidden-strength", forbidden, rounds=1, strategy=strategy, compute_test=False)
        failed = json.loads((self.root / "forbidden-strength" / "archive.json").read_text(encoding="utf-8"))
        self.assertEqual(failed["execution_status"], "failed")
        self.assertEqual(failed["trials"], [])
        self.assertFalse((self.root / "forbidden-strength" / "validation.json").exists())

    def test_strategy_evidence_focus_changes_examples_actually_sent_to_researcher(self):
        dataset = coat_fixture(heterogeneous_development=True)
        examples = {}
        for focus in ("overall", "tradeoff_users", "tail_relevance"):
            strategy = deepcopy(benchmark.DEFAULT_STRATEGY)
            strategy["evidence_focus"] = focus
            model = StructuredResearchModel([("retain", 0)])
            result = self.run_study(focus, model, dataset, rounds=1, strategy=strategy, compute_test=False)
            initial = model.calls[0]["payload"]["evidence"]["initial_examples"]
            learned = model.calls[1]["payload"]["trial"]["examples"]
            self.assertEqual(learned, result["trials"][0]["examples"])
            self.assertEqual(initial, [{k: v for k, v in row.items() if k not in ("quality_change", "bias_change")}
                                      for row in learned])
            self.assertLessEqual(len(initial), 6)
            examples[focus] = [row["user"] for row in initial]
        self.assertNotEqual(examples["overall"], examples["tail_relevance"])
        self.assertNotEqual(examples["overall"], examples["tradeoff_users"])

    def test_initial_evidence_describes_baseline_without_fabricating_previous_treatment(self):
        model = StructuredResearchModel([("popularity_penalty", .05)])
        self.run_study("baseline-only-evidence", model, rounds=1, compute_test=False)
        evidence = model.calls[0]["payload"]["evidence"]
        self.assertNotIn("initial_change_statistics", evidence)
        self.assertIn("No current-task intervention has run yet", evidence["initial_state"])
        self.assertIn("excluded from quality_users", evidence["population_definitions"]["zero_ideal_users"])
        for row in evidence["initial_examples"]:
            self.assertNotIn("quality_change", row)
            self.assertNotIn("bias_change", row)
        self.assertIn("change_statistics", model.calls[1]["payload"]["trial"])

    def test_source_experience_keeps_aggregates_without_transferring_user_cases(self):
        dataset = coat_fixture()
        dataset["dataset_id"] = "coat_source_task"
        marker = "source-user-should-not-be-a-transfer-case"
        for row in dataset["development"]:
            row["user"] = marker + "-" + row["user"]
        source = self.run_study("source-task", StructuredResearchModel(), dataset, rounds=2)
        # A model may mention an inspected user in its narrative. Such text is
        # not a portable lesson about users of the next dataset.
        for trial in source["trials"]:
            trial["proposal"]["reason"] = f"检查 {marker} 后提出的原因"
            trial["research_update"]["lesson"] = f"{marker} 的用户案例"
            trial["per_user"] = [{"user": marker, "ndcg": .7}]
        source["per_user"] = [{"user": marker, "ndcg": .7}]
        before = deepcopy(source)
        experience = benchmark.source_experience(source)
        serialized = json.dumps(experience, ensure_ascii=False)
        self.assertEqual(experience["source_dataset"], "coat_source_task")
        self.assertEqual(len(experience["policy_trace"]), 2)
        self.assertEqual(experience["validation"]["metrics"], source["validation"]["metrics"])
        self.assertIn(json.dumps(source["trials"][0]["policy"], ensure_ascii=False), serialized)
        self.assertNotIn(marker, serialized)
        for key in ('"examples"', '"per_user"', '"proposal"', '"reason"', '"lesson"'):
            self.assertNotIn(key, serialized)
        self.assertEqual(source, before)

    def test_transfer_context_distinguishes_current_cases_from_source_aggregates(self):
        source_dataset = coat_fixture()
        source_dataset["dataset_id"] = "coat_source_task"
        marker = "source-private-person"
        for row in source_dataset["development"]:
            row["user"] = marker + "-" + row["user"]
        source = self.run_study("source-for-transfer", StructuredResearchModel(), source_dataset, rounds=1)
        experience = benchmark.source_experience(source)
        current = coat_fixture()
        current["dataset_id"] = "kuairand_current_task"
        model = StructuredResearchModel()
        self.run_study("transfer-context", model, current, rounds=2,
                       prior_history=[experience], compute_test=False)
        for call in model.calls:
            self.assertEqual(call["payload"]["current_dataset"], "kuairand_current_task")
            self.assertNotIn(marker, call["prompt"])
            prior = call["payload"]["history"][0]
            self.assertEqual(prior["source_dataset"], "coat_source_task")
            self.assertEqual(prior["validation"]["metrics"], source["validation"]["metrics"])
            examples = (call["payload"]["evidence"]["initial_examples"]
                        if call["stage"] == "observed_plan" else call["payload"]["trial"]["examples"])
            self.assertTrue(examples)
            self.assertTrue(all(row["dataset"] == "kuairand_current_task" for row in examples))
        second_plan = [call for call in model.calls if call["stage"] == "observed_plan"][1]
        current_history = second_plan["payload"]["history"][1]
        self.assertTrue(all(row["dataset"] == "kuairand_current_task" for row in current_history["examples"]))

    def test_research_without_final_validation_does_not_read_test_labels(self):
        class UnreadableTestRow(dict):
            def __getitem__(row, key):
                if key == "rating":
                    raise AssertionError("held-out labels were read during development")
                return super().__getitem__(key)

        dataset = coat_fixture()
        dataset["test"] = [UnreadableTestRow(row) for row in dataset["test"]]
        result = self.run_study("development-only", StructuredResearchModel(), dataset,
                                rounds=3, compute_test=False)
        self.assertEqual(result["research_status"], "candidate_frozen")
        self.assertNotIn("validation", result)
        self.assertFalse((self.root / "development-only" / "validation.json").exists())


if __name__ == "__main__":
    unittest.main()
