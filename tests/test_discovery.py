"""Only tests affecting claim validity and blind tool boundaries."""
import json
import tempfile
import unittest
from pathlib import Path

from rsi_researcher.research.discovery_lab import (
    make_private, query, evaluate, FIELDS, CONDITIONS,
)

class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.private = Path(self.tmp.name)/"private"/"hidden.json"
        self.public = make_private(73513, self.private)
        self.secret = json.loads(self.private.read_text())["cases"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_public_boundary(self):
        public_json = json.dumps(self.public)
        self.assertNotIn("amplitude", public_json)
        self.assertNotIn("direction", public_json)
        self.assertNotIn("single", public_json)
        self.assertNotIn("interaction", public_json)
        self.assertEqual(set(self.public["tasks"]), set(self.secret))
        self.assertTrue(all(task.startswith("task-") for task in self.public["tasks"]))
        for task in self.public["tasks"]:
            result = query(self.private, task, "observe")
            self.assertEqual(result["n_users"], 384)
            self.assertNotIn("mechanism", json.dumps(result))

    def test_real_paired_ranking_and_null(self):
        for task, cfg in self.secret.items():
            tag = cfg["tag"]
            factor = cfg["factor"] or "device"
            scope = cfg["segment"]
            result = query(self.private, task, "contrast",
                           factor=factor, condition=scope, tag=tag)
            again = query(self.private, task, "contrast",
                          factor=factor, condition=scope, tag=tag)
            self.assertEqual(result, again)
            self.assertEqual(result["n_randomization_units"],
                             384 if scope == "all" else 192)
            if cfg["type"] == "null":
                self.assertEqual(result["effect"], 0)
                self.assertEqual(result["top1_changed_fraction"], 0)
            else:
                self.assertGreater(abs(result["effect"]), 0.05,
                                   f"{cfg} {result}")
                self.assertGreater(result["top1_changed_fraction"], 0.02)
                self.assertEqual(result["effect"] > 0, cfg["direction"] > 0)
                if scope == "segment_1":
                    opposite = query(self.private, task, "contrast",
                                     factor=factor, condition="segment_0", tag=tag)
                    self.assertEqual(opposite["effect"], 0)

    def test_exact_evaluator(self):
        hypotheses = {
            case: {"factor": cfg["factor"], "condition": cfg["segment"],
                   "tag": cfg["tag"], "direction":
                   "increase" if cfg["direction"] > 0 else "decrease",
                   "status": "supported" if cfg["factor"] else "inconclusive"}
            for case, cfg in self.secret.items()
        }
        report = evaluate(self.private, hypotheses)
        self.assertEqual(report["valid_discoveries"], 3)
        self.assertEqual(report["false_discoveries"], 0)
        self.assertEqual(report["mechanism_recall"], 1)
        for claim in hypotheses.values():
            claim["status"] = "inconclusive"
        report = evaluate(self.private, hypotheses)
        self.assertEqual(report["valid_discoveries"], 0)

    def test_manipulation_check_does_not_confuse_null_with_failed_tool(self):
        task = next(task for task, cfg in self.secret.items() if cfg["type"] == "null")
        result = query(self.private, task, "contrast",
                       factor="slot", condition="all", tag=1)
        self.assertEqual(result["effect"], 0)
        self.assertEqual(result["top1_changed_fraction"], 0)
        self.assertEqual(
            result["manipulation_check"]["registered_setting_changed_fraction"], 1.0)
        self.assertTrue(result["manipulation_check"]["same_users_candidates_and_random_variates"])


    def test_public_observation_contrast_cards_are_not_oracle(self):
        for case in self.public["tasks"]:
            obs = query(self.private, case, "observe")
            cards = obs["contrast_cards"]
            self.assertEqual(len(cards), 8)
            self.assertEqual({c["field"] for c in cards}, set(FIELDS))
            self.assertEqual({c["user_segment"] for c in cards}, {0, 1})
            self.assertTrue(all(c["meaning"].startswith("observational")
                                for c in cards))
            self.assertNotIn("single", json.dumps(cards))
            self.assertNotIn("interaction", json.dumps(cards))
            self.assertNotIn("true_mechanism", json.dumps(cards))

    def test_independent_confirmation_checks_claim_scope(self):
        from rsi_researcher.research.discovery_cycle import confirm
        interaction_case, cfg = next(
            (task, row) for task, row in self.secret.items()
            if row["type"] == "interaction")
        claim = {
            "status": "supported", "factor": cfg["factor"], "tag": cfg["tag"],
            "direction": "increase" if cfg["direction"] > 0 else "decrease",
            "scope": "all"}
        # A significant *pooled* contrast is insufficient for claiming
        # invariance across both user segments.
        broad = confirm(self.private, interaction_case, claim)
        self.assertEqual(broad["status"], "inconclusive")
        self.assertEqual(set(broad["scope_subgroup_checks"]),
                         {"segment_0", "segment_1"})
        claim["scope"] = "segment_1"
        local = confirm(self.private, interaction_case, claim)
        self.assertEqual(local["status"], "confirmed")
        self.assertEqual(local["scope_negative_control"]["effect"], 0)


    def test_bounded_concept_revision_is_supported_only_by_real_contrasts(self):
        from unittest.mock import patch
        from rsi_researcher.research.discovery_cycle import concept_test, confirm, static_hyp
        task, cfg = next((task, row) for task, row in self.secret.items()
                         if row["type"] == "single")
        initial = static_hyp(cfg["factor"], "segment_0", cfg["tag"],
                             "increase" if cfg["direction"] < 0 else "decrease",
                             "concept_test")
        # The first model hypothesis is deliberately wrong about direction
        # and subgroup scope. The revised claim may only use measured evidence.
        def standin(model, role, instruction, context, schema, episode):
            if role == "discovery-concept-explorer":
                return {
                    "factor":cfg["factor"],
                    "observational_signal":"Public paired groups suggest a condition effect.",
                    "why_informative":"Potentially measurable exposure contrast.",
                    "competing_explanation":"Differences could be due to random preference.",
                    "falsification_plan":"Compare fixed users under controlled levels."}
            if role == "discovery-concept":
                return initial
            if role == "discovery-claim-revision":
                corrected = dict(initial)
                corrected["condition"] = "all"
                corrected["direction"] = ("increase" if cfg["direction"] > 0
                                          else "decrease")
                return corrected
            raise AssertionError(role)
        with patch("rsi_researcher.research.discovery_cycle.ask", standin):
            claim, experiments, trajectory = concept_test(
                None, self.private, task, query(self.private, task, "observe"), 0)
        self.assertEqual(claim["status"], "supported")
        self.assertTrue(claim["changed_after_experiment"])
        self.assertEqual(len(experiments), 2)
        self.assertEqual(confirm(self.private, task, claim)["status"], "confirmed")
        self.assertEqual([e["id"] for e in experiments], ["e00", "e01"])


    def test_discovered_mechanism_has_separate_utility_reference(self):
        task, cfg = next((t,v) for t,v in self.secret.items() if v["type"] == "single")
        result = query(self.private, task, "contrast", split="confirmation",
                       factor=cfg["factor"], condition="all", tag=cfg["tag"])
        utility = result["reference_utility"]
        self.assertIn("treatment_minus_control", utility)
        self.assertLessEqual(
            abs(utility["treatment"] - utility["control"]
                - utility["treatment_minus_control"]),0.00011)
        self.assertEqual(len(utility["paired_user_ci95"]),2)
        self.assertIn("utility_harm_detected",utility)
        null = next(t for t,v in self.secret.items() if v["type"]=="null")
        null_result=query(self.private,null,"contrast",factor="slot")
        self.assertEqual(null_result["reference_utility"]["treatment_minus_control"],0)

    def test_real_data_user_temporal_holdout_contract(self):
        from rsi_researcher.research.kuairand_discovery import _split, _duration
        for user in ("0","1","42","19000","27284"):
            early=_split(user,20220428,20261023)
            late=_split(user,20220502,20261023)
            self.assertNotEqual(early,late)
            self.assertEqual(set([early,late]) & {"development","confirmation"},
                             {early or late})
        self.assertEqual(_duration("14000"),"short_lt15s")
        self.assertEqual(_duration("45000"),"medium_15to60s")
        self.assertEqual(_duration("90000"),"long_ge60s")

if __name__ == "__main__":
    unittest.main(verbosity=2)
