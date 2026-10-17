"""Regression tests for the chat-owned CPU-only unified research runner."""
from pathlib import Path
import json
import math
import tempfile
import unittest

import rsi_researcher.research.unified as u


class UnifiedResearchTests(unittest.TestCase):
    def test_immutable_sources_and_registered_task(self):
        hashes = u.frozen_sources()
        self.assertEqual(len(hashes), 7)
        self.assertEqual(u.situation().stage, "development")

    def test_replay_is_exact_archived_qwen(self):
        arms = u.replay_arms()
        self.assertEqual(len(arms["qwen_no_history_replay"]), 3)
        self.assertEqual(len(arms["qwen_with_history_replay"]), 3)
        self.assertEqual(
            [x["policy"] for x in arms["qwen_with_history_replay"]],
            ["estimated_propensity_clip_5", "estimated_propensity_clip_20",
             "estimated_propensity_clip_100"])

    def test_gate_accepts_registered_cpu_candidate_only(self):
        verdict = u.gate(u.situation(), u.fixed_choice("estimated_propensity_clip_20"))
        self.assertEqual(verdict["status"], "DEVELOPMENT_CPU_ONLY")
        self.assertFalse(verdict["execution_authorized"])

    def test_gate_denies_unregistered_policy(self):
        candidate = u.fixed_choice("estimated_propensity_clip_20")
        candidate["policy"] = "arbitrary_python_or_gpu_action"
        self.assertEqual(u.gate(u.situation(), candidate)["status"], "REJECTED")

    def test_gate_denies_tool_injection_and_no_falsifier(self):
        c = u.fixed_choice("estimated_propensity_clip_5")
        c["shell"] = "rm -rf /"
        self.assertEqual(u.gate(u.situation(), c)["status"], "REJECTED")
        c.pop("shell")
        c["falsifier"] = ""
        self.assertEqual(u.gate(u.situation(), c)["status"], "REJECTED")

    def test_evidence_tamper_detection(self):
        ctx = u.situation()
        from dataclasses import replace
        ctx.evidence["TASK_REGISTRY"] = replace(
            ctx.evidence["TASK_REGISTRY"], sha256="0" * 64)
        with self.assertRaises(ValueError):
            u.gate(ctx, u.fixed_choice("estimated_propensity_clip_5"))

    def test_only_chat_owned_output_path_allowed(self):
        with self.assertRaises(ValueError):
            u.run(Path("/data/cjh/rsi/experiments/unified-research-20261017-chatgpt/runs/forbidden-output"))

    def test_complete_cpu_episode_reproduces_saved_development(self):
        with tempfile.TemporaryDirectory(dir=u.HERE / "runs") as parent:
            result = u.run(Path(parent) / "replay_case")
            self.assertEqual(result["status"], "CPU_DEVELOPMENT_REPLAY_COMPLETE")
            self.assertEqual(result["new_llm_requests"], 0)
            self.assertEqual(result["gpu_runs"], 0)
            self.assertFalse(result["holdout_metrics_calculated"])
            self.assertEqual(result["holdout_rows_withheld"], 2209)
            selected = {k: v["selected_development_policy"]
                        for k, v in result["arms"].items()}
            self.assertEqual(selected, {
                "fixed_grid": "estimated_propensity_clip_100",
                "qwen_no_history_replay": "estimated_propensity_clip_5",
                "qwen_with_history_replay": "estimated_propensity_clip_100",
            })
            reference = json.loads(u.DEV_REFERENCE.read_text())
            for arm in result["arms"].values():
                for episode in arm["episodes"]:
                    self.assertEqual(episode["gate"]["status"], "DEVELOPMENT_CPU_ONLY")
                    if episode["status"] != "EVALUATED":
                        continue
                    key = episode["policy"]
                    score = reference["policies"][key]["development_metrics"]
                    self.assertTrue(math.isclose(
                        episode["evaluation"]["Brier"],
                        score["Brier_random_development"], abs_tol=1e-12))
            saved = Path(parent) / "replay_case"
            self.assertTrue((saved / "research_episodes.json").is_file())
            memory = json.loads((saved / "research_memory_cards.json").read_text())
            self.assertEqual(len(memory), 3)
            self.assertTrue(all(x["evidence_grade"] == "development_only"
                                and not x["confirmed"] for x in memory))


if __name__ == "__main__":
    unittest.main()
