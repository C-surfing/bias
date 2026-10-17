"""External JSON proposal path: bounded user/LLM-provided decisions, CPU only."""
from pathlib import Path
import json
import tempfile
import unittest

import rsi_researcher.research.unified as u


class ExternalProposalTests(unittest.TestCase):
    def test_external_proposal_must_be_own_file(self):
        with self.assertRaises(ValueError):
            u.arm_choices(Path("/tmp/outside-chat-proposal.json"))

    def test_external_decisions_keep_fixed_control(self):
        with tempfile.TemporaryDirectory(dir=u.HERE / "runs") as parent:
            source = Path(parent) / "proposals.json"
            source.write_text(json.dumps({
                "arm_name": "external_valid",
                "decisions": [u.fixed_choice("estimated_propensity_clip_20"),
                              u.fixed_choice("estimated_propensity_clip_100")]},
                ensure_ascii=False), encoding="utf8")
            result = u.run(Path(parent) / "external-run", source)
            self.assertEqual(result["proposal_mode"], "provided_proposals_cpu_only")
            self.assertEqual(result["new_llm_requests"], 0)
            self.assertEqual(set(result["arms"]), {"fixed_grid", "external_valid"})
            self.assertEqual(result["arms"]["external_valid"]["selected_development_policy"],
                             "estimated_propensity_clip_100")

    def test_external_policy_injection_denied_by_gate(self):
        with tempfile.TemporaryDirectory(dir=u.HERE / "runs") as parent:
            source = Path(parent) / "proposals.json"
            bad = u.fixed_choice("estimated_propensity_clip_20")
            bad["shell"] = "unregistered_tool"
            source.write_text(json.dumps({
                "arm_name": "external_negative",
                "decisions": [bad]}), encoding="utf8")
            result = u.run(Path(parent) / "rejected-run", source)
            episode = result["arms"]["external_negative"]["episodes"][0]
            self.assertEqual(episode["status"], "REJECTED")
            self.assertEqual(result["arms"]["external_negative"]["selected_development_policy"],
                             "unweighted")
            self.assertFalse(result["independent_confirmation"])


if __name__ == "__main__":
    unittest.main()
