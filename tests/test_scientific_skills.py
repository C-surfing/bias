"""No unverified reflection may become a promoted research skill."""
from hashlib import sha256
from pathlib import Path
import json
import tempfile
import unittest

from rsi_researcher.research import scientific_skills as skills
from rsi_researcher.research.scientific_loop import campaign, public_evidence
from tests.test_scientific_loop import FakeModel


class ScientificSkillTests(unittest.TestCase):
    def test_candidate_is_frozen_and_never_auto_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            ep=root/"episode-001"
            ep.mkdir()
            audit=ep/"audit.json"
            audit.write_text(json.dumps({"task":"agentcf_memory",
                "independently_confirmed_discoveries":0,
                "reason":"only new development state"}))
            card=skills.proposal(ep, "agentcf_memory",
                                 "Require paired no-change rankings before claiming mechanism.", audit)
            self.assertFalse(card["validated_skill"])
            self.assertEqual(card["status"],"unverified_candidate")
            with self.assertRaises(FileExistsError):
                skills.proposal(ep, "agentcf_memory", "Retry same candidate skill",audit)
            audit.write_text("{}")
            library=root/"skills.json"
            library.write_text(json.dumps([{**card,"status":"promoted",
                "independent_evaluation_file":str(audit),
                "evaluation_sha256":card["source_audit_sha256"],
                "external_approval_reference":"invalid"}]))
            with self.assertRaises(ValueError):
                skills.select_promoted(library,"agentcf_memory")

    def matched_trials(self, root, eligible=True):
        rows=[]
        for i in range(8):
            task=f"heldout-new-mechanism-{i}"
            cohort=sha256(f"cohort-{i}".encode()).hexdigest()
            trial={"task_id":task,"cohort_sha256":cohort,
                   "budget":{"max_requests":300,"max_tokens":50000},"arms":{}}
            for arm in skills.ARMS:
                audit={
                    "research_task":task,"cohort_sha256":cohort,
                    "evaluation_split":"independent",
                    "pre_registered_confirmation":True,"arm":arm,
                    "independently_confirmed_discoveries":int(arm=="evolved_skill") if eligible else 0,
                    "model_requests":35,"total_tokens":5000}
                filename=root/f"{i}_{arm}.json"
                filename.write_text(json.dumps(audit))
                trial["arms"][arm]={"audit_file":str(filename),
                    "audit_sha256":skills.fingerprint(filename)}
            rows.append(trial)
        return rows

    def test_heldout_skill_requires_real_comparison_arms_and_explicit_protocol(self):
        candidate={"status":"unverified_candidate",
                   "validated_skill":False,
                   "scope_task":"agentcf_memory"}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            good=self.matched_trials(root)
            self.assertTrue(skills.evaluate_candidate(candidate,good)["eligible"])
            self.assertFalse(skills.evaluate_candidate(candidate,good[:4])["eligible"])
            bad=self.matched_trials(root,eligible=False)
            self.assertFalse(skills.evaluate_candidate(candidate,bad)["eligible"])
            bad[0]["arms"]["strong_rule"]["audit_sha256"]="0"*64
            with self.assertRaises(ValueError):
                skills.evaluate_candidate(candidate,bad)

    def test_shuffled_or_heldout_history_cannot_impersonate_verified_skill(self):
        cards=[
            {"task":"coat_propensity","action":"gamma_stress","evidence_grade":"development_only",
             "panel_measurements":[],"source_sha256":"x","limitation":"development"},
            {"task":"agentcf_memory","action":"item_memory_ablation","evidence_grade":"development_only",
             "panel_measurements":[],"source_sha256":"y","limitation":"development"},
        ]
        fresh=public_evidence("agentcf_popularity",cards,arm="qwen_audited_history")
        none=public_evidence("agentcf_popularity",cards,arm="qwen_no_history")
        shuffle=public_evidence("agentcf_popularity",cards,arm="qwen_shuffled_history")
        self.assertEqual([x["task"] for x in fresh["relevant_prior_numeric_evidence"]],
                         ["agentcf_memory"])
        self.assertEqual(none["relevant_prior_numeric_evidence"],[])
        self.assertEqual([x["task"] for x in shuffle["relevant_prior_numeric_evidence"]],
                         ["coat_propensity"])
        with self.assertRaises(ValueError):
            public_evidence("agentcf_popularity",cards,arm="qwen_promoted_skill")

    def test_fixed_rule_baseline_never_uses_scientific_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"fixed"
            model=FakeModel()
            result=campaign(out,["agentcf_popularity"],"fresh",model,
                            cohort_size=4,arm="strong_rule")
            self.assertEqual(result["episodes"][0]["proposal_origin"],"fixed_rule_baseline")
            self.assertNotIn("diagnosis",model.roles)
            self.assertNotIn("design",model.roles)
            self.assertNotIn("critic",model.roles)
            self.assertNotIn("reflection",model.roles)
            self.assertIn("rank",model.roles)
            self.assertFalse((out/"episode-001/candidate_skill.json").exists())
            self.assertEqual(result["independently_confirmed_discoveries"],0)


if __name__=="__main__":
    unittest.main()
