"""Qwen pilot policy contract tests without launching the model."""
import unittest
from rsi_researcher.research import decision_benchmark as b
from rsi_researcher.research import decision_qwen_pilot as pilot


class QwenPilotTests(unittest.TestCase):
    def test_budget_and_all_arms(self):
        self.assertEqual(len(pilot.ARMS),4)
        self.assertEqual(pilot.MAX_TOTAL_CALLS,60)
        self.assertEqual(pilot.MAX_CALLS_PER_ARM,15)

    def test_unknown_logging_propensity_forces_abstention(self):
        unknown=b.diagnostic(b.spec("missing_propensity",42,"evaluation"))
        chosen,error=pilot.validate_decision(
            {"action":"ips","rationale":"Try IPS","falsifier":"Insufficient evidence"},unknown)
        self.assertEqual(chosen,"abstain")
        self.assertEqual(error,"missing_logging_propensity")

    def test_zero_positivity_forces_abstention(self):
        bad=b.diagnostic(b.spec("unsupported",42,"evaluation"))
        chosen,error=pilot.validate_decision(
            {"action":"snips","rationale":"test","falsifier":"test"},bad)
        self.assertEqual((chosen,error),("abstain","positivity_violated"))

    def test_malformed_output_abstains(self):
        known=b.diagnostic(b.spec("head_shift",37,"evaluation"))
        chosen,error=pilot.validate_decision(
            {"action":"python_shell","rationale":"x","falsifier":"y"},known)
        self.assertEqual((chosen,error),("abstain","unregistered_action"))

    def test_only_development_cards_guide_memory(self):
        p=b.frozen_payload()
        ev=p["evaluation_tasks"][0]
        for arm in pilot.ARMS:
            cards=pilot.history_cards(p,arm,ev)
            self.assertFalse(any("oracle" in str(c) or
                "evaluation_feedback" in str(c) for c in cards))
        self.assertEqual(pilot.history_cards(p,"no_history",ev),[])
        self.assertTrue(pilot.history_cards(p,"related_history",ev))


if __name__=="__main__":unittest.main()
