"""Independent audit/ResearchMemory regression on frozen new state."""
import unittest
from rsi_researcher.research import fresh_audit,fresh_memory
from rsi_researcher.research.memory import ResearchMemory,validate_card


class FreshAuditTests(unittest.TestCase):
    def test_tokens_and_preforecasts_match_frozen_files(self):
        a=fresh_audit.audit()
        self.assertEqual(a["status"],"INDEPENDENT_NEW_STATE_FORECAST_AUDIT_COMPLETE")
        self.assertEqual(a["total_cost"]["calls"],92)
        self.assertTrue(a["forecast_precedes_training"])
        self.assertEqual(a["by_arm"]["qwen_no_history"]["matching_direction_and_effect_and_quality_in_both_panels"],0)
        self.assertEqual(a["by_arm"]["qwen_verified_history"]["matching_direction_and_effect_and_quality_in_both_panels"],0)
        self.assertEqual(a["strict_independently_confirmed_scientific_discoveries"],0)

    def test_fresh_research_memory_is_numerical_and_unconfirmed(self):
        cards=fresh_memory.build()
        self.assertEqual(len(cards["cards"]),2)
        self.assertFalse(cards["qwen_free_text_used"])
        for c in cards["cards"]:
            v=ResearchMemory(**c)
            validate_card(v)
            self.assertFalse(v.confirmed)
            self.assertEqual(v.evidence_grade,"development_only")
            self.assertIn("safety gates",v.conditional_lesson)


if __name__=="__main__":unittest.main()
