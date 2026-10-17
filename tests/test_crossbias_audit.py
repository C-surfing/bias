"""Guard against invalid Qwen memory promotion and audit cost errors."""
import unittest
from rsi_researcher.research import crossbias_memory as m
from rsi_researcher.research import crossbias_audit as audit
from rsi_researcher.research.memory import ResearchMemory,validate_card


class ResearchEvidenceAuditTests(unittest.TestCase):
    def test_pilot_cost_and_source_validity(self):
        report=audit.audit()
        self.assertEqual(report["model_usage_totals"]["calls"],60)
        self.assertTrue(report["qwen_history_did_not_change_actions"])
        self.assertEqual(report["by_arm"]["rule"]["independent_scientific_discoveries"],0)
        self.assertEqual(report["by_arm"]["qwen_no_history"]["model_research_calls"],8)

    def test_no_unvalidated_llm_text_in_promotable_memory(self):
        cards=m.build_cards()
        self.assertEqual(len(cards),5)
        for c in cards:
            obj=ResearchMemory(**c)
            validate_card(obj)
            self.assertEqual(obj.evidence_grade,"development_only")
            self.assertFalse(obj.confirmed)
            self.assertNotIn("The current evidence does not conclusively support",obj.conditional_lesson)
            self.assertIn("development",obj.limitation.lower())

    def test_falsifier_failure_is_not_counted_as_discovery(self):
        cards=m.build_cards()
        failed=[c for c in cards if "Item" in c["conditional_lesson"]
                or "item_memory_ablation" in c["card_id"]]
        self.assertTrue(failed)
        self.assertTrue(any("FAIL/INCONCLUSIVE" in x["conditional_lesson"] for x in failed))


if __name__=="__main__":unittest.main()
