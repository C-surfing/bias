"""Regression: repair effects must never be mistaken for independent debias."""
from unittest import TestCase

from rsi_researcher.llmrank.evidence import paired_clean_evidence
from rsi_researcher.llmrank.science import audit


def metrics(ndcg,arp):
    return {"5":{"users":14,"valid_rankings":14,"invalid_rankings":0,
                 "HR@5":0.30,"NDCG@5":ndcg,"ARP@5":arp,"APLT@5":0.4,
                 "policy_execution_failures":0}}


def rows(repaired,users=14):
    items=[f"i{j}" for j in range(20)]
    return [dict(user=f"u{i}",target=items[10],ranking=items,
                 observed_votes=1,repaired_votes=int(i in repaired),
                 policy_execution_failures=0) for i in range(users)]


class FormatAwareHarnessTests(TestCase):
    def setUp(self):
        self.titles={f"i{i}":f"title {i}" for i in range(20)}
        self.pop={f"i{i}":i+1 for i in range(20)}
        self.baseline=metrics(0.045,19.0)
        self.candidate=metrics(0.071,15.3)

    def test_legacy_metrics_can_appear_positive_but_format_support_blocks(self):
        self.assertTrue(audit(self.baseline,self.candidate)["eligible_development_only"])
        left={"random":rows({0}),"first":rows({1})}
        right={"random":rows({2,3,4}),"first":rows({5,6,7})}
        matched=paired_clean_evidence(left,right,self.pop,self.titles)
        self.assertEqual(matched["paired_clean_users"],6)
        self.assertFalse(matched["sufficient_support"])
        result=audit(self.baseline,self.candidate,clean_evidence=matched)
        self.assertFalse(result["eligible_development_only"])
        self.assertEqual(result["failure_mode"],"inconclusive_due_to_ranking_format")
        self.assertEqual(result["format_assessment"],"insufficient_clean_paired_support")

    def test_clean_pair_can_reject_apparent_full_panel_success(self):
        clean=rows(set())
        matched=paired_clean_evidence(
            {"random":clean,"first":clean},
            {"random":clean,"first":clean},self.pop,self.titles)
        self.assertTrue(matched["sufficient_support"])
        result=audit(self.baseline,self.candidate,clean_evidence=matched)
        self.assertFalse(result["eligible_development_only"])
        self.assertEqual(result["failure_mode"],"format_sensitive_candidate")
        self.assertEqual(result["format_assessment"],"not_robust_to_format_exclusion")

    def test_identical_repaired_case_must_be_excluded_from_both_arms(self):
        left={"random":rows({1}),"first":rows(set())}
        right={"random":rows(set()),"first":rows({3})}
        matched=paired_clean_evidence(left,right,self.pop,self.titles)
        self.assertEqual(matched["paired_clean_users"],12)
        self.assertEqual(matched["baseline"]["5"]["users"],12)
        self.assertEqual(matched["candidate"]["5"]["users"],12)
        self.assertEqual(matched["excluded_users"],2)
