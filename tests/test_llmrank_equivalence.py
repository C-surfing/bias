from unittest import TestCase
from rsi_researcher.llmrank.strategy import BASELINE,functional_fingerprint
from rsi_researcher.llmrank.tools import candidate_snapshot

class CohortEquivalenceTests(TestCase):
    def test_history_windows_collapse_without_available_history(self):
        cases=[{"user":"u","history":["1","2","3","4"],
                "candidates":["1","2"],"target":"2"}]
        a={**BASELINE,"history_n":20}
        b={**BASELINE,"history_n":50}
        c={**BASELINE,"history_n":3}
        self.assertEqual(functional_fingerprint(a,cases),functional_fingerprint(b,cases))
        self.assertNotEqual(functional_fingerprint(a,cases),functional_fingerprint(c,cases))
    def test_neutral_penalty_mechanisms_are_identical(self):
        cases=[{"user":"u","history":["1","2","3","4"],"candidates":["1","2"],"target":"2"}]
        a={**BASELINE,"rerank_policy":"global"}
        b={**BASELINE,"rerank_policy":"protect_top3"}
        c={**BASELINE,"popularity_lambda":0.2,"rerank_policy":"protect_top3"}
        self.assertEqual(functional_fingerprint(a,cases),functional_fingerprint(b,cases))
        self.assertNotEqual(functional_fingerprint(a,cases),functional_fingerprint(c,cases))
    def test_diagnostic_reports_effective_history_fraction(self):
        cases=[{"user":"u","history":["1","2","3","4"],"candidates":["1","2"],"target":"2"},
               {"user":"v","history":["1","2","3"],"candidates":["1","2"],"target":"2"}]
        m=candidate_snapshot(cases,{"1":1,"2":1})
        self.assertEqual(m["history_window_effective_fraction"]["3"],0.5)
        self.assertEqual(m["history_window_effective_fraction"]["5"],0.0)
