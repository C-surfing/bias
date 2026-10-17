"""Paired synthetic-research analysis validation, without model calls."""
import unittest
from rsi_researcher.research.decision_analysis import mean_diff_ci
from rsi_researcher.research.decision_benchmark import tasks


class DecisionAnalysisTests(unittest.TestCase):
    def test_mechanism_cluster_bootstrap(self):
        _,eval_cases=tasks()
        a={c["task_id"]:0.2 for c in eval_cases}
        b={c["task_id"]:0.1 for c in eval_cases}
        result=mean_diff_ci(a,b,draws=400)
        self.assertEqual(result["mechanism_clusters"],7)
        self.assertAlmostEqual(result["mean_excess_regret"],0.1)
        self.assertAlmostEqual(result["cluster_bootstrap_95"][0],0.1)
        self.assertAlmostEqual(result["cluster_bootstrap_95"][1],0.1)

    def test_mismatched_tasks_are_rejected(self):
        _,eval_cases=tasks()
        a={c["task_id"]:1 for c in eval_cases}
        b={c["task_id"]:1 for c in eval_cases[:-1]}
        with self.assertRaises(ValueError):
            mean_diff_ci(a,b,draws=100)


if __name__=="__main__":unittest.main()
