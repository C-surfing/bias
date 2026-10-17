"""Cross-Bias discovery gate and CPU falsifier regression tests."""
from pathlib import Path
import math
import unittest
from rsi_researcher.research import crossbias_discovery as b
from rsi_researcher.research import crossbias_runner as runner


class CrossBiasTests(unittest.TestCase):
    def test_frozen_registered_task_contract(self):
        p=b.frozen_protocol()
        self.assertEqual(len(p["tasks"]),4)
        self.assertEqual(len(b.ARMS),3)
        self.assertEqual(p["author_query_count"],10)
        self.assertEqual(set(b.RULE),set(b.CATEGORIES))
        self.assertEqual(p["hypothesis_success_grade"].startswith("replicated_development"),True)

    def test_proposal_rejects_unauthorized_actions(self):
        for task,plan in b.rule_plans().items():
            valid,error=b.valid_plan(task,plan)
            self.assertTrue(valid, (task,error))
            copied=plan.copy()
            copied["action"]="arbitrary_python"
            self.assertFalse(b.valid_plan(task,copied)[0])
            copied=plan.copy()
            copied["shell"]="anything"
            self.assertFalse(b.valid_plan(task,copied)[0])

    def test_coat_all_dev_only_falsifiers(self):
        for name in ("gamma_stress","propensity_shuffle","split_utility"):
            result=b.coat_observation(name)
            self.assertEqual(result["status"],"EXECUTED_DEVELOPMENT")
            self.assertFalse(result["confirmation_used"])
            task="coat_propensity" if name!="split_utility" else "coat_utility"
            self.assertIn("passes",b.assess(task,name,result)["falsifier"])
            self.assertFalse(b.assess(task,name,result)["independently_confirmed_scientific_discovery"])

    def test_author_initial_and_frozen_state_same_query_membership(self):
        source,state,profile,queries=b.author_inputs()
        self.assertEqual(len(queries),10)
        self.assertTrue(all(q["user"] in state["users"] for q in queries))
        self.assertTrue(all(set(q["candidates"])<=state["items"].keys() for q in queries))
        self.assertEqual(len({q["user"] for q in queries}),10)

    def test_agentcf_evaluation_only_predeclared_metrics(self):
        same=lambda q:{"ranking":[str(q)]*10,"excess_popularity_exposure":.3,
                       "ndcg_at_5":.5}
        original=[same(i) for i in range(5)]
        # Candidate and repeat intentionally identical, so neither mechanism counts.
        mock={"panels":{"a":{"original":original,"repeat":original,
              "item_memory_ablation":original,"popularity_prompt":original},
              "b":{"original":original,"repeat":original,
              "item_memory_ablation":original,"popularity_prompt":original}}}
        self.assertFalse(b.evaluate_author("item_memory_ablation",mock)["passes"])
        self.assertFalse(b.evaluate_author("popularity_prompt",mock)["passes"])

    def test_aggregate_does_not_promote_dev_signals_to_independent_discovery(self):
        plans={a:b.rule_plans() for a in b.ARMS}
        results={a:{t:{"replicated_development_signal":True,
                 "independently_confirmed_scientific_discovery":False,
                 "falsifier":{"passes":True}} for t in b.CATEGORIES} for a in b.ARMS}
        costs={a:{"llm_calls":0 if a=="rule" else 8} for a in b.ARMS}
        result=b.aggregate(plans,results,costs)
        self.assertEqual(result["rule"]["development_signals_replicated"],4)
        self.assertTrue(all(x["independent_valid_scientific_discoveries"]==0
                            for x in result.values()))

    def test_no_qwen_startup_in_unit_tests(self):
        self.assertEqual(runner.PORT,18329)
        self.assertGreaterEqual(runner.MAX_REQUESTS,50)
        self.assertLessEqual(runner.MAX_REQUESTS,120)


if __name__=="__main__":unittest.main()
