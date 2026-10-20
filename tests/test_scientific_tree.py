"""Scientific tree tests use the real episode and experiment adapter.

FakeModel drives software paths only; it does not establish any Qwen finding.
"""
import json
from pathlib import Path
import tempfile
import unittest

from rsi_researcher.research.scientific_tree import search
from tests.test_scientific_loop import FakeModel


class TreeResearchTests(unittest.TestCase):
    def test_real_fresh_agentcf_branch_backtrack_and_shared_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"fresh_tree"
            model=FakeModel()
            result=search(output,"agentcf_memory",model=model,
                          cohort_size=12,seed=20270001,
                          width=2,max_depth=2,max_nodes=3,
                          max_proposals=4,max_requests=300)
            self.assertEqual(result["status"],"COMPLETED_DEVELOPMENT_SEARCH")
            self.assertEqual(result["attempted_nodes"],3)
            self.assertEqual(result["independently_confirmed_discoveries"],0)
            nodes=result["nodes"]
            self.assertEqual({nodes["N001"]["parent_id"],nodes["N002"]["parent_id"]},{"ROOT"})
            self.assertEqual(nodes["N003"]["parent_id"],"N001")
            self.assertEqual([r["node_id"] for r in result["selection_history"]],
                             ["N001","N003","N002"])
            self.assertEqual(result["selection_history"][2]["backtracked_from"],"N003")
            self.assertTrue(result["state_ref"])
            state_path=output/result["state_ref"]
            self.assertTrue(state_path.is_file())
            for node_id in ("N001","N002","N003"):
                node=nodes[node_id]
                self.assertEqual(node["state_ref"],result["state_ref"])
                self.assertEqual(node["status"],"COMPLETED_DEVELOPMENT")
                plan=json.loads((output/node["plan_ref"]).read_text())
                audit=json.loads((output/node["audit_ref"]).read_text())
                frozen=json.loads((output/"nodes"/node_id/"frozen_plan.json").read_text())
                training=json.loads((output/"nodes"/node_id/"agentcf_training.json").read_text())
                self.assertEqual(plan["proposal"],frozen["proposal"])
                self.assertEqual(plan["proposal"]["action"],audit["action"])
                self.assertEqual(training["trained_state_sha256"],result["training_sha256"])
                self.assertEqual(audit["independently_confirmed_discoveries"],0)
            # All distinct, real registered interventions; no duplicate search nodes.
            self.assertEqual(len({nodes[x]["action"] for x in nodes}),len(nodes))
            # Parent audit only appears in its children, never the root sibling.
            sibling=json.loads((output/"nodes/N002/evidence.json").read_text())
            child=json.loads((output/"nodes/N003/evidence.json").read_text())
            self.assertNotIn("tree_parent_development_evidence",sibling)
            self.assertEqual(child["tree_parent_development_evidence"]["parent_id"],"N001")
            self.assertNotIn('"target":',json.dumps(child["tree_parent_development_evidence"]))
            self.assertTrue((output/"protocol.sha256.json").is_file())
            with self.assertRaises(ValueError):
                search(output,"agentcf_memory",model=FakeModel())

    def test_archive_branching_is_replay_not_new_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=search(Path(tmp)/"archive","coat_utility",model=FakeModel(),
                          mode="archive",width=2,max_depth=2,max_nodes=2)
            self.assertEqual(result["attempted_nodes"],2)
            self.assertEqual(len(result["nodes"]),2)
            self.assertEqual(result["independently_confirmed_discoveries"],0)
            self.assertEqual(result["status"],"COMPLETED_ARCHIVE_REPLAY_SEARCH")
            self.assertEqual(result["archived_replay_nodes"],2)
            self.assertEqual(result["executed_development_nodes"],0)
            for node in result["nodes"].values():
                audit=json.loads((Path(tmp)/"archive"/node["audit_ref"]).read_text())
                self.assertEqual(audit["evidence_grade"],"previous_development_only")
            self.assertEqual(result["termination_reason"],"max_executed_nodes")

    def test_critic_rejections_remain_and_budget_leaves_frontier(self):
        with tempfile.TemporaryDirectory() as tmp:
            refused=search(Path(tmp)/"reject","coat_propensity",
                           model=FakeModel(reject=True),mode="archive",
                           width=2,max_nodes=2)
            self.assertEqual(refused["attempted_nodes"],2)
            self.assertEqual(refused["executed_development_nodes"],0)
            self.assertTrue(all(n["status"]=="DESIGN_REJECTED"
                                for n in refused["nodes"].values()))
            self.assertTrue(all(n["audit_ref"] is None
                                for n in refused["nodes"].values()))
            partial=search(Path(tmp)/"budget","coat_utility",
                           model=FakeModel(),mode="archive",
                           width=2,max_nodes=1)
            self.assertEqual(partial["attempted_nodes"],1)
            self.assertEqual(partial["termination_reason"],"max_executed_nodes")
            self.assertEqual(len(partial["unselected_frontier_nodes"]),1)
            self.assertEqual(partial["nodes"][partial["unselected_frontier_nodes"][0]]
                             ["status"],"FRONTIER")


if __name__=="__main__":
    unittest.main()
