from copy import deepcopy
import unittest

from rsi_researcher.recommenders.memory_dependencies import affected_events, dependency_graph


class DependenciesTests(unittest.TestCase):
    def records(self):
        state = {"users": dict.fromkeys(("a", "b", "c"), "initial"),
                 "items": {i: {"memory": "initial"} for i in ("p", "q", "r", "s")}, "completed": []}
        records = []
        for eid, user, positive, negative in (("e0", "a", "p", "q"), ("e1", "c", "r", "s"),
                                               ("e2", "b", "q", "p"), ("e3", "b", "s", "r")):
            before = deepcopy(state)
            state["users"][user] = eid
            state["items"][positive]["memory"] = eid
            state["completed"].append(eid)
            records.append({"event": {"event_id": eid, "user": user, "positive": positive,
                                      "negative": negative, "candidates": [negative, positive]},
                            "before": before, "after": deepcopy(state)})
        return records

    def test_cross_user_reads_and_transitive_repair_exclude_independent_event(self):
        graph = dependency_graph(self.records())
        self.assertEqual(graph["e2"]["parents"], ["e0"])
        self.assertEqual(graph["e3"]["parents"], ["e1", "e2"])
        self.assertEqual(affected_events(graph, ["e0"]), ["e0", "e2", "e3"])

    def test_corrupt_or_partial_chain_is_rejected(self):
        records = self.records()
        with self.assertRaises(ValueError):
            dependency_graph(records[1:])
        records[1]["before"]["users"]["a"] = "corrupt"
        with self.assertRaises(ValueError):
            dependency_graph(records)
        records = self.records()
        records[0]["after"]["items"]["r"]["memory"] = "illegal"
        with self.assertRaises(ValueError):
            dependency_graph(records)
        with self.assertRaises(ValueError):
            affected_events(dependency_graph(self.records()), ["missing"])
        records = self.records()
        records[0]["memory_policy"] = "no_sharing"
        with self.assertRaises(ValueError):
            dependency_graph(records)


if __name__ == "__main__":
    unittest.main()
