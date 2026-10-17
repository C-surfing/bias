from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from rsi_researcher.datasets.controlled import WORLD_PROTOCOL, build_world, ordered_events, weighted_pick
from rsi_researcher.evaluation.metrics import evaluate


class WorldTests(unittest.TestCase):
    def test_reproducible_paired_inputs_and_separate_truth(self):
        world = build_world(17)
        self.assertEqual(world, build_world(17))
        self.assertNotEqual(world, build_world(18))
        self.assertEqual(set(world["public"]), {"initial", "events", "queries", "exposure"})
        self.assertNotIn("utilities", world["public"]["initial"])
        for stream in world["public"]["events"].values():
            self.assertEqual(len(stream), 8)
            seen = {}
            for event in stream:
                history = seen.setdefault(event["user"], set())
                self.assertNotIn(event["negative"], history)
                self.assertNotEqual(event["positive"], event["negative"])
                history.add(event["positive"])
        for stratum in ("unrelated", "aligned"):
            world = build_world(17, stratum=stratum)
            rankings = {q["query_id"]: sorted(q["candidates"], key=lambda i: (-world["private"]["utilities"][q["user"]][i], i))
                        for q in world["public"]["queries"]}
            score = evaluate(world["public"]["queries"], rankings, world["private"]["utilities"], world["private"]["popular"])
            self.assertEqual((score["utility"], score["bias"]), (1, 0))

    def test_actual_sampling_probabilities_and_draws_are_auditable(self):
        for stratum in ("unrelated", "aligned"):
            world = build_world(31, users=3, items=60, events_per_user=4, stratum=stratum)
            private, public = world["private"], world["public"]
            catalog = list(public["initial"]["items"])
            self.assertEqual(world["protocol"]["world_protocol"], WORLD_PROTOCOL)
            pre_weights = [sum(private["utilities"][u][i] for u in private["utilities"])
                           if stratum == "aligned" else 1 for i in catalog]
            for i, weight in zip(catalog, pre_weights):
                self.assertAlmostEqual(private["preperiod_q"][i], weight / sum(pre_weights))
            for arm, multiplier in (("low", 1), ("high", 4)):
                counts = dict.fromkeys(catalog, 0)
                history = {u: set() for u in private["utilities"]}
                weights = [multiplier if i in private["popular"] else 1 for i in catalog]
                for i, weight in zip(catalog, weights):
                    self.assertAlmostEqual(public["exposure"]["design_q"][arm][i], weight / sum(weights))
                for event, trace in zip(public["events"][arm], private["sampling"][arm]):
                    u = event["user"]
                    self.assertEqual(trace["history_before"], sorted(history[u]))
                    pos_weights = [w * private["utilities"][u][i] for i, w in zip(catalog, weights)]
                    self.assertAlmostEqual(sum(trace["positive_q"].values()), 1)
                    self.assertAlmostEqual(sum(trace["negative_q"].values()), 1)
                    for i, weight in zip(catalog, pos_weights):
                        self.assertGreater(trace["positive_q"][i], 0)
                        self.assertAlmostEqual(trace["positive_q"][i], weight / sum(pos_weights))
                    self.assertEqual(event["positive"], weighted_pick(catalog, pos_weights, trace["positive_draw"]))
                    pool = [i for i in catalog if i not in history[u] and i != event["positive"]]
                    neg_weights = [multiplier if i in private["popular"] else 1 for i in pool]
                    self.assertEqual(event["negative"], weighted_pick(pool, neg_weights, trace["negative_draw"]))
                    for i in catalog:
                        expected = (multiplier if i in private["popular"] else 1) / sum(neg_weights) if i in pool else 0
                        self.assertAlmostEqual(trace["negative_q"][i], expected)
                    counts[event["positive"]] += 1
                    self.assertEqual(trace["positive_count_after"], counts[event["positive"]])
                    history[u].add(event["positive"])
                self.assertEqual(private["dynamic_positive_counts"][arm], counts)
            for a, b in zip(private["sampling"]["low"], private["sampling"]["high"]):
                self.assertEqual((a["positive_draw"], a["negative_draw"]), (b["positive_draw"], b["negative_draw"]))
            self.assertNotIn('"utilities"', json.dumps(public))
            self.assertNotIn('"positive_q"', json.dumps(public))
            public["exposure"]["popular"].clear()
            public["exposure"]["preperiod_counts"].clear()
            self.assertTrue(private["popular"])
            self.assertEqual(sum(private["preperiod_counts"].values()), 1000)

    def test_repeated_consumption_keeps_unobserved_comparisons_and_fixed_twenty_item_panels(self):
        with patch("rsi_researcher.datasets.controlled.random.Random.random", return_value=0):
            world = build_world(19, users=2, items=60, events_per_user=4)
        for stream in world["public"]["events"].values():
            for user in world["private"]["utilities"]:
                events = [e for e in stream if e["user"] == user]
                self.assertEqual(len({e["positive"] for e in events}), 1)
                self.assertTrue(all(e["negative"] != events[0]["positive"] for e in events))
        self.assertEqual(world["protocol"]["evaluation_panel_size"], 20)
        for query in world["public"]["queries"]:
            self.assertEqual(len(query["candidates"]), 20)
            self.assertEqual(len(set(query["candidates"])), 20)
        self.assertEqual(build_world(19, items=60)["public"]["queries"],
                         build_world(19, items=60, events_per_user=5)["public"]["queries"])

    def test_pairing_changes_only_order_and_keeps_parent_intact(self):
        world = build_world(23)
        events = world["public"]["events"]["high"]
        before = deepcopy(events)
        popular = world["private"]["popular"]
        first = ordered_events(events, popular, "pair_0")
        second = ordered_events(events, popular, "pair_1")
        for a, b in zip(first, second):
            self.assertEqual(a["candidates"], list(reversed(b["candidates"])))
            self.assertEqual(a["positive"], b["positive"])
        for event in ordered_events(events, popular, "head_first"):
            if sum(i in popular for i in event["candidates"]) == 1:
                self.assertIn(event["candidates"][0], popular)
        self.assertEqual(events, before)


if __name__ == "__main__":
    unittest.main()
