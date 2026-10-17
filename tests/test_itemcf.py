"""Train-only ranking contracts for observed-rating recommendation data."""

import math
import unittest

from rsi_researcher.recommenders.itemcf import ItemCF


class ItemCFTests(unittest.TestCase):
    def setUp(self):
        self.items = ["a", "b", "c", "d", "e"]
        self.train = [
            {"user": "u", "item": "a", "rating": 5},
            {"user": "u", "item": "b", "rating": 5},
            {"user": "v", "item": "a", "rating": 5},
            {"user": "v", "item": "c", "rating": 5},
            {"user": "w", "item": "b", "rating": 5},
            {"user": "w", "item": "c", "rating": 0},
        ]
        self.model = ItemCF(self.train, self.items)

    def test_positive_cosine_ranking_matches_hand_calculation(self):
        # a/b have two positive users, c one.  u's score(c)=1/sqrt(2),
        # score(a)=score(b)=1/2.  Unscored catalogue entries tie by ID.
        scores, scale = self.model._scores("u")
        self.assertAlmostEqual(scores["c"], 1 / math.sqrt(2))
        self.assertAlmostEqual(scores["a"], .5)
        self.assertAlmostEqual(scores["b"], .5)
        self.assertAlmostEqual(scale, 1 / math.sqrt(2))
        self.assertEqual(self.model.rank("u", self.items), ["c", "a", "b", "d", "e"])

    def test_binary_zero_is_observed_exposure_but_not_a_positive(self):
        model = ItemCF([
            {"user": "u", "item": "clicked", "rating": 1},
            {"user": "u", "item": "unclicked", "rating": 0},
        ], ["clicked", "unclicked", "unexposed"], positive_threshold=1)
        self.assertEqual(model.train_counts, {"clicked": 1, "unclicked": 1, "unexposed": 0})
        self.assertEqual(model._positive["u"], {"clicked"})
        self.assertEqual(model._neighbors["clicked"], ())
        self.assertEqual(model.rank_catalog(["u"]), {"u": ["unexposed"]})
        self.assertEqual(model._fallback["unclicked"], 0)
        self.assertEqual(model._fallback["unexposed"], .5)

    def test_repeated_exposures_count_but_do_not_duplicate_cooccurrence(self):
        duplicate = self.train + [self.train[0].copy()]
        model = ItemCF(duplicate, self.items)
        self.assertEqual(model.train_counts["a"], 3)
        self.assertEqual(model._neighbors, self.model._neighbors)
        self.assertEqual(model.rank("u", self.items), self.model.rank("u", self.items))

    def test_panel_labels_are_never_read(self):
        class CandidateOnly(dict):
            def __getitem__(self, key):
                if key not in ("user", "item"):
                    raise AssertionError("ranking tried to read an evaluation label")
                return super().__getitem__(key)

        rows = [CandidateOnly(user="u", item=item, rating=float("nan")) for item in self.items]
        self.assertEqual(self.model.rank_panels(rows), {"u": self.model.rank("u", self.items)})
        rows.reverse()
        rows.append(rows[0])
        self.assertEqual(self.model.rank_panels(rows), {"u": self.model.rank("u", self.items)})

    def test_panel_membership_cannot_change_pairwise_order(self):
        for operator in ("retain", "popularity_penalty", "user_calibrated", "support_preserving"):
            for strength in (0, .1, .5, 1):
                policy = {"operator": operator, "strength": strength}
                complete = self.model.rank("u", self.items, policy)
                subset = ["b", "c", "e"]
                self.assertEqual(self.model.rank("u", subset, policy),
                                 [item for item in complete if item in subset])

    def test_policy_penalty_and_personal_calibration_are_train_only(self):
        self.assertEqual(self.model.head_items, {"a"})
        self.assertEqual(self.model._policy("u", {"operator": "user_calibrated", "strength": 1}), .5)
        # The unknown user's fallback ratio uses all five training positives,
        # two of which are for the one head item.
        self.assertAlmostEqual(self.model._policy("unknown", {"operator": "user_calibrated", "strength": 1}), .6)
        self.assertEqual(self.model._policy("u", {"operator": "retain", "strength": 1}), 0)
        self.assertEqual(self.model._popularity["a"], 1)
        self.assertEqual(self.model._popularity["d"], 0)
        base = self.model.rank("unknown", self.items)
        changed = self.model.rank("unknown", self.items, {"operator": "popularity_penalty", "strength": 1})
        self.assertLess(base.index("a"), base.index("d"))
        self.assertLess(changed.index("d"), changed.index("a"))

    def test_proportional_penalty_preserves_positive_support_above_zero_scores(self):
        scores, _ = self.model._scores("u")
        supported = {item for item in self.items if scores.get(item, 0) > 0}
        for strength in (.01, .02, .05, .99):
            ranking = self.model.rank("u", self.items, {"operator": "support_preserving", "strength": strength})
            self.assertEqual(set(ranking[:len(supported)]), supported)
        self.assertEqual(self.model.rank("u", self.items, {"operator": "support_preserving", "strength": 0}),
                         self.model.rank("u", self.items))

    def test_score_support_diagnosis_uses_ids_and_training_scores_only(self):
        class IDsOnly(dict):
            def __getitem__(self, key):
                if key not in ("user", "item"):
                    raise AssertionError("diagnosis read an evaluation label")
                return super().__getitem__(key)
        result = self.model.diagnose_panels([IDsOnly(user="u", item=item) for item in self.items])
        self.assertEqual(result["candidate_pairs"], 5)
        self.assertEqual(result["zero_score_fraction"], 2 / 5)
        self.assertEqual(result["per_user"][0]["positive_score_candidates"], 3)

    def test_rank_catalog_excludes_all_seen_items_and_returns_bounded_lists(self):
        result = self.model.rank_catalog(["u", "u", "w", "new"], k=2)
        self.assertEqual(set(result), {"u", "w", "new"})
        self.assertEqual(result["u"], ["c", "d"])
        self.assertFalse(set(result["w"]) & {"b", "c"})
        self.assertTrue(all(len(items) == 2 and len(set(items)) == 2 for items in result.values()))

    def test_top_neighbors_limit_and_ties_are_deterministic(self):
        model = ItemCF(self.train, self.items, neighbors=1)
        self.assertTrue(all(len(neighbors) <= 1 for neighbors in model._neighbors.values()))
        self.assertEqual(model._neighbors["a"][0][0], "c")
        self.assertEqual(model._neighbors["b"][0][0], "a")
        reversed_model = ItemCF(list(reversed(self.train)), list(reversed(self.items)), neighbors=1)
        self.assertEqual(model._neighbors, reversed_model._neighbors)
        self.assertEqual(model.rank("u", self.items), reversed_model.rank("u", self.items))

    def test_empty_and_unseen_histories_have_documented_fallbacks(self):
        self.assertEqual(self.model.rank("new", self.items), ["a", "b", "d", "e", "c"])
        empty = ItemCF([], self.items)
        for operator in ("retain", "popularity_penalty", "user_calibrated"):
            self.assertEqual(empty.rank("new", list(reversed(self.items)),
                             {"operator": operator, "strength": 1}), self.items)
        self.assertEqual(empty.metadata["missing_feedback"], "absent_row_not_rating_zero")

    def test_invalid_candidate_and_policy_inputs_fail_explicitly(self):
        for candidates in (["a", "a"], ["not_in_training_catalogue"]):
            with self.assertRaises(ValueError):
                self.model.rank("u", candidates)
        for policy in ({"operator": "not_implemented"},
                       {"operator": "popularity_penalty", "strength": -1},
                       {"operator": "popularity_penalty", "strength": True},
                       {"operator": "popularity_penalty", "strength": float("nan")}):
            with self.assertRaises(ValueError):
                self.model.rank("u", self.items, policy)
        with self.assertRaises(ValueError):
            self.model.rank_catalog(["u"], k=0)

    def test_train_rows_have_no_implicit_missing_placeholder(self):
        for rating in (-1, True, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                ItemCF([{"user": "u", "item": "a", "rating": rating}], self.items)
        with self.assertRaises(ValueError):
            ItemCF([{"user": "u", "item": "outside", "rating": 5}], self.items)
        with self.assertRaises(ValueError):
            ItemCF([], ["a", "a"])


if __name__ == "__main__":
    unittest.main()
