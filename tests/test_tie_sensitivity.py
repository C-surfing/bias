from itertools import permutations, product
import math
import unittest

from rsi_researcher.evaluation.tie_sensitivity import ideal_head_range


class TieSensitivityTests(unittest.TestCase):
    def test_all_equal_utilities_can_permit_popular_first_without_quality_loss(self):
        panel = [str(i) for i in range(12)]
        result = ideal_head_range(panel, dict.fromkeys(panel, 1), set(panel[:2]))
        self.assertEqual(result["minimum"], 0)
        self.assertAlmostEqual(result["random_tie_expectation"], 2 / 12)
        self.assertGreater(result["maximum"], result["random_tie_expectation"])

    def test_bounds_and_expectation_match_exhaustive_tie_permutations(self):
        groups = [list("ab"), list("c"), list("de"), list("fghijk")]
        panel = sum(groups, [])
        utility = {i: 4 - n for n, group in enumerate(groups) for i in group}
        popular = set("adik")
        discount = [1 / math.log2(j + 2) for j in range(10)]
        heads = []
        for order in product(*(permutations(g) for g in groups)):
            ranking = sum((list(g) for g in order), [])
            heads.append(sum(d for d, i in zip(discount, ranking) if i in popular) / sum(discount))
        result = ideal_head_range(panel, utility, popular)
        self.assertAlmostEqual(result["minimum"], min(heads))
        self.assertAlmostEqual(result["maximum"], max(heads))
        self.assertAlmostEqual(result["random_tie_expectation"], sum(heads) / len(heads))


if __name__ == "__main__":
    unittest.main()
