import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsi_researcher.datasets.coat import COAT_PROTOCOL, load_coat


class CoatTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "coat"
        features = self.data / "user_item_features"
        features.mkdir(parents=True)
        # Both matrices retain unobserved zeroes. One random pair overlaps train.
        (self.data / "train.ascii").write_text("5 0 0 0 0 0\n0 0 1 0 0 0\n", encoding="utf-8")
        (self.data / "test.ascii").write_text("4 2 3 4 5 1\n5 4 0 3 2 1\n", encoding="utf-8")
        (features / "item_features_map.txt").write_text("color:black\nonfrontpage:yes\n", encoding="utf-8")
        (features / "item_features.ascii").write_text("1 0\n0 1\n1 0\n0 1\n1 0\n0 1\n", encoding="utf-8")
        (self.root / "source.json").write_text(json.dumps({"sha256": "fixture-archive-hash"}), encoding="utf-8")

    @staticmethod
    def pairs(rows):
        return {(row["user"], row["item"]) for row in rows}

    def test_missing_remains_unknown_and_train_random_overlap_is_removed(self):
        result = load_coat(self.root, split_seed=7)
        self.assertEqual(result["statistics"]["original_train_rows"], 2)
        self.assertEqual(result["statistics"]["original_random_rows"], 11)
        self.assertEqual(result["statistics"]["random_train_overlap_removed"], 1)
        self.assertEqual((len(result["development"]), len(result["test"])), (4, 6))
        train, development, test = [self.pairs(result[key]) for key in ("train", "development", "test")]
        self.assertFalse(train & development)
        self.assertFalse(train & test)
        self.assertFalse(development & test)
        self.assertEqual(len(train | development | test), 12)
        self.assertEqual(sum(len(result[key]) for key in ("train", "development", "test")), 12)
        self.assertTrue(all(row["rating"] >= 1 for key in ("train", "development", "test") for row in result[key]))
        self.assertNotIn("utilities", result)
        self.assertEqual(result["protocol"]["dataset_protocol"], COAT_PROTOCOL)
        self.assertIn("not temporal", result["protocol"]["validation_scope"])

    def test_split_is_reproducible_and_changes_with_seed(self):
        first = load_coat(self.root, split_seed=7)
        self.assertEqual(first, load_coat(self.root, split_seed=7))
        other = load_coat(self.root, split_seed=8)
        self.assertNotEqual(first["development"], other["development"])
        self.assertNotEqual(first["fingerprint"], other["fingerprint"])
        self.assertEqual(first["train"], other["train"])
        self.assertEqual(self.pairs(first["development"] + first["test"]),
                         self.pairs(other["development"] + other["test"]))

    def test_real_feature_names_are_retained_and_propensities_are_not_read(self):
        # A malformed propensity file cannot influence a loader which never uses it.
        (self.data / "propensities.ascii").write_text("not true probabilities", encoding="utf-8")
        result = load_coat(self.data)
        self.assertEqual(result["item_features"]["0"], {"color:black": 1.0, "onfrontpage:yes": 0.0})
        self.assertEqual(result["user_ids"], ["0", "1"])
        self.assertEqual(result["item_ids"], list(map(str, range(6))))
        self.assertFalse(result["source"]["propensities_used"])
        self.assertNotIn("propensities.ascii", result["source"]["loaded_file_sha256"])
        self.assertEqual((result["positive_threshold"], result["gain_floor"]), (4, 1))

    def test_fingerprint_tracks_loaded_data_and_protocol(self):
        before = load_coat(self.root)
        (self.data / "train.ascii").write_text("4 0 0 0 0 0\n0 0 1 0 0 0\n", encoding="utf-8")
        after = load_coat(self.root)
        self.assertNotEqual(before["fingerprint"], after["fingerprint"])
        self.assertEqual(before["protocol"], after["protocol"])

    def test_incompatible_matrices_and_out_of_range_ratings_are_rejected(self):
        (self.data / "test.ascii").write_text("1 0\n0 5\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            load_coat(self.root)
        (self.data / "test.ascii").write_text("6 0 0 0 0 0\n0 0 0 0 0 0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integers from 1 to 5"):
            load_coat(self.root)


if __name__ == "__main__":
    unittest.main()
