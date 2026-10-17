import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsi_researcher.datasets.kuairand import FEATURE_FILE, RANDOM_FILE, TRAIN_FILE, load_kuairand


class KuaiRandTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "KuaiRand-Pure" / "data"
        self.data.mkdir(parents=True)
        self.train = [
            {"user_id": "u1", "video_id": "a", "date": 20220408, "time_ms": 1, "is_click": 1, "is_rand": 0},
            {"user_id": "u2", "video_id": "b", "date": 20220421, "time_ms": 2, "is_click": 0, "is_rand": 0},
        ]
        self.random = [
            {"user_id": "u1", "video_id": "a", "date": 20220422, "time_ms": 3, "is_click": 1, "is_rand": 1},
            {"user_id": "u1", "video_id": "c", "date": 20220423, "time_ms": 5, "is_click": 1, "is_rand": 1},
            {"user_id": "u1", "video_id": "c", "date": 20220422, "time_ms": 4, "is_click": 0, "is_rand": 1},
            {"user_id": "u1", "video_id": "c", "date": 20220501, "time_ms": 6, "is_click": 1, "is_rand": 1},
            {"user_id": "u1", "video_id": "d", "date": 20220502, "time_ms": 7, "is_click": 0, "is_rand": 1},
            {"user_id": "u2", "video_id": "d", "date": 20220430, "time_ms": 8, "is_click": 1, "is_rand": 1},
            {"user_id": "u2", "video_id": "e", "date": 20220508, "time_ms": 9, "is_click": 1, "is_rand": 1},
        ]
        self.write(TRAIN_FILE, self.train)
        self.write(RANDOM_FILE, self.random)
        self.write(FEATURE_FILE, [{"video_id": item, "tag": "12,65", "visible_status": 1} for item in "abcde"])

    def write(self, filename, rows):
        with (self.data / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_observed_zero_retained_earliest_repeat_and_time_partitions(self):
        result = load_kuairand(self.root)
        self.assertEqual(result["dataset_id"], "kuairand-pure")
        self.assertEqual((len(result["train"]), len(result["development"]), len(result["test"])), (2, 2, 2))
        self.assertIn({"user": "u1", "item": "c", "rating": 0.0}, result["development"])
        self.assertNotIn({"user": "u1", "item": "c", "rating": 1.0}, result["development"])
        self.assertNotIn(("u1", "c"), {(row["user"], row["item"]) for row in result["test"]})
        self.assertEqual(result["statistics"]["random_train_overlap_rows_removed"], 1)
        self.assertEqual(result["statistics"]["development_repeat_rows_removed"], 1)
        self.assertEqual(result["statistics"]["test_development_overlap_pairs_removed"], 1)
        self.assertEqual(result["statistics"]["train_observed_zero_rows"], 1)
        self.assertEqual((result["positive_threshold"], result["gain_floor"]), (1, 0))
        self.assertNotIn("utilities", result)

    def test_cohort_never_depends_on_future_positive_labels(self):
        before = load_kuairand(self.root, user_limit=1, split_seed=7)
        for row in self.random:
            row["is_click"] = 1 - row["is_click"]
        self.write(RANDOM_FILE, self.random)
        after = load_kuairand(self.root, user_limit=1, split_seed=7)
        self.assertEqual(before["user_ids"], after["user_ids"])
        self.assertEqual(before["train"], after["train"])
        self.assertEqual(before["item_ids"], after["item_ids"])
        self.assertEqual(before["user_ids"], load_kuairand(self.root, user_limit=1, split_seed=7)["user_ids"])
        self.assertNotEqual(before["fingerprint"], after["fingerprint"])

    def test_more_training_users_do_not_change_evaluation_cohort_or_panels(self):
        small = load_kuairand(self.root, user_limit=1, split_seed=7)
        complete = load_kuairand(self.root, user_limit=1, split_seed=7, training_scope="all_preperiod_users")
        self.assertEqual(small["user_ids"], complete["user_ids"])
        self.assertEqual(small["development"], complete["development"])
        self.assertEqual(small["test"], complete["test"])
        self.assertEqual(complete["statistics"]["training_users"], 2)
        self.assertEqual(len(complete["train"]), 2)
        self.assertGreater(len(complete["train"]), len(small["train"]))
        self.assertNotEqual(small["fingerprint"], complete["fingerprint"])
        for row in self.random:
            row["is_click"] = 1 - row["is_click"]
        self.write(RANDOM_FILE, self.random)
        after = load_kuairand(self.root, user_limit=1, split_seed=7, training_scope="all_preperiod_users")
        self.assertEqual(complete["train"], after["train"])

    def test_training_cap_keeps_latest_and_full_history_seen_pairs_still_excluded(self):
        self.train = [{"user_id": "u1", "video_id": "a", "date": 20220408,
                       "time_ms": index, "is_click": index % 2, "is_rand": 0} for index in range(201)]
        self.train[0]["video_id"] = "b"
        self.write(TRAIN_FILE, self.train)
        self.random.append({"user_id": "u1", "video_id": "b", "date": 20220501,
                            "time_ms": 1000, "is_click": 1, "is_rand": 1})
        self.write(RANDOM_FILE, self.random)
        result = load_kuairand(self.data)
        self.assertEqual(len(result["train"]), 200)
        self.assertTrue(all(row["item"] == "a" for row in result["train"]))
        self.assertFalse(any(row["item"] == "b" for row in result["test"]))
        self.assertEqual(result["statistics"]["train_history_cap_rows_removed"], 1)

    def test_future_statistics_and_dynamic_visibility_are_not_features(self):
        (self.data / "video_features_statistic_pure.csv").write_text("future labels must never be read", encoding="utf-8")
        result = load_kuairand(self.root)
        self.assertFalse(result["source"]["video_statistics_used"])
        self.assertEqual(result["item_features"]["a"], {"tag": "12,65"})
        self.assertFalse(any("statistic" in filename for filename in result["source"]["loaded_file_sha256"]))

    def test_nonrandom_or_outside_date_rows_are_not_silently_admitted(self):
        self.random.extend([
            {"user_id": "u1", "video_id": "e", "date": 20220421, "time_ms": 10, "is_click": 1, "is_rand": 1},
            {"user_id": "u1", "video_id": "e", "date": 20220509, "time_ms": 11, "is_click": 1, "is_rand": 1},
            {"user_id": "u1", "video_id": "e", "date": 20220425, "time_ms": 12, "is_click": 1, "is_rand": 0},
        ])
        self.write(RANDOM_FILE, self.random)
        result = load_kuairand(self.root)
        self.assertEqual(result["statistics"]["random_outside_protocol_rows"], 3)
        self.assertFalse(any(row["user"] == "u1" and row["item"] == "e"
                             for row in result["development"] + result["test"]))


if __name__ == "__main__":
    unittest.main()
