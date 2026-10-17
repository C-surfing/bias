"""Load observed Coat ratings without inventing feedback for missing entries."""

from hashlib import sha256
import json
from pathlib import Path
from random import Random


COAT_PROTOCOL = "coat_observed_random_holdout_v1"


def _matrix(path):
    rows = [list(map(float, line.split())) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{path.name} must contain a nonempty rectangular matrix")
    return rows


def _ratings(matrix):
    result = []
    for user, values in enumerate(matrix):
        for item, rating in enumerate(values):
            if rating not in (0, 1, 2, 3, 4, 5):
                raise ValueError("Coat ratings must be missing (0) or observed integers from 1 to 5")
            if rating:
                result.append({"user": str(user), "item": str(item), "rating": rating})
    return result


def load_coat(directory, split_seed=20261014):
    """Return self-selected training ratings and two disjoint random-rating panels.

    ``directory`` can be the extraction directory containing ``coat/`` or the
    inner directory containing ``train.ascii``. Randomly observed pairs already
    present in training are removed before partitioning. The per-user split is
    reproducible, but is not a temporal or a new-user validation experiment.
    """
    directory = Path(directory)
    data_dir = directory if (directory / "train.ascii").is_file() else directory / "coat"
    train_matrix = _matrix(data_dir / "train.ascii")
    random_matrix = _matrix(data_dir / "test.ascii")
    if (len(train_matrix), len(train_matrix[0])) != (len(random_matrix), len(random_matrix[0])):
        raise ValueError("Coat train and random rating matrices must have identical shapes")
    user_ids = list(map(str, range(len(train_matrix))))
    item_ids = list(map(str, range(len(train_matrix[0]))))
    train, random_ratings = _ratings(train_matrix), _ratings(random_matrix)
    train_pairs = {(row["user"], row["item"]) for row in train}
    remaining = [row for row in random_ratings if (row["user"], row["item"]) not in train_pairs]
    by_user = {user: [] for user in user_ids}
    for row in remaining:
        by_user[row["user"]].append(row)
    development, test = [], []
    for user in user_ids:
        rows = list(by_user[user])
        Random(f"{split_seed}:{user}").shuffle(rows)
        midpoint = len(rows) // 2
        development.extend(rows[:midpoint])
        test.extend(rows[midpoint:])

    feature_dir = data_dir / "user_item_features"
    feature_names = [line.strip() for line in (feature_dir / "item_features_map.txt").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    feature_matrix = _matrix(feature_dir / "item_features.ascii")
    if len(feature_matrix) != len(item_ids) or len(feature_matrix[0]) != len(feature_names):
        raise ValueError("Coat item feature dimensions must match the catalog and feature map")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Coat item feature map contains duplicate names")
    item_features = {item: dict(zip(feature_names, values))
                     for item, values in zip(item_ids, feature_matrix)}

    used_files = ["train.ascii", "test.ascii", "user_item_features/item_features.ascii",
                  "user_item_features/item_features_map.txt"]
    file_hashes = {name: sha256((data_dir / name).read_bytes()).hexdigest() for name in used_files}
    source_path = next((path for path in (directory / "source.json", data_dir.parent / "source.json",
                                          data_dir / "source.json") if path.is_file()), None)
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path else {
        "dataset": "Coat Shopping", "official_page": "https://www.cs.cornell.edu/~schnabts/mnar/",
        "archive_verification": "No source.json supplied; loaded-file hashes recorded below",
    }
    source = source | {"loaded_file_sha256": file_hashes,
                       "propensities_used": False,
                       "propensities_note": "Author-estimated propensities are not loaded or treated as truth"}
    protocol = {
        "dataset_protocol": COAT_PROTOCOL,
        "split_seed": split_seed,
        "training": "All nonzero self-selected train.ascii ratings",
        "random_pool": "Nonzero test.ascii ratings excluding every pair observed in training",
        "development_test_split": "Per-user shuffle seeded by '<split_seed>:<user>', floor(n/2) to development",
        "validation_scope": "Random observed pairs for known users; not temporal or new-user validation",
        "missing_rating": "0 means unobserved; never converted to a negative rating",
        "item_id": "Zero-based anonymous matrix column index",
        "rating_scale": [1, 5],
    }
    fingerprint = sha256(json.dumps({"protocol": protocol, "archive_sha256": source.get("sha256"),
                                     "loaded_file_sha256": file_hashes}, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()
    statistics = {
        "users": len(user_ids), "items": len(item_ids),
        "original_train_rows": len(train), "original_random_rows": len(random_ratings),
        "random_train_overlap_removed": len(random_ratings) - len(remaining),
        "random_after_overlap_removal": len(remaining),
        "train_rows": len(train), "development_rows": len(development), "test_rows": len(test),
        "development_users": len({row["user"] for row in development}),
        "test_users": len({row["user"] for row in test}),
    }
    return {"dataset_id": "coat", "train": train, "development": development, "test": test,
            "user_ids": user_ids, "item_ids": item_ids, "item_features": item_features,
            "feature_names": feature_names, "positive_threshold": 4, "gain_floor": 1,
            "protocol": protocol, "statistics": statistics, "source": source,
            "fingerprint": fingerprint}
