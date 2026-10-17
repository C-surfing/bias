"""A chronological, observed-feedback panel from the official KuaiRand-Pure."""

from collections import Counter
import csv
from hashlib import sha256
from heapq import heappush, heapreplace
import json
from pathlib import Path


KUAIRAND_PROTOCOL = "kuairand_pure_preperiod_temporal_random_panel_v1"
TRAIN_FILE = "log_standard_4_08_to_4_21_pure.csv"
RANDOM_FILE = "log_random_4_22_to_5_08_pure.csv"
FEATURE_FILE = "video_features_basic_pure.csv"
MAX_TRAIN_EVENTS_PER_USER = 200


def _csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name} lacks required columns: {sorted(set(required) - set(reader.fieldnames or []))}")
        yield from reader


def _order(row, index):
    return int(row["date"]), int(row.get("time_ms") or 0), index


def _label(row):
    rating = float(row["is_click"])
    if rating not in (0, 1):
        raise ValueError("KuaiRand is_click must be 0 or 1; zero is observed feedback")
    return {"user": row["user_id"], "item": row["video_id"], "rating": rating}


def _hash_file(path):
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_kuairand(directory, user_limit=200, split_seed=20261014, training_scope="cohort"):
    """Load a cohort chosen before viewing future feedback.

    Training retains at most 200 latest preperiod events per selected user.
    Random exposures form development (April 22--30) and test (May 1--8)
    panels. Repeated pairs keep their earliest event, never their best label.
    """
    if not isinstance(user_limit, int) or user_limit < 1:
        raise ValueError("user_limit must be a positive integer")
    if training_scope not in ("cohort", "all_preperiod_users"):
        raise ValueError("training_scope must be cohort or all_preperiod_users")
    directory = Path(directory)
    data_dir = next((candidate for candidate in (
        directory, directory / "data", directory / "KuaiRand-Pure" / "data")
        if (candidate / TRAIN_FILE).is_file()), None)
    if data_dir is None:
        raise FileNotFoundError(f"Cannot find official {TRAIN_FILE} below {directory}")
    required = ("user_id", "video_id", "date", "is_click")
    statistics = Counter({key: 0 for key in (
        "train_rows_scanned", "preperiod_rows", "train_outside_protocol_rows", "cohort_preperiod_rows",
        "basic_feature_rows", "random_rows_scanned", "random_outside_cohort_rows", "cohort_random_rows",
        "random_outside_protocol_rows", "random_train_overlap_rows_removed",
        "development_repeat_rows_removed", "test_repeat_rows_removed")})
    eligible_users = set()
    for row in _csv(data_dir / TRAIN_FILE, required):
        statistics["train_rows_scanned"] += 1
        if 20220408 <= int(row["date"]) <= 20220421 and row.get("is_rand", "0") == "0":
            eligible_users.add(row["user_id"])
            statistics["preperiod_rows"] += 1
        else:
            statistics["train_outside_protocol_rows"] += 1
    user_ids = sorted(eligible_users, key=lambda user: (
        sha256(f"{split_seed}:{user}".encode("utf-8")).hexdigest(), user))[:user_limit]
    selected = set(user_ids)
    training_users = eligible_users if training_scope == "all_preperiod_users" else selected
    histories = {user: [] for user in sorted(training_users)}
    train_seen = set()
    training_pool_rows = 0
    for index, row in enumerate(_csv(data_dir / TRAIN_FILE, required)):
        if row["user_id"] not in training_users or not 20220408 <= int(row["date"]) <= 20220421:
            continue
        if row.get("is_rand", "0") != "0":
            continue
        training_pool_rows += 1
        if row["user_id"] in selected:
            statistics["cohort_preperiod_rows"] += 1
            train_seen.add((row["user_id"], row["video_id"]))
        event = (_order(row, index), _label(row))
        heap = histories[row["user_id"]]
        if len(heap) < MAX_TRAIN_EVENTS_PER_USER:
            heappush(heap, event)
        elif event[0] > heap[0][0]:
            heapreplace(heap, event)
    train = [event[1] for event in sorted(event for heap in histories.values() for event in heap)]

    # Basic metadata has no interaction outcomes. Dynamic visibility and the
    # separate monthly statistics table are intentionally not recommendation features.
    feature_fields = {"author_id", "video_type", "upload_dt", "upload_type", "video_duration",
                      "server_width", "server_height", "music_id", "music_type", "tag"}
    item_features, catalog = {}, set()
    for row in _csv(data_dir / FEATURE_FILE, ("video_id",)):
        statistics["basic_feature_rows"] += 1
        item = row["video_id"]
        catalog.add(item)
        item_features[item] = {key: value for key, value in row.items() if key in feature_fields}
    development_pairs, test_pairs = {}, {}
    for index, row in enumerate(_csv(data_dir / RANDOM_FILE, required)):
        statistics["random_rows_scanned"] += 1
        catalog.add(row["video_id"])
        if row["user_id"] not in selected:
            statistics["random_outside_cohort_rows"] += 1
            continue
        statistics["cohort_random_rows"] += 1
        date = int(row["date"])
        if not 20220422 <= date <= 20220508 or row.get("is_rand", "1") != "1":
            statistics["random_outside_protocol_rows"] += 1
            continue
        pair = row["user_id"], row["video_id"]
        if pair in train_seen:
            statistics["random_train_overlap_rows_removed"] += 1
            continue
        panel = development_pairs if date <= 20220430 else test_pairs
        key = "development" if date <= 20220430 else "test"
        event = (_order(row, index), _label(row))
        if pair in panel:
            statistics[f"{key}_repeat_rows_removed"] += 1
        if pair not in panel or event[0] < panel[pair][0]:
            panel[pair] = event
    cross_pairs = set(development_pairs) & set(test_pairs)
    for pair in cross_pairs:
        del test_pairs[pair]
    development = [event[1] for event in sorted(development_pairs.values())]
    test = [event[1] for event in sorted(test_pairs.values())]
    # The catalog is metadata + random-pool IDs, never a selection by test label.
    missing_training_ids = {row["item"] for row in train} - catalog
    if missing_training_ids:
        raise ValueError("Training contains item IDs absent from basic metadata and random pool")
    item_ids = sorted(catalog)
    for item in item_ids:
        item_features.setdefault(item, {})
    statistics.update({
        "eligible_training_users": len(eligible_users), "users": len(user_ids), "items": len(item_ids),
        "training_users": len(training_users), "training_pool_preperiod_rows": training_pool_rows,
        "train_history_cap_rows_removed": training_pool_rows - len(train),
        "test_development_overlap_pairs_removed": len(cross_pairs),
        "train_rows": len(train), "development_rows": len(development), "test_rows": len(test),
        "development_users": len({row["user"] for row in development}),
        "test_users": len({row["user"] for row in test}),
        "development_users_with_positive": len({row["user"] for row in development if row["rating"] == 1}),
        "test_users_with_positive": len({row["user"] for row in test if row["rating"] == 1}),
        "train_observed_zero_rows": sum(row["rating"] == 0 for row in train),
        "development_observed_zero_rows": sum(row["rating"] == 0 for row in development),
        "test_observed_zero_rows": sum(row["rating"] == 0 for row in test),
    })
    source_path = next((path for path in (directory / "source.json", data_dir.parent / "source.json",
                                          data_dir.parent.parent / "source.json") if path.is_file()), None)
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path else {
        "dataset": "KuaiRand-Pure", "official_page": "https://kuairand.com/",
        "download_url": "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz",
    }
    file_hashes = {name: _hash_file(data_dir / name) for name in (TRAIN_FILE, RANDOM_FILE, FEATURE_FILE)}
    source = source | {"loaded_file_sha256": file_hashes, "video_statistics_used": False,
                       "user_feature_snapshot_used": False,
                       "basic_features_note": "Static metadata only; snapshot availability at the training cutoff is not verified"}
    protocol = {
        "dataset_protocol": KUAIRAND_PROTOCOL, "split_seed": split_seed, "user_limit": user_limit,
        "cohort": "Training-user IDs ordered by sha256('<seed>:<user>'); no future-label eligibility check",
        "training_dates": [20220408, 20220421], "development_dates": [20220422, 20220430],
        "test_dates": [20220501, 20220508], "max_training_events_per_user": MAX_TRAIN_EVENTS_PER_USER,
        "training_history": "Latest preperiod events by date,time_ms,file row; positive and zero feedback retained",
        "unseen_pair_filter": "Exclude pairs from the cohort's full preperiod, including events outside retained history",
        "repeated_random_pairs": "Earliest event per partition by date,time_ms,file row; remove test pairs seen in development",
        "rating": "Native is_click: 0 is exposed non-click/non-valid-play, 1 is positive; UI semantics vary",
        "ranking_scope": "Within randomly exposed observed-item panels; not complete catalog preference truth",
        "popularity_scope": "Only the retained at-most-200 training events per cohort user",
        "catalog": "All basic metadata IDs plus random-pool IDs, without inspecting feedback for membership",
        "sequence_limit": "Pure filters ordinary logs to the random pool and does not preserve complete sequences",
        "metadata_limit": "Basic static snapshot retained for descriptions; availability at cutoff is unverified",
    }
    if training_scope != "cohort":
        protocol.update(dataset_protocol="kuairand_pure_all_preperiod_training_temporal_panel_v2",
            training_scope=training_scope,
            training_history="Latest at-most-200 preperiod events for every training-period user; evaluation cohort stays fixed",
            popularity_scope="Retained preperiod exposure events from all eligible training users, never future random feedback")
    fingerprint = sha256(json.dumps({"protocol": protocol, "archive_sha256": source.get("sha256"),
                                     "loaded_file_sha256": file_hashes}, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"dataset_id": "kuairand-pure", "train": train, "development": development, "test": test,
            "item_ids": item_ids, "user_ids": user_ids, "item_features": item_features,
            "feature_names": sorted(feature_fields), "positive_threshold": 1, "gain_floor": 0,
            "protocol": protocol, "statistics": dict(statistics), "source": source, "fingerprint": fingerprint}
