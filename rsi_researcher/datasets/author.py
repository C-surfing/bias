"""Load the pinned author CD input for engineering checks."""
from collections import Counter
import csv
from pathlib import Path
import random

AUTHOR_DATA_PROTOCOL = "cds_basic_9_comparisons_plus_target_user_round_v2"

def author_data(directory, user_count, events_per_user, seed=20261013, selection="first", excluded_users=None):
    directory = Path(directory)
    with (directory / "CDs.item").open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))[1:]
    items = {}
    for row in rows:
        if len(row) not in (2, 3) or row[0] in items:
            raise ValueError("invalid or duplicate item metadata")
        item_id, title = row[:2]
        category = row[2] if len(row) == 3 else "CDs"
        items[item_id] = {"title": title, "memory": f"The CD is called '{title}'. The category of this CD is: '{category}'."}
    with (directory / "CDs-100-user-dense.train.inter").open(encoding="utf-8") as stream:
        train = list(csv.DictReader(stream, delimiter="\t"))
    users = select_users(train, user_count, seed, selection, excluded_users=excluded_users)
    by_user = {user: [row for row in train if row["user_id:token"] == user][:events_per_user] for user in users}
    if len(users) != user_count or any(len(rows) != events_per_user for rows in by_user.values()):
        raise ValueError("requested smoke size exceeds available train data")
    counts = Counter(row["item_id:token"] for row in train)
    catalog = sorted(counts)
    rng = random.Random(seed)
    events = []
    # Author benchmark uses user-round order, not a verified global timestamp.
    for step in range(events_per_user):
        for user in users:
            row = by_user[user][step]
            positive = row["item_id:token"]
            known = set(row["item_id_list:token_seq"].split()) | {positive}
            pool = [item for item in catalog if item not in known]
            negative = rng.choices(pool, weights=[counts[item] for item in pool], k=1)[0]
            events.append({"event_id": f"train-{len(events):04d}", "user": user,
                           "positive": positive, "negative": negative, "candidates": [negative, positive]})
    with (directory / "CDs-100-user-dense.random").open(encoding="utf-8") as stream:
        panels = {user: values.split() for user, values in csv.reader(stream, delimiter="\t")}
    with (directory / "CDs-100-user-dense.valid.inter").open(encoding="utf-8") as stream:
        valid = list(csv.DictReader(stream, delimiter="\t"))
    queries = []
    for row in valid:
        user, target = row["user_id:token"], row["item_id:token"]
        if user not in users:
            continue
        negatives = list(dict.fromkeys(item for item in panels[user] if item != target))[:9]
        candidates = negatives + [target]
        rng.shuffle(candidates)
        if len(candidates) != 10:
            raise ValueError("candidate panel does not contain 10 unique items")
        queries.append({"query_id": "valid-" + user, "user": user, "candidates": candidates, "target": target})
    needed = {i for event in events for i in event["candidates"]} | {i for q in queries for i in q["candidates"]}
    if not needed <= items.keys():
        raise ValueError("item metadata is incomplete")
    return {"users": {u: "I enjoy listening to CDs very much." for u in users},
            "items": {i: items[i] for i in sorted(needed)}, "completed": []}, events, queries


def select_users(train, count, seed, selection="first", excluded_users=None):
    """Freeze a cohort using training support only, before reading validation."""
    ordered = list(dict.fromkeys(row["user_id:token"] for row in train))
    if not 0 < count <= len(ordered):
        raise ValueError("requested user count exceeds author training data")
    if selection == "exclude_known":
        if excluded_users is None:
            raise ValueError("Explicit exclusion list required")
        eligible=[u for u in ordered if u not in set(excluded_users)]
        if len(eligible)<count:
            raise ValueError("Not enough disjoint author users")
        return eligible[:count]
    if selection == "first":
        return ordered[:count]
    if selection == "after_first_20":
        # Fresh training-only cohort, disjoint from previous first-20 AgentCF
        # development run; selection does not depend on validation or test.
        if count > len(ordered) - 20:
            raise ValueError("insufficient users beyond old cohort")
        return ordered[20:20 + count]
    if selection != "training_popularity_quartiles":
        raise ValueError("unknown author user selection")
    counts = Counter(row["item_id:token"] for row in train)
    support = {u: [] for u in ordered}
    for row in train:
        support[row["user_id:token"]].append(counts[row["item_id:token"]])
    ranked = sorted(ordered, key=lambda u: (sum(support[u]) / len(support[u]), u))
    bands = [ranked[len(ranked)*i//4:len(ranked)*(i+1)//4] for i in range(4)]
    rng = random.Random(seed)
    chosen = set()
    for i, band in enumerate(bands):
        chosen.update(rng.sample(band, count//4 + (i < count % 4)))
    return [u for u in ordered if u in chosen]


def author_profile(directory):
    """Describe training support and candidate pools; never read test labels."""
    directory = Path(directory)
    with (directory / "CDs-100-user-dense.train.inter").open(encoding="utf-8") as stream:
        train = list(csv.DictReader(stream, delimiter="\t"))
    counts = Counter(row["item_id:token"] for row in train)
    users = Counter(row["user_id:token"] for row in train)
    frequencies = sorted(counts.values(), reverse=True)
    head_min = frequencies[max(0, (len(frequencies) + 4) // 5 - 1)]
    tail_max = frequencies[min(len(frequencies) - 1, len(frequencies) * 4 // 5)]
    histories = {user: set() for user in users}
    for row in train:
        histories[row["user_id:token"]].update(row["item_id_list:token_seq"].split() + [row["item_id:token"]])
    with (directory / "CDs-100-user-dense.random").open(encoding="utf-8") as stream:
        pool_sizes = Counter(len(values.split()) for _, values in csv.reader(stream, delimiter="\t"))
    with (directory / "CDs.item").open(encoding="utf-8") as stream:
        catalog_items = sum(1 for _ in stream) - 1
    groups = Counter(popularity_group(item, counts, head_min, tail_max) for item in counts)
    return {"dataset": "CDs-100-user-dense", "training_users": len(users),
            "training_events": len(train), "training_items": len(counts), "metadata_items": catalog_items,
            "events_per_user": {"min": min(users.values()), "max": max(users.values())},
            "distinct_observed_items_per_user": {"min": min(map(len, histories.values())), "max": max(map(len, histories.values()))},
            "candidate_pool_sizes": dict(pool_sizes), "ranking_candidates": 10,
            "training_item_counts": dict(counts), "head_min_count": head_min, "tail_max_count": tail_max,
            "training_group_sizes": dict(groups),
            "group_definition": "Train target-event frequency only: head >= top-20%-item boundary; tail <= bottom-20%-item boundary; ties kept together and head has priority; other observed items mid; count 0 unseen.",
            "feedback_scope": ".random contains candidate IDs, not exposures or ratings; one known validation target per user; other relevance unknown.",
            "implementation_scope": "Qwen-compatible semantic-memory implementation; serial user-round training, Python shuffle, JSON IDs; not numerical reproduction of the author's batched GPT run."}


def popularity_group(item, counts, head_min, tail_max):
    count = counts.get(item, 0)
    if count == 0:
        return "unseen"
    if count >= head_min:
        return "head"
    return "tail" if count <= tail_max else "mid"


