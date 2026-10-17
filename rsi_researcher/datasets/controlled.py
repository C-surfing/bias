"""Small paired semantic worlds for workflow/pilot checks, not real-user evidence."""

import argparse
from copy import deepcopy
import random

from rsi_researcher.runtime.client import atomic_json


GENRES = ("jazz", "classical", "folk", "electronic")
WORLD_PROTOCOL = "static_genre_full_support_panel20_v2"
EVALUATION_PANEL_SIZE = 20


def probabilities(items, weights):
    total = sum(weights)
    return {item: weight / total for item, weight in zip(items, weights)}


def weighted_pick(items, weights, draw):
    threshold = draw * sum(weights)
    cumulative = 0
    for item, weight in zip(items, weights):
        cumulative += weight
        if threshold < cumulative:
            return item
    return items[-1]


def build_world(seed, users=4, items=12, events_per_user=2, stratum="unrelated"):
    if users < 1 or items < 12 or events_per_user < 1 or events_per_user >= items:
        raise ValueError("need users, >=12 items, and fewer events per user than items")
    if stratum not in ("unrelated", "aligned"):
        raise ValueError("unknown popularity stratum")
    rng = random.Random(seed)
    catalog = [f"i{i:03d}" for i in range(items)]
    genre_order = [GENRES[i % len(GENRES)] for i in range(items)]
    rng.shuffle(genre_order)
    genres = dict(zip(catalog, genre_order))
    user_ids = [f"u{u:03d}" for u in range(users)]
    utilities = {}
    for user in user_ids:
        tastes = list(range(1, 5))
        rng.shuffle(tastes)
        preference = dict(zip(GENRES, tastes))
        utilities[user] = {item: preference[genres[item]] for item in catalog}
    # A pre-period exposure simulation freezes H before any training interactions.
    counts = dict.fromkeys(catalog, 0)
    weights = [sum(utilities[u][i] for u in user_ids) if stratum == "aligned" else 1 for i in catalog]
    preperiod_q = probabilities(catalog, weights)
    for _ in range(1000):
        counts[weighted_pick(catalog, weights, rng.random())] += 1
    popular = sorted(catalog, key=lambda i: (-counts[i], i))[:max(1, items // 4)]
    initial = {"users": {u: "I enjoy listening to CDs very much." for u in user_ids},
               "items": {i: {"title": f"Collection {i}", "memory": f"This CD contains {genres[i]} music."}
                         for i in catalog}, "completed": []}
    queries = []
    for user in user_ids:
        panel = catalog.copy()
        rng.shuffle(panel)
        queries.append({"query_id": "dev-" + user, "user": user,
                        "candidates": panel[:EVALUATION_PANEL_SIZE]})
    # Common random draws pair exposure arms without forcing identical outcomes.
    draws = [(rng.random(), rng.random()) for _ in range(users * events_per_user)]
    events = {}
    design_q, sampling, dynamic_counts = {}, {}, {}
    for exposure, multiplier in (("low", 1), ("high", 4)):
        exposure_weights = {i: multiplier if i in popular else 1 for i in catalog}
        design_q[exposure] = probabilities(catalog, list(exposure_weights.values()))
        history = {u: set() for u in user_ids}
        stream, trace = [], []
        positive_counts = dict.fromkeys(catalog, 0)
        for step in range(events_per_user):
            for user_index, user in enumerate(user_ids):
                event_index = step * users + user_index
                positive_draw, negative_draw = draws[event_index]
                positive_weights = [exposure_weights[i] * utilities[user][i] for i in catalog]
                positive_q = probabilities(catalog, positive_weights)
                # Consumed CDs remain possible future positives. The comparison CD is unobserved.
                positive = weighted_pick(catalog, positive_weights, positive_draw)
                negative_pool = [i for i in catalog if i not in history[user] and i != positive]
                negative_q = dict.fromkeys(catalog, 0.0)
                negative_q.update(probabilities(negative_pool, [exposure_weights[i] for i in negative_pool]))
                negative = weighted_pick(negative_pool, [exposure_weights[i] for i in negative_pool], negative_draw)
                positive_counts[positive] += 1
                trace.append({"event_id": f"train-{event_index:04d}", "user": user,
                              "history_before": sorted(history[user]),
                              "positive": positive, "negative": negative,
                              "positive_draw": positive_draw, "negative_draw": negative_draw,
                              "positive_q": positive_q, "negative_q": negative_q,
                              "positive_count_after": positive_counts[positive]})
                history[user].add(positive)
                stream.append({"event_id": f"train-{event_index:04d}", "user": user,
                               "positive": positive, "negative": negative,
                               "candidates": [negative, positive]})
        events[exposure] = stream
        sampling[exposure] = trace
        dynamic_counts[exposure] = positive_counts
    return {"public": {"initial": initial, "events": events, "queries": queries,
                       "exposure": {"popular": popular.copy(), "preperiod_counts": counts.copy(),
                                    "design_q": design_q}},
            "private": {"utilities": utilities, "popular": popular, "preperiod_counts": counts,
                        "preperiod_q": preperiod_q, "sampling": sampling,
                        "dynamic_positive_counts": dynamic_counts},
            "protocol": {"world_protocol": WORLD_PROTOCOL, "evaluation_panel_size": min(items, EVALUATION_PANEL_SIZE),
                         "seed": seed, "stratum": stratum, "users": users, "items": items,
                         "events_per_user": events_per_user, "high_exposure_multiplier": 4,
                         "sampling_sources": {
                             "preperiod": "1000 draws; uniform for unrelated, population utility weighted for aligned",
                             "design_q": "normalized frozen-H exposure weights; low=1, high=4 for H and 1 elsewhere",
                             "positive_q": "normalized exposure weight times latent user utility over the full catalog; repeats allowed",
                             "negative_q": "normalized exposure weights over CDs never consumed by this user, excluding current positive; zero elsewhere"},
                         "utility_gain": "linear", "ideal_tie_break": "ascending item ID",
                         "purpose": "search-development workflow/pilot only; no final-test claim"}}


def ordered_events(events, popular, position):
    """Use paired seed runs for pair_0/pair_1; never average away their discrepancy."""
    if position not in ("pair_0", "pair_1", "head_first"):
        raise ValueError("unknown position policy")
    result = deepcopy(events)
    for n, event in enumerate(result):
        if position == "head_first":
            event["candidates"].sort(key=lambda i: (i not in popular, i))
        elif (n + (position == "pair_1")) % 2:
            event["candidates"].reverse()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    from pathlib import Path
    directory = Path(options.output)
    directory.mkdir(parents=True, exist_ok=False)
    world = build_world(options.seed)
    for name, value in world.items():
        atomic_json(directory / (name + ".json"), value)
