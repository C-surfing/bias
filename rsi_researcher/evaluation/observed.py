"""Independent evaluation on explicitly observed recommendation feedback.

Missing ratings never become negative labels.  Every method ranks the same
observed candidate panel for a user.  These are panel ranking measurements,
not estimates of full-catalogue counterfactual NDCG or causal popularity bias.
"""

from collections import defaultdict
import hashlib
import json
import math
import random
import statistics


SCOPE = "observed_candidate_panel_only_not_catalog_counterfactual"
PROTOCOL = "observed_panel_linear_gain_tie_average_head_v1"


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _ideal_head_share(items, gains, head, discounts):
    """Average head exposure across all equally good ideal rankings.

    Breaking equal-relevance ties by item ID could create apparent popularity
    excess from an arbitrary naming choice, so each tied group shares its rank
    discounts uniformly.
    """
    groups = defaultdict(list)
    for item in items:
        groups[gains[item]].append(item)
    position, numerator = 0, 0.0
    for gain in sorted(groups, reverse=True):
        group = groups[gain]
        group_discounts = discounts[position:position + len(group)]
        numerator += sum(group_discounts) * sum(item in head for item in group) / len(group)
        position += len(group)
    return numerator / sum(discounts)


def evaluate_observed(rows, rankings, train_counts, *, k=5, positive_threshold=4,
                      gain_floor=1, head_fraction=.2, catalog_items=None):
    """Score complete permutations of the observed items for each user.

    ``rows`` contain ``user``, ``item``, and ``rating``.  ``rankings`` may use
    candidate IDs but must never be given these held-out ratings by a caller.
    Head membership is frozen from training counts over ``catalog_items``.
    Quality and panel excess use users whose ideal DCG is positive; users with
    no relevant tail item do not enter the tail-recall denominator.  Invalid
    rankings receive quality 0, head share 1, and panel excess 1 where defined.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not all(_number(v) for v in (positive_threshold, gain_floor, head_fraction)):
        raise ValueError("evaluation settings must be finite numbers")
    if positive_threshold <= gain_floor or not 0 < head_fraction <= 1:
        raise ValueError("positive threshold must exceed gain floor and head fraction must be in (0, 1]")
    if not isinstance(rankings, dict) or not isinstance(train_counts, dict):
        raise ValueError("rankings and training counts must be mappings")
    if any(not isinstance(item, str) or not item or type(count) is not int or count < 0
           for item, count in train_counts.items()):
        raise ValueError("training counts need string IDs and nonnegative integer counts")

    panels = defaultdict(dict)
    for row in rows:
        user, item, rating = row["user"], row["item"], row["rating"]
        if not isinstance(user, str) or not user or not isinstance(item, str) or not item:
            raise ValueError("observations need nonempty string user/item IDs")
        if not _number(rating):
            raise ValueError("observed ratings must be finite numbers")
        if item in panels[user]:
            raise ValueError("duplicate user-item observations do not define one panel")
        panels[user][item] = float(rating)
    if not panels:
        raise ValueError("evaluation requires observed panels")
    if set(rankings) - set(panels):
        raise ValueError("unexpected user in rankings")
    observed_items = {item for panel in panels.values() for item in panel}
    supplied_catalog = catalog_items is not None
    catalog = list(catalog_items) if supplied_catalog else sorted(set(train_counts) | observed_items)
    if not catalog or any(not isinstance(item, str) or not item for item in catalog) or len(set(catalog)) != len(catalog):
        raise ValueError("catalog items must be distinct nonempty string IDs")
    if not observed_items <= set(catalog) or not set(train_counts) <= set(catalog):
        raise ValueError("catalog must include observed items and all training-count items")
    counts = {item: train_counts.get(item, 0) for item in sorted(catalog)}
    head_size = max(1, math.ceil(len(catalog) * head_fraction))
    popularity_order = sorted(catalog, key=lambda item: (-counts[item], item))
    head = set(popularity_order[:head_size])
    boundary = counts[popularity_order[head_size - 1]]
    protocol = {"version": PROTOCOL, "scope": SCOPE, "k": k,
                "gain": "max(observed_rating - gain_floor, 0)", "gain_floor": gain_floor,
                "positive_threshold": positive_threshold, "head_fraction": head_fraction,
                "head_items": sorted(head), "catalog_size": len(catalog),
                "catalog_fingerprint": _fingerprint(sorted(catalog)),
                "training_counts_fingerprint": _fingerprint(counts),
                "catalog_was_explicit": supplied_catalog,
                "head_boundary_tie_items": sum(count == boundary for count in counts.values()),
                "head_tie_break": "training_count_descending_then_item_id",
                "ideal_relevance_ties": "expected_head_share_over_all_tied_orderings",
                "unobserved_items": "excluded_not_imputed_as_negatives"}
    per_user = []
    for user in sorted(panels):
        truth = panels[user]
        items = sorted(truth)
        gains = {item: max(truth[item] - gain_floor, 0.0) for item in items}
        cutoff = min(k, len(items))
        discounts = [1 / math.log2(position + 2) for position in range(cutoff)]
        ideal = sorted(items, key=lambda item: (-gains[item], item))
        ideal_dcg = sum(discount * gains[item] for discount, item in zip(discounts, ideal))
        evaluable = ideal_dcg > 0
        ideal_head = _ideal_head_share(items, gains, head, discounts) if evaluable else None
        tail_relevant = {item for item in items if item not in head and truth[item] >= positive_threshold}
        order = rankings.get(user)
        valid = (isinstance(order, list) and all(isinstance(item, str) for item in order)
                 and len(order) == len(items) and set(order) == set(items))
        if valid:
            ndcg = sum(discount * gains[item] for discount, item in zip(discounts, order)) / ideal_dcg if evaluable else None
            head_share = sum(discount for discount, item in zip(discounts, order) if item in head) / sum(discounts)
            signed_excess = head_share - ideal_head if evaluable else None
            excess = max(0.0, signed_excess) if evaluable else None
            recall = len(tail_relevant.intersection(order[:cutoff])) / len(tail_relevant) if tail_relevant else None
        else:
            ndcg, head_share = (0.0 if evaluable else None), 1.0
            signed_excess, excess = (1.0, 1.0) if evaluable else (None, None)
            recall = 0.0 if tail_relevant else None
        per_user.append({"user": user, "valid": valid, "observed_items": len(items),
                         "cutoff": cutoff, "quality_evaluable": evaluable,
                         "panel_fingerprint": _fingerprint(sorted(truth.items())),
                         "ndcg": ndcg, "head_share": head_share,
                         "panel_ideal_head_share": ideal_head,
                         "panel_signed_head_excess": signed_excess,
                         "panel_excess_head_share": excess,
                         "tail_relevant_recall": recall, "tail_relevant_items": len(tail_relevant)})
    quality = [row for row in per_user if row["quality_evaluable"]]
    tail_supported = [row for row in per_user if row["tail_relevant_items"]]
    return {"scope": SCOPE, "protocol": protocol, "evaluated_users": len(per_user),
            "quality_users": len(quality), "zero_ideal_users": len(per_user) - len(quality),
            "tail_supported_users": len(tail_supported), "no_tail_positive_users": len(per_user) - len(tail_supported),
            "observed_ratings": sum(row["observed_items"] for row in per_user),
            "invalid_rankings": sum(not row["valid"] for row in per_user),
            "ndcg": _mean(row["ndcg"] for row in quality),
            "head_share": _mean(row["head_share"] for row in per_user),
            "panel_excess_head_share": _mean(row["panel_excess_head_share"] for row in quality),
            "panel_signed_head_excess": _mean(row["panel_signed_head_excess"] for row in quality),
            "tail_relevant_recall": _mean(row["tail_relevant_recall"] for row in tail_supported),
            "per_user": per_user}


def _quantile(values, probability):
    values = sorted(values)
    position = probability * (len(values) - 1)
    low = math.floor(position)
    fraction = position - low
    return values[low] * (1 - fraction) + values[math.ceil(position)] * fraction


def paired_comparison(baseline, candidate, seed=20261014, bootstrap_samples=500):
    """Keep point gates separate from paired-user uncertainty intervals.

    The bootstrap resamples users, not individual ratings.  The intervals are
    descriptive uncertainty on this panel split, with no correction for trying
    many development candidates and no causal or final-test promotion claim.
    """
    if type(seed) is not int or type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap needs an integer seed and positive sample count")
    if baseline["protocol"] != candidate["protocol"]:
        raise ValueError("paired comparison requires the same evaluation protocol and training counts")
    left = {row["user"]: row for row in baseline["per_user"]}
    right = {row["user"]: row for row in candidate["per_user"]}
    if len(left) != len(baseline["per_user"]) or len(right) != len(candidate["per_user"]) or set(left) != set(right):
        raise ValueError("paired comparison requires the same distinct users")
    for user in left:
        if any(left[user][key] != right[user][key] for key in
               ("panel_fingerprint", "quality_evaluable", "cutoff", "tail_relevant_items")):
            raise ValueError("paired comparison requires the same observed labels and candidate panels")
    users = sorted(user for user, row in left.items() if row["quality_evaluable"])
    differences = {metric: [right[user][metric] - left[user][metric] for user in users]
                   for metric in ("ndcg", "panel_excess_head_share")}
    random_state = random.Random(seed)
    samples = {metric: [] for metric in differences}
    if users:
        for _ in range(bootstrap_samples):
            positions = [random_state.randrange(len(users)) for _ in users]
            for metric, values in differences.items():
                samples[metric].append(statistics.mean(values[position] for position in positions))
    uncertainty = {metric: {"mean_difference": _mean(values),
                           "ci95": [_quantile(samples[metric], .025), _quantile(samples[metric], .975)] if users else None}
                   for metric, values in differences.items()}
    base_quality, new_quality = baseline["ndcg"], candidate["ndcg"]
    base_bias, new_bias = baseline["panel_excess_head_share"], candidate["panel_excess_head_share"]
    quality_gate = new_quality >= .99 * base_quality if base_quality is not None and base_quality > 0 else None
    bias_gate = new_bias <= .90 * base_bias if base_bias is not None and base_bias >= .001 else None
    valid_gate = candidate["invalid_rankings"] == 0 and baseline["invalid_rankings"] == 0
    if not valid_gate or quality_gate is False or bias_gate is False:
        status = "rejected"
    elif quality_gate is None or bias_gate is None:
        status = "inconclusive"
    else:
        status = "point_gates_passed"
    return {"scope": SCOPE, "status": status, "paired_users": len(users),
            "quality_change": new_quality - base_quality if base_quality is not None else None,
            "bias_change": new_bias - base_bias if base_bias is not None else None,
            "head_share_change": candidate["head_share"] - baseline["head_share"],
            "tail_relevant_recall_change": candidate["tail_relevant_recall"] - baseline["tail_relevant_recall"]
            if baseline["tail_relevant_recall"] is not None else None,
            "point_gates": {"quality": quality_gate, "bias": bias_gate, "valid_rankings": valid_gate},
            "quality_floor_ratio": .99, "bias_ceiling_ratio": .90, "minimum_reference_bias": .001,
            "uncertainty": {"method": "paired_user_percentile_bootstrap", "seed": seed,
                            "bootstrap_samples": bootstrap_samples, "metrics": uncertainty,
                            "interpretation": "panel_split_uncertainty_not_corrected_for_candidate_selection"},
            "statistical_promotion": False, "causal_claim": False}
