"""A reproducible *model-level* feedback-write intervention on near-full KuaiRec.

This is an external-relevance research environment, not a causal claim about
KuaiRec's production ranking policy. Development/confirmation users are disjoint;
the actual user's held-out item feedback is never used to create the recommender.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from statistics import mean

DEFAULT_DATA = "data/external/kuairec/KuaiRec.zip"
WINDOW = 45
RECENT_WRITES = 12
PROFILE_STRENGTH = 0.48
K = 10
SEED = 20261026


def bucket(uid: int) -> str:
    key = hashlib.sha256(f"discovery-state-v1:{uid}".encode()).digest()
    b = int.from_bytes(key[:4], "big") % 10
    return "train" if b < 7 else "development" if b < 9 else "confirmation"


def reference_panel(uid: int, vid: int) -> bool:
    # Item-level split within the evaluation user's nearly complete feedback.
    key = hashlib.blake2b(f"panel-v1:{uid}:{vid}".encode(), digest_size=4).digest()
    return int.from_bytes(key, "big") % 5 == 0


def read_data(path: str, phase: str):
    if phase not in ("development", "confirmation"):
        raise ValueError("Unknown phase")
    categories = {}
    item_positive = Counter()
    item_total = Counter()
    history = defaultdict(list)
    heldout = defaultdict(list)
    with zipfile.ZipFile(path) as z:
        with z.open("KuaiRec 2.0/data/item_categories.csv") as raw:
            for r in csv.DictReader(io.TextIOWrapper(raw)):
                categories[int(r["video_id"])] = tuple(ast.literal_eval(r["feat"]))
        with z.open("KuaiRec 2.0/data/small_matrix.csv") as raw:
            cohort = {}
            for r in csv.DictReader(io.TextIOWrapper(raw)):
                uid, vid = int(r["user_id"]), int(r["video_id"])
                group = cohort.setdefault(uid, bucket(uid))
                relevant = int(float(r["watch_ratio"]) >= 1)
                if group == "train":
                    item_total[vid] += 1
                    item_positive[vid] += relevant
                elif group == phase and vid in categories:
                    if reference_panel(uid, vid):
                        heldout[uid].append((vid, relevant))
                    else:
                        # Some official rows have missing timestamp values.
                        # Place these at the start, not at an invented recent time.
                        stamp = float(r["timestamp"]) if r["timestamp"].strip() else 0.0
                        history[uid].append((stamp, vid, relevant))
    # Frozen item prior uses training users only.
    priors = {vid: (item_positive[vid] + 8) / (n + 16)
              for vid, n in item_total.items()}
    return categories, priors, history, heldout, Counter(cohort.values())


def profile(events, categories):
    counts, sums = Counter(), Counter()
    for _, vid, feedback in events[-WINDOW:]:
        for tag in categories.get(vid, ()):
            counts[tag] += 1
            sums[tag] += 2 * feedback - 1
    return {tag: value / (counts[tag] + 6) for tag, value in sums.items()}


def rank(candidate_labels, state_events, categories, priors):
    state = profile(state_events, categories)
    scores = []
    for vid, label in candidate_labels:
        if vid not in priors:
            continue
        tags = categories.get(vid, ())
        user_signal = (sum(state.get(t, 0) for t in tags) / len(tags)) if tags else 0
        scores.append((priors[vid] + PROFILE_STRENGTH * user_signal, vid, label))
    scores.sort(key=lambda record: (-record[0], record[1]))
    return scores[:K], sum(label for _, label in candidate_labels)


def ndcg(top, positive):
    if positive <= 0:
        return None
    actual = sum(label / math.log2(position + 2)
                 for position, (_, _, label) in enumerate(top))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(K, positive)))
    return actual / ideal


def ci_paired(differences, seed):
    rng = Random(seed)
    estimates = sorted(mean(rng.choices(differences, k=len(differences)))
                       for _ in range(400))
    return [estimates[10], estimates[389]]


def evaluate(zip_path: str, phase: str):
    categories, priors, histories, panels, counts = read_data(zip_path, phase)
    per_user = []
    for uid in sorted(panels):
        history = sorted(histories[uid])
        candidates = panels[uid]
        if len(history) < WINDOW + RECENT_WRITES + 5 or len(candidates) < K:
            continue
        older, recent = history[:-RECENT_WRITES], history[-RECENT_WRITES:]
        # Intervention modifies what the executable recommender receives.
        # The held-out candidates and their genuine feedback are held constant.
        stale, positive = rank(candidates, older, categories, priors)
        restored, _ = rank(candidates, older + recent, categories, priors)
        # Same-count historical deletion outside the model's bounded window.
        # A zero effect is a prespecified negative control.
        placebo, _ = rank(candidates, older[RECENT_WRITES:] + recent,
                          categories, priors)
        sa, ra, pa = (ndcg(top, positive) for top in (stale, restored, placebo))
        if any(value is None for value in (sa, ra, pa)):
            continue
        per_user.append({
            "uid": uid, "reference_candidates": len(candidates),
            "reference_positive": positive,
            "ndcg_stale": sa, "ndcg_restored": ra,
            "gain": ra-sa, "placebo_gain": ra-pa,
            "ranking_changed": [item[1] for item in stale] !=
                               [item[1] for item in restored]})
    if len(per_user) < 12:
        raise ValueError("Too few evaluable independent users")
    effects = [item["gain"] for item in per_user]
    placebo = [item["placebo_gain"] for item in per_user]
    return {
        "source": str(zip_path), "phase": phase,
        "protocol": {
            "mechanism": "controlled feedback-write omission in an explicitly implemented category-memory recommender",
            "intervention": f"replay versus omit final {RECENT_WRITES} feedback writes",
            "negative_control": f"remove {RECENT_WRITES} old writes outside the {WINDOW}-event rolling state",
            "candidate_split": "stable item hash; holdout labels never used in state or training",
            "train_and_test_users_disjoint": True,
            "model_window": WINDOW, "rank_k": K, "profile_strength": PROFILE_STRENGTH,
            "seed": SEED, "utility_reference": "KuaiRec near-full observed watch_ratio>=1",
            "causal_identification": "paired intervention in OUR recommender only",
            "platform_algorithm_bias_identified": False,
            "real_world_novel_bias_confirmed": False},
        "user_folds": dict(counts),
        "evaluated_users": len(per_user),
        "model_ndcg_stale": mean(x["ndcg_stale"] for x in per_user),
        "model_ndcg_restored": mean(x["ndcg_restored"] for x in per_user),
        "paired_ndcg_gain": mean(effects),
        "gain_user_bootstrap_ci95": ci_paired(effects, SEED + (phase == "confirmation")),
        "fraction_rankings_changed": mean(x["ranking_changed"] for x in per_user),
        "placebo_gain": mean(placebo),
        "placebo_nonzero_users": sum(abs(v) > 1e-12 for v in placebo),
        "per_user": per_user,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.data, args.phase)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_user"}, indent=2))


if __name__ == "__main__":
    main()
