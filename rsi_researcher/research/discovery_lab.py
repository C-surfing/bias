"""Black-box recommendation laboratory and private evaluator.

Truth is held in a separate private file and is never embedded in public
observations, prompts, journal records, filenames, or benchmark IDs.
All score differences arise from actually rerunning deterministic ranking.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from random import Random
from statistics import mean

FIELDS = ("device", "context", "slot", "history")
CONDITIONS = ("all", "segment_0", "segment_1")
TAG_IDS = (0, 1)

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

def make_private(seed, destination):
    rng = Random(seed)
    recipes = ["single", "interaction", "dynamic", "null"]
    rng.shuffle(recipes)
    private = {"protocol": "discovery-lab-v2", "seed": seed, "cases": {}}
    for i, recipe in enumerate(recipes):
        cfg = {"type": recipe, "factor": None, "segment": "all",
               "tag": rng.randint(0, 1), "direction": rng.choice([-1, 1]),
               "amplitude": 0.0}
        if recipe == "single":
            cfg.update(factor=rng.choice(["device", "slot"]),
                       amplitude=0.36)
        elif recipe == "interaction":
            cfg.update(factor=rng.choice(["device", "context"]),
                       segment="segment_1", amplitude=0.43)
        elif recipe == "dynamic":
            cfg.update(factor="history", segment="segment_1", amplitude=0.38)
        else:
            cfg.update(direction=0)
        private["cases"][f"task-{i:02d}"] = cfg
    save(destination, private)
    Path(destination).chmod(0o600)
    return {"protocol": private["protocol"], "tasks": list(private["cases"]),
            "field_descriptions": {
                "segment": "User segment (0 or 1).",
                "device": "Client context flag, 0 or 1.",
                "context": "Request-context flag, 0 or 1.",
                "slot": "Presentation slot configuration, 0 or 1.",
                "history": "Number of preceding exposure events (0 or 2).",
                "item_tag": "Observable item category, 0 or 1.",
                "exposure_at_3": "Fraction of top-three ranked items with requested tag."},
            "interventions": list(FIELDS), "conditions": list(CONDITIONS),
            "research_question": "Discover invariant-violating systematic ranking effects and their scope; null findings are valid."}

def _unit(seed, uid, index):
    raw = f"{seed}:{uid}:{index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / (2**64)

def simulate(cfg, seed, uid, override=None):
    # One user, six fixed candidates, fixed relevance and random realization.
    segment = uid % 2
    device = int(_unit(seed, uid, 1) > 0.5)
    context = int(_unit(seed, uid, 2) > 0.5)
    slot = int(_unit(seed, uid, 3) > 0.5)
    history = 2 * int(_unit(seed, uid, 4) > 0.5)
    attrs = dict(segment=segment, device=device, context=context,
                 slot=slot, history=history)
    attrs.update(override or {})
    affinity = int(_unit(seed, uid, 5) > 0.5)
    ranked = []
    for item in range(6):
        tag = item % 2
        # Stable item preferences and perturbation are identical in all paired arms.
        score = 0.60 + (0.11 if affinity == tag else 0.0)
        score += 0.035 * (_unit(seed, uid, 10 + item) - 0.5)
        score += 0.032 * (item // 2 - 1)
        active = False
        if cfg["type"] == "single":
            active = attrs[cfg["factor"]] == 1
        elif cfg["type"] == "interaction":
            active = attrs["segment"] == 1 and attrs[cfg["factor"]] == 1
        elif cfg["type"] == "dynamic":
            # Replay a two-event exposure history, updating a user-specific state.
            # Historical top-1 choices come from the same frozen recommendation
            # candidates under the pre-update preference model.
            remembered = 0
            for event in range(attrs["history"]):
                previous_scores = []
                for past_item in range(6):
                    past_tag = past_item % 2
                    previous_score = 0.60 + (0.11 if affinity == past_tag else 0.0)
                    previous_score += 0.035 * (_unit(seed, uid, 10 + past_item) - 0.5)
                    previous_score += 0.032 * (past_item // 2 - 1)
                    # Earlier events have genuine exploration fluctuations:
                    # randomized preference perturbations induce exposure
                    # beyond a deterministic always-repeat-top-item loop.
                    previous_score += 0.46 * (
                        _unit(seed, uid, 100 + event * 6 + past_item) - 0.5)
                    if past_tag == cfg["tag"]:
                        previous_score += cfg["direction"] * cfg["amplitude"] * (remembered / 2)
                    previous_scores.append((previous_score, past_item))
                previous_top = max(previous_scores, key=lambda x: (x[0], -x[1]))[1]
                remembered += int(previous_top % 2 == cfg["tag"])
            if attrs["segment"] == 1 and tag == cfg["tag"]:
                score += cfg["direction"] * cfg["amplitude"] * remembered / 2
        if active and tag == cfg["tag"]:
            score += cfg["direction"] * cfg["amplitude"]
        ranked.append((score, item, tag))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return {"segment": segment, "device": attrs["device"], "context": attrs["context"],
            "slot": attrs["slot"], "history": attrs["history"],
            "top3_tags": [item[2] for item in ranked[:3]],
            "top1_tag": ranked[0][2]}

def _subset(uid, condition):
    return condition == "all" or uid % 2 == int(condition[-1])

def _fraction(row, tag):
    return sum(x == tag for x in row["top3_tags"]) / 3

def paired(cfg, seed, factor, condition, tag, n=384):
    if factor not in FIELDS or condition not in CONDITIONS or tag not in TAG_IDS:
        raise ValueError("unregistered experimental action")
    if n < 48 or n > 4096:
        raise ValueError("invalid cohort size")
    # Distinct condition-local participants; uid is randomized unit.
    uids = [uid for uid in range(n) if _subset(uid, condition)]
    v1 = 2 if factor == "history" else 1
    diffs, change_top1, changed_setting = [], [], []
    utility0, utility1, utility_diffs = [], [], []
    for uid in uids:
        a = simulate(cfg, seed, uid, {factor: 0})
        b = simulate(cfg, seed, uid, {factor: v1})
        diffs.append(_fraction(b, tag) - _fraction(a, tag))
        change_top1.append(int(a["top1_tag"] != b["top1_tag"]))
        changed_setting.append(int(a[factor] != b[factor]))
        # Evaluator-side relevance standard: actual synthetic user preference,
        # not the model's stated preference or the intervention target tag.
        relevant_tag=int(_unit(seed, uid, 5) > .5)
        ua=_fraction(a, relevant_tag)
        ub=_fraction(b, relevant_tag)
        utility0.append(ua)
        utility1.append(ub)
        utility_diffs.append(ub-ua)
    effect = mean(diffs)
    # User-level paired normal CI for development; not sequential Type-I control.
    variance = sum((v-effect)**2 for v in diffs) / (len(diffs)-1)
    halfwidth = 1.96 * math.sqrt(variance / len(diffs))
    du=mean(utility_diffs)
    utility_var=sum((d-du)**2 for d in utility_diffs)/(len(uids)-1)
    hu=1.96*math.sqrt(utility_var/len(uids))
    return {"intervention": {"factor": factor, "levels": [0, v1],
                              "condition": condition, "item_tag": tag},
            "estimand": "paired difference in tag exposure@3, level1 - level0",
            "effect": round(effect, 5),
            "ci95": [round(effect-halfwidth, 5), round(effect+halfwidth, 5)],
            "n_randomization_units": len(uids),
            "reference_utility": {
                "measure":"fraction of top-three candidates matching frozen synthetic user affinity",
                "control":round(mean(utility0),5),
                "treatment":round(mean(utility1),5),
                "treatment_minus_control":round(du,5),
                "paired_user_ci95":[round(du-hu,5),round(du+hu,5)],
                "utility_harm_detected":bool(du+hu<0),
                "debias_candidate":"remove only the irrelevant setting effect, then rerank and independently test utility",
                "caveat":"Simulator has independent underlying relevance; real logs need external relevance evidence"},
            "top1_changed_fraction": round(mean(change_top1), 5),
            "manipulation_check": {
                "registered_setting_changed_fraction": round(mean(changed_setting), 5),
                "same_users_candidates_and_random_variates": True,
                "top1_changed_fraction_is_an_outcome_not_a_manipulation_check": True},
            "causal_scope": "controlled simulator intervention only",
            "inference": "exploratory paired user-level CI; no sequential-error guarantee"}

def observe(cfg, seed, n=384):
    rows = [simulate(cfg, seed, uid) for uid in range(n)]
    cells = []
    for factor in FIELDS:
        levels = (0, 2) if factor == "history" else (0, 1)
        for segment in (0, 1):
            for level in levels:
                subset = [r for r in rows if r["segment"] == segment and r[factor] == level]
                cells.append({"field": factor, "segment": segment,
                              "value": level, "n": len(subset),
                              "tag1_exposure_at_3": round(mean(_fraction(r, 1) for r in subset), 4)})
    # Model-independent evidence cards: summarize observable heterogeneity,
    # without defining Bias types or revealing hidden mechanisms.
    cards = []
    for factor in FIELDS:
        for segment in (0, 1):
            levels = (0, 2) if factor == "history" else (0, 1)
            values = [[_fraction(r, 1) for r in rows
                       if r["segment"] == segment and r[factor] == level]
                      for level in levels]
            x, y = values
            mx, my = mean(x), mean(y)
            vx = sum((v-mx)**2 for v in x)/(len(x)-1)
            vy = sum((v-my)**2 for v in y)/(len(y)-1)
            delta = my-mx
            se = math.sqrt(vx/len(x)+vy/len(y))
            cards.append({
                "field":factor, "user_segment":segment,
                "baseline_level":levels[0], "comparison_level":levels[1],
                "n_by_level":[len(x),len(y)],
                "tag1_exposure_at_3_by_level":[round(mx,5),round(my,5)],
                "observed_difference_tag1":round(delta,5),
                "descriptive_interval95":[round(delta-1.96*se,5),round(delta+1.96*se,5)],
                "meaning":"observational association, not a causal effect"})
    return {"observation_type": "non_interventional_group_summaries",
            "n_users": n, "candidate_items_per_user": 6, "fixed_candidate_set": True,
            "cells": cells, "contrast_cards": cards, "sample": rows[:6],
            "note": "Observational contrasts are confounded; use paired interventions to identify effects."}

def query(secret_path, task, action, split="development", factor=None,
          condition="all", tag=1, n=384):
    cfgs = json.loads(Path(secret_path).read_text(encoding="utf8"))
    cfg = cfgs["cases"][task]
    seed = cfgs["seed"] + (29 if split == "development" else 32771)
    if split not in ("development", "confirmation"):
        raise ValueError("unknown split")
    if action == "observe":
        return observe(cfg, seed, n)
    if action == "contrast":
        return paired(cfg, seed, factor, condition, tag, n)
    raise ValueError("unknown research action")

def evaluate(secret_path, submissions):
    private = json.loads(Path(secret_path).read_text(encoding="utf8"))
    graded = {}
    for task, cfg in private["cases"].items():
        claim = submissions.get(task, {})
        accepted = claim.get("status") in ("supported", "confirmed")
        nonnull = cfg["type"] != "null"
        # A binary ranking is invariant to the representation
        # "boost tag 0" <-> "suppress tag 1". Absolute additive score
        # shifts cannot be identified from relative rankings alone.
        # Grade the identifiable relative contrast, not its gauge.
        predicted_sign = 1 if claim.get("direction") == "increase" else (
            -1 if claim.get("direction") == "decrease" else 0)
        same_tag = claim.get("tag") == cfg["tag"]
        equivalent_tag = claim.get("tag") in (0, 1)
        correct_sign = (predicted_sign == cfg["direction"] *
                        (1 if same_tag else -1)) if equivalent_tag else False
        factor_correct = nonnull and claim.get("factor") == cfg["factor"]
        scope_correct = nonnull and claim.get("condition") == cfg["segment"]
        raw_match = nonnull and factor_correct and scope_correct and correct_sign
        correct = accepted and raw_match
        graded[task] = {"true_kind": cfg["type"], "valid_discovery": bool(correct),
                        "structural_hypothesis_match": bool(raw_match),
                        "false_discovery": bool(accepted and not correct),
                        "missed_discovery": bool(nonnull and not correct),
                        "correct_abstention": bool(not nonnull and not accepted),
                        "factor_correct": bool(factor_correct),
                        "scope_correct": bool(scope_correct),
                        "relative_direction_correct": bool(nonnull and correct_sign),
                        "binary_tag_gauge_equivalent": True}
    tp = sum(x["valid_discovery"] for x in graded.values())
    fp = sum(x["false_discovery"] for x in graded.values())
    n_positive = sum(x["true_kind"] != "null" for x in graded.values())
    return {"mechanism_recall": tp / n_positive,
            "raw_hypothesis_recall": sum(x["structural_hypothesis_match"] for x in graded.values()) / n_positive,
            "factor_identification_accuracy": sum(x["factor_correct"] for x in graded.values()) / n_positive,
            "scope_identification_accuracy": sum(x["scope_correct"] for x in graded.values()) / n_positive,
            "relative_direction_accuracy": sum(x["relative_direction_correct"] for x in graded.values()) / n_positive,
            "null_rejection_accuracy": sum(x["correct_abstention"] for x in graded.values() if x["true_kind"] == "null"),
            "valid_discovery_precision": tp/(tp+fp) if tp+fp else None,
            "false_discovery_proportion": fp/(tp+fp) if tp+fp else 0.0,
            "valid_discoveries": tp, "false_discoveries": fp,
            "cases": graded}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["make", "query", "grade"])
    p.add_argument("--private", required=True)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--task")
    p.add_argument("--action")
    p.add_argument("--split", default="development")
    p.add_argument("--factor")
    p.add_argument("--condition", default="all")
    p.add_argument("--tag", type=int, default=1)
    p.add_argument("--n", type=int, default=384)
    p.add_argument("--submissions")
    args = p.parse_args()
    if args.mode == "make":
        result = make_private(args.seed, args.private)
    elif args.mode == "query":
        result = query(args.private, args.task, args.action, args.split,
                       args.factor, args.condition, args.tag, args.n)
    else:
        result = evaluate(args.private, json.loads(Path(args.submissions).read_text()))
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
