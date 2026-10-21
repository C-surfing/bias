"""Frozen official LLMRank candidate-file adapter.

Reconstructs SelectedUserTrainer.has_gt=True's top-(recall_budget-1),
append ground truth and shuffle (fix_pos=-1). Not full RecBole reproduction:
dataset-specific evaluation targets/histories must also match official splits.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import random

def source_candidates(path):
    result = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            user, *items = line.strip().split("\t", 1)
            if len(items) != 1:
                continue
            result[user] = items[0].split()
    return result

def seeded(seed, user, split):
    return int.from_bytes(hashlib.sha256(f"{seed}/{user}/{split}".encode()).digest()[:8],"big")

def cases_from_source(dataset, split, selected_users, train, positives, titles,
                      source_file, seed, recall_budget=20, hist_len=50,
                      strict_candidate_size=True):
    if recall_budget < 2:
        raise ValueError("recall_budget must be at least 2")
    source = source_candidates(source_file)
    cases = []
    skipped = {}
    for user in selected_users:
        if user not in source:
            skipped["not_in_source"] = skipped.get("not_in_source",0)+1
            continue
        target = positives[user][split]
        pool = source[user]
        # Official code deletes existing ground truth, shifts list left, takes
        # first budget-1, then inserts the true item and shuffles the list.
        seen = set()
        negatives = []
        for item in pool:
            if item == target:
                continue
            if item in seen:
                continue
            seen.add(item)
            if item not in titles:
                continue
            negatives.append(item)
            if len(negatives) == recall_budget-1:
                break
        if len(negatives) != recall_budget-1 or target not in titles:
            skipped["insufficient_candidates"] = skipped.get("insufficient_candidates",0)+1
            continue
        cand = negatives+[target]
        random.Random(seeded(seed,user,split)).shuffle(cand)
        assert len(set(cand))==recall_budget
        cases.append({"user":user,"history":train[user][-hist_len:],
                      "target":target,"candidates":cand})
    if strict_candidate_size and not cases:
        raise ValueError(f"No users with sufficient valid official candidate items: {skipped}")
    return cases, skipped

def describe(source_file, recall_budget):
    import hashlib
    p=Path(source_file)
    with p.open("rb") as f:
        digest=hashlib.file_digest(f,"sha256").hexdigest()
    return {"candidate_source":p.name,"sha256":digest,"recall_budget":recall_budget,
            "selection":"official has_gt=True: top K-1 source candidates excluding GT; append GT; seeded shuffle",
            "reproduction_limit":"Official RecBole user/target split and model prompt are not yet exactly reproduced."}
