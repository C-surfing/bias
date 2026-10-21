"""Read-only, leakage-safe observations for the EXISTING RSI Diagnosis Agent."""
from __future__ import annotations
from collections import Counter
from statistics import median

def candidate_snapshot(cases,train_popularity):
    if not cases:
        raise ValueError("empty development cohort")
    lens=[len(x["history"]) for x in cases]
    counts=[train_popularity.get(item,0) for x in cases for item in x["candidates"]]
    unique=set(item for case in cases for item in case["candidates"])
    if any(case["target"] not in case["candidates"] for case in cases):
        raise ValueError("invalid sampled candidate set")
    return {
        "tool":"inspect_frozen_candidate_distribution",
        "role":"read-only prior to intervention",
        "users":len(cases),
        "history_length":{"minimum":min(lens),"median":median(lens),
                          "maximum":max(lens)},
        "candidate_total":len(counts),
        "candidate_distinct":len(unique),
        "training_popularity":{"minimum":min(counts),"median":median(counts),
                               "maximum":max(counts),"unseen_ratio":sum(x==0 for x in counts)/len(counts)},
        "target_labels_sent_to_agent":False,
        "interpretation":"candidate retrieval confounds global popularity; no causal bias identified",
    }

def metric_tradeoff(baseline,changed):
    b,c=baseline["5"],changed["5"]
    return {"tool":"paired_metric_tradeoff",
        "ndcg_delta":c["NDCG@5"]-b["NDCG@5"],
        "hit_delta":c["HR@5"]-b["HR@5"],
        "arp_delta":c["ARP@5"]-b["ARP@5"] if None not in (c["ARP@5"],b["ARP@5"]) else None,
        "aplt_delta":c["APLT@5"]-b["APLT@5"] if None not in (c["APLT@5"],b["APLT@5"]) else None,
        "note":"Empirical paired development changes; not an identified causal correction"}
