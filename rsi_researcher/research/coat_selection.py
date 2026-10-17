"""Coat selection-bias *development* benchmark, CPU and Qwen-free.

Compares equal-capacity unweighted vs estimated-propensity-weighted item mean
recommendation scores on the SAME independently sampled Coat rating panels.
The supplied learned propensities are estimates, not logged randomization
probabilities or causal truth. No held-out test labels enter this run.
"""
from collections import defaultdict,Counter
from datetime import datetime,timezone
from hashlib import sha256
from math import isfinite
from pathlib import Path
import argparse,json,random

from rsi_researcher.datasets.coat import load_coat
from rsi_researcher.evaluation.observed import evaluate_observed
from rsi_researcher.recommenders.itemcf import ItemCF

DATA=Path("/data/cjh/rsi/project/data/external/coat/coat")
SEED=20261014
THRESHOLD=4
CAPS=(5.,20.,100.)
SMOOTH=5.0

def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")

def estimated_p_matrix():
    path=DATA/"propensities.ascii"
    rows=[[float(s) for s in ln.split()] for ln in path.read_text(encoding="ascii").splitlines() if ln.strip()]
    if len(rows)!=290 or any(len(row)!=300 for row in rows):
        raise ValueError("Coat propensity estimates are not a 290x300 matrix")
    if any(not isfinite(p) or not 0 < p <= 1 for row in rows for p in row):
        raise ValueError("Invalid propensity estimates, not safe to weight")
    return rows

def fit_item_mean(train,items,p_matrix=None,cap=None,prior_strength=SMOOTH):
    """Observed-row item relevance prevalence estimator with common model form.
    Weights only affect observed rating contributions; never fill missing entries.
    """
    rows=[]
    for row in train:
        u=int(row["user"]);i=int(row["item"])
        p=(p_matrix[u][i] if p_matrix is not None else 1.0)
        raw=1/p
        w=min(raw,float(cap)) if cap is not None else raw
        rows.append((row["item"],int(row["rating"]>=THRESHOLD),w))
    mean_w=sum(w for _,_,w in rows)/len(rows)
    normalized=[(i,y,w/mean_w) for i,y,w in rows]
    pos=sum(w*y for _,y,w in normalized)
    total=sum(w for _,_,w in normalized)
    global_prior=pos/total
    counts=defaultdict(float);positives=defaultdict(float)
    for i,y,w in normalized:
        counts[i]+=w
        positives[i]+=w*y
    probabilities={
      item:(positives[item]+prior_strength*global_prior)/(counts[item]+prior_strength)
      for item in items
    }
    squared_weights=sum(w*w for _,_,w in normalized)
    eff=total*total/squared_weights
    return probabilities,{"weight_cap":cap,"global_prevalence_train":global_prior,
       "effective_train_rows":eff,"original_train_rows":len(rows),
       "normalized_weight_min":min(w for _,_,w in normalized),
       "normalized_weight_max":max(w for _,_,w in normalized),
       "all_finite":all(isfinite(v) and 0<=v<=1 for v in probabilities.values())}

def metrics(random_dev,probs,train_counts,items):
    grouped=defaultdict(list)
    for row in random_dev:grouped[row["user"]].append(row)
    rankings={u:sorted((x["item"] for x in panel),key=lambda i:(-probs[i],i))
              for u,panel in grouped.items()}
    panel=evaluate_observed(random_dev,rankings,train_counts,catalog_items=items,
                             k=5,positive_threshold=THRESHOLD,gain_floor=1)
    sum_sq=0.0;per_user={}
    for u,rs in grouped.items():
        losses=[(probs[x["item"]]-int(x["rating"]>=THRESHOLD))**2 for x in rs]
        sum_sq+=sum(losses)
        per_user[u]=sum(losses)/len(losses)
    observed_rate=sum(x["rating"]>=THRESHOLD for x in random_dev)/len(random_dev)
    positive_panels={u for u,rs in grouped.items() if any(x["rating"]>=THRESHOLD for x in rs)}
    hit_count=sum(any(x["item"] in set(rankings[u][:5]) and x["rating"]>=THRESHOLD
                      for x in grouped[u]) for u in positive_panels)
    hit_at_5=hit_count/len(positive_panels) if positive_panels else None
    return {"Brier_random_development":sum_sq/len(random_dev),
       "NDCG5_random_development":panel["ndcg"],
       "Hit5_random_development":hit_at_5,
       "head_share_random_development":panel["head_share"],
       "tail_relevant_recall":panel["tail_relevant_recall"],
       "quality_users":panel["quality_users"],
       "positive_rate_random_development":observed_rate,
       "scored_rows":len(random_dev),
       "per_user_Brier":per_user,"per_user_NDCG":{v["user"]:v["ndcg"] for v in panel["per_user"] if v["ndcg"] is not None},
       "ranking_ids_only":True}

def bootstrap_paired(before,after,seed=SEED,draws=1000):
    matched=sorted(set(before)&set(after))
    if not matched:return None
    differences=[after[u]-before[u] for u in matched]
    point=sum(differences)/len(differences)
    rng=random.Random(seed)
    sims=sorted(sum(differences[rng.randrange(len(differences))]
                    for _ in differences)/len(differences) for _ in range(draws))
    return {"users":len(matched),"mean_after_minus_before":point,
       "illustrative_user_bootstrap_95":[sims[25],sims[975]],
       "limitation":"users share training item statistics; bootstrap is conditional, not independent training repeats"}

def run(outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=False)
    ds=load_coat(DATA,split_seed=SEED)
    train=ds["train"];dev=ds["development"]
    # IMPORTANT: although the standard loader has produced a separate test array,
    # no heldout outcome is read, scored or saved here. Never change this protocol after scoring.
    prop=estimated_p_matrix()
    train_counts=Counter(x["item"] for x in train)
    all_counts={i:train_counts.get(i,0) for i in ds["item_ids"]}
    policies=[("unweighted",None,None)]+[(f"estimated_propensity_clip_{int(c)}",prop,c) for c in CAPS]
    results={}
    for name,propensity,cap in policies:
        scores,fit=fit_item_mean(train,ds["item_ids"],propensity,cap)
        evaluation=metrics(dev,scores,all_counts,ds["item_ids"])
        results[name]={"model":"smoothed_item_positive_rate_same_capacity",
          "configuration":{"weight_cap":cap,"prior_strength":SMOOTH,
             "weight_source":"Coat learned propensity estimates" if cap else "unit weights"},
          "train_diagnostics":fit,
          "development_metrics":{k:v for k,v in evaluation.items() if not k.startswith("per_user_")}}
        # Evaluation per-user arrays remain in memory for paired uncertainty only.
        results[name]["paired_difference_to_unweighted"]=None
        if name=="unweighted":base=evaluation
        else:
            results[name]["paired_difference_to_unweighted"]={
                "Brier":bootstrap_paired(base["per_user_Brier"],evaluation["per_user_Brier"]),
                "NDCG5":bootstrap_paired(base["per_user_NDCG"],evaluation["per_user_NDCG"])}
    # Evaluator is criterion-driven; not an independent-confirmation claim.
    brier0=results["unweighted"]["development_metrics"]["Brier_random_development"]
    ndcg0=results["unweighted"]["development_metrics"]["NDCG5_random_development"]
    eligible=[(name,val) for name,val in results.items() if name!="unweighted"
       and val["development_metrics"]["Brier_random_development"] < brier0
       and val["development_metrics"]["NDCG5_random_development"] >= .99*ndcg0]
    selected=min(eligible,key=lambda r:r[1]["development_metrics"]["Brier_random_development"])[0] if eligible else "unweighted"
    # An additional pre-existing ItemCF baseline is a harder ranking sanity check,
    # not an equal-capacity causal contrast with weighted item means.
    itemcf=ItemCF(train,ds["item_ids"],THRESHOLD)
    itemcf_panels=itemcf.rank_panels([{"user":x["user"],"item":x["item"]} for x in dev])
    itemcf_metrics=evaluate_observed(dev,itemcf_panels,all_counts,
                                    catalog_items=ds["item_ids"],k=5,positive_threshold=THRESHOLD,gain_floor=1)
    status={"run_status":"DEVELOPMENT_COMPLETE_NO_CONFIRMATION","created_utc":datetime.now(timezone.utc).isoformat(),
       "task_id":"selection","registry_version":"classic-five-bias-tasks-v1.0",
       "source_dataset_fingerprint":ds["fingerprint"],
       "split_seed":SEED,"train_rows":len(train),"random_dev_rows":len(dev),
       "random_holdout_rows_withheld":ds["statistics"]["test_rows"],
       "author_estimated_propensity_sha256":sha256((DATA/"propensities.ascii").read_bytes()).hexdigest(),
       "propensity_note":"Learned author estimates from propensity_data; not true randomized logging probabilities",
       "protocol":{"candidate_policy_names":[x[0] for x in policies],
          "positive_threshold":4,"smoothing":SMOOTH,
          "primary_quality":"random-panel Brier, lower better",
          "quality_safeguard":"NDCG@5 at least 99% of unweighted item mean baseline",
          "dev_only_selection":"minimum Brier among candidates passing safeguard; otherwise unweighted",
          "random_panel_scope":"observed randomized-rated user-item pairs; never impute missing preferences"},
       "policies":results,"separately_reported_itemcf_ranking_reference":{
          k:itemcf_metrics[k] for k in ("ndcg","head_share","tail_relevant_recall","quality_users")},
       "selected_development_policy":selected,
       "independent_confirmation":False,"test_evaluation_done":False,
       "scientific_limitations":[
         "Empirical rating-selection difference is not a causal identification result",
         "Author-learned propensities are not verified randomized assignment probabilities",
         "Only item means were compared; the separate ItemCF is a quality sanity baseline",
         "User bootstrap intervals do not capture full training-state or logging dependencies",
         "Do not make adaptive post-hoc claims from frozen random holdout without fresh protocol"]}
    write(out/"coat_selection_development.json",status)
    print(json.dumps({"status":status["run_status"],"observed_rows":len(train),
      "random_dev":len(dev),"random_test_withheld":ds["statistics"]["test_rows"],
      "baseline_Brier":brier0,"baseline_NDCG5":ndcg0,
      "weighted_Brier":{k:round(v["development_metrics"]["Brier_random_development"],5)
                        for k,v in results.items() if k!="unweighted"},
      "selected_dev":selected,"itemcf_NDCG5":itemcf_metrics["ndcg"]},ensure_ascii=False))
    return status

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True)
    run(p.parse_args().output)
