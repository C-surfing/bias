"""Development-only robustness audit of Coat estimated propensity weights.

Tests whether a selected clipping parameter remains stable under prespecified
miscalibration perturbations of the *estimated* propensity matrix. This does
not infer true exposure probabilities or use confirmation ratings.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import math

from rsi_researcher.datasets.coat import load_coat
from rsi_researcher.research.coat_selection import (
    DATA, SEED, estimated_p_matrix, fit_item_mean, metrics,
)

ROOT=Path("/data/cjh/rsi/project")
GAMMAS=(0.75,1.0,1.25)
CAPS=(5,20,100)
FLOOR=0.99


def source_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def percentile(vals, q):
    seq=sorted(vals)
    if not seq:
        raise ValueError("Empty values")
    return seq[round(q*(len(seq)-1))]


def choose(candidates, baseline):
    valid=[c for c in candidates if c["Brier"] < baseline["Brier"]
           and c["NDCG5"] >= FLOOR*baseline["NDCG5"]]
    return min(valid,key=lambda c:c["Brier"])["cap"] if valid else None


def run(output: Path):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/"runs"):
        raise ValueError("Output must be in project/runs")
    registry=ROOT/"configs/bias_tasks.json"
    if source_digest(registry)!=registry.with_suffix(".sha256").read_text().strip():
        raise ValueError("Registry fingerprint mismatch")
    ds=load_coat(DATA,split_seed=SEED)
    if len(ds["train"])!=6960 or len(ds["development"])!=2065 or len(ds["test"])!=2209:
        raise ValueError("Input split changed; do not compare different protocols")
    source=estimated_p_matrix()
    train=ds["train"]
    counts=Counter(x["item"] for x in train)
    all_counts={i:counts.get(i,0) for i in ds["item_ids"]}
    observed_p=[source[int(r["user"])][int(r["item"])] for r in train]
    probabilities,fit=fit_item_mean(train,ds["item_ids"])
    base_metric=metrics(ds["development"],probabilities,all_counts,ds["item_ids"])
    baseline={"Brier":base_metric["Brier_random_development"],
              "NDCG5":base_metric["NDCG5_random_development"],
              "Hit5":base_metric["Hit5_random_development"],
              "tail_relevant_recall":base_metric["tail_relevant_recall"]}
    sweeps=[]
    for gamma in GAMMAS:
        shifted=[[min(1.0,max(1e-8,p**gamma)) for p in row] for row in source]
        cases=[]
        for cap in CAPS:
            trained,d=fit_item_mean(train,ds["item_ids"],shifted,float(cap))
            scored=metrics(ds["development"],trained,all_counts,ds["item_ids"])
            raw=[1.0/shifted[int(r["user"])][int(r["item"])] for r in train]
            cand={
                "cap":cap,
                "Brier":scored["Brier_random_development"],
                "NDCG5":scored["NDCG5_random_development"],
                "Hit5":scored["Hit5_random_development"],
                "tail_relevant_recall":scored["tail_relevant_recall"],
                "ESS":d["effective_train_rows"],
                "fraction_capped":sum(v>cap for v in raw)/len(raw),
                "Brier_delta":scored["Brier_random_development"]-baseline["Brier"],
                "NDCG_safeguard_pass":scored["NDCG5_random_development"]>=FLOOR*baseline["NDCG5"],
            }
            if any(not math.isfinite(x) for x in (cand["Brier"],cand["NDCG5"],cand["ESS"])):
                raise ValueError("Nonfinite numerical result")
            cases.append(cand)
        sweeps.append({"gamma":gamma,"cases":cases,"best_cap":choose(cases,baseline)})
    out={
      "status":"DEV_PROPENSITY_SENSITIVITY_ONLY",
      "created_utc":datetime.now(timezone.utc).isoformat(),
      "dataset_fingerprint":ds["fingerprint"],
      "source_sha256":{"propensities":source_digest(DATA/"propensities.ascii"),
                       "task_registry":source_digest(registry)},
      "train_rows":len(ds["train"]),"development_rows":len(ds["development"]),
      "withheld_random_holdout_rows":len(ds["test"]),
      "holdout_evaluated":False, "model_requests":0,"gpu_runs":0,
      "observed_training_propensity_quantiles":{
          "min":min(observed_p),"p01":percentile(observed_p,0.01),
          "p50":percentile(observed_p,0.5),"p99":percentile(observed_p,0.99),
          "max":max(observed_p)},
      "baseline":baseline,"gammas":list(GAMMAS),"caps":list(CAPS),
      "sweep":sweeps,"stable_best_cap_across_perturbations":
          len({x["best_cap"] for x in sweeps})==1,
      "limitations":[
         "gamma is a hypothetical distortion of author-estimated propensity scores; it is not true randomization.",
         "Selected hyperparameters come from one known-user randomized-rating development panel.",
         "The fixed model and item-level fitted statistics share training data; user bootstrap alone understates uncertainty.",
         "The original Coat loader creates a predeclared holdout split; no test ratings are scored in this module.",
         "Do not select a new gamma on this same panel and then claim independent gains."
      ]
    }
    output.mkdir(parents=True,exist_ok=False)
    with (output/"sensitivity.json").open("x",encoding="utf8") as f:
        json.dump(out,f,indent=2,ensure_ascii=False)
    text=["# Coat estimated propensity sensitivity audit","",
      "Development-only, no new model calls and no confirmation outcomes.",
      f"Baseline Brier: {baseline['Brier']:.8f}; baseline NDCG@5: {baseline['NDCG5']:.8f}.",
      "", "| Gamma perturbation | Best clipped cap | ESS of cap 100 | Brier of cap 100 | Tail recall of cap 100 |",
      "|---:|---:|---:|---:|---:|"]
    for row in sweeps:
        c=next(v for v in row["cases"] if v["cap"]==100)
        text.append(f"| {row['gamma']:.2f} | {row['best_cap']} | {c['ESS']:.1f} | {c['Brier']:.6f} | {c['tail_relevant_recall']:.4f} |")
    text += ["",f"Stable best cap: {out['stable_best_cap_across_perturbations']}.",
             "Any improvement is exploratory; the supplied propensity matrix is estimated, not logged treatment assignment."]
    (output/"REPORT.md").write_text("\n".join(text)+"\n",encoding="utf8")
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True,type=Path)
    v=run(p.parse_args().output)
    print(json.dumps({"status":v["status"],
        "best_caps":[s["best_cap"] for s in v["sweep"]],
        "stable":v["stable_best_cap_across_perturbations"],
        "new_model_requests":0,"holdout_evaluated":False}))
