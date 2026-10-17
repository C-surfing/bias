"""Controlled synthetic Exposure Bias laboratory with known logging propensities.

Not a claim about KuaiRand or any real-user causal effect.
Observes Bernoulli rewards only for logged impressions and evaluates target
policies by IPS and SNIPS against an independent mathematically known oracle.
"""
from __future__ import annotations
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
from random import Random
import argparse
import json
import math

ROOT=Path("/data/cjh/rsi/project")
VERSION="exposure-identifiability-v1"
USERS=50
ITEMS=20
REPS=32
EVENTS=3000
SEED=20261017
EXPLORATION=0.10
POPULARITY_STRENGTH=4.0
TARGETS=("uniform","personalized_oracle")


def q(u:int,i:int)->float:
    """Synthetic potential-outcome probability; never observed directly by estimators."""
    return 0.1+0.8*(1+math.cos(2*math.pi*(u/USERS-i/ITEMS)))/2


def logging_distribution():
    """Known epsilon-mixed propensity, independent of rewards conditional on user/item."""
    logits=[POPULARITY_STRENGTH*i/(ITEMS-1) for i in range(ITEMS)]
    m=max(logits)
    weights=[math.exp(v-m) for v in logits]
    z=sum(weights)
    probs=[EXPLORATION/ITEMS+(1-EXPLORATION)*v/z for v in weights]
    if not all(p>0 for p in probs) or abs(sum(probs)-1)>1e-12:
        raise ValueError("Logging probabilities violate positivity/normalization")
    return probs


def target_distribution(u, policy):
    if policy=="uniform":
        return [1/ITEMS]*ITEMS
    if policy=="personalized_oracle":
        best=max(range(ITEMS),key=lambda i:(q(u,i),-i))
        return [0.1/ITEMS+(0.9 if i==best else 0) for i in range(ITEMS)]
    raise ValueError("Unregistered target policy")


def true_value(policy):
    return sum(sum(pi*y for pi,y in zip(target_distribution(u,policy),
                   (q(u,i) for i in range(ITEMS)))) for u in range(USERS))/USERS


def choose(probs,rng):
    t=rng.random()
    cumulative=0
    for i,p in enumerate(probs):
        cumulative+=p
        if t<cumulative:return i
    return len(probs)-1


def simulate(seed, events, *, propensities_visible=True):
    if events<=0:raise ValueError("Events must be positive")
    rng=Random(seed)
    logging=logging_distribution()
    numerators={t:0.0 for t in TARGETS}
    denominators={t:0.0 for t in TARGETS}
    squares={t:0.0 for t in TARGETS}
    raw=0
    for _ in range(events):
        u=rng.randrange(USERS)
        i=choose(logging,rng)
        reward=int(rng.random()<q(u,i))
        raw+=reward
        if not propensities_visible:
            continue
        for t in TARGETS:
            pi=target_distribution(u,t)[i]
            weight=pi/logging[i]
            numerators[t]+=weight*reward
            denominators[t]+=weight
            squares[t]+=weight*weight
    if not propensities_visible:
        return {"status":"ABSTAIN_NO_PROPENSITY","observed_logging_click_rate":raw/events,
                "ips":None,"snips":None,
                "reason":"Logging probabilities unavailable to estimator; no identified OPE"}
    return {"status":"IDENTIFIED_SYNTHETIC_OPE",
       "observed_logging_click_rate":raw/events,
       "targets":{t:{
           "IPS":numerators[t]/events,
           "SNIPS":numerators[t]/denominators[t] if denominators[t] else None,
           "ESS":denominators[t]**2/squares[t] if squares[t] else 0,
           "weight_sum":denominators[t]} for t in TARGETS}}


def stats(xs, truth):
    n=len(xs)
    bias=sum(v-truth for v in xs)/n
    rmse=math.sqrt(sum((v-truth)**2 for v in xs)/n)
    return {"mean":sum(xs)/n,"bias":bias,"rmse":rmse,
            "min":min(xs),"max":max(xs)}


def evaluate(reps=REPS,events=EVENTS,seed=SEED):
    if reps<2 or events<1:raise ValueError("Invalid experiment budget")
    truth={t:true_value(t) for t in TARGETS}
    logs=[]
    for j in range(reps):
        logs.append(simulate(seed+j,events))
    opaque=simulate(seed,events,propensities_visible=False)
    targets={}
    for t in TARGETS:
        target_true=truth[t]
        targets[t]={
          "oracle_value":target_true,
          "naive_logged_click_rate":stats([x["observed_logging_click_rate"] for x in logs],target_true),
          "IPS":stats([x["targets"][t]["IPS"] for x in logs],target_true),
          "SNIPS":stats([x["targets"][t]["SNIPS"] for x in logs],target_true),
          "ESS_mean":sum(x["targets"][t]["ESS"] for x in logs)/reps,
          "ESS_min":min(x["targets"][t]["ESS"] for x in logs),
        }
    p=logging_distribution()
    return {"status":"SYNTHETIC_KNOWN_PROPENSITY_BENCHMARK_COMPLETE",
      "version":VERSION,"created_utc":datetime.now(timezone.utc).isoformat(),
      "config":{"users":USERS,"items":ITEMS,"repetitions":reps,
                "events_per_repetition":events,"seed":seed,
                "logging_exploration_probability":EXPLORATION,
                "popularity_strength":POPULARITY_STRENGTH,
                "target_policies":list(TARGETS)},
      "randomization":{"all_logged_propensities_known":True,
                        "logging_policy_positive_min":min(p),
                        "logging_policy_positive_max":max(p),
                        "outcomes_only_for_logged_impressions":True},
      "targets":targets,
      "missing_propensity_control":opaque,
      "independent_real_data_confirmation":False,"test_labels_read":False,
      "new_model_calls":0,"gpu_calls":0,
      "limitations":[
        "Preference probabilities are programmed synthetic truth, not real customer feedback.",
        "Personalized oracle uses hidden simulator truth, which is unavailable to a real recommender.",
        "The naive logging reward is NOT an unbiased estimate of alternative policy value.",
        "SNIPS is self-normalized and can have finite-sample bias; no universal improvement is guaranteed.",
        "Synthetic known propensities cannot resolve KuaiRand's missing true logging probabilities."
      ]}


def run(output, *, reps=REPS,events=EVENTS,seed=SEED):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/"runs"):
        raise ValueError("Outputs must stay under project/runs")
    out=evaluate(reps,events,seed)
    output.mkdir(parents=True,exist_ok=False)
    (output/"exposure_synthetic.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    lines=["# Controlled Exposure Bias: known propensity benchmark","",
      "Synthetic data ONLY. Does not score Coat/KuaiRand confirmation labels.",
      f"Seed {seed}; {reps} independent repetitions × {events} logged interactions.",
      "Logging policy has guaranteed positivity and known probabilities.",
      "", "| Target | Oracle value | Naive RMSE | IPS RMSE | SNIPS RMSE | Mean ESS |",
      "|---|---:|---:|---:|---:|---:|"]
    for t,row in out["targets"].items():
        lines.append(f"| {t} | {row['oracle_value']:.4f} | "
            f"{row['naive_logged_click_rate']['rmse']:.4f} | "
            f"{row['IPS']['rmse']:.4f} | "
            f"{row['SNIPS']['rmse']:.4f} | {row['ESS_mean']:.1f} |")
    lines +=["","The missing-propensity control abstains, rather than computing fictitious IPS.",
            "This controlled reference is appropriate for estimator validation, not real-data causal proof."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    return out


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--reps",type=int,default=REPS)
    a.add_argument("--events",type=int,default=EVENTS)
    args=a.parse_args()
    result=run(args.output,reps=args.reps,events=args.events)
    print(json.dumps({"status":result["status"],
      "targets":{x:{"ips_rmse":y["IPS"]["rmse"],"snips_rmse":y["SNIPS"]["rmse"]}
                 for x,y in result["targets"].items()},
      "missing_propensities":result["missing_propensity_control"]["status"],
      "gpu":0,"llm":0},ensure_ascii=False))
