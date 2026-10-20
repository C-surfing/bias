"""Single-use, preregistered Coat RANDOM-RATING holdout confirmation.

No LLM, new fit, propensity estimation, development selection or threshold
adaptation happens after seeing holdout outcomes. Record same-cohort caveat.
"""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json, argparse

from rsi_researcher.research.coat_selection import (
    DATA, SEED, THRESHOLD, estimated_p_matrix, fit_item_mean,
    metrics, bootstrap_paired,
)
from rsi_researcher.datasets.coat import load_coat
from rsi_researcher.runtime.client import atomic_json


def check_protocol(protocol, devpath):
    source=Path(devpath)
    if sha256(source.read_bytes()).hexdigest()!=protocol["development_source_sha256"]:
        raise ValueError("development source checksum mismatch")
    dev=json.loads(source.read_text(encoding="utf8"))
    if (dev["selected_development_policy"]!=protocol["primary_policy"] or
        dev["independent_confirmation"] is not False or
        dev["test_evaluation_done"] is not False or
        protocol["primary_policy"]!="estimated_propensity_clip_100" or
        protocol["baseline_policy"]!="unweighted" or
        protocol["holdout_split_seed"]!=SEED or
        protocol["min_brier_gain"]!=.002 or
        protocol["min_relative_ndcg"]!=.99 or
        protocol["max_hit_loss"]!=.02 or
        protocol["paired_bootstrap_confidence"]!=.95):
        raise ValueError("confirmation policy/threshold not frozen as expected")
    return dev


def verdict(base, treatment, pair, protocol):
    b=base["Brier_random_development"]
    t=treatment["Brier_random_development"]
    n0=base["NDCG5_random_development"]
    n1=treatment["NDCG5_random_development"]
    h0=base["Hit5_random_development"]
    h1=treatment["Hit5_random_development"]
    ci=pair["illustrative_user_bootstrap_95"]
    requirements={
        "brier_gain_min_0_002": b-t>=protocol["min_brier_gain"],
        "user_bootstrap_improvement_upper_below_zero":ci[1]<0,
        "ndcg_at_least_99pct_baseline":n1>=protocol["min_relative_ndcg"]*n0,
        "hit_loss_max_0_02":h1>=h0-protocol["max_hit_loss"],
    }
    return {"passes_all_frozen_requirements":all(requirements.values()),
            "requirements":requirements,
            "brier_gain":b-t,
            "ndcg_ratio":n1/n0 if n0 else None,
            "hit_change":h1-h0,
            "paired_user_ci_of_after_minus_before":ci,
            "confirmatory_scope":"one untouched split from randomized ratings of existing users; NOT a new-user replication",
            "propensities_are_estimated":True,
            "not_proof_of_full_causal_identification":True}


def confirm(output,protocol_path,development_path):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("cannot overwrite confirmation or view holdout twice")
    protocol_path=Path(protocol_path)
    protocol=json.loads(protocol_path.read_text(encoding="utf8"))
    if protocol.get("mode")!="single_use_independent_random_rating_holdout":
        raise ValueError("confirmation mode is not preregistered")
    dev=check_protocol(protocol,development_path)
    ds=load_coat(DATA,split_seed=SEED)
    if dev["source_dataset_fingerprint"]!=ds["fingerprint"]:
        raise ValueError("Coat source/frozen split fingerprint mismatch")
    if len(ds["test"])!=dev["random_holdout_rows_withheld"]:
        raise ValueError("heldout split has changed")
    train=ds["train"]
    counts=Counter(r["item"] for r in train)
    scores_base,_=fit_item_mean(train,ds["item_ids"],None,None)
    scores_weighted,_=fit_item_mean(train,ds["item_ids"],estimated_p_matrix(),100)
    base=metrics(ds["test"],scores_base,counts,ds["item_ids"])
    treatment=metrics(ds["test"],scores_weighted,counts,ds["item_ids"])
    pair=bootstrap_paired(base["per_user_Brier"],treatment["per_user_Brier"],
                          seed=protocol["bootstrap_seed"],draws=2000)
    primary=verdict(base,treatment,pair,protocol)
    output.mkdir(parents=True)
    atomic_json(output/"confirmation.json",{
       "status":("SINGLE_HOLDOUT_CRITERIA_PASS" if
                 primary["passes_all_frozen_requirements"]
                 else "SINGLE_HOLDOUT_CRITERIA_FAILED"),
       "bias_id":"selection",
       "protocol_sha256":sha256(protocol_path.read_bytes()).hexdigest(),
       "development_source_sha256":sha256(Path(development_path).read_bytes()).hexdigest(),
       "dataset_fingerprint":ds["fingerprint"],
       "scored_holdout_ratings":len(ds["test"]),
       "training_rows":len(train),
       "primary_policy":protocol["primary_policy"],
       "baseline_policy":"unweighted",
       "baseline":{k:v for k,v in base.items() if not k.startswith("per_user_")},
       "treatment":{k:v for k,v in treatment.items() if not k.startswith("per_user_")},
       "predeclared_verdict":primary,
       "no_hypothesis_or_method_selection_after_holdout":True,
       "same_users_as_development":True,
       "independent_scientific_debias_discovery":False,
       "reason":"A single heldout rating split can validate this conditional policy comparison; "
                "it cannot show new-user transfer, propensity identification, or RSI superiority."})
    return primary


def main(argv=None):
    a=argparse.ArgumentParser()
    a.add_argument("--protocol",required=True)
    a.add_argument("--development",required=True)
    a.add_argument("--output",required=True)
    args=a.parse_args(argv)
    print(json.dumps(confirm(args.output,args.protocol,args.development),
                     ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
