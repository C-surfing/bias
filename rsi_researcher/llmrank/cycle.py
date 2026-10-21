"""Budgeted recursive RSI research over the existing science-tree task adapter.

Development is repeated with a frozen user/candidate cohort. Failed and
dominated exact strategies are not executed again. Only once ALL search rounds
stop is the untouched holdout opened for an independent final audit.
This is NOT an 'until lucky p-value' loop.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rsi_researcher.llmrank.science import search, select_case_panels
from rsi_researcher.llmrank.strategy import (
    BASELINE, fingerprint, evaluate_strategy, bootstrap_paired
)
from rsi_researcher.llmrank.agents import RankGateway
from rsi_researcher.runtime.client import atomic_json, usage_summary
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.models import create_model

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def ensure_frozen(expected, actual):
    for key in ("seed", "dataset", "dev_users", "holdout_users",
                "data_archive_sha256", "candidate_source", "retrieval"):
        if expected[key]!=actual[key]:
            raise ValueError("confirmation cohort changed: "+key)

def independent_confirmation(directory, *, config, data, dataset, retrieval,
                             users, seeds, frozen_campaign, max_requests):
    """One-shot confirmation: creates a single immutable held-out report."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    atomic_json(directory/"confirmation_protocol.json",{
        "source_campaign":str(frozen_campaign),"holdout_opened_once":True,
        "frozen_selection_only":True,"max_requests":max_requests,
        "interpretation":"sample-based holdout, not causal identification"})
    limits=RunLimits(directory,max_requests=max_requests,
        max_tokens=config["budget"]["max_tokens"],
        max_seconds=config["budget"]["max_seconds"])
    model=create_model(directory,limits.max_requests,limits,config)
    model.health()
    ranker=RankGateway(model,directory/"rank_calls.jsonl")
    output=[]
    for seed in seeds:
        saved=read_json(Path(frozen_campaign)/f"seed-{seed}"/"result.json")
        if saved["holdout"] is not None:
            raise ValueError("candidate has already consumed test labels")
        dev,test,titles,pop,protocol=select_case_panels(
            dataset,data,retrieval,users,seed)
        ensure_frozen(saved["cohort"],protocol)
        rsi=saved["development"]["selected_rsi"]["strategy"]
        fixed=saved["development"]["selected_fixed"]["strategy"]
        cache={}
        def eval_one(label,strategy):
            key=fingerprint(strategy)
            if key not in cache:
                cache[key]=evaluate_strategy(ranker,test,titles,pop,strategy,
                    seed,"confirm."+key)
            return cache[key]
        baseline=eval_one("baseline",BASELINE)
        fixed_arm=eval_one("fixed",fixed)
        rsi_arm=eval_one("rsi",rsi)
        paired=bootstrap_paired(baseline,rsi_arm,pop,titles,seed=seed)
        fixed_paired=bootstrap_paired(baseline,fixed_arm,pop,titles,seed=seed+1)
        against_fixed=bootstrap_paired(fixed_arm,rsi_arm,pop,titles,seed=seed+2)
        row={
            "seed":seed,"cohort":protocol,
            "frozen_rsi":rsi,"frozen_fixed":fixed,
            "baseline":baseline["metrics"],"rsi":rsi_arm["metrics"],
            "fixed":fixed_arm["metrics"],"paired_rsi":paired,
            "paired_fixed":fixed_paired,"paired_rsi_minus_fixed":against_fixed,
            "source_selection":str(frozen_campaign),
            "independent_causal_discoveries":0,
        }
        atomic_json(directory/f"seed-{seed}.json",row)
        output.append(row)
    atomic_json(directory/"result.json",{
        "status":"completed","seeds":seeds,
        "comparisons":output,"model_requests":ranker.calls,
        "cache_hits":ranker.cache_hits,
        "scientific_conclusion":"Holdout measured after freezing development strategy; "
            "no claim of statistically significant or causal debias from these pilot sizes."})
    atomic_json(directory/"usage.json",usage_summary(directory))
    return output,ranker.calls

def conduct(directory, *,config,data,dataset,retrieval,users,seeds,
            max_rounds,max_nodes,requests_per_round,max_total_requests,
            patience):
    root=Path(directory).resolve()
    if root.exists():
        raise FileExistsError(root)
    if not 1<=max_rounds<=12 or not 1<=patience<=max_rounds or not 1<=max_nodes<=6:
        raise ValueError("rounds/nodes/patience outside the registered finite search")
    if requests_per_round<20 or max_total_requests<=requests_per_round:
        raise ValueError("invalid research/confirmation budget")
    root.mkdir(parents=True)
    atomic_json(root/"cycle_protocol.json",{
        "dataset":dataset,"retrieval":retrieval,"users":users,"seeds":seeds,
        "max_rounds":max_rounds,"max_nodes_per_round":max_nodes,
        "requests_per_round":requests_per_round,
        "max_total_model_requests":max_total_requests,
        "patience":patience,"holdout_access":"AFTER stopping only",
        "stopping":"budget OR early scientific candidate OR patience OR maximum rounds",
        "not_until_success_by_repeated_test_peeking":True})
    history=[]
    best_merit=-1.0
    misses=0
    prior=None
    requests=0
    stopped=None
    # Reserve enough capacity for a final once-only heldout comparison.
    n_test=max(2,users-int(users*.6))
    confirmation_reserve=len(seeds)*n_test*6+12
    for iteration in range(1,max_rounds+1):
        remaining=max_total_requests-requests-confirmation_reserve
        if remaining<20:
            stopped="reserve_confirmation_budget"
            break
        budget=min(requests_per_round,remaining)
        destination=root/f"round-{iteration:02d}"
        result=search(destination,config=config,data=data,dataset=dataset,
            retrieval=retrieval,users=users,seeds=seeds,max_nodes=max_nodes,
            max_requests=budget,prior_campaign=prior,development_only=True)
        used=result["requests"]
        requests+=used
        summary=read_json(destination/"summary.json")
        merits=[]
        for seed in seeds:
            record=read_json(destination/f"seed-{seed}"/"result.json")
            if record["holdout"] is not None:
                raise RuntimeError("development iteration leaked the confirmation holdout")
            merits.append(record["development"]["selected_rsi"]["merit"])
        merit=min(merits)  # reproducible guard: weakest dev seed
        progressed=merit>best_merit+1e-9
        if progressed:
            best_merit=merit
            misses=0
        else:
            misses+=1
        history.append({
            "round":iteration,"campaign":str(destination),
            "model_requests":used,"selected_development_merit_by_seed":merits,
            "worst_seed_merit":merit,"progressed":progressed,
            "development_only":True})
        prior=destination
        atomic_json(root/"cycle_index.json",{
            "rounds":history,"cumulative_requests":requests,
            "current_incumbent":str(prior),"remaining_budget":
                max_total_requests-requests,"independent_confirmation":False})
        if misses>=patience:
            stopped="patience_no_verified_development_progress"
            break
        if iteration==max_rounds:
            stopped="maximum_rounds"
    if prior is None:
        atomic_json(root/"status.json",{"status":"stopped_without_confirmation",
            "reason":stopped,"rounds":history,"requests":requests})
        return {"status":"stopped_without_confirmation","rounds":len(history),
                "reason":stopped,"requests":requests}
    allowed=max_total_requests-requests
    if allowed<confirmation_reserve:
        atomic_json(root/"status.json",{"status":"insufficient_confirmation_budget",
            "reason":stopped,"rounds":history,"requests":requests,
            "remaining":allowed})
        return {"status":"insufficient_confirmation_budget",
                "rounds":len(history),"requests":requests}
    verified,confirmed_calls=independent_confirmation(root/"confirmation",
        config=config,data=data,dataset=dataset,retrieval=retrieval,
        users=users,seeds=seeds,frozen_campaign=prior,max_requests=allowed)
    outcome={"status":"completed_development_and_one_holdout",
        "rounds":len(history),"termination":stopped,
        "research_requests":requests,"confirmation_requests":confirmed_calls,
        "combined_requests":requests+confirmed_calls,
        "independent_confirmed_scientific_discoveries":0,
        "selected_from":str(prior),"summary":[{
            "seed":r["seed"],
            "baseline_NDCG5":r["baseline"]["5"]["NDCG@5"],
            "rsi_NDCG5":r["rsi"]["5"]["NDCG@5"],
            "baseline_ARP5":r["baseline"]["5"]["ARP@5"],
            "rsi_ARP5":r["rsi"]["5"]["ARP@5"]}
            for r in verified]}
    atomic_json(root/"status.json",outcome)
    return outcome

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--dataset",choices=["ml-1m","Games","Beauty","Sports","CDs"],
                   default="ml-1m")
    p.add_argument("--retrieval",choices=["random","sasrec"],default="random")
    p.add_argument("--users",type=int,default=12)
    p.add_argument("--seeds",default="20261021")
    p.add_argument("--max-rounds",type=int,default=3)
    p.add_argument("--max-nodes",type=int,default=2)
    p.add_argument("--requests-per-round",type=int,default=220)
    p.add_argument("--max-total-requests",type=int,default=900)
    p.add_argument("--patience",type=int,default=2)
    a=p.parse_args(argv)
    config=json.loads(Path(a.config).read_text(encoding="utf-8"))
    if config.get("llm",{}).get("provider")!="vllm":
        p.error("requires local Qwen vLLM")
    seeds=[int(x) for x in a.seeds.split(",")]
    print(json.dumps(conduct(a.output,config=config,data=a.data,dataset=a.dataset,
        retrieval=a.retrieval,users=a.users,seeds=seeds,
        max_rounds=a.max_rounds,max_nodes=a.max_nodes,
        requests_per_round=a.requests_per_round,
        max_total_requests=a.max_total_requests,patience=a.patience),
        ensure_ascii=False))

if __name__=="__main__":
    main()
