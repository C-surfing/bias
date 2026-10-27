"""Strict recovery of a frozen Exposure confirmation whose JSON write failed.

Never runs a scientist, retunes parameters, selects a new winner, or reads
development data. First confirmation was computed but not saved (numpy int64).
This is an identical, auditable replay of the same locked evaluator.
"""
import argparse,json
from pathlib import Path
from rsi_researcher.research.four_bias_benchmarks import ExposurePanel,SEEDS,OBD,BTS,digest
from rsi_researcher.runtime.client import atomic_json,usage_summary

def recover(path):
    out=Path(path).resolve()
    freeze=json.loads((out/"frozen_before_confirmation.json").read_text())
    prior=json.loads((out/"summary.json").read_text())
    proto=json.loads((out/"protocol.json").read_text())
    if freeze["task"]!="exposure" or proto["task"]!="exposure":
        raise ValueError("not the exact frozen Exposure task")
    if not freeze["confirmation_allowed"] or freeze["candidate"] is None:
        raise ValueError("no development-passing frozen candidate")
    if (out/"confirmation.json").exists() or (out/"recovered_confirmation.json").exists():
        raise ValueError("confirmation already persisted")
    if prior.get("status")!="development_running" or prior.get("holdout_opened"):
        raise ValueError("confirmation recovery requires interrupted pre-final state")
    hashes={p.name:digest(p) for p in (OBD,BTS)}
    if hashes!=freeze["freeze_dataset_hashes"]:
        raise ValueError("data source changed since frozen candidate selection")
    arms={"baseline":{"top":20,"mass":.25}}
    for name,selection in freeze["method_winners"].items():
        if selection:
            arms[name]=selection["strategy"]
    atomic_json(out/"CONFIRMATION_RECOVERY_AUDIT.json",{
        "reason":"initial frozen holdout was computed, but save failed due numpy.int64 JSON field",
        "exception":"TypeError: Object of type int64 is not JSON serializable",
        "prior_confirmation_result_read_by_agent":False,
        "same_locked_params":arms,"source_hashes":hashes,
        "replay_count":1,"policy_modifications":False,
        "scientist_model_calls_during_replay":0,
        "interpretation":"a technical replay, not a second adaptive holdout choice"})
    panel=ExposurePanel("confirm")
    outcomes={}
    for name,params in arms.items():
        if name=="baseline":
            outcomes[name]={"strategy":params,
               "by_seed":[panel.measure(params,seed) for seed in SEEDS]}
        else:
            by_seed=[panel.compare(arms["baseline"],params,seed) for seed in SEEDS]
            outcomes[name]={"strategy":params,"by_seed":by_seed,
                "consistent_positive_CI":all(x["CI95_gain"][0]>0 for x in by_seed),
                "utility_guard_pass":all(
                    x["candidate"]["ESS"]>=x["min_ESS_required"]
                    for x in by_seed)}
    confirm={
        "phase":"frozen_confirmation_replayed_after_serialization_failure",
        "task":"exposure",
        "cohort":"full official OBD randomized actions, fixed disjoint impression split",
        "summary":outcomes,"confirmed_bias_causal_claims":0,
        "independent_confirmation_of_specific_estimand":{
           name:bool(row.get("consistent_positive_CI") and row.get("utility_guard_pass"))
           for name,row in outcomes.items() if name!="baseline"},
        "scope":"static randomized-logger OPE value, NOT exposure bias reduction",
        "technical_replay_disclosed":True}
    atomic_json(out/"recovered_confirmation.json",confirm)
    status={**prior,"status":"completed_after_fixed_technical_replay",
      "development_seeds":list(SEEDS),"pre_confirmation_freeze":freeze,
      "confirmation":confirm,"holdout_opened":True,
      "qwen_requests":prior["qwen_model_requests"],
      "strict_equal_compute":False,"confirmed_general_debias_discoveries":0,
      "cost_per_confirmed_general_discovery":None}
    atomic_json(out/"summary.json",status)
    atomic_json(out/"usage.json",usage_summary(out))
    print("FROZEN_CONFIRMATION_RECOVERED",json.dumps({
      "task":"exposure","qwen_calls_replay":0,
      "confirmed_policy_value":confirm["independent_confirmation_of_specific_estimand"],
      "status":status["status"]}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",required=True)
    a=p.parse_args();recover(a.run)
