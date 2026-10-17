"""Prospective new-cohort AgentCF mechanism forecasts and numeric adjudication.

Separate the forecast contract (registered before model run) from mechanism
outcomes (observed only after frozen-state training and new model rankings).
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean
import json

from rsi_researcher.paths import ROOT
from rsi_researcher.datasets.author import author_data, select_users, author_profile

OLD=ROOT/"runs/qwen-author-dev20-baseline-20261015"
NEW=ROOT/"runs/agentcf-newstate-forecast-20261017"
USER_COUNT=6
EVENTS_PER_USER=2
COHORT_SEED=20261107
TASKS={
 "item_memory":{
    "experiment":"item_memory_ablation",
    "question":"Restoring only item memories to pretraining descriptions, with users and candidates unchanged, affects matched rankings beyond repeat noise.",
    "primary_metric":"excess_rank_change",
    "legal_metrics":["excess_rank_change","ndcg_delta","head_excess_delta"]
 },
 "popularity":{
    "experiment":"popularity_prompt",
    "question":"Explicit ranking-time popularity guidance alters head-item excess relative to candidates beyond repeat noise, without degrading known-target quality.",
    "primary_metric":"head_excess_delta",
    "legal_metrics":["head_excess_delta","ndcg_delta","excess_rank_change"]
 },
}
ARMS=["rule","qwen_no_history","qwen_verified_history"]
METRICS=["excess_rank_change","head_excess_delta","ndcg_delta"]
DIRECTIONS=["increase","decrease","approximately_zero"]
SCHEMA={"type":"object","properties":{
 "action":{"type":"string","enum":["item_memory_ablation","popularity_prompt"]},
 "metric":{"type":"string","enum":METRICS},
 "direction":{"type":"string","enum":DIRECTIONS},
 "minimum_abs_effect":{"type":"number","minimum":0,"maximum":0.5},
 "hypothesis":{"type":"string","minLength":15},
 "competing_explanation":{"type":"string","minLength":15},
 "falsifier":{"type":"string","minLength":15},
 "quality_constraint":{"type":"string","enum":["ndcg_99pct_floor_both_panels"]},
},"required":["action","metric","direction","minimum_abs_effect",
             "hypothesis","competing_explanation","falsifier","quality_constraint"],
 "additionalProperties":False}
PROTO="new-cohort-structured-forecast-v1"
MAX_REQUESTS=160


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def serialize(obj):return json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True)+"\n"


def atomic_new(path,obj):
    with Path(path).open("x",encoding="utf8") as f:
        f.write(serialize(obj))


def cohort_data():
    old=json.loads((OLD/"input.json").read_text(encoding="utf8"))
    excluded=set(old["initial"]["users"])
    source,events,queries=author_data(ROOT/"data",USER_COUNT,EVENTS_PER_USER,
                                     COHORT_SEED,"exclude_known",excluded_users=excluded)
    new_users=set(source["users"])
    if new_users & set(old["initial"]["users"]):
        raise ValueError("Fresh users overlap old twenty")
    if len(new_users)!=USER_COUNT or len(queries)!=USER_COUNT:
        raise ValueError("Incorrect fresh user selection")
    if len(events)!=USER_COUNT*EVENTS_PER_USER:
        raise ValueError("Incorrect fresh event count")
    return source,events,queries


def protocol():
    source,events,queries=cohort_data()
    fingerprints={}
    for p in (ROOT/"data/CDs-100-user-dense.train.inter",
              ROOT/"data/CDs-100-user-dense.valid.inter",
              ROOT/"data/CDs-100-user-dense.random",
              ROOT/"data/author_prompts.json",
              OLD/"input.json",
              ROOT/"rsi_researcher/recommenders/agentcf.py",
              ROOT/"rsi_researcher/datasets/author.py",
              ROOT/"rsi_researcher/research/fresh_predictions.py"):
        fingerprints[str(p.relative_to(ROOT))]=digest(p)
    return {"version":PROTO,"tasks":TASKS,"arms":ARMS,"forecast_schema":SCHEMA,
       "source_sha256":fingerprints,"cohort_users":sorted(source["users"]),
       "cohort_seed":COHORT_SEED,"selection":"exclude_known_from_prior_exact_user_ids",
       "training_events":len(events),"queries":len(queries),
       "split":{"A":[q["query_id"] for q in sorted(queries,key=lambda x:x["query_id"])[:3]],
                "B":[q["query_id"] for q in sorted(queries,key=lambda x:x["query_id"])[3:]]},
       "predeclared_numeric_contract":{
          "primary_metric_per_task":{k:v["primary_metric"] for k,v in TASKS.items()},
          "max_effect":0.5,"minimum_effect_inclusive":True,
          "null_band_absolute":0.02,"quality_floor_ratio":0.99,
          "repeat_control":"same-condition complete ranking repeat",
          "two_panels_required":True,
          "independent_confirmation":False,
          "the_new_training_state_is_distinct_but_same_source_dataset":True},
       "cost_cap":{"model_requests":MAX_REQUESTS,
                   "prompt_token_cap":850000,"wall_seconds":1250},
       "not_a_new_bias_task":"Same item memory/popularity mechanism families; fresh cohort does NOT give new mechanism generalization",
       "no_coat_random_holdout":True}


def freeze():
    if NEW.exists():raise FileExistsError("Never overwrite a frozen forecast pilot")
    # mkdir after protocol data integrity preconditions have passed
    frozen=protocol()
    NEW.mkdir(parents=True,exist_ok=False)
    atomic_new(NEW/"protocol.json",frozen)
    (NEW/"protocol.sha256").write_text(digest(NEW/"protocol.json")+"\n",encoding="ascii")
    return frozen


def verify():
    data=json.loads((NEW/"protocol.json").read_text(encoding="utf8"))
    if digest(NEW/"protocol.json")!=(NEW/"protocol.sha256").read_text().strip():
        raise ValueError("Frozen forecast file modified")
    if data!=protocol():raise ValueError("Forecast code or source changed after freeze")
    return data


def rule_forecast(task):
    return {"action":TASKS[task]["experiment"],
      "metric":TASKS[task]["primary_metric"],
      "direction":"increase" if task=="item_memory" else "decrease",
      "minimum_abs_effect":0.10 if task=="item_memory" else 0.02,
      "hypothesis":"A single registered intervention has a replicable effect after accounting for repeat ranking noise.",
      "competing_explanation":"Repeated rankings may vary because of stochastic sampling or learned user preferences.",
      "falsifier":"Effect must meet the numeric sign and minimum magnitude on BOTH frozen panels and pass the NDCG floor.",
      "quality_constraint":"ndcg_99pct_floor_both_panels"}


def validate(task,p):
    if not isinstance(p,dict) or set(p)!=set(SCHEMA["required"]):
        return False,"invalid_fields"
    if p.get("action")!=TASKS[task]["experiment"]:
        return False,"unregistered_action_for_task"
    if p.get("metric")!=TASKS[task]["primary_metric"]:
        return False,"wrong_primary_metric"
    if p.get("direction") not in DIRECTIONS:
        return False,"invalid_direction"
    t=p.get("minimum_abs_effect")
    if isinstance(t,bool) or not isinstance(t,(float,int)) or not 0.0<=t<=0.5:
        return False,"invalid_effect"
    if p.get("quality_constraint")!="ndcg_99pct_floor_both_panels":
        return False,"quality_constraint_missing"
    if not all(isinstance(p.get(k),str) and len(p[k].strip())>=15
               for k in ("hypothesis","competing_explanation","falsifier")):
        return False,"missing_rationale"
    return True,None


def score_panel(original,repeat,changed):
    a={row["query_id"]:row for row in original}
    b={row["query_id"]:row for row in repeat}
    c={row["query_id"]:row for row in changed}
    if not a or a.keys()!=b.keys() or a.keys()!=c.keys():
        raise ValueError("Required paired cohorts differ")
    for key in a:
        if not all(set(row["ranking"])==set(a[key]["ranking"]) and
                   len(row["ranking"])==10 and row["valid"]
                   for row in (a[key],b[key],c[key])):
            raise ValueError("Invalid or unmatched ranking")
    total=len(a)
    repeat_change=sum(a[k]["ranking"]!=b[k]["ranking"] for k in a)/total
    intervention_change=sum(a[k]["ranking"]!=c[k]["ranking"] for k in a)/total
    d={"excess_rank_change":intervention_change-repeat_change,
       "head_excess_delta":mean(c[k]["excess_popularity_exposure"]-
                                a[k]["excess_popularity_exposure"] for k in a),
       "ndcg_delta":mean(c[k]["ndcg_at_5"]-a[k]["ndcg_at_5"] for k in a)}
    baseline_ndcg=mean(a[k]["ndcg_at_5"] for k in a)
    changed_ndcg=mean(c[k]["ndcg_at_5"] for k in a)
    return {"n":total,**d,"repeat_change_rate":repeat_change,
            "intervention_change_rate":intervention_change,
            "baseline_NDCG5":baseline_ndcg,"candidate_NDCG5":changed_ndcg,
            "quality_pass":baseline_ndcg>0 and changed_ndcg>=0.99*baseline_ndcg}


def score_contract(forecast,panels):
    if set(panels)!={"A","B"}:
        raise ValueError("Both predeclared panels A and B required")
    for cohort,panel in panels.items():
        if not panel or "quality_pass" not in panel:
            raise ValueError("Missing programmatic panel evidence")
    metric=forecast["metric"]
    minimum=forecast["minimum_abs_effect"]
    def directional(value):
        if forecast["direction"]=="increase":return value>=minimum
        if forecast["direction"]=="decrease":return value<=-minimum
        return abs(value)<=0.02
    checks={cohort:{"metric_value":panel[metric],
       "directional_prediction_matches":directional(panel[metric]),
       "quality_pass":panel["quality_pass"]}
        for cohort,panel in panels.items()}
    valid=all(x["directional_prediction_matches"] and x["quality_pass"]
              for x in checks.values()) and len(checks)==2
    return {"both_panels_pass":valid,"panels":checks,
      "formal_development_prediction_supported":valid,
      "independent_discovery":False,
      "note":"Same dataset and only one fresh training state; this is conditional development evidence"}


def retrospective_evaluate(decisions,panels_by_task):
    evaluated={}
    for arm,forecasts in decisions.items():
        evaluated[arm]={}
        for task,proposal in forecasts.items():
            ok,why=validate(task,proposal)
            if not ok:raise ValueError(arm+" "+task+" "+str(why))
            panel=panels_by_task[task]
            evaluated[arm][task]=score_contract(proposal,panel)
    return evaluated
