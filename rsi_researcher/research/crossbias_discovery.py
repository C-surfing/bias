"""Registered Coat/AgentCF discovery experiments, prospective development only.

Task definitions and falsification actions are frozen BEFORE Qwen plans.
Coat uses disjoint *development* user panels, never randomized holdout.
AgentCF reranks one frozen author-trained state; Qwen model is a component
under test, not an oracle/quality judge.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from statistics import mean
from random import Random
import json,math,time

ROOT=Path("/data/cjh/rsi/project")
RUN=ROOT/"runs/crossbias-discovery-pilot-20261017"
AUTHOR=ROOT/"runs/qwen-author-dev20-baseline-20261015"
CATEGORIES={
 "coat_propensity":{
   "domain":"Coat Selection","actions":["gamma_stress","propensity_shuffle"],
   "primary":"Brier on randomized development users",
   "question":"Does weighted evaluation improvement survive estimated propensity misspecification rather than arbitrary propensity mapping?",
   "falsifier":"Direction/benefit must persist across both predeclared development user panels and relevant negative controls."
 },
 "coat_utility":{
   "domain":"Coat Selection","actions":["split_utility","gamma_stress"],
   "primary":"Brier with >=99% unweighted NDCG@5 and tracked tail-recall",
   "question":"Is an apparent calibration gain accompanied by ranking-quality or long-tail damage?",
   "falsifier":"Benefit must be replicated without violating the quality safeguard, or mechanism claim is rejected."
 },
 "agentcf_memory":{
   "domain":"AgentCF Item Memory","actions":["item_memory_ablation","popularity_prompt"],
   "primary":"paired ranking membership + NDCG, and repeat noise",
   "question":"Does shared Item Memory itself affect downstream rankings when User Memory and queries are held constant?",
   "falsifier":"Item memory ablation must change rankings beyond same-condition repeat noise on both disjoint user panels."
 },
 "agentcf_popularity":{
   "domain":"AgentCF Popularity","actions":["popularity_prompt","item_memory_ablation"],
   "primary":"head excess exposure, NDCG and repeat noise",
   "question":"Does explicit popularity/reputation wording increase head exposure in AgentCF, independent of learned memory?",
   "falsifier":"Removing only popularity wording must lower head excess beyond noise without violating NDCG safeguard on both panels."
 }
}
ALL_ACTIONS=sorted({a for cat in CATEGORIES.values() for a in cat["actions"]})
RULE={key:cat["actions"][0] for key,cat in CATEGORIES.items()}
ARMS=("rule","qwen_no_history","qwen_related_memory")
PROTOCOL_VERSION="crossbias-discovery-pilot-v1"
N_AUTHOR_QUERIES=10
SCHEMA={
 "type":"object","properties":{
  "action":{"type":"string","enum":ALL_ACTIONS},
  "hypothesis":{"type":"string","minLength":12},
  "competing_explanation":{"type":"string","minLength":12},
  "falsifier":{"type":"string","minLength":12},
  "predicted_direction":{"type":"string","enum":["decrease","increase","no_change","uncertain"]},
  "budget_justification":{"type":"string","minLength":8}
 },"required":["action","hypothesis","competing_explanation","falsifier",
               "predicted_direction","budget_justification"],"additionalProperties":False
}
REFLECT_SCHEMA={"type":"object","properties":{
 "evidence_state":{"type":"string","enum":["supported","weakened","inconclusive"]},
 "lesson":{"type":"string","minLength":10},
 "remaining_competing_explanation":{"type":"string","minLength":10},
 "next_discriminating_test":{"type":"string","minLength":10}
},"required":["evidence_state","lesson","remaining_competing_explanation",
               "next_discriminating_test"],"additionalProperties":False}


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"))


def sha_file(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def source_manifest():
    from rsi_researcher.research.coat_selection import DATA
    p=[ROOT/"configs/bias_tasks.json",
       ROOT/"rsi_researcher/research/coat_selection.py",
       ROOT/"rsi_researcher/research/author_experiment.py",
       ROOT/"rsi_researcher/recommenders/agentcf.py",
       AUTHOR/"original_update/state.json",
       AUTHOR/"input.json",AUTHOR/"data_profile.json",
       DATA/"propensities.ascii"]
    return {str(x.relative_to(ROOT)):sha_file(x) for x in p}


def frozen_protocol():
    return {
      "protocol":PROTOCOL_VERSION,"tasks":CATEGORIES,
      "registered_actions":ALL_ACTIONS,"rule":RULE,"arms":list(ARMS),
      "source_sha256":source_manifest(),
      "author_query_count":N_AUTHOR_QUERIES,
      "author_source":"frozen 20-user author training state; subset of 10 validation queries",
      "coat_scope":"historically used random DEVELOPMENT panel only; new within-panel split NOT independent holdout",
      "coat_user_partition_seed":20261017,
      "author_panel_seed":20261017,
      "minimum_replicated_coat_brier_gain":1e-4,
      "ranking_change_min_fraction":0.2,
      "popularity_mean_excess_reduction_min":0.02,
      "quality_floor_ratio":0.99,
      "agentcf_stochastic_repeat_required":True,
      "qwen_planner_calls_per_arm":4,
      "qwen_reflection_calls_per_arm":4,
      "hypothesis_success_grade":"replicated_development_mechanism_only; NEVER independent scientific discovery",
      "predeclared_cost_accounting":"separate physical calls vs logical action-evaluation costs",
    }


def save_new(path,obj):
    with Path(path).open("x",encoding="utf8") as f:
        f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n")


def freeze():
    RUN.mkdir(parents=True,exist_ok=False)
    p=frozen_protocol()
    save_new(RUN/"protocol.json",p)
    (RUN/"protocol.sha256").write_text(sha_file(RUN/"protocol.json")+"\n",encoding="ascii")
    return p


def verify():
    p=json.loads((RUN/"protocol.json").read_text(encoding="utf8"))
    if sha_file(RUN/"protocol.json")!=(RUN/"protocol.sha256").read_text().strip():
        raise ValueError("Frozen protocol hash changed")
    if p!=frozen_protocol():
        raise ValueError("Sources or protocol changed since freeze")
    return p


def author_inputs():
    source=json.loads((AUTHOR/"input.json").read_text(encoding="utf8"))
    state=json.loads((AUTHOR/"original_update/state.json").read_text(encoding="utf8"))
    profile=json.loads((AUTHOR/"data_profile.json").read_text(encoding="utf8"))
    queries=source["queries"]
    if len(queries)!=20:raise ValueError("Expected fixed 20-user author cohort")
    chosen=sorted(queries,key=lambda x:sha256(
        (x["query_id"]+"-20261017").encode()).hexdigest())[:N_AUTHOR_QUERIES]
    chosen=sorted(chosen,key=lambda x:x["query_id"])
    if len({x["user"] for x in chosen})!=N_AUTHOR_QUERIES:
        raise ValueError("Author paired cohort has duplicate users")
    return source,state,profile,chosen


def coat_inputs():
    from rsi_researcher.datasets.coat import load_coat
    from rsi_researcher.research.coat_selection import DATA,SEED,estimated_p_matrix
    ds=load_coat(DATA,split_seed=SEED)
    if (len(ds["train"]),len(ds["development"]),len(ds["test"]))!=(6960,2065,2209):
        raise ValueError("Coat partition changed")
    users=sorted({r["user"] for r in ds["development"]},
         key=lambda u:sha256((str(u)+"_20261017").encode()).hexdigest())
    halves=[set(users[:len(users)//2]),set(users[len(users)//2:])]
    if not all(len(s)>20 for s in halves):raise ValueError("Too few development users")
    panels=[[r for r in ds["development"] if r["user"] in group] for group in halves]
    return ds,estimated_p_matrix(),panels


def coat_observation(action):
    from rsi_researcher.research.coat_selection import fit_item_mean,metrics
    if action not in ("gamma_stress","propensity_shuffle","split_utility"):
        raise ValueError("Invalid Coat action")
    ds,p,panels=coat_inputs()
    counts=Counter(r["item"] for r in ds["train"])
    counts={i:counts.get(i,0) for i in ds["item_ids"]}
    configs=[("unweighted",None,None)]
    if action in ("gamma_stress","split_utility"):
        configs.extend([("gamma1_cap20",p,20),("gamma1_cap100",p,100)])
        if action=="gamma_stress":
            configs.extend((f"gamma{g}_cap{cap}",
                            [[max(1e-8,min(1.0,v**g)) for v in row] for row in p],cap)
                           for g in (0.75,1.25) for cap in (20,100))
    else:
        configs.append(("matched_propensity",p,100))
        for seed in (31,47):
            shuffled=deepcopy(p)
            Random(seed).shuffle(shuffled)
            configs.append((f"user_permuted_propensity_{seed}",shuffled,100))
    rows={}
    for label,matrix,cap in configs:
        predicted,fit=fit_item_mean(ds["train"],ds["item_ids"],matrix,cap)
        rows[label]={"ess":fit["effective_train_rows"],"panels":[]}
        for panel in panels:
            scored=metrics(panel,predicted,counts,ds["item_ids"])
            rows[label]["panels"].append({
                "rows":len(panel),"users":len(set(r["user"] for r in panel)),
                "Brier":scored["Brier_random_development"],
                "NDCG5":scored["NDCG5_random_development"],
                "tail_relevant_recall":scored["tail_relevant_recall"]})
    report={"action":action,"status":"EXECUTED_DEVELOPMENT",
        "method":"same shared training fitted item means, disjoint development users",
        "records":rows,"confirmation_used":False,
        "source":"Coat original random development already used in prior exploratory runs"}
    return report


def make_evidence():
    from rsi_researcher.research.coat_selection import DATA
    from rsi_researcher.research.agentcf_loop import bias_summary
    ds,p,panels=coat_inputs()
    _,state,profile,chosen=author_inputs()
    source_summary=bias_summary(AUTHOR)
    changed=sum(state["items"][i]["memory"]!=
        json.loads((AUTHOR/"input.json").read_text(encoding="utf8"))["initial"]["items"][i]["memory"]
        for i in {x for q in chosen for x in q["candidates"]})
    return {
      "coat":{"training_rows":len(ds["train"]),
        "random_development_rows":len(ds["development"]),
        "development_user_panels":[{"users":len({r["user"] for r in x}),
                                    "rows":len(x)} for x in panels],
        "estimated_propensities_not_known_assignment":True,
        "training_only_propensity_min":min(p[int(x["user"])][int(x["item"])] for x in ds["train"]),
        "historical_hypothesis":"IPS clipping improves calibration on original development but may have NDCG/tail cost; historical outcomes must not be treated as new confirmation"},
      "agentcf":{"dataset":"CDs author training and 20-user development query cohort",
        "queries_selected":len(chosen),
        "trained_state_frozen":True,
        "ranked_candidates":10,
        "candidate_item_memories_different_from_initial":changed,
        "head_excess_prior_observational":source_summary["quality_popularity"]["excess_popularity_exposure"],
        "known_target_only":True,
        "causal_popularity_unidentified":True}
    }


def rule_plans():
    return {t:{
       "action":RULE[t],
       "hypothesis":"Targeted controlled perturbation affects measured development mechanism while preserving quality",
       "competing_explanation":"Sample noise, prompt randomness or previously tuned development parameters",
       "falsifier":"Effect absent in second disjoint development cohort or quality control fails",
       "predicted_direction":"uncertain",
       "budget_justification":"One registered falsification action at a fixed bounded cost"
    } for t in CATEGORIES}


def valid_plan(task,plan):
    if not isinstance(plan,dict) or set(plan)!=set(SCHEMA["required"]):
        return False,"invalid_fields"
    if plan.get("action") not in CATEGORIES[task]["actions"]:
        return False,"unregistered_task_action"
    if any(not isinstance(plan.get(field),str) or len(plan[field].strip())<8
         for field in ("hypothesis","competing_explanation","falsifier","budget_justification")):
        return False,"missing_substantive_research_reason"
    if plan.get("predicted_direction") not in SCHEMA["properties"]["predicted_direction"]["enum"]:
        return False,"invalid_direction"
    return True,None


def evaluate_coat(action,data):
    rows=data["records"];baseline=rows["unweighted"]["panels"]
    if action=="split_utility":
        weighted=rows["gamma1_cap100"]["panels"]
        per=[{"brier_delta":c["Brier"]-b["Brier"],
              "ndcg_ratio":c["NDCG5"]/b["NDCG5"] if b["NDCG5"] else None,
              "tail_delta":c["tail_relevant_recall"]-b["tail_relevant_recall"]
              if c["tail_relevant_recall"] is not None and b["tail_relevant_recall"] is not None else None}
              for b,c in zip(baseline,weighted)]
        passes=all(x["brier_delta"]<=-1e-4 and x["ndcg_ratio"] is not None
                   and x["ndcg_ratio"]>=.99 for x in per)
        return {"test":"per_split_quality_safeguard","cohorts":per,"passes":passes}
    if action=="gamma_stress":
        winners=[]
        for k in range(2):
            best=min(("gamma0.75_cap20","gamma0.75_cap100","gamma1_cap20",
                      "gamma1_cap100","gamma1.25_cap20","gamma1.25_cap100"),
                     key=lambda label:rows[label]["panels"][k]["Brier"])
            winners.append(best)
        return {"test":"gamma_cross_panel_stability",
                "best_weighting_by_panel":winners,
                "passes":winners[0]==winners[1] and "gamma1_" in winners[0],
                "interpretation":"Strict invariance under misspecification and both disjoint development user panels"}
    matched=rows["matched_propensity"]["panels"]
    negative=[rows[k]["panels"] for k in rows if k.startswith("user_permuted")]
    differences=[]
    for k in range(2):
        best_placebo=min(x[k]["Brier"] for x in negative)
        differences.append({"matched_Brier":matched[k]["Brier"],"best_placebo_Brier":best_placebo,
            "delta_matched_minus_placebo":matched[k]["Brier"]-best_placebo})
    return {"test":"wrong_propensity_negative_control","cohorts":differences,
            "passes":all(x["delta_matched_minus_placebo"]<=-1e-4 for x in differences),
            "interpretation":"A shuffled propensity is intentionally invalid; improvement alone cannot identify causality"}


def evaluate_author(action,record):
    panels=record["panels"]
    results=[]
    for group in ("a","b"):
        rows=panels[group]
        original=rows["original"]
        repeat=rows["repeat"]
        candidate=rows[action]
        match=mean(x["ranking"]==y["ranking"] for x,y in zip(original,repeat))
        change=mean(x["ranking"]!=y["ranking"] for x,y in zip(original,candidate))
        before=mean(x["excess_popularity_exposure"] for x in original)
        after=mean(x["excess_popularity_exposure"] for x in candidate)
        ndcg0=mean(x["ndcg_at_5"] for x in original)
        ndcg1=mean(x["ndcg_at_5"] for x in candidate)
        results.append({"cohort":group,"queries":len(original),"repeat_exact_agreement":match,
          "candidate_ranking_change_rate":change,
          "head_excess_delta":after-before,"NDCG5_baseline":ndcg0,
          "NDCG5_candidate":ndcg1,
          "NDCG5_quality_pass":ndcg0>0 and ndcg1>=.99*ndcg0})
    if action=="item_memory_ablation":
        qualifies=all(x["repeat_exact_agreement"]>=.8
                      and x["candidate_ranking_change_rate"]>=.2
                      and x["NDCG5_quality_pass"] for x in results)
    else:
        qualifies=all(x["repeat_exact_agreement"]>=.8
                       and x["head_excess_delta"]<=-.02
                       and x["NDCG5_quality_pass"] for x in results)
    return {"test":action+"_paired_falsifier","cohorts":results,"passes":qualifies,
            "limitations":"10 author development users; stochastic repeat and shared trained item memory; outcome relevance only one known target per user"}


def assess(task,action,data):
    if action not in CATEGORIES[task]["actions"]:raise ValueError("Unregistered task action")
    if CATEGORIES[task]["domain"].startswith("Coat"):
        verdict=evaluate_coat(action,data)
    else:
        verdict=evaluate_author(action,data)
    # Even strict within-development replication cannot produce independent confirmation.
    return {"task":task,"action":action,"falsifier":verdict,
      "replicated_development_signal":bool(verdict["passes"]),
      "independently_confirmed_scientific_discovery":False,
      "evidence_grade":"development_only_conditional",
      "reason":"All data/tasks were developed from previously consulted panels/states; no independent confirmatory evidence"}


def aggregate(plans,results,costs):
    output={}
    for arm in ARMS:
        rows=[results[arm][t] for t in CATEGORIES]
        supported=sum(row["replicated_development_signal"] for row in rows)
        output[arm]={"tasks":len(rows),
           "valid_proposals":sum(valid_plan(t,plans[arm][t])[0] for t in CATEGORIES),
           "falsification_experiments_completed":sum("falsifier" in x for x in rows),
           "development_signals_replicated":supported,
           "independent_valid_scientific_discoveries":0,
           "strict_discoveries_per_100_llm_calls":0,
           "development_signals_per_100_llm_calls":None if costs[arm]["llm_calls"]==0
                 else 100*supported/costs[arm]["llm_calls"],
           "costs":costs[arm]}
    return output
