"""Synthetic unseen-decision benchmark: exposure-OPE research actions.

The researcher observes a single diagnostic log, never simulator reward truth or
evaluation repetitions. Public task data and hidden scoring are kept in separate
structures. Development cases are disjoint from evaluation scenario identifiers.
This is *simulated research decision making*, not proof of real-world RSI.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
from random import Random
from statistics import mean
import json
import math

from rsi_researcher.research.exposure_synthetic import USERS, ITEMS, q, choose

ROOT=Path("/data/cjh/rsi/project")
VERSION="exposure-decision-benchmark-v1"
SEED=20261017
MODES=("matched","head_shift","uniform_shift","tail_shift",
       "sparse_shift","missing_propensity","unsupported")
ACTIONS=("naive","ips","snips","abstain")
REPETITIONS=24
ABSTAIN_LOSS=0.018
INVALID_LOSS=1.0


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                             ensure_ascii=False).encode("utf8")).hexdigest()


def probabilities(mode,variant):
    strength=1.6+(variant%4)*0.6
    eps=0.04+(variant%3)*0.045
    x=[math.exp(strength*(i/(ITEMS-1)-1)) for i in range(ITEMS)]
    z=sum(x)
    p=[eps/ITEMS+(1-eps)*v/z for v in x]
    if mode=="unsupported":
        z=1-p[0]
        p=[0.0]+[v/z for v in p[1:]]
    return p


def target(u,mode,variant,logging):
    if mode=="matched":
        return logging[:]
    if mode=="head_shift":
        weights=[math.exp(0.18*i) for i in range(ITEMS)]
    elif mode=="uniform_shift" or mode=="missing_propensity":
        weights=[1.0]*ITEMS
    elif mode=="tail_shift" or mode=="unsupported":
        weights=[math.exp(-0.23*i) for i in range(ITEMS)]
    else:
        center=(u*3+variant*7)%ITEMS
        weights=[math.exp(-0.30*min(abs(i-center),ITEMS-abs(i-center))**2)
                 for i in range(ITEMS)]
    z=sum(weights)
    return [v/z for v in weights]


def spec(mode,variant,stage):
    if mode not in MODES or stage not in ("development","evaluation"):
        raise ValueError("unknown benchmark mode or split")
    return {"task_id":f"{stage}-{mode}-{variant:02d}","stage":stage,
            "mode":mode,"variant":variant,
            "n":(200,450,900)[variant%3],
            "logging_propensities_visible":mode!="missing_propensity"}


def sample(sc,seed):
    rng=Random(seed)
    logging=probabilities(sc["mode"],sc["variant"])
    observations=[]
    for _ in range(sc["n"]):
        u=rng.randrange(USERS)
        item=choose(logging,rng)
        reward=int(rng.random()<q(u,item))
        observations.append((u,item,reward))
    return observations


def estimator(sc,observations):
    mode=sc["mode"];variant=sc["variant"]
    logging=probabilities(mode,variant)
    raw=mean(row[2] for row in observations)
    if not sc["logging_propensities_visible"]:
        return {"naive":raw,"ips":None,"snips":None,"ess":None,"max_weight":None,
                "supported":None}
    numerator=denom=sq=0.0
    supported=all(not (p==0 and any(target(u,mode,variant,logging)[i]>1e-12
                          for u in range(USERS))) for i,p in enumerate(logging))
    for u,i,reward in observations:
        if logging[i]<=0:
            raise ValueError("Observed logging probability cannot be zero")
        w=target(u,mode,variant,logging)[i]/logging[i]
        numerator+=w*reward;denom+=w;sq+=w*w
    return {"naive":raw,
            "ips":numerator/len(observations) if supported else None,
            "snips":numerator/denom if denom and supported else None,
            "ess":denom*denom/sq if sq else 0.0,
            "max_weight":max(target(u,mode,variant,logging)[i]/logging[i]
                for u,i,_ in observations),
            "supported":supported}


def true_target(sc):
    logging=probabilities(sc["mode"],sc["variant"])
    return sum(sum(target(u,sc["mode"],sc["variant"],logging)[i]*q(u,i)
               for i in range(ITEMS)) for u in range(USERS))/USERS


def diagnostic(sc):
    data=sample(sc,SEED*1000+sc["variant"]*137+MODES.index(sc["mode"])*11
                      +(0 if sc["stage"]=="development" else 1000003))
    obs=estimator(sc,data)
    disclosed={
      "task_id":sc["task_id"],"dataset":"synthetic_exposure_development",
      "n_logged":sc["n"],"requested_policy":{"matched":"logging","head_shift":"head","uniform_shift":"uniform","tail_shift":"tail","sparse_shift":"user_conditional","missing_propensity":"uniform","unsupported":"tail"}[sc["mode"]],
      "logging_propensities_available":sc["logging_propensities_visible"],
      "observed_logged_click_rate":round(obs["naive"],6),
      "allowed_actions":list(ACTIONS),
      "identification_note":"Observed rewards only on actually logged impressions",
    }
    if obs["supported"] is not None:
        disclosed.update({"positivity_verified":obs["supported"],
            "diagnostic_effective_sample_size":round(obs["ess"],2),
            "diagnostic_max_weight":round(obs["max_weight"],2),
            "diagnostic_ips":round(obs["ips"],6) if obs["ips"] is not None else None,
            "diagnostic_snips":round(obs["snips"],6) if obs["snips"] is not None else None})
    # No q, hidden true policy value, holdout rewards or optimal decision exposed.
    assert not any(k in disclosed for k in ("oracle","true_value","best_action"))
    return disclosed


def risk_table(sc,repetitions=REPETITIONS):
    if repetitions<2: raise ValueError("At least two independent repetitions required")
    visible=sc["logging_propensities_visible"]
    support=estimator(sc,sample(sc,SEED+sc["variant"]+7))["supported"]
    identified=bool(visible and support)
    truth=true_target(sc)
    losses=defaultdict(list)
    for replicate in range(repetitions):
        observations=sample(sc,SEED*311+100000*MODES.index(sc["mode"])
                    +sc["variant"]*1009+replicate*19
                    +(0 if sc["stage"]=="development" else 70000000))
        row=estimator(sc,observations)
        for action in ACTIONS:
            if action=="abstain":
                losses[action].append(0.0 if not identified else ABSTAIN_LOSS)
            elif not identified:
                losses[action].append(INVALID_LOSS)
            else:
                losses[action].append((row[action]-truth)**2)
    averaged={key:mean(val) for key,val in losses.items()}
    best=min(ACTIONS,key=lambda a:(averaged[a],ACTIONS.index(a)))
    return {"loss":averaged,"best_action":best,"identified":identified,
            "oracle_hidden_policy_value":truth}


def tasks():
    dev=[spec(m,i,"development") for i,m in enumerate(MODES)]
    evaluation=[spec(m,30+j*7+i,"evaluation")
                for j in range(2) for i,m in enumerate(MODES)]
    Random(SEED).shuffle(evaluation)
    if {x["task_id"] for x in dev}&{x["task_id"] for x in evaluation}:
        raise AssertionError("Development/evaluation IDs overlap")
    return dev,evaluation


def frozen_payload():
    dev,eval_cases=tasks()
    public_development=[{"task":diagnostic(sc),
                         "development_feedback":risk_table(sc)["best_action"]}
                        for sc in dev]
    public_evaluation=[diagnostic(sc) for sc in eval_cases]
    return {
      "version":VERSION,"seed":SEED,"allowed_actions":list(ACTIONS),
      "development_tasks":public_development,
      "evaluation_tasks":public_evaluation,
      "scenario_counts":{"development":len(dev),"evaluation":len(eval_cases)},
      "budget":{"max_per_arm_decisions":len(eval_cases),
                "max_model_calls_per_qwen_arm":len(eval_cases)+1,
                "scoring_repetitions":REPETITIONS},
      "disclosure":"No oracle values or evaluation outcomes are passed to the planner",
      "scope":"controlled synthetic exposure; no real-data debias or RSI confirmation"
    }


def rule(task):
    if not task["logging_propensities_available"]:
        return "abstain"
    if task.get("positivity_verified") is not True:
        return "abstain"
    if task["requested_policy"]=="matched":
        return "naive"
    if task["diagnostic_effective_sample_size"]<0.06*task["n_logged"]:
        return "abstain"
    return "snips"


def shuffled_cards(public):
    cards=[{"mechanism":e["task"]["requested_policy"],
            "action":e["development_feedback"]} for e in public]
    values=[c["action"] for c in cards]
    rotate=values[2:]+values[:2]
    return [dict(card,action=rotate[i]) for i,card in enumerate(cards)]


def score_arm(actions,eval_cases=None):
    _,cases=tasks()
    if eval_cases is not None:
        cases=eval_cases
    if len(actions)!=len(cases):
        raise ValueError("One registered action per frozen evaluation task required")
    rows=[]
    for sc,action in zip(cases,actions):
        if action not in ACTIONS:
            raise ValueError("Unregistered research action")
        outcomes=risk_table(sc)
        regret=outcomes["loss"][action]-min(outcomes["loss"].values())
        rows.append({"task_id":sc["task_id"],"mode":sc["mode"],
             "action":action,"best_hidden_action":outcomes["best_action"],
             "selected_risk":outcomes["loss"][action],"oracle_best_risk":min(outcomes["loss"].values()),
             "regret":regret,"identified":outcomes["identified"],
             "wrong_nonabstention":not outcomes["identified"] and action!="abstain",
             "unnecessary_abstention":outcomes["identified"] and action=="abstain"})
    return {"mean_regret":mean(x["regret"] for x in rows),
            "mean_selected_risk":mean(x["selected_risk"] for x in rows),
            "action_agreement":sum(x["action"]==x["best_hidden_action"] for x in rows)/len(rows),
            "invalid_causal_decisions":sum(x["wrong_nonabstention"] for x in rows),
            "unnecessary_abstentions":sum(x["unnecessary_abstention"] for x in rows),
            "rows":rows}


def freeze(output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/"runs"):
        raise ValueError("Output must be under project/runs")
    payload=frozen_payload()
    output.mkdir(parents=True,exist_ok=False)
    serialized=json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2)+"\n"
    (output/"frozen_protocol.json").write_text(serialized,encoding="utf8")
    (output/"protocol.sha256").write_text(sha256(serialized.encode()).hexdigest()+"\n",encoding="ascii")
    rule_result=score_arm([rule(c) for c in payload["evaluation_tasks"]])
    summary={"status":"PROTOCOL_FROZEN_RULE_BASELINE_COMPLETE",
       "created_utc":datetime.now(timezone.utc).isoformat(),
       "rule":rule_result,"model_calls":0,"no_qwen_claim":True,
       "evaluation_cases":len(payload["evaluation_tasks"]),
       "external_confirmation":False}
    (output/"rule_baseline.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    return summary


def read_frozen(output):
    output=Path(output).resolve()
    protocol=output/"frozen_protocol.json"
    expected=(output/"protocol.sha256").read_text(encoding="ascii").strip()
    if sha256(protocol.read_bytes()).hexdigest()!=expected:
        raise ValueError("Frozen protocol tampered")
    payload=json.loads(protocol.read_text(encoding="utf8"))
    if payload!=frozen_payload():
        raise ValueError("Frozen task generator has changed since freeze")
    return payload
