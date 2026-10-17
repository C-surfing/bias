"""Prospective bounded cross-Bias hypothesis -> falsifier -> evidence -> reflection.

Uses one owned, short-lived Qwen3-8B process. Does not train AgentCF, score the
Coat random holdout, modify shared server state, or grant Qwen filesystem tools.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from time import monotonic,sleep
import json,os,signal,socket,subprocess,traceback
from urllib.request import build_opener,ProxyHandler
from urllib.error import URLError

from rsi_researcher.research import crossbias_discovery as b
from rsi_researcher.research.decision_qwen_pilot import shutdown_own,gpu_state,GPU_UUID
from rsi_researcher.runtime.models import create_model
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.client import atomic_json,usage_summary,TEXT_SCHEMA,object_schema
from rsi_researcher.recommenders.agentcf import AgentCF
from rsi_researcher.research.author_experiment import author_intervention
from rsi_researcher.evaluation.author import score_author_ranking

PORT=18329
MAX_SECONDS=1250
MAX_REQUESTS=100
SYS_DESCRIPTION=("Researcher works with recommendation-selection bias and AgentCF. "
 "This is a controlled, exploratory development experiment; propose one "
 "falsifiable hypothesis, a competing explanation, one registered discriminating "
 "experiment, and a testable failure condition. Avoid unjustified causal claims. "
 "Historical reports are fallible and are not independent new evidence. "
 "Do not request shell actions, training, hidden labels or new data.")
HIST={
 "coat_propensity":"Prior Coat gamma sensitivity exploratory: optimal clipping shifted under gamma=1.25; estimated propensities are not logging-truth.",
 "coat_utility":"Prior Coat development Brier improved for cap100, but Hit5 and long-tail recall declined; strong utility constraints matter.",
 "agentcf_memory":"Prior AgentCF author run suggests false dislike statements propagate through shared item/user notes, but there is no human gold label.",
 "agentcf_popularity":"Historical author popularity-neutral prompt probe showed candidate-mix and prompt stochasticity can masquerade as bias; head share alone is not fairness."
}


def now():return datetime.now(timezone.utc).isoformat()


def persist(name,obj):
    atomic_json(b.RUN/name,obj)


def proposal_text(task,context,arm):
    memory=[] if arm=="qwen_no_history" else [HIST[task]]
    return json.dumps({
       "research_task_id":task,"domain":b.CATEGORIES[task]["domain"],
       "question":b.CATEGORIES[task]["question"],
       "primary_metric":b.CATEGORIES[task]["primary"],
       "available_experiments":b.CATEGORIES[task]["actions"],
       "required_falsifier":b.CATEGORIES[task]["falsifier"],
       "observed_development_evidence":context["coat" if task.startswith("coat") else "agentcf"],
       "historical_development_lessons":memory,
       "constraint":"Choose exactly one available experiment, all other assumptions held fixed. No test outcomes visible."},
       ensure_ascii=False)


def ask_plan(model,task,context,arm):
    model.scope={"phase":"proposal","arm":arm,"task":task}
    plan=model.ask(SYS_DESCRIPTION+"\n"+proposal_text(task,context,arm),
                   b.SCHEMA,{"stage":"propose"})
    ok,reason=b.valid_plan(task,plan)
    return plan,ok,reason


def run_author_rankings(model):
    source,state,profile,selected=b.author_inputs()
    prompts=source["prompts"]
    initial=source["initial"]
    mutated=deepcopy(state)
    changed=0
    relevant={i for q in selected for i in q["candidates"]}
    for item in relevant:
        old=initial["items"][item]["memory"]
        if mutated["items"][item]["memory"]!=old:changed+=1
        mutated["items"][item]["memory"]=old
    if not changed:
        raise ValueError("No candidate item memory differs from initial; cannot test ablation")
    neutral=author_intervention(prompts,"neutralize_rank_popularity")
    conditions={
      "original":(state,prompts),
      "repeat":(state,prompts),
      "item_memory_ablation":(mutated,prompts),
      "popularity_prompt":(state,neutral),
    }
    rows={"a":{},"b":{}}
    chosen_sorted=sorted(selected,key=lambda q:q["query_id"])
    halves={"a":chosen_sorted[:5],"b":chosen_sorted[5:]}
    if any(len(x)!=5 for x in halves.values()):raise ValueError("Author panel split changed")
    start=model.requests
    for group,queries in halves.items():
        for condition,(current,prompt) in conditions.items():
            kernel=AgentCF(model,prompt)
            panel=[]
            for query in queries:
                model.scope={"phase":"falsification","domain":"AgentCF",
                              "action":condition,"cohort":group}
                before=model.requests
                try:
                    ranking=kernel.rank(current,query)
                    if set(ranking)!=set(query["candidates"]):
                        raise ValueError("Invalid ranking membership after repair")
                    panel.append({"query_id":query["query_id"],
                         "ranking":ranking,"valid":True,
                         **score_author_ranking(ranking,query,profile),
                         "calls_used":model.requests-before})
                except Exception as exc:
                    # Preserve failure rather than impute a fabricated ranking.
                    panel.append({"query_id":query["query_id"],
                        "valid":False,"error":type(exc).__name__,
                        "ranking":[],"calls_used":model.requests-before})
                if panel[-1]["valid"] is False:
                    raise RuntimeError("Author ranking panel incomplete: "+
                                       str(panel[-1]["error"]))
            rows[group][condition]=panel
            print(json.dumps({"phase":"author","cohort":group,
                               "condition":condition,"rankings":len(panel),
                               "model_calls":model.requests-start}),flush=True)
    return {"status":"NEW_PAIRED_QWEN_RERANK_COMPLETE",
       "source":str(b.AUTHOR),"state_source_sha256":b.sha_file(b.AUTHOR/"original_update/state.json"),
       "n_queries":len(selected),"changed_candidate_item_memories":changed,
       "shared_training_state":True,"no_training_calls":True,"panels":rows,
       "physical_rank_model_requests":model.requests-start,
       "limitations":[
        "Ten fixed author development targets; other candidate labels unknown",
        "Item memory ablation resets trained item notes, not a full re-training or online exposure intervention",
        "Popularity prompt is a ranking-stage text intervention, not proof of causal population popularity bias",
        "Conditional on a shared trained user/item memory state; no independent confirmation"
       ]}


def main():
    if b.RUN.exists():raise FileExistsError("Cannot overwrite prospective run")
    status={"status":"STARTING","created_utc":now(),
        "gpu_uuid":GPU_UUID,"port":PORT,"calls":0,
        "confirmation_used":False,"server_owned":True,
        "model_max_requests":MAX_REQUESTS,
        "max_wall_seconds":MAX_SECONDS}
    b.freeze()
    persist("status.json",status)
    svc=None
    start=monotonic()
    model=None
    try:
        inputs=b.make_evidence()
        persist("public_evidence.json",inputs)
        mem,util=gpu_state()
        if mem>=512 or util!=0:raise RuntimeError("Selected GPU occupied; no service launched")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1",PORT))==0:
                raise RuntimeError("Own port unavailable")
        env=os.environ.copy()
        env.update(RSI_BASE="/data/cjh/rsi",RSI_GPU_UUID=GPU_UUID,
                   RSI_MODEL_PORT=str(PORT),PYTHONUNBUFFERED="1")
        log=(b.RUN/"vllm.log").open("x",encoding="utf8")
        try:
            svc=subprocess.Popen(["bash","/data/cjh/rsi/deployment/qwen3/serve.sh"],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        finally:log.close()
        status["service_pid"]=svc.pid;persist("status.json",status)
        opener=build_opener(ProxyHandler({}))
        while monotonic()-start<420:
            if svc.poll() is not None:raise RuntimeError("Owned model exited unexpectedly")
            try:
                with opener.open(f"http://127.0.0.1:{PORT}/health",timeout=3) as response:
                    if response.status==200:break
            except (URLError,TimeoutError):pass
            sleep(3)
        else:raise TimeoutError("Qwen service startup deadline exceeded")
        cfg={"llm":{"provider":"vllm","base_url":f"http://127.0.0.1:{PORT}/v1",
           "model":"qwen3-8b","enable_thinking":False,"temperature":0.0,
           "max_tokens":1000,"top_p":1.0,"top_k":-1,"seed":20261017,
           "timeout_seconds":120}}
        limits=RunLimits(b.RUN,max_requests=MAX_REQUESTS,max_tokens=700000,
                         max_seconds=MAX_SECONDS)
        model=create_model(b.RUN,MAX_REQUESTS,limits,cfg)
        status["status"]="COLLECTING_BLIND_PLANS";persist("status.json",status)
        plans={"rule":b.rule_plans(),"qwen_no_history":{},
               "qwen_related_memory":{}}
        rejections=[]
        # No evaluation or treatment execution until *all* plans collected.
        for arm in ("qwen_no_history","qwen_related_memory"):
            for task in b.CATEGORIES:
                try:
                    proposed,ok,reason=ask_plan(model,task,inputs,arm)
                except Exception as exc:
                    proposed={"error":type(exc).__name__,"message":str(exc)}
                    ok,reason=False,"model_proposal_failure"
                if not ok:
                    rejections.append({"arm":arm,"task":task,"reason":reason,
                                       "original":proposed})
                    # Explicit deterministic fallback with no extra model requests.
                    proposed=b.rule_plans()[task]
                    proposed["fallback_used"]=True
                plans[arm][task]=proposed
                status["calls"]=model.requests
                persist("plans.json",plans);persist("rejected_proposals.json",rejections)
                persist("status.json",status)
                print(json.dumps({"phase":"proposal","arm":arm,"task":task,
                        "action":proposed["action"],"accepted":ok,
                        "calls":model.requests}),flush=True)
        status["status"]="EXECUTING_FALSIFIERS";persist("status.json",status)
        actions={p["action"] for arm in plans.values() for p in arm.values()}
        coat_actions=sorted(a for a in actions if a in
                            ("gamma_stress","propensity_shuffle","split_utility"))
        agent_actions=sorted(a for a in actions if a in
                            ("item_memory_ablation","popularity_prompt"))
        observations={}
        action_costs={}
        for action in coat_actions:
            tick=monotonic()
            observations[action]=b.coat_observation(action)
            action_costs[action]={"physical_qwen_calls":0,
                                "execution_cpu_seconds":round(monotonic()-tick,3)}
            persist("executions.json",observations)
        if agent_actions:
            t=monotonic()
            author=b.make_evidence()["agentcf"]
            observations["agentcf_shared_rank_panels"]=run_author_rankings(model)
            action_costs["agentcf_paired_conditions"]={
                "physical_qwen_calls":observations["agentcf_shared_rank_panels"]["physical_rank_model_requests"],
                "wall_seconds":round(monotonic()-t,2)}
            persist("executions.json",observations)
        results={}
        for arm in b.ARMS:
            results[arm]={}
            for task,proposal in plans[arm].items():
                action=proposal["action"]
                evidence=(observations[action] if task.startswith("coat") else
                          observations["agentcf_shared_rank_panels"])
                results[arm][task]=b.assess(task,action,evidence)
        persist("assessments.json",results)
        # Reflections happen after all hypothesis/falsifier choices are frozen.
        status["status"]="REFLECTING_AFTER_FALSIFICATION";persist("status.json",status)
        lessons={}
        for arm in ("qwen_no_history","qwen_related_memory"):
            lessons[arm]={}
            for task in b.CATEGORIES:
                model.scope={"phase":"reflection","arm":arm,"task":task}
                payload={"original_proposal":plans[arm][task],
                         "independent_python_assessment":results[arm][task],
                         "research_constraint":"Development evidence only. Do not claim causality or scientific confirmation."}
                try:
                    answer=model.ask(json.dumps(payload,ensure_ascii=False),
                                     b.REFLECT_SCHEMA,{"stage":"reflect"})
                except Exception as exc:
                    answer={"evidence_state":"inconclusive",
                            "lesson":"Reflection failed; cannot interpret as evidence",
                            "remaining_competing_explanation":type(exc).__name__,
                            "next_discriminating_test":"Acquire independent controlled evidence"}
                lessons[arm][task]=answer
                persist("reflections.json",lessons)
                print(json.dumps({"phase":"reflection","arm":arm,"task":task,
                                  "model_calls":model.requests}),flush=True)
        usage=usage_summary(b.RUN)
        costs={}
        plan_calls=4;reflection_calls=4
        for arm in b.ARMS:
            selected={p["action"] for p in plans[arm].values()}
            costs[arm]={"llm_calls":0 if arm=="rule" else plan_calls+reflection_calls,
                "planned_experiment_actions":len(selected),
                "logical_agentcf_condition_rankings":(
                    10*len([a for a in selected if a in ("item_memory_ablation","popularity_prompt")])
                    if any(a in selected for a in ("item_memory_ablation","popularity_prompt")) else 0),
                "shared_physical_executor_calls_not_attributed_to_single_arm":True}
        aggregates=b.aggregate(plans,results,costs)
        final={"status":"PROSPECTIVE_CROSSBIAS_PILOT_COMPLETED",
            "protocol_sha256":b.sha_file(b.RUN/"protocol.json"),
            "study_scope":"development_only_coat_and_author_agentcf_mechanisms",
            "model":"qwen3-8b","model_requests_total":model.requests,
            "model_usage":usage,"actions_executed":sorted(actions),
            "action_costs":action_costs,"proposal_rejections":rejections,
            "by_arm":aggregates,"independent_confirmed_discoveries":0,
            "no_confirmation_labels_accessed":True,
            "interpretation":"Development falsification; effective confirmed discovery count requires independent data and stronger uncertainty checks",
            "key_limits":[
              "Previously used Coat development and AgentCF author training state",
              "Hypotheses are not independent externally adjudicated truth",
              "Programmatic repeated-condition checks cannot establish classical causal debiasing",
              "Logical execution budgets differ when arms choose different registered actions",
              "Physical candidate scoring reused across arms; GPU tokens must be reported separately",
              "All action selection done before experiment outputs became available to the planner"
            ]}
        persist("comparison.json",final)
        lines=["# Cross-Bias Qwen falsification pilot","",
          "Coat Selection and AgentCF Popularity/Item Memory, prospective proposals before experiment execution.",
          "Development-only: no independent scientifically validated discovery yet.",
          "","| Arm | Accepted proposals | Completed falsifiers | Replicated dev signals | Independently validated discoveries | Qwen planner/reflection calls |",
          "|---|---:|---:|---:|---:|---:|"]
        for arm,score in aggregates.items():
            lines.append(f"| {arm} | {score['valid_proposals']} | "
               f"{score['falsification_experiments_completed']} | "
               f"{score['development_signals_replicated']} | "
               f"{score['independent_valid_scientific_discoveries']} | "
               f"{score['costs']['llm_calls']} |")
        lines+=["",f"Total Qwen calls including ranking execution: {model.requests}.",
                "Full traces: plans.json, executions.json, assessments.json, reflections.json, calls.jsonl, comparison.json.",
                "This is a development screening study, NOT independent confirmation or proof of RSI."]
        (b.RUN/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
        status["status"]="COMPLETED";status["calls"]=model.requests
    except BaseException as exc:
        status["status"]="FAILED"
        status["error"]=type(exc).__name__+": "+str(exc)
        status["traceback_tail"]=traceback.format_exc()[-1800:]
        raise
    finally:
        shutdown_own(svc)
        status["own_model_exited"]=svc is None or svc.poll() is not None
        status["finished_utc"]=now()
        status["elapsed_s"]=round(monotonic()-start,2)
        status["calls"]=model.requests if model else 0
        if svc:
            for _ in range(15):
                mem,util=gpu_state()
                if mem<512:break
                sleep(2)
            status["gpu_released"]=mem<512
            status["gpu_mib"]=mem
        persist("status.json",status)
        print(json.dumps({"status":status["status"],"calls":status["calls"],
            "gpu_released":status.get("gpu_released")}),flush=True)
    return status


if __name__=="__main__":
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
