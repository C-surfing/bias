"""Run fresh AgentCF memory/popularity forecast study on a new disjoint cohort.

Single bounded owned vLLM server; source-verified memory vs no-history Qwen;
all numeric hypotheses are frozen before any new state is trained or ranked.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from time import monotonic,sleep
from urllib.request import build_opener,ProxyHandler
from urllib.error import URLError
import json,os,signal,socket,subprocess,traceback

from rsi_researcher.paths import ROOT
from rsi_researcher.research import fresh_predictions as f
from rsi_researcher.research.decision_qwen_pilot import GPU_UUID,gpu_state,shutdown_own
from rsi_researcher.research.memory import ResearchMemory,validate_card
from rsi_researcher.research.author_experiment import author_intervention
from rsi_researcher.runtime.models import create_model
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.client import atomic_json,usage_summary,validate_ranking
from rsi_researcher.recommenders.agentcf import AgentCF
from rsi_researcher.evaluation.author import score_author_ranking
from rsi_researcher.datasets.author import author_profile

PORT=18341
WALL=1250
SYSTEM=("You are a scientific hypothesis designer. Provide a precise falsifiable "
   "numerical prediction for the registered recommendation mechanism on a fresh "
   "AgentCF author cohort. You MUST use the requested primary metric, registered "
   "experiment and quality safeguard. Predict direction and minimum absolute "
   "effect in the interval 0 to 0.5. Competing explanation and falsifier must "
   "contain actual testable scientific distinctions. You cannot inspect or request "
   "unobserved test labels, edit code or access a shell. Development observations "
   "are not confirmed scientific results. Reply JSON only.")
REFLECT={"type":"object","properties":{
 "agreement":{"type":"string","enum":["supported","refuted","inconclusive"]},
 "evidence_anchored_lesson":{"type":"string","minLength":25},
 "alternative_explanation":{"type":"string","minLength":25},
 "next_separating_test":{"type":"string","minLength":25}
},"required":["agreement","evidence_anchored_lesson","alternative_explanation","next_separating_test"],
 "additionalProperties":False}


def now():return datetime.now(timezone.utc).isoformat()


def put(name,value):atomic_json(f.NEW/name,value)


def verified_cards():
    p=ROOT/"runs/crossbias-discovery-pilot-20261017/SOURCE_CHECKED_RESEARCH_MEMORY.json"
    raw=json.loads(p.read_text(encoding="utf8"))
    if raw["status"]!="FACTUAL_MEMORY_DEVELOPMENT_ONLY" or raw["llm_written_claims_admitted"]!=0:
        raise ValueError("Unreviewed or promoted research history")
    cards=[]
    for row in raw["cards"]:
        c=ResearchMemory(**row)
        validate_card(c)
        cards.append(c)
    return cards


def prior_for(task,cards):
    keep=[c for c in cards if
          (("memory" in c.mechanism_tags and task=="item_memory") or
           ("popularity" in c.mechanism_tags and task=="popularity"))]
    return [{"grade":c.evidence_grade,"lesson":c.conditional_lesson,
             "limitation":c.limitation,"source_sha256":c.source_sha256}
            for c in keep[:2]]


def proposed(model,task,arm,evidence):
    model.scope={"task":task,"arm":arm,"phase":"prospective_prediction"}
    p={"question":f.TASKS[task]["question"],
       "registered_action":f.TASKS[task]["experiment"],
       "required_primary_metric":f.TASKS[task]["primary_metric"],
       "metrics_definitions":{
          "excess_rank_change":"fraction of complete rankings changed after item-memory ablation MINUS fraction changed between two unchanged repeats",
          "head_excess_delta":"post-intervention top5-head excess minus baseline top5-head excess",
          "ndcg_delta":"post-intervention NDCG@5 minus baseline NDCG@5"},
       "quality_constraint":"NDCG@5 >= 99% of baseline in both 3-user panels",
       "new_cohort":{"users":f.USER_COUNT,"training_events":f.USER_COUNT*f.EVENTS_PER_USER,
                     "new_user_ids_hidden":True,"outcomes_hidden":True,
                     "history_split":"development_only"},
       "source_verified_prior_evidence":evidence if arm=="qwen_verified_history" else [],
       "rule":"Do not make causal claims or cite independent replication from old development cards."}
    return model.ask(SYSTEM+"\n"+json.dumps(p,ensure_ascii=False),
                     f.SCHEMA,{"stage":"hypothesis"})


def execute_training(model,initial,events):
    kernel=AgentCF(model,json.loads((ROOT/"data/author_prompts.json").read_text(encoding="utf8")))
    state=deepcopy(initial)
    train_dir=f.NEW/"training"
    train_dir.mkdir(parents=True,exist_ok=False)
    for j,event in enumerate(events):
        model.scope={"phase":"new_cohort_training","event_index":j}
        before=model.requests
        after=kernel.event(state,event,"original_update")
        if after["completed"]!=state["completed"]+[event["event_id"]]:
            raise AssertionError("Broken training event chain")
        put(f"training/event-{j:03d}.json",{"event_id":event["event_id"],
             "before_state_sha256":f.sha256(f.serialize(state).encode()).hexdigest(),
             "after_state_sha256":f.sha256(f.serialize(after).encode()).hexdigest(),
             "requests":model.requests-before})
        state=after
        print(json.dumps({"phase":"training","event":j+1,
              "total_events":len(events),"model_calls":model.requests}),flush=True)
    put("frozen_state.json",state)
    return state


def execute_rankings(model,initial,state,queries):
    base=json.loads((ROOT/"data/author_prompts.json").read_text(encoding="utf8"))
    neutral=author_intervention(base,"neutralize_rank_popularity")
    profile=author_profile(ROOT/"data")
    changed=deepcopy(state)
    modified=0
    for item in {x for q in queries for x in q["candidates"]}:
        if changed["items"][item]["memory"]!=initial["items"][item]["memory"]:
            modified+=1
        changed["items"][item]["memory"]=initial["items"][item]["memory"]
    if modified<1:raise ValueError("None of the candidate Item Memories changed")
    setups={"original":(state,base),"repeat":(state,base),
        "item_memory_ablation":(changed,base),
        "popularity_prompt":(state,neutral)}
    halves={"A":sorted(queries,key=lambda q:q["query_id"])[:3],
            "B":sorted(queries,key=lambda q:q["query_id"])[3:]}
    panels={cohort:{} for cohort in halves}
    for label,subset in halves.items():
        if len(subset)!=3:raise ValueError("cohort size changed")
        for action,(world,prompt) in setups.items():
            agent=AgentCF(model,prompt)
            rows=[]
            for query in subset:
                model.scope={"phase":"new_state_falsifier","condition":action,"cohort":label}
                before=model.requests
                ranking=agent.rank(world,query)
                validate_ranking({"ranking":ranking},query["candidates"])
                rows.append({"query_id":query["query_id"],"ranking":ranking,
                             "valid":True,"calls_used":model.requests-before,
                             **score_author_ranking(ranking,query,profile)})
            panels[label][action]=rows
            put("ranking_panels.json",panels)
            print(json.dumps({"phase":"rank","condition":action,
                  "cohort":label,"n":len(rows),"calls":model.requests}),flush=True)
    return panels,modified


def main():
    frozen=f.freeze()
    status={"status":"FROZEN","created_utc":now(),"protocol_sha256":f.digest(f.NEW/"protocol.json"),
            "new_training_cohort_disjoint":True,"model_requests":0,
            "gpu_uuid":GPU_UUID,"port":PORT,"independent_confirmation":False}
    put("status.json",status)
    svc=None;model=None
    start=monotonic()
    try:
        cards=verified_cards()
        initial,events,queries=f.cohort_data()
        mem,util=gpu_state()
        if mem>=512 or util!=0:raise RuntimeError("GPU 5 is not idle")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1",PORT))==0:raise RuntimeError("Port occupied")
        env=os.environ.copy()
        env.update(RSI_BASE="/data/cjh/rsi",RSI_GPU_UUID=GPU_UUID,
                    RSI_MODEL_PORT=str(PORT),PYTHONUNBUFFERED="1")
        with (f.NEW/"vllm.log").open("x",encoding="utf8") as fh:
            svc=subprocess.Popen(["bash","/data/cjh/rsi/deployment/qwen3/serve.sh"],
                 env=env,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
        status["service_pid"]=svc.pid;put("status.json",status)
        opener=build_opener(ProxyHandler({}))
        while monotonic()-start<390:
            if svc.poll() is not None:raise RuntimeError("Own vLLM process exited")
            try:
                with opener.open(f"http://127.0.0.1:{PORT}/health",timeout=2) as response:
                    if response.status==200:break
            except (URLError,TimeoutError):pass
            sleep(3)
        else:raise TimeoutError("Own model service startup timeout")
        config={"llm":{"provider":"vllm",
             "base_url":f"http://127.0.0.1:{PORT}/v1","model":"qwen3-8b",
             "enable_thinking":False,"temperature":0.0,
             "max_tokens":800,"top_p":1.0,"top_k":-1,
             "seed":20261107,"timeout_seconds":120}}
        limits=RunLimits(f.NEW,f.MAX_REQUESTS,850000,1250)
        model=create_model(f.NEW,f.MAX_REQUESTS,limits,config)
        status["status"]="BLIND_NUMERIC_FORECASTS";put("status.json",status)
        plans={"rule":{t:f.rule_forecast(t) for t in f.TASKS},
               "qwen_no_history":{},"qwen_verified_history":{}}
        rejected=[]
        for arm in ("qwen_no_history","qwen_verified_history"):
            for task in f.TASKS:
                try:
                    plan=proposed(model,task,arm,prior_for(task,cards))
                    ok,why=f.validate(task,plan)
                except Exception as exc:
                    plan={"error":str(exc)};ok,why=False,"model_error"
                if not ok:
                    rejected.append({"arm":arm,"task":task,"why":why,"raw":plan})
                    plan=f.rule_forecast(task)
                plans[arm][task]=plan
                put("predictions.json",plans)
                put("prediction_rejections.json",rejected)
                print(json.dumps({"phase":"forecast","arm":arm,"task":task,
                   "valid":ok,"direction":plan["direction"],
                   "min_effect":plan["minimum_abs_effect"]}),flush=True)
        # Frozen plans must be persisted before training the new state.
        put("forecasts_frozen_at.json",{"time_utc":now(),"hash_sha256":
              f.digest(f.NEW/"predictions.json"),
              "training_started":False,"model_calls":model.requests})
        status["status"]="NEW_STATE_TRAINING";put("status.json",status)
        state=execute_training(model,initial,events)
        status["status"]="NEW_STATE_RANKING";put("status.json",status)
        panels,changed=execute_rankings(model,initial,state,queries)
        evaluated={}
        for task,data in f.TASKS.items():
            action=data["experiment"]
            evaluated[task]={cohort:f.score_panel(
                 conditions["original"],conditions["repeat"],conditions[action])
                  for cohort,conditions in panels.items()}
        put("measured_numeric_evidence.json",evaluated)
        score=f.retrospective_evaluate(plans,evaluated)
        put("machine_scored_forecasts.json",score)
        status["status"]="REFLECTION_AFTER_ALL_NUMERIC_SCORING";put("status.json",status)
        reflections={}
        reflect_schema=REFLECT
        for arm in ("qwen_no_history","qwen_verified_history"):
            reflections[arm]={}
            for task in f.TASKS:
                model.scope={"phase":"reflection","arm":arm,"task":task}
                payload={"frozen_predeclared_forecast":plans[arm][task],
                         "independent_numeric_measurement":evaluated[task],
                         "programmatic_gate":score[arm][task],
                         "constraint":"Do not claim independent scientific discoveries"}
                try:
                    row=model.ask(json.dumps(payload,ensure_ascii=False),
                                  reflect_schema,{"stage":"post_hoc_reflection"})
                except Exception as exc:
                    row={"error":type(exc).__name__,"not_verified":True}
                reflections[arm][task]=row
                put("reflections.json",reflections)
        usage=usage_summary(f.NEW)
        by_arm={}
        for arm in f.ARMS:
            values=score[arm]
            n=sum(v["both_panels_pass"] for v in values.values())
            by_arm[arm]={"formal_predictions_evaluated":len(values),
                "conditional_development_predictions_supported":n,
                "independently_confirmed_discoveries":0,
                "research_calls":0 if arm=="rule" else 4,
                "supported_per_research_call":n/4 if arm!="rule" else None,
                "measurements_shared_across_arms":True}
        result={"status":"NEW_COHORT_STRUCTURED_FORECAST_COMPLETED",
            "protocol_sha256":status["protocol_sha256"],"model":"qwen3-8b",
            "sources_verified":True,"new_user_count":f.USER_COUNT,
            "new_training_events":len(events),"new_training_state":True,
            "changed_item_memories_in_candidates":changed,
            "overall_model_usage":usage,
            "by_arm":by_arm,
            "formal_forecast_scores":score,
            "prediction_rejections":rejected,
            "physical_calls_shared_across_arms":True,
            "confirmation_labels_used":False,
            "independently_confirmed_discoveries":0,
            "limitations":["This is a new user/AgentCF state, not a new source dataset",
                "Same mechanism families were already studied; test-task novelty is limited",
                "Single Qwen generation/model training seed, six author users in two 3-user panels",
                "New users share item metadata and global item popularity measurement",
                "Quality is one known validation target per user; unobserved preferences unknown",
                "Model-generated reflections not eligible as verified Research Memory"]}
        put("comparison.json",result)
        text=["# New AgentCF cohort prospective numeric hypothesis study","",
          "Distinct users from old first-20; 6 users, 2 training events each.",
          "Two panels of 3 users are small and are NOT independent scientific confirmation.","",
          "| Arm | Formal forecasts | Passed numeric prediction AND quality gate | Confirmed scientific discoveries | Research calls |",
          "|---|---:|---:|---:|---:|"]
        for arm,v in by_arm.items():
            text.append(f"| {arm} | {v['formal_predictions_evaluated']} | "
                        f"{v['conditional_development_predictions_supported']} | "
                        f"{v['independently_confirmed_discoveries']} | "
                        f"{v['research_calls']} |")
        text += ["",f"All model calls including training and ranks: {usage['requests']}.",
          f"Prompt / output tokens: {usage['prompt_tokens']} / {usage['completion_tokens']}.",
          "No training result was visible to the forecasting Qwen before predictions froze.",
          "Mechanism effects/quality were independently measured in Python.",
          "Development-only, never promote model-written post hoc conclusions to confirmation."]
        (f.NEW/"REPORT.md").write_text("\n".join(text)+"\n",encoding="utf8")
        status["status"]="COMPLETED"
    except BaseException as exc:
        status["status"]="FAILED"
        status["error"]=type(exc).__name__+": "+str(exc)
        status["traceback_tail"]=traceback.format_exc()[-2400:]
        raise
    finally:
        shutdown_own(svc)
        status["own_model_exited"]=svc is None or svc.poll() is not None
        status["duration_s"]=round(monotonic()-start,2)
        status["model_requests"]=model.requests if model else 0
        if svc is not None:
            for _ in range(15):
                mem,util=gpu_state()
                if mem<512:break
                sleep(2)
            status["gpu_released"]=mem<512
            status["gpu_mib"]=mem
        status["end_utc"]=now()
        put("status.json",status)
        print(json.dumps({"status":status["status"],"model_requests":status["model_requests"],
                          "gpu_released":status.get("gpu_released")}),flush=True)


if __name__=="__main__":
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
