"""Bounded Qwen3-8B decision comparison on frozen synthetic exposure tasks.

One owned vLLM process on an idle named GPU, never a global kill.
Four Qwen arms each receive exactly 1 calibration + 14 decision calls;
fixed rule consumes zero model calls. No feedback from evaluation tasks is
included in model prompts. All scoring is independent Python evaluation.
"""
from __future__ import annotations
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler
from urllib.error import URLError
import json
import os
import signal
import socket
import subprocess
import sys
import time

from rsi_researcher.research import decision_benchmark as b

RUN=b.ROOT/"runs/decision-benchmark-v1-20261017"
DIR=RUN/"qwen-pilot-01"
MODEL="qwen3-8b"
PORT=18317
GPU_UUID="GPU-e9bc2d1d-cb92-bc3e-b145-855b0cddae0b"
ARMS=("no_history","related_history","shuffled_history","strategy_revision")
MAX_CALLS_PER_ARM=15
MAX_TOTAL_CALLS=60
MAX_WALL_SECONDS=1100


def utc():
    return datetime.now(timezone.utc).isoformat()


def persist(name,obj):
    path=DIR/name
    tmp=DIR/(name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    tmp.replace(path)


def gpu_state():
    raw=subprocess.check_output(["nvidia-smi","-i",GPU_UUID,
           "--query-gpu=memory.used,utilization.gpu",
           "--format=csv,noheader,nounits"],text=True).strip()
    return tuple(int(x.strip()) for x in raw.split(","))


def validate_decision(data,task):
    if not isinstance(data,dict):
        return "abstain","non_object"
    action=data.get("action")
    if action not in b.ACTIONS:
        return "abstain","unregistered_action"
    if not isinstance(data.get("rationale"),str) or not data["rationale"].strip():
        return "abstain","missing_rationale"
    if not isinstance(data.get("falsifier"),str) or not data["falsifier"].strip():
        return "abstain","missing_falsifier"
    if action!="abstain" and not task["logging_propensities_available"]:
        return "abstain","missing_logging_propensity"
    if action!="abstain" and task.get("positivity_verified") is False:
        return "abstain","positivity_violated"
    return action,None


def model_request(opener,system,user):
    payload={"model":MODEL,"temperature":0.0,"max_tokens":250,
      "chat_template_kwargs":{"enable_thinking":False},
      "response_format":{"type":"json_object"},
      "messages":[{"role":"system","content":system},
                  {"role":"user","content":json.dumps(user,ensure_ascii=False)}]}
    req=Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
       data=json.dumps(payload,ensure_ascii=False).encode("utf8"),
       headers={"Content-Type":"application/json"},method="POST")
    with opener.open(req,timeout=110) as result:
        response=json.loads(result.read().decode("utf8"))
    content=response["choices"][0]["message"].get("content","")
    data=json.loads(content)
    if not isinstance(data,dict):
        raise ValueError("non_json_object")
    return data,{"usage":response.get("usage",{}),
                 "finish_reason":response["choices"][0].get("finish_reason"),
                 "raw_content":content}


def shutdown_own(service):
    if service is None:return
    if service.poll() is None:
        try:os.killpg(service.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:service.wait(timeout=35)
        except subprocess.TimeoutExpired:
            try:os.killpg(service.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            service.wait(timeout=15)


SYSTEM=("You are a cautious research-method decision planner for SYNTHETIC "
  "recommendation exposure logs. Select exactly ONE estimator action among "
  "naive, ips, snips, abstain. IPS/SNIPS require known logging propensities "
  "AND positivity for the target policy; reject causal inference without either. "
  "Logged click mean may be biased for a shifted target. SNIPS may trade "
  "finite-sample bias for variance reduction. Avoid unsafe guesses and "
  "weigh effective sample size. You are NOT permitted to execute code, "
  "use tools, change metrics, request holdout labels or claim model success. "
  "Return a JSON object with exactly action, rationale, falsifier; "
  "each rationale/falsifier must be a nonempty string. No extra prose.")


def history_cards(protocol,arm,task):
    cards=[{"requested_policy":c["task"]["requested_policy"],
            "development_action":c["development_feedback"],
            "n_logged":c["task"]["n_logged"]}
           for c in protocol["development_tasks"]]
    if arm=="no_history": return []
    if arm=="shuffled_history":
        actions=[x["development_action"] for x in cards]
        shuffled=actions[2:]+actions[:2]
        cards=[dict(x,development_action=shuffled[i]) for i,x in enumerate(cards)]
    matched=[x for x in cards if x["requested_policy"]==task["requested_policy"]]
    return matched or cards[:2]


def calibration_payload(protocol,arm):
    cards=[{"requested_policy":x["task"]["requested_policy"],
            "propensities_available":x["task"]["logging_propensities_available"],
            "positivity_verified":x["task"].get("positivity_verified"),
            "development_action":x["development_feedback"]}
           for x in protocol["development_tasks"]]
    if arm=="no_history":cards=[]
    if arm=="shuffled_history":
        acts=[x["development_action"] for x in cards]
        cards=[dict(x,development_action=acts[(i+2)%len(acts)])
               for i,x in enumerate(cards)]
    return {
      "calibration_goal":("Rewrite the research selection policy into at most "
         "three concise, cautious condition-action guidelines using ONLY these "
         "development examples. Do not guess evaluation labels."
         if arm=="strategy_revision"
         else "Explain cautious research policy selection on synthetic logs."),
      "development_cards":cards,
      "return":"JSON object with action='abstain', rationale=policy guidance, falsifier=limitation",
      "heldout_results":"not supplied"
    }


def run():
    protocol=b.read_frozen(RUN)
    if DIR.exists():raise FileExistsError("Cannot overwrite previous model pilot")
    DIR.mkdir(parents=True)
    start=time.monotonic()
    status={"status":"STARTING","start_utc":utc(),"model":MODEL,
            "gpu_uuid":GPU_UUID,"port":PORT,"requests_sent":0,
            "per_arm_budget":MAX_CALLS_PER_ARM,
            "frozen_protocol_sha256":(RUN/"protocol.sha256").read_text().strip(),
            "evaluation_labels_to_model":False,
            "new_evaluation_scoring_before_completion":False}
    persist("status.json",status)
    service=None
    history=[];records=[]
    opener=build_opener(ProxyHandler({}))
    try:
        mem,util=gpu_state()
        if mem>=512 or util>0:
            raise RuntimeError("GPU5 not idle; refuse to displace other workloads")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1",PORT))==0:
                raise RuntimeError("Private model port occupied")
        env=os.environ.copy()
        env.update(RSI_BASE="/data/cjh/rsi",RSI_GPU_UUID=GPU_UUID,
                   RSI_MODEL_PORT=str(PORT),PYTHONUNBUFFERED="1")
        with (DIR/"vllm.log").open("x",encoding="utf8") as logfile:
            service=subprocess.Popen(["bash","/data/cjh/rsi/deployment/qwen3/serve.sh"],
                env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
        status["service_pid"]=service.pid
        persist("status.json",status)
        ready=time.monotonic()+420
        while time.monotonic()<ready:
            if service.poll() is not None:
                raise RuntimeError("Own vLLM server exited during startup")
            try:
                with opener.open(f"http://127.0.0.1:{PORT}/health",timeout=2) as response:
                    if response.status==200:break
            except (URLError,TimeoutError):pass
            time.sleep(3)
        else:raise TimeoutError("Model service health timeout")
        status["status"]="RUNNING"
        persist("status.json",status)
        for arm in ARMS:
            if time.monotonic()-start>MAX_WALL_SECONDS:
                raise TimeoutError("Model pilot total time budget exhausted")
            if status["requests_sent"]>=MAX_TOTAL_CALLS:
                raise RuntimeError("Model request limit reached")
            calibration=calibration_payload(protocol,arm)
            status["requests_sent"]+=1;persist("status.json",status)
            try:
                preflight,meta=model_request(opener,SYSTEM,calibration)
            except (ValueError,KeyError,TypeError) as exc:
                preflight={"rationale":"Calibration unavailable; no strategy acquired",
                          "falsifier":type(exc).__name__}
                meta={"error":str(exc)}
            guidance=(str(preflight.get("rationale",""))[:600]
                      if arm=="strategy_revision" else "")
            history.append({"arm":arm,"calibration":preflight,"metadata":meta,
                            "guidance_consumed":arm=="strategy_revision"})
            persist("calibrations.json",history)
            arm_records=[]
            for task in protocol["evaluation_tasks"]:
                if time.monotonic()-start>MAX_WALL_SECONDS:
                    raise TimeoutError("Model pilot total wall-time budget exhausted")
                if status["requests_sent"]>=MAX_TOTAL_CALLS:
                    raise RuntimeError("Model request limit reached")
                context=history_cards(protocol,arm,task)
                # This payload contains the public diagnostic ONLY. Never include
                # simulator oracle/hidden best action/eval rewards.
                user={"decision_task":task,
                      "development_memory":context,
                      "revised_method_guidance":guidance,
                      "tools_allowed":False,"evaluation_feedback":None}
                if "oracle_hidden_policy_value" in json.dumps(user):
                    raise AssertionError("Hidden oracle leaked")
                status["requests_sent"]+=1
                persist("status.json",status)
                try:
                    decision,meta=model_request(opener,SYSTEM,user)
                    action,rejection=validate_decision(decision,task)
                except (ValueError,KeyError,TypeError) as exc:
                    decision={};meta={"error":str(exc)}
                    action,rejection="abstain",type(exc).__name__
                row={"arm":arm,"task_id":task["task_id"],"chosen_action":action,
                     "validation_rejection":rejection,
                     "model_response":decision,"usage":meta.get("usage"),
                     "model_metadata":{k:v for k,v in meta.items() if k!="raw_content"}}
                arm_records.append(row);records.append(row)
                persist("model_decisions.json",records)
                status["last_completed"]={"arm":arm,"task_id":task["task_id"],
                                          "action":action,"rejection":rejection}
                persist("status.json",status)
                print(json.dumps({"arm":arm,"task":task["task_id"],
                      "action":action,"requests":status["requests_sent"]}),flush=True)
            if len(arm_records)!=len(protocol["evaluation_tasks"]):
                raise AssertionError("Incomplete decision arm")
        # No numerical evaluation feedback passed to Qwen before all arms finish.
        status["status"]="SCORING_AFTER_ALL_DECISIONS"
        persist("status.json",status)
        scored={"fixed_rule":b.score_arm([b.rule(t) for t in protocol["evaluation_tasks"]])}
        for arm in ARMS:
            rows=[r for r in records if r["arm"]==arm]
            scored[arm]=b.score_arm([r["chosen_action"] for r in rows])
        usage={}
        for arm in ARMS:
            calls=[x["metadata"].get("usage") for x in history if x["arm"]==arm]
            calls.extend([x["usage"] for x in records if x["arm"]==arm])
            usage[arm]={"calls":len(calls),
                        "prompt_tokens":sum((x or {}).get("prompt_tokens",0) for x in calls),
                        "completion_tokens":sum((x or {}).get("completion_tokens",0) for x in calls)}
        out={"status":"MODEL_PILOT_COMPLETE",
          "experiment_scope":"synthetic development evaluation, not independently validated RSI",
          "frozen_protocol_sha256":status["frozen_protocol_sha256"],
          "model":MODEL,"gpu_uuid":GPU_UUID,
          "arm_order":list(ARMS),"equal_request_count":True,
          "different_prompt_token_budget":True,
          "evaluation_labels_to_model":False,"model_calls":status["requests_sent"],
          "separate_calibration_call_each_arm":True,
          "calibration_output_used_only_for_strategy_revision":True,
          "usage":usage,"scores":scored,
          "scientific_limitations":[
             "Only 14 synthetic evaluation tasks from seven families, insufficient for population-level RSI claims.",
             "Model arms share the same predefined action space and oracle scorer but have unequal prompt tokens.",
             "All Qwen arms receive one calibration call; only strategy revision may use its output.",
             "Synthetic oracle values are computed after all decisions; not sent to the model.",
             "No real-world de-bias confirmation, no human expert evaluation.",
          ]}
        persist("comparison.json",out)
        columns=["arm","calls","mean_regret","accuracy","invalid","abstentions"]
        table=["| "+" | ".join(columns)+" |",
               "|---|---:|---:|---:|---:|---:|"]
        for arm in ("fixed_rule",)+ARMS:
            sc=scored[arm]
            table.append(f"| {arm} | {usage.get(arm,{}).get('calls',0)} | "
                 f"{sc['mean_regret']:.6f} | {sc['action_agreement']:.3f} | "
                 f"{sc['invalid_causal_decisions']} | {sc['unnecessary_abstentions']} |")
        (DIR/"REPORT.md").write_text("# Frozen synthetic research-decision pilot\n\n"+
           "\n".join(table)+
           "\n\nExploratory, no independent RSI claim. Calibration + decision calls are equal across model arms; prompt token use varies. Wrong causal actions are blocked locally.\n",
           encoding="utf8")
        status["status"]="COMPLETED"
    except BaseException as exc:
        status["status"]="FAILED"
        status["error"]=type(exc).__name__+": "+str(exc)
        raise
    finally:
        shutdown_own(service)
        status["own_model_exited"]=(service is None or service.poll() is not None)
        if service is not None:
            for _ in range(15):
                mem,util=gpu_state()
                if mem<512:break
                time.sleep(2)
            status["gpu_memory_released"]=mem<512
            status["remaining_gpu_mib"]=mem
        status["end_utc"]=utc()
        status["elapsed_s"]=round(time.monotonic()-start,1)
        persist("status.json",status)
        print(json.dumps({"status":status["status"],"requests":status["requests_sent"],
                          "own_model_exited":status["own_model_exited"]}),flush=True)
    return status


def on_sig(*_):raise KeyboardInterrupt("Model pilot interrupted")
if __name__=="__main__":
    signal.signal(signal.SIGTERM,on_sig)
    signal.signal(signal.SIGINT,on_sig)
    run()
