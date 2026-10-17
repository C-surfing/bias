"""Linux server: one idle GPU, fresh baseline and two registered research trials.

Starts only its own service in a new process group. Completion, exception and
deadline paths stop that service; never scans/kills another user's processes.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    temporary.replace(path)


def stop_group(process):
    """Signal the isolated process group we created, never a name/port match."""
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=15)
    # A parent can exit before its model workers. The group is exclusively ours.
    try:os.killpg(process.pid,0)
    except ProcessLookupError:return
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass


def request_stop(signum,frame):
    """Let normal finally cleanup run when this batch receives a stop signal."""
    raise SystemExit("batch received signal "+str(signum))


def run(args):
    if os.name!="posix":
        raise ValueError("this launcher runs only inside the user's Linux server directory")
    base=Path(args.base).resolve()
    output=Path(args.output).resolve()
    if not output.is_relative_to(base) or not ROOT.is_relative_to(base):
        raise ValueError("batch and source release must stay inside own RSI directory")
    if args.max_seconds<60:
        raise ValueError("batch duration must allow service startup")
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("dataset")!="CDs-100-user-dense" or config.get("llm",{}).get("provider")!="vllm":
        raise ValueError("bounded batch uses only author CDs and the local model")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1",8000))==0:
            raise ValueError("port 8000 is occupied; inspect the existing service instead")
    config["data_dir"]=str(ROOT/"data")
    config["budget"]={"max_requests":750,"max_tokens":3000000,"max_seconds":3600}
    save(output/"config.json",config)
    allowed=["balanced_observed_replay","neutral_candidate_order"]
    protocol={"created_at_utc":datetime.now(timezone.utc).isoformat(),"source_release":str(ROOT),
              "dataset":config["dataset"],"users":config["users"],"events_per_user":config["events_per_user"],
              "seed":config["seed"],"registered_operators":allowed,"rounds":2,
              "fresh_baseline":True,"position_design":"six same-state original/repeat/reverse panels",
              "quality_floor_ratio":.99,"minimum_target_reduction":.10,"position_noise_tolerance":.02,
              "max_seconds":args.max_seconds,
              "max_requests_total_ceiling":1500+6*config["users"]+max(6,config["users"])+6,
              "scope":"Development mechanism verification, not independent holdout confirmation or RSI proof."}
    save(output/"protocol.json",protocol)
    if args.journal:
        save(output/"research-journal.json",json.loads(Path(args.journal).read_text(encoding="utf-8")))
    deadline=time.monotonic()+args.max_seconds
    status={"status":"starting","jobs":[],"service":{},"shutdown":{}}
    service=None
    child=None
    env=os.environ.copy();env["RSI_BASE"]=str(base);env["RSI_GPU_UUID"]=args.gpu_uuid
    env["PYTHONUNBUFFERED"]="1"
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with (output/"vllm.log").open("w") as log:
            service=subprocess.Popen(["bash",str(base/"deployment/qwen3/serve.sh")],env=env,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status["service"]={"pid":service.pid,"gpu_uuid":args.gpu_uuid}
        save(output/"status.json",status)
        ready_until=min(deadline,time.monotonic()+600)
        while time.monotonic()<ready_until:
            if service.poll() is not None:
                raise RuntimeError("own model service exited; inspect vllm.log")
            try:
                with opener.open("http://127.0.0.1:8000/health",timeout=2) as response:
                    if response.status==200:break
            except OSError:pass
            time.sleep(3)
        else:raise TimeoutError("model startup deadline exceeded")
        python=str(base/"envs/research/bin/python")
        jobs=[("baseline",[python,"-m","rsi_researcher","author-check","--config",str(output/"config.json"),
                           "--output",str(output/"baseline")]),
              ("research",[python,"-m","rsi_researcher.research.agentcf_loop","--config",str(output/"config.json"),
                           "--baseline",str(output/"baseline"),"--output",str(output/"research"),"--rounds","2",
                           "--operators",*allowed,"--journal",str(output/"research-journal.json"),
                           "--max-requests","6","--max-tokens","80000","--max-seconds",str(args.max_seconds)])]
        for name,command in jobs:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError("batch deadline exceeded")
            status["status"]="running";status["active_job"]=name
            with (output/(name+".log")).open("w") as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status["active_pid"]=child.pid;save(output/"status.json",status)
            code=child.wait(timeout=remaining)
            status["jobs"].append({"name":name,"pid":child.pid,"returncode":code})
            if code!=0:raise RuntimeError(name+" execution failed")
            child=None
        status["status"]="completed"
    except BaseException as exc:
        status.update(status="failed",error=type(exc).__name__+": "+str(exc))
    finally:
        try:
            if child is not None:stop_group(child)
        finally:
            if service is not None:stop_group(service)
        if service is not None:
            # Wait for this service's workers to release their CUDA context.
            released=False
            for _ in range(20):
                usage=subprocess.check_output(["nvidia-smi","-i",args.gpu_uuid,"--query-gpu=memory.used,utilization.gpu",
                                               "--format=csv,noheader,nounits"],text=True).strip()
                memory,util=[int(v.strip()) for v in usage.split(",")]
                if memory<512:released=True;break
                time.sleep(2)
            status["shutdown"]={"service_pid_exited":service.poll() is not None,
                                "gpu_memory_mib":memory,"gpu_utilization":util,"memory_released":released}
        status["finished_at_utc"]=datetime.now(timezone.utc).isoformat()
        save(output/"status.json",status)
    print(json.dumps(status,ensure_ascii=False))
    return 0 if status["status"]=="completed" and status["shutdown"].get("memory_released") else 1


if __name__=="__main__":
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",default="/data/cjh/rsi")
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--gpu-uuid",required=True)
    parser.add_argument("--journal")
    parser.add_argument("--max-seconds",type=int,default=5400)
    raise SystemExit(run(parser.parse_args()))
