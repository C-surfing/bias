"""Four-bias RSI research workflow, reusing the existing scientific agent roles.

Amazon invokes the production LLMRank science tree. Selection/Exposure use
the same Diagnosis/Designer/Critic/Reflection agents around frozen, real-data
benchmark adapters. All claims remain development-only.
"""
from __future__ import annotations
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from rsi_researcher.research.four_bias_contracts import CONTRACTS, assess
from rsi_researcher.research.scientific_agents import (
    DiagnosisAgent, DesignerAgent, CriticAgent, ReflectionAgent,
)
from rsi_researcher.runtime.client import atomic_json, usage_summary
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.models import create_model
from rsi_researcher.llmrank.science import search

class FrozenBiasDesigner(DesignerAgent):
    instruction = (DesignerAgent.instruction +
       " You are choosing a bounded numeric intervention for a FROZEN DEVELOPMENT "
       "experiment. A falsifier MUST be measurable on the current reference data "
       "without opening new holdout or requesting unavailable samples. "
       "For selection, IPW clipping alters known-mask weights; for exposure, "
       "the policy changes item-selection probabilities under the known random "
       "logger, not the logging propensity itself. Never call a favorable policy "
       "value difference proof that exposure bias was eliminated.")

class FrozenBiasCritic(CriticAgent):
    instruction = (
       "You are the Critic for a frozen development-only numerical debias study. "
       "APPROVE an executable strategy with a relevant measured falsifier, bounded "
       "parameter, identical evaluation reference and no label leakage. "
       "Reject actual unsupported identification claims, impossible tests, "
       "unidentifiable propensities, or wrong estimands. "
       "Do not require a fresh held-out confirmation before a legitimate DEVELOPMENT "
       "diagnostic experiment. Independent causal claims remain prohibited regardless "
       "of this review's decision. Selection can test clipped IPW weights under a "
       "known extra MNAR mask. Exposure can test static target policies with random "
       "logger probabilities and ESS; it cannot prove bias elimination from SNIPS alone. "
       "Name the exact reason and a testable revision if rejecting."
    )

PROJECT=Path("/data/cjh/rsi/project")

def experimental_data(output, name, seed):
    folder=output/"data"
    folder.mkdir()
    if name=="selection":
        command=[sys.executable,"-B",str(PROJECT/"scripts/kuairec_selection_popularity_20261022.py"),
                 "--seed",str(seed),"--output",str(folder/"reference")]
    elif name=="exposure":
        command=[sys.executable,"-B",str(PROJECT/"scripts/open_bandit_ope_four_bias_scope_20261022.py"),
                 "--output",str(folder/"reference")]
    else: raise ValueError(name)
    completed=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True,check=True)
    atomic_json(folder/"execution.json",{"command":command,"stdout":completed.stdout,
                    "returncode":completed.returncode})
    result=json.loads((folder/"reference/results.json").read_text())
    if name=="selection":
        arms={key:{"Brier":v["Brier"],"NDCG@50":v["NDCG@50"],
                    "Recall@50":v["Recall@50"],"PopREO@50":v["PopREO@50"]}
              for key,v in result["metrics"].items()}
        baseline="selection_naive"
        fixed="known_mask_oracle_ipw"
        options=("ipw_clip_2","ipw_clip_3","ipw_clip_4")
        grade="known_extra_MNAR_mask_on_near_full_KuaiRec"
        caveat="Only artificial extra MNAR corrected, not original real-world selection."
    else:
        arms={key:{"SNIPS":v["SNIPS"],"IPS":v["IPS"],"DR":v["DR"],
                  "ESS":v["ESS"],"SNIPS_CI95":v["ci95"]["SNIPS"]}
              for key,v in result["results"].items()}
        baseline="uniform"
        fixed="bts_90head10uniform"
        options=("bts_head20","bts_tail20")
        grade="known_random_logger_propensity"
        caveat="Static policy OPE; value improvement is not in itself exposure-bias reduction."
    return arms,baseline,fixed,options,grade,caveat,result

def constrained_report(name,output,seed,config):
    output.mkdir(parents=True,exist_ok=False)
    arms,baseline,fixed,options,grade,caveat,raw=experimental_data(output,name,seed)
    choices=list(options)
    rand=random.Random(seed+100)
    random_arm=rand.choice(choices)
    base_metrics=arms[baseline]
    def evaluate(chosen):
        return assess(name,base_metrics,arms[chosen],
                      identification=grade,evidence_grade="development_only",
                      ess=arms[chosen].get("ESS"))
    modeldir=output/"scientific_agent_calls"
    modeldir.mkdir()
    limits=RunLimits(modeldir,max_requests=16,max_tokens=140000,max_seconds=1800)
    model=create_model(modeldir,limits.max_requests,limits,config)
    model.health()
    fixed_eval=evaluate(fixed)
    random_eval=evaluate(random_arm)
    snapshot={
        "task":name,"phase":"development","scientific_target":CONTRACTS[name].__dict__,
        "data_identification":grade,"baseline":base_metrics,
        "fixed_reference": {"arm":fixed,"metrics":arms[fixed],"audit":fixed_eval},
        "known_limitations":caveat,
        "allowed_new_experiments":choices,
        "frozen_experiment_parameters":"data, evaluation, seed, labels and candidates cannot change",
        "model_cost_policy":"Record scientist inference separately from fixed/random CPU evaluations",
        "no_test_results_or_holdout_in_agent_context":True,
    }
    atomic_json(output/"program.json",snapshot)
    steps={}
    diag=DiagnosisAgent(model).act({"observations":snapshot},f"{name}-{seed}-diagnosis")
    steps["diagnosis"]=diag
    proposals=[]
    actual=None
    critical=None
    chosen=None
    for revision in range(2):
        plan=FrozenBiasDesigner(model).propose({
            "diagnosis":diag,"data":snapshot,
            "previous_review":proposals[-1] if proposals else None,
            "revision":revision},f"{name}-{seed}-proposal-{revision}",
            choices,[CONTRACTS[name].metric])
        chosen=plan["action"]
        critical=FrozenBiasCritic(model).act({
            "diagnosis":diag,"proposal":plan,
            "frozen_data_scope":snapshot["data_identification"],
            "baseline_metrics":base_metrics,
            "fixed_reference":snapshot["fixed_reference"],
            "executability":chosen in choices,
            "phase":"development_not_confirmation",
            "comparison_available":"registered baseline and fixed",
            "independent_holdout_used":False},f"{name}-{seed}-critique-{revision}")
        proposals.append({"plan":plan,"review":critical})
        if critical["decision"]=="approve" and chosen in choices:
            actual=chosen
            break
    steps["proposals"]=proposals
    steps["design"]=proposals[-1]["plan"]
    steps["critic"]=critical
    approved=actual is not None
    agent_eval=evaluate(actual) if actual else None
    outcome={"proposed":chosen,"approved":approved,"measured_arm":actual,
             "metrics":arms.get(actual) if actual else None,
             "evaluation":agent_eval,
             "identification_scope":caveat,"independent_confirmation":False}
    steps["experiment"]=outcome
    reflection=ReflectionAgent(model).act({
        "diagnosis":diag,"proposed":plan,"critic":critical,
        "numeric_audit":outcome,"independent_confirmation":False,
        "cost_so_far":model.requests,
        "scientific_status":"development_only_not_confirmed"},
        f"{name}-{seed}-reflection")
    steps["reflection"]=reflection
    scientific_cost=model.requests
    report={
        "task":name,"seed":seed,"status":"completed_development_only",
        "frozen_program":snapshot,"scientific_roles":steps,
        "baseline":{"arm":baseline,"metrics":base_metrics},
        "fixed":{"arm":fixed,"metrics":arms[fixed],"audit":fixed_eval,"model_requests":0},
        "random":{"arm":random_arm,"metrics":arms[random_arm],"audit":random_eval,"model_requests":0},
        "agent":{"arm":actual,"metrics":arms.get(actual) if actual else None,
                 "audit":agent_eval,"scientist_model_requests":scientific_cost},
        "cost":{"fixed_qwen_requests":0,"random_qwen_requests":0,
                "agent_scientist_requests":scientific_cost,
                "reference_data_runs":1,
                "strict_equal_model_compute":False,
                "discovery_per_request":None},
        "scientific_conclusion":"No independently confirmed bias discovery, no holdout.",
        "limitation":caveat,"confirmed_discoveries":0,"independent_holdout_opened":False,
    }
    atomic_json(output/"result.json",report)
    atomic_json(output/"usage.json",usage_summary(modeldir))
    print("WORKFLOW_RESULT",json.dumps({
      "task":name,"baseline":baseline,"fixed":fixed,"random":random_arm,
      "agent":actual,"audit":agent_eval,"qwen_requests":scientific_cost
    },ensure_ascii=False),flush=True)
    return report

def amazon_run(output,seed,users,dataset,bias_target,config,prior=None,max_requests=1240):
    if bias_target not in ("popularity","candidate_position"):
        raise ValueError(bias_target)
    dataset_dir=PROJECT/"datasets/amazon/processed"/dataset
    result=search(output,config=config,data=dataset_dir,dataset=dataset,
                  retrieval="bm25",users=users,seeds=[seed],max_nodes=1,
                  max_requests=max_requests,development_only=True,bias_target=bias_target,prior_campaign=prior,
                  arm="four_bias_registered_workflow")
    print("AMAZON_RSI_RESULT",json.dumps({
        "target":bias_target,"dataset":dataset,"seed":seed,"result":result},ensure_ascii=False),flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--mode",choices=["selection","exposure","amazon"],required=True)
    p.add_argument("--dataset",default="Beauty")
    p.add_argument("--target",default="popularity")
    p.add_argument("--seed",type=int,default=20261026)
    p.add_argument("--users",type=int,default=200)
    p.add_argument("--prior",default=None)
    p.add_argument("--requests",type=int,default=1240)
    args=p.parse_args()
    config=json.loads((PROJECT/"configs/llmrank_qwen.json").read_text())
    folder=Path(args.output)
    if args.mode=="amazon":
        amazon_run(folder,args.seed,args.users,args.dataset,args.target,config,args.prior,args.requests)
    else:
        constrained_report(args.mode,folder,args.seed,config)

if __name__=="__main__":main()
