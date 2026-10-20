"""Finite autonomous scientific research across multiple hypothesis-search trees.

A single owner coordinates tasks and evidence/termination. Each search tree
remains the only frontier owner; the established episode executes and audits.
No agent can change evaluators, re-label holdouts, or approve its own discovery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsi_researcher.research import scientific_experiments as experiments
from rsi_researcher.research.scientific_loop import (
    digest, store_new, utc, verified_experience,
)
from rsi_researcher.research.scientific_tree import search
from rsi_researcher.runtime.client import atomic_json, usage_summary
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.models import create_model, model_description


def choose_next(pending, last=None):
    """Deterministic evidence-guided task order within preregistered candidates.

    A failed-memory-support result redirects to the registered popularity
    diagnostic on the SAME cohort, rather than repeatedly training memory
    trials until one happens to yield a writer-reader pathway. Task/seed
    inventory is frozen; never search for lucky seeds or reinterpret labels.
    """
    if last and last["task"]=="agentcf_memory" and last["status"] in (
            "ABSTAIN_PREDECISION_MECHANISM_SUPPORT",
            "ABSTAIN_INSUFFICIENT_MECHANISM_SUPPORT",
            "COMPLETED_NO_EXECUTION"):
        for i,entry in enumerate(pending):
            if entry["task"]=="agentcf_popularity" and entry["seed"]==last["seed"]:
                return i,"memory_support_failure_switch_to_same_cohort_popularity"
    return 0,"predeclared_queue_order"


def eligibility(output, tree, *, min_links_per_panel=1):
    """Evidence-based *development readiness*, NEVER an independent finding."""
    candidates = []
    for node in tree["nodes"].values():
        ref = node.get("audit_ref")
        if not ref:
            continue
        audit = json.loads((Path(output)/ref).read_text(encoding="utf8"))
        if audit.get("execution_kind") == "ARCHIVED_DEVELOPMENT_REPLAY":
            continue
        # The independent evaluator is the ONE owner of scientific meaning.
        # A passing numerical prediction or popularity proxy is not Debias.
        qualified=experiments.classify_debias_evidence(audit)
        if qualified["debias_candidate_eligible"] and (
                audit.get("evidence_grade")=="new_unconfirmed_development"):
            candidates.append({
                "task":audit["task"],"action":audit["action"],
                "bias_task_id":qualified["bias_task_id"],
                "debias_evidence_stage":qualified["debias_evidence_stage"],
                "audit_ref":str(Path(output)/ref),
                "audit_sha256":digest(Path(output)/ref),
                "status":"DEBIAS_DEVELOPMENT_READY_FOR_INDEPENDENT_TEST"})
    # Absence is scientifically meaningful; do not fabricate eligibility.
    return candidates


def decide_acceptance(cycles, *, min_ready=2, min_families=2):
    candidates = [c for cycle in cycles for c in cycle.get("ready_candidates",[])]
    unique_tasks = {c["task"] for c in candidates}
    families = {c["bias_task_id"] for c in candidates
                if c.get("bias_task_id") in {
                    t["bias_id"] for t in experiments.five_bias_coverage()}}
    ready = len(unique_tasks) >= min_ready and len(families) >= min_families
    uncovered=[t["bias_id"] for t in experiments.five_bias_coverage()
               if t["bias_id"] not in families]
    return {
        "development_readiness_met": ready,
        "unique_development_ready_tasks":len(unique_tasks),
        "distinct_ready_families":sorted(families),
        "five_bias_targets_without_ready_debias":uncovered,
        "registry": "classic-five-bias-tasks-v1.0",
        "default_two_bias_acceptance_currently_blocked_by_missing_identified_methods":
            bool(min_families>=2 and len(families)<2),
        "minimum_ready_tasks":min_ready,
        "minimum_distinct_families":min_families,
        "independently_confirmed_discoveries":0,
        "scientific_success":False,
        "reason": ("Development-only Debias candidates; independent pre-registered "
                   "tests and matched strong baselines still required"
                   if ready else "Not enough distinct five-bias targets with "
                                 "an identifiable Debias intervention, registered "
                                 "falsifier and quality-preserving benefit; "
                                 "mechanism probes do not count"),
        "confirmation_gate":"NOT_AVAILABLE_IN_THIS_DEVELOPMENT_EXECUTOR",
    }


def conduct(output, tasks, *, model=None, config=None,
            seed=20270001, cohort_size=8, max_cycles=4,
            max_visits_per_task=1, max_nodes_per_cycle=2,
            max_proposals=4, max_total_requests=120,
            max_stagnant_cycles=3, min_ready=2, min_families=2,
            mode="fresh", prepared_state=None, arm="qwen_audited_history"):
    """Run bounded multiple research cycles; never a forever GPU daemon.

    Existing task sources and action registry are authorities. Task/seed
    schedule is frozen before model inference, preventing positive-seed
    cherry-picking. Repeat AgentCF trials require disjoint training users.
    """
    output = Path(output).resolve()
    if (output.exists() or mode not in ("fresh","archive") or
        arm not in ("qwen_no_history","qwen_audited_history",
                    "qwen_shuffled_history") or
        not tasks or len(set(tasks)) != len(tasks) or
        any(task not in experiments.reference.CATEGORIES for task in tasks) or
        not 1 <= max_cycles <= 12 or
        not 1 <= max_visits_per_task <= 3 or
        not 1 <= max_nodes_per_cycle <= 4 or
        not 2 <= max_proposals <= 12 or
        not 1 <= max_stagnant_cycles <= max_cycles or
        not 1 <= min_ready <= 12 or
        not 1 <= min_families <= 2 or
        max_total_requests < 1 or
        cohort_size not in (4,6,8,10,12)):
        raise ValueError("unsafe or invalid bounded scientific director protocol")
    if prepared_state is not None and (mode!="fresh" or
                                       not any(t.startswith("agentcf_") for t in tasks) or
                                       max_visits_per_task != 1):
        raise ValueError("frozen state reuse needs fresh AgentCF and one visit per task")
    if model is None and (config is None or
                          config.get("llm",{}).get("provider")!="vllm"):
        raise ValueError("local Qwen/vLLM is required; no external API fallback")
    if config is not None and (config.get("llm",{}).get("provider")!="vllm" or
                               max_total_requests>config["budget"]["max_requests"]):
        raise ValueError("director budget may not exceed frozen Qwen budget")

    # One preregistered schedule, no post-result seed search.
    schedule=[]
    for visit in range(max_visits_per_task):
        for task in tasks:
            if task.startswith("coat_") and visit:
                continue   # Replaying the same Coat development panel is not new evidence.
            schedule.append({"task":task,"visit":visit+1,
                             "seed":seed+1009*visit})
    if len(schedule) > max_cycles:
        schedule=schedule[:max_cycles]
    if prepared_state is not None:
        initial=experiments.prepare_cohort(set(),seed,cohort_size)
        experiments.load_prepared_agentcf(prepared_state,initial)

    output.mkdir(parents=True)
    (output/"cycles").mkdir()
    protocol={
        "version":"autonomous-scientific-director-development-v1",
        "tasks":tasks,"schedule":schedule,
        "method":"predeclared task/seed inventory with registered "
                 "memory-support redirection and evidence-based development stop",
        "cohort_size":cohort_size,"mode":mode,"arm":arm,
        "max_cycles":max_cycles,
        "max_nodes_per_cycle":max_nodes_per_cycle,
        "max_generated_proposals_per_cycle":max_proposals,
        "max_total_model_requests":max_total_requests,
        "max_consecutive_cycles_without_ready_candidate":max_stagnant_cycles,
        "acceptance":{"min_development_ready_tasks":min_ready,
                      "min_distinct_task_families":min_families,
                      "cannot_self_confirm_science":True},
        "prepared_state":str(Path(prepared_state).resolve()) if prepared_state else None,
        "llm":model_description(config) if config else {"kind":"offline_test_double"},
        "only_source_verified_audits_feed_research_memory":True,
        "no_gpu_launch_or_external_code_execution":True,
        "no_holdout_reading":True,
    }
    store_new(output/"protocol.json",protocol)
    store_new(output/"protocol.sha256.json",{"sha256":digest(output/"protocol.json")})
    if model is None:
        limits=RunLimits(output,**{
            **config["budget"],"max_requests":max_total_requests})
        model=create_model(output,max_total_requests,limits,config)

    status={"state":"RUNNING","started":utc(),"cycles":[],
            "selection_history":[],
            "acceptance":decide_acceptance([],min_ready=min_ready,
                                           min_families=min_families),
            "confirmed_discoveries":0}
    def save():
        atomic_json(output/"director_state.json",status)
    save()
    observed_users_by_seed={}
    frozen_by_seed={}
    stagnant=0
    pending=list(schedule)
    try:
        while pending:
            if model.requests>=max_total_requests:
                status["state"]="STOP_RESOURCE_BUDGET"
                break
            if len(status["cycles"])>=max_cycles:
                status["state"]="STOP_CYCLE_CEILING"
                break
            index,choice=choose_next(
                pending,status["cycles"][-1] if status["cycles"] else None)
            entry=pending.pop(index)
            status["selection_history"].append({
                "task":entry["task"],"seed":entry["seed"],
                "reason":choice,
                "candidate_task_count_at_selection":len(pending)+1})
            i=len(status["cycles"])
            task,run_seed=entry["task"],entry["seed"]
            if task.startswith("agentcf_") and mode=="fresh":
                cohort=experiments.prepare_cohort(set(),run_seed,cohort_size)
                users=set(cohort[0]["users"])
                prior_users=set().union(
                    *(value for cohort_seed,value in observed_users_by_seed.items()
                      if cohort_seed != run_seed))
                if prior_users & users:
                    # Changing the seed does not make an overlapping participant
                    # a fresh independent observation. Do NOT cherry-pick a seed.
                    status["cycles"].append({
                        **entry,"status":"SKIP_REUSED_TRAINING_USERS",
                        "overlap_count":len(prior_users & users),
                        "new_experimental_observations":0,
                        "ready_candidates":[]})
                    stagnant+=1
                    save()
                    if stagnant>=max_stagnant_cycles:
                        status["state"]="STOP_NO_IDENTIFIABLE_PROGRESS"
                        break
                    continue
                observed_users_by_seed[run_seed]=users

            cycle_name=f"cycle-{i+1:03d}-{task}"
            destination=output/"cycles"/cycle_name
            before=model.requests
            shared=(frozen_by_seed.get(run_seed,prepared_state)
                    if task.startswith("agentcf_") and entry["visit"]==1
                    else None)
            tree=search(destination,task,model=model,config=config,
                        mode=mode,arm=arm,seed=run_seed,
                        cohort_size=cohort_size,
                        width=2,max_depth=2,
                        max_nodes=max_nodes_per_cycle,
                        max_proposals=max_proposals,
                        max_requests=max_total_requests,
                        prior_campaign=output,
                        prepared_state=shared,
                        search_policy="tree")
            if (task.startswith("agentcf_") and
                    tree.get("state_ref") and
                    run_seed not in frozen_by_seed):
                frozen_by_seed[run_seed]=tree["state_ref"]
            new=eligibility(destination,tree)
            stat={
                **entry,"status":tree["status"],
                "shared_development_state_reused":bool(shared),
                "same_cohort_is_not_independent_confirmation":bool(shared),
                "tree_ref":str(destination.relative_to(output)),
                "attempted_nodes":tree["attempted_nodes"],
                "executed_development_nodes":tree["executed_development_nodes"],
                "model_requests_this_cycle":model.requests-before,
                "ready_candidates":new,
                "mechanism_support_gate":tree.get("identifiability_gate"),
                "observed_before_design":bool(tree.get("identifiability_gate")),
                "independently_confirmed_discoveries":0}
            status["cycles"].append(stat)
            status["acceptance"]=decide_acceptance(status["cycles"],
                min_ready=min_ready,min_families=min_families)
            stagnant=0 if new else stagnant+1
            save()

            if status["acceptance"]["development_readiness_met"]:
                status["state"]="STOP_DEVELOPMENT_READY_FOR_INDEPENDENT_CONFIRMATION"
                break
            if stagnant>=max_stagnant_cycles:
                status["state"]="STOP_NO_IDENTIFIABLE_PROGRESS"
                break
        else:
            status["state"]="STOP_SCHEDULE_EXHAUSTED"

        if status["state"]=="RUNNING":
            status["state"]="STOP_SCHEDULE_EXHAUSTED"
        status["finished"]=utc()
        status["total_model_requests"]=model.requests
        status["actual_model_usage"]=(
            usage_summary(output) if (output/"calls.jsonl").is_file()
            else {"requests":model.requests,"usage_measured":False,
                  "reason":"offline software model"})
        status["independently_confirmed_discoveries"]=0
        status["scientific_goal_accepted"]=False
    except Exception as e:
        status["state"]="FAILED_CLOSED"
        status["error_type"]=type(e).__name__
        save()
        raise
    save()
    return status


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--tasks",nargs="+",required=True,
                        choices=sorted(experiments.reference.CATEGORIES))
    parser.add_argument("--seed",type=int,default=20270001)
    parser.add_argument("--cohort-size",type=int,default=8)
    parser.add_argument("--max-cycles",type=int,default=4)
    parser.add_argument("--max-visits-per-task",type=int,default=1)
    parser.add_argument("--max-nodes-per-cycle",type=int,default=2)
    parser.add_argument("--max-total-requests",type=int,default=120)
    parser.add_argument("--max-stagnant-cycles",type=int,default=3)
    parser.add_argument("--min-ready",type=int,default=2)
    parser.add_argument("--min-families",type=int,default=2)
    parser.add_argument("--prepared-state")
    parser.add_argument("--mode",choices=["fresh","archive"],default="fresh")
    parser.add_argument("--arm",choices=["qwen_no_history","qwen_audited_history"],
                        default="qwen_audited_history")
    args=parser.parse_args(argv)
    config=json.loads(Path(args.config).read_text(encoding="utf8"))
    # Fail before creating any run or using any GPU if service is not authorized.
    create_model(args.output,config["budget"]["max_requests"],None,config).health()
    result=conduct(args.output,args.tasks,config=config,seed=args.seed,
                   cohort_size=args.cohort_size,max_cycles=args.max_cycles,
                   max_visits_per_task=args.max_visits_per_task,
                   max_nodes_per_cycle=args.max_nodes_per_cycle,
                   max_total_requests=args.max_total_requests,
                   max_stagnant_cycles=args.max_stagnant_cycles,
                   min_ready=args.min_ready,min_families=args.min_families,
                   mode=args.mode,prepared_state=args.prepared_state,
                   arm=args.arm)
    print(json.dumps({"state":result["state"],
                      "cycles":len(result["cycles"]),
                      "request_count":result["total_model_requests"],
                      "scientific_confirmations":0},ensure_ascii=False))


if __name__=="__main__":
    main()
