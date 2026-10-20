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


"""A bounded *Debias* search on the five registered tasks, within the Director.

Four EXISTING research roles, original Bias Registry and frozen real-data
numeric actions are reused. No new research agents, evaluators or holdout use.
This is a development-stage strategy test, not independent discovery.
"""
from pathlib import Path
import json
from time import perf_counter
from rsi_researcher.research.five_bias_development import (
    DEBIAS_ACTIONS, DEBIAS_METRICS, evaluate_debias_action,
    public_debias_context,
)
from rsi_researcher.research.scientific_agents import (
    DiagnosisAgent, DesignerAgent, CriticAgent, ReflectionAgent, strict, TEXT,
    authoritative_evidence,
)
from rsi_researcher.research.scientific_loop import utc, digest
from rsi_researcher.runtime.client import atomic_json, usage_summary
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.models import create_model, model_description


def conduct_five_bias_debias(output, *, evidence, model=None, config=None,
                            policy="qwen", max_rounds=14, max_requests=90,
                            min_development_debias_tasks=1,
                            joint_design_only=False,
                            max_joint_proposals=1):
    """Continue until real development Debias goal or honest bounded exhaustion.

    Method candidates, effect floors, evaluation data, and stop criteria are
    frozen BEFORE proposals. Evaluated numeric outcomes stay hidden from the
    model until that method has been chosen and audited. A rejection triggers
    another legal method on the next cycle; no lucky-seed reruns.
    """
    begin_wall=perf_counter()
    output=Path(output).resolve()
    evidence=Path(evidence).resolve()
    if (output.exists() or policy not in ("qwen","strong_rule") or
        not 5<=max_rounds<=50 or not 1<=max_requests<=500 or
        not 1<=min_development_debias_tasks<=5 or
        not 1<=max_joint_proposals<=5):
        raise ValueError("invalid five-bias Debias scientific campaign")
    if model is None and (config is None or
                          config.get("llm",{}).get("provider")!="vllm"):
        raise ValueError("local vLLM Qwen configuration required")
    if config is not None and max_requests>config["budget"]["max_requests"]:
        raise ValueError("model requests exceed configured budget")
    frozen=json.loads(evidence.read_text(encoding="utf8"))
    if (frozen.get("mode")!="fixed_development_protocol_never_confirmation"
        or frozen.get("confirmed_debias_findings")!=0 or
        set(frozen.get("data",{}))!=set(DEBIAS_ACTIONS)):
        raise ValueError("unsupported or altered five-bias evidence contract")
    # Lock every candidate's original measured outcome to this input digest.
    protocol={
        "version":"five-bias-constrained-scientific-debias-v1",
        "policy":policy,"max_rounds":max_rounds,
        "joint_design_only":joint_design_only,
        "maximum_frozen_joint_proposals":max_joint_proposals,
        "hard_max_requests":max_requests,
        "min_development_debias_tasks":min_development_debias_tasks,
        "five_bias_tasks":list(DEBIAS_ACTIONS),
        "predeclared_actions":{k:list(v) for k,v in DEBIAS_ACTIONS.items()},
        "predeclared_metrics":DEBIAS_METRICS,
        "frozen_real_development_evidence":str(evidence),
        "evidence_sha256":digest(evidence),
        "no_holdout_use":True,
        "evaluator_and_acceptance_rules_never_modified_by_agent":True,
        "agent_can_propose_training_policy_parameters":True,
        "no_independent_scientific_confirmation":True,
        "model":model_description(config) if config else {"kind":"offline-test-double"},
    }
    output.mkdir(parents=True)
    atomic_json(output/"protocol.json",protocol)
    atomic_json(output/"protocol.sha256.json",{"sha256":digest(output/"protocol.json")})
    if model is None:
        limits=RunLimits(output,**{**config["budget"],
                                    "max_requests":max_requests})
        model=create_model(output,max_requests,limits,config)
    actions_left={k:(
        ["joint_propensity_popularity_design"] if k=="selection" else []
        ) if joint_design_only else list(v)
        for k,v in DEBIAS_ACTIONS.items()}
    attempted={k:0 for k in DEBIAS_ACTIONS}
    joint_proposed=[]
    completed=set()
    pending=list(DEBIAS_ACTIONS)
    sequence=[]
    state={
        "status":"RUNNING","started_at":utc(),"rounds":sequence,
        "task_status":{},"policy":policy,"actual_model_requests":0,
        "development_debias_candidates":[],
        "observed_proxy_effects":[],
        "independently_confirmed_debias":0,
        "science_goal_met":False,
        "development_goal_met":False,
    }
    def save():
        state["actual_model_requests"]=model.requests
        atomic_json(output/"director_state.json",state)
    save()
    try:
        # One-pass explicit abstention, no fake action for unidentifiable bubble.
        for task in pending:
            if not actions_left[task]:
                state["task_status"][task]=(
                    "ABSTAIN_UNIDENTIFIABLE_FILTER_BUBBLE" if
                    task=="filter_bubble" and not joint_design_only
                    else "NOT_IN_REGISTERED_SEARCH_MODE")
                completed.add(task)
        pending=[t for t in pending if t not in completed]
        save()

        while pending and len(sequence)<max_rounds:
            # One scientific episode needs diagnosis + design + review +
            # reflection. Do not start an episode without its call budget.
            if model.requests >= max_requests or (
                    policy=="qwen" and model.requests+5>max_requests):
                state["status"]="STOP_MODEL_REQUEST_BUDGET"
                break
            bias=pending.pop(0)
            if not actions_left[bias]:
                completed.add(bias)
                state["task_status"].setdefault(bias,"EXHAUSTED_METHODS_NO_DEBIAS")
                continue
            previous=[{"action":r["action"],"audit":r.get("numeric_audit"),
                       "decision":r["status"]} for r in sequence
                      if r["bias_id"]==bias and r.get("numeric_audit")]
            context=public_debias_context(frozen,bias,previous)
            metric,threshold=DEBIAS_METRICS[bias]
            claim_contract=context["falsification_protocol"]
            round_id=f"round-{len(sequence)+1:03d}-{bias}"
            choice={"round_id":round_id,"bias_id":bias,
                    "available_actions":list(actions_left[bias]),
                    "metric":metric,"numeric_threshold":threshold,
                    "prior_audit_count":len(previous)}
            if policy=="strong_rule":
                choice["action"]=actions_left[bias][0]
                choice["status"]="FIXED_RULE_PREDECLARED"
                if choice["action"]=="joint_propensity_popularity_design":
                    candidates=((.75,.05),(.9,0.),(.5,.02),(.65,.01),(.3,0.))
                    alpha,beta=candidates[min(len(joint_proposed),len(candidates)-1)]
                    choice["joint_parameters"]={"alpha":alpha,"beta":beta}
            else:
                diag=DiagnosisAgent(model).act(context,round_id)
                design=DesignerAgent(model).propose(
                    {"evidence":context,"diagnosis":diag,
                     "metric_definition":metric,
                     "minimum_effect":threshold,
                     "do_not_infer_unseen_method_outcomes":True},
                    round_id,actions_left[bias],[metric])
                choice.update({"diagnosis":diag,"frozen_proposal":design})
                proposed=design["action"]
                if proposed=="abstain":
                    choice["status"]="AGENT_ABSTAIN_NO_ACTION"
                    actions_left[bias].clear()
                    completed.add(bias)
                    state["task_status"][bias]="ABSTAIN_BY_DESIGNER"
                    sequence.append(choice)
                    save()
                    continue
                if proposed not in actions_left[bias]:
                    raise ValueError("Qwen proposed unregistered method")
                choice["action"]=proposed
                if proposed=="joint_propensity_popularity_design":
                    values=DesignerAgent(model).act({
                        "legal_expression":"(1-alpha)*unweighted_item_probability + alpha*propensity_weighted_item_probability; ranking subtracts beta*log(1+training_frequency)/normalizer",
                        "unknown_development_outcome":True,
                        "predeclared_goal":"Brier improvement >=0.002, NDCG >=99% of baseline, Hit loss <=0.02",
                        "alpha_range":[0,1],"beta_range":[0,.15],
                        "previous_joint_development_trials": [
                            {"parameters":x["parameters"],
                             "outcome":x["result"]} for x in joint_proposed],
                        "must_avoid_duplicate_parameters":True,
                        "no_holdout_labels_visible":True
                    },round_id,schema=strict({
                        "alpha":{"type":"number","minimum":0,"maximum":1},
                        "beta":{"type":"number","minimum":0,"maximum":.15},
                        "mechanism":TEXT,"falsifier":TEXT}))
                    choice["joint_parameters"]={"alpha":values["alpha"],
                                                "beta":values["beta"]}
                    choice["joint_hypothesis"]=values
                duplicate_proposal = (
                    proposed=="joint_propensity_popularity_design" and
                    any(x["parameters"]==choice.get("joint_parameters")
                        for x in joint_proposed))
                if duplicate_proposal:
                    # An invalid repeat cannot gain new evidence. Save the
                    # Critic request: numeric evaluation would also be skipped.
                    choice["status"]="REJECTED_DUPLICATE_DEVELOPMENT_PROPOSAL"
                    choice["critic_review"]={
                        "decision":"not_called","reason":"duplicate frozen parameters",
                        "numeric_claim_authority":"none"}
                else:
                    critique=CriticAgent(model).review_debias({
                        "five_bias_task":context,
                        "frozen_design":design,
                        "joint_parameters":choice.get("joint_parameters"),
                        "real_numeric_results_hidden":True,
                        "registered_evaluation_guards":{
                            "metric":metric,"minimum_effect":threshold,
                            "never_claim_causal_confirmation":True,
                            "development_assessment_only":True},
                        "relevance_challenge":{
                            "primary_null":claim_contract["primary_null_hypothesis"],
                            "observable_falsifier":claim_contract["registered_falsifier"],
                            "allowed_conclusion":claim_contract["admissible_conclusion"],
                            "potential_counterexample":claim_contract["counterexample_challenge"],
                            "forbidden_inferences":claim_contract["missing_or_nonidentified_if"],
                            "no_sequential_significance_test":True}},round_id)
                    choice["critic_review"]=critique
                    if design["metric"]!=metric or design["direction"]!="increase" or (
                       design["minimum_abs_effect"]<threshold):
                        choice["status"]="REJECTED_PREDECLARED_CONTRACT"
                    elif (critique["decision"]!="approve" or not
                          critique["registered_falsifier_measurable"] or
                          critique["relevance_to_frozen_null"]=="not_relevant"):
                        choice["status"]="REJECTED_BY_CRITIC"
                    else:
                        choice["status"]="CRITIC_APPROVED_DEVELOPMENT_TEST"
            actions_left[bias].remove(choice["action"])
            # Sequential proposals must be distinct; repeated lookups at the
            # same development metrics provide no additional scientific data.
            if choice["action"]=="joint_propensity_popularity_design":
                params=choice.get("joint_parameters")
                if params is None:
                    raise ValueError("joint action missing parameters")
                if any(old["parameters"]==params for old in joint_proposed):
                    choice["status"]="REJECTED_DUPLICATE_DEVELOPMENT_PROPOSAL"
                joint_proposed.append({"parameters":params,
                                        "result":"not_evaluated_yet"})
            if choice["status"] not in ("REJECTED_PREDECLARED_CONTRACT",
                                       "REJECTED_DUPLICATE_DEVELOPMENT_PROPOSAL",
                                       "REJECTED_BY_CRITIC"):
                if digest(evidence)!=protocol["evidence_sha256"]:
                    raise ValueError("frozen numeric action source changed during research")
                numeric_start=perf_counter()
                measurement=evaluate_debias_action(
                    frozen,bias,choice["action"],
                    parameters=choice.get("joint_parameters"))
                choice["numeric_evaluation_wall_seconds"]=round(
                    perf_counter()-numeric_start,4)
                choice["numeric_audit"]=measurement
                choice["authoritative_evidence"]=authoritative_evidence(measurement)
                choice["registered_falsifier"]=claim_contract
                choice["numeric_audit_source_sha256"]=digest(evidence)
                choice["status"]=("DEVELOPMENT_EFFECT_WITH_QUALITY_GUARD"
                                   if measurement["development_effect_passes"]
                                   else "DEVELOPMENT_FALSIFIER_OR_UTILITY_FAILED")
                if choice["action"]=="joint_propensity_popularity_design":
                    joint_proposed[-1]["result"]={
                        "measured_improvement":measurement["measured_improvement"],
                        "quality_guard_passes":measurement["quality_guard_passes"],
                        "passed":measurement["development_effect_passes"]}
                if measurement["development_effect_passes"]:
                    state["observed_proxy_effects"].append({
                        "bias_id":bias,"method":choice["action"],
                        "evidence_class":measurement["evidence_class"]})
                if measurement["development_debias_candidate"]:
                    state["development_debias_candidates"].append({
                        "bias_id":bias,"method":choice["action"],
                        "audit_round":round_id})
            if policy=="qwen":
                reflection=ReflectionAgent(model).reflect({
                    "bias_id":bias,"decision":choice["status"],
                    "measured_numeric_audit":choice.get("numeric_audit"),
                    "critic_reason":choice.get("critic_review",{}).get("reason"),
                    "never_promote_proxy_or_dev_as_scientific_confirmation":True},
                    round_id,[])
                choice["reflection_hypotheses_only"]=reflection
                choice["reflection_evidence_authority"]=authoritative_evidence(
                    choice.get("numeric_audit"))
                choice["reflection_does_not_override_numerical_audit"]=True
            attempted[bias]+=1
            if (choice.get("action")=="joint_propensity_popularity_design"
                    and attempted[bias]<max_joint_proposals and
                    not choice.get("numeric_audit",{}).get(
                        "development_effect_passes")):
                actions_left[bias].append("joint_propensity_popularity_design")
            sequence.append(choice)
            state["task_status"][bias]=choice["status"]
            if (choice.get("numeric_audit",{}).get("development_effect_passes")):
                completed.add(bias)
            elif actions_left[bias]:
                pending.append(bias)
            else:
                completed.add(bias)
                state["task_status"][bias]="EXHAUSTED_METHODS_NO_DEBIAS"
            save()
            # This is a DEVELOPMENT goal, not scientific success. Every frozen
            # bias must be inspected/abstained before accepting a five-bias run.
            covered=all(attempted[t] or t in completed for t in DEBIAS_ACTIONS)
            qualified={x["bias_id"] for x in state["development_debias_candidates"]}
            if covered and len(qualified)>=min_development_debias_tasks:
                state["status"]="STOP_DEVELOPMENT_DEBIAS_CANDIDATE_PENDING_CONFIRMATION"
                state["development_goal_met"]=True
                break
        if state["status"]=="RUNNING":
            state["status"]=("STOP_ROUND_CEILING" if pending else
                             "STOP_NO_FURTHER_REGISTERED_METHODS")
    except Exception as e:
        state["status"]="FAILED_CLOSED"
        state["error_type"]=type(e).__name__
        save()
        raise
    state["finished_at"]=utc()
    state["remaining_actions"]={k:v for k,v in actions_left.items() if v}
    state["evaluated_biases"]=sorted(set(DEBIAS_ACTIONS)-set(pending))
    state["development_debias_targets_reached"]=len({
        c["bias_id"] for c in state["development_debias_candidates"]})
    state["science_goal_met"]=False
    state["independently_confirmed_debias"]=0
    state["evidence_authority"]=authoritative_evidence()
    state["real_confirmation_protocol"]=(
        "Coat cap100 previously failed its one-shot holdout ranking guard; "
        "the old holdout must not be reused for new proposals. "
        "Development success is not independent confirmation.")
    state["frozen_action_tests"]={k:v for k,v in DEBIAS_METRICS.items()}
    state["campaign_wall_seconds"]=round(perf_counter()-begin_wall,3)
    state["total_numeric_execution_wall_seconds"]=round(sum(
        x.get("numeric_evaluation_wall_seconds",0) for x in sequence),3)
    state["executed_development_experiments"]=sum(
        "numeric_audit" in x for x in sequence)
    state["falsified_or_rejected_rounds"]=sum(x["status"] in (
        "REJECTED_BY_CRITIC","REJECTED_PREDECLARED_CONTRACT",
        "REJECTED_DUPLICATE_DEVELOPMENT_PROPOSAL",
        "DEVELOPMENT_FALSIFIER_OR_UTILITY_FAILED") for x in sequence)
    state["actual_model_usage"]=(
        usage_summary(output) if (output/"calls.jsonl").is_file() else
        {"requests":model.requests,"usage_measured":False})
    save()
    return state


def debias_main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--evidence",required=True)
    parser.add_argument("--policy",choices=("qwen","strong_rule"),default="qwen")
    parser.add_argument("--config")
    parser.add_argument("--max-rounds",type=int,default=14)
    parser.add_argument("--max-requests",type=int,default=90)
    parser.add_argument("--min-development-debias-tasks",type=int,default=1)
    parser.add_argument("--joint-design-only",action="store_true")
    parser.add_argument("--max-joint-proposals",type=int,default=1)
    args=parser.parse_args(argv)
    conf=json.loads(Path(args.config).read_text()) if args.config else None
    if args.policy=="qwen":
        if conf is None:
            parser.error("Qwen requires --config")
        create_model(args.output,args.max_requests,None,conf).health()
    elif conf is None:
        # A fixed-rule run needs no model service or credentials; pass an
        # explicit zero-call scorer using the same numerical executor.
        class ZeroModel:
            requests=0
        return _print_debias(conduct_five_bias_debias(
            args.output,evidence=args.evidence,model=ZeroModel(),
            policy="strong_rule",max_rounds=args.max_rounds,
            max_requests=args.max_requests,
            min_development_debias_tasks=args.min_development_debias_tasks,
            joint_design_only=args.joint_design_only,
            max_joint_proposals=args.max_joint_proposals))
    return _print_debias(conduct_five_bias_debias(
        args.output,evidence=args.evidence,config=conf,
        policy=args.policy,max_rounds=args.max_rounds,
        max_requests=args.max_requests,
        min_development_debias_tasks=args.min_development_debias_tasks,
        joint_design_only=args.joint_design_only,
        max_joint_proposals=args.max_joint_proposals))


def _print_debias(result):
    print(json.dumps({"status":result["status"],
                      "attempted_rounds":len(result["rounds"]),
                      "development_debias_targets_reached":
                        result["development_debias_targets_reached"],
                      "independently_confirmed_debias":0,
                      "actual_model_requests":result["actual_model_requests"]},
                     ensure_ascii=False))
