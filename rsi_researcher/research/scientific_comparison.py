"""Matched, bounded comparisons of research strategies on the SAME development cohort.

This only runs development pilots. It NEVER invents independent confirmation.
Each arm has its own model budget, event logs, frozen plan and audit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from rsi_researcher.research import crossbias_discovery as source
from rsi_researcher.research.scientific_loop import campaign, digest
from rsi_researcher.research import scientific_experiments as experiments
from rsi_researcher.runtime.client import atomic_json, usage_summary
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.models import create_model, model_description


DEFAULT_ARMS=("strong_rule","qwen_no_history",
              "qwen_audited_history","qwen_shuffled_history")



def prepare_shared_states(output, tasks, config, seed, cohort_size):
    """Train/probe once per new AgentCF task, then use the same state for all arms."""
    if not any(task.startswith("agentcf_") for task in tasks):
        return {}, {"requests":0, "usage_measured":True}
    folder = Path(output) / "shared_training"
    folder.mkdir()
    source.verify()
    limits = RunLimits(folder, **config["budget"])
    model = create_model(folder, limits.max_requests, limits, config)
    prepared_by_task, excluded = {}, set()
    for i, task in enumerate(tasks):
        if not task.startswith("agentcf_"):
            continue
        cohort = experiments.prepare_cohort(excluded, seed+i, cohort_size)
        directory = folder / f"task-{i+1:03d}-{task}"
        directory.mkdir()
        experiments.train_and_observe_agentcf(cohort, model, directory)
        prepared_by_task[task] = directory / "prepared_agentcf.json"
        excluded.update(cohort[0]["users"])
    cost = (usage_summary(folder) if (folder/"calls.jsonl").is_file()
            else {"requests":model.requests,"usage_measured":False,
                  "reason":"test model without real token logs"})
    return prepared_by_task, cost

def compare_development(output, tasks, config, arms=DEFAULT_ARMS, *,
                        cohort_size=6, seed=20261017, prior_campaign=None,
                        skill_library=None):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("Comparison runs are immutable: output must be new")
    if (not tasks or len(set(tasks))!=len(tasks) or not arms or
        len(set(arms))!=len(arms) or
        any(a not in (*DEFAULT_ARMS,"qwen_promoted_skill") for a in arms)):
        raise ValueError("comparison uses unique registered tasks and arms")
    if config.get("llm",{}).get("provider")!="vllm":
        raise ValueError("Qwen/vLLM required")
    for task in tasks:
        if task not in source.CATEGORIES:
            raise ValueError("unregistered research task")
    if not 4 <= cohort_size <= 12 or cohort_size%2:
        raise ValueError("invalid development cohort size")
    if "qwen_promoted_skill" in arms and not skill_library:
        raise ValueError("evolved skill arm requires verified independent skill library")
    output.mkdir(parents=True)
    protocol={
        "protocol":"matched-research-policy-development-v1",
        "arms":list(arms),"tasks":list(tasks),"cohort_size":cohort_size,
        "cohort_seed":seed,
        "same_seed_and_causal_operator_registry_across_arms":True,
        "agentcf_training_and_predecision_baseline_shared_across_arms":True,
        "fixed_model_request_and_token_budgets":config["budget"],
        "model":model_description(config),
        "reference_protocol_hash":digest(source.RUN/"protocol.json"),
        "prior_campaign":str(prior_campaign) if prior_campaign else None,
        "independent_confirmation_split_used":False,
        "skill_library":str(skill_library) if skill_library else None,
    }
    atomic_json(output/"protocol.json",protocol)
    atomic_json(output/"protocol.sha256.json",{"sha256":digest(output/"protocol.json")})
    result={"status":"RUNNING","arms":{},"independent_discoveries":0}
    atomic_json(output/"comparison.json",result)
    try:
        # Shared training and pre-decision diagnostic ranking is a COMMON cost.
        # All arms see the same trained state, not independent Qwen trajectories.
        prepared_by_task, shared_cost = prepare_shared_states(
            output, tasks, config, seed, cohort_size)
        result["shared_training_model_usage"] = shared_cost
        result["shared_training_state_by_task"] = {
            task: str(path) for task,path in prepared_by_task.items()}
        atomic_json(output/"comparison.json",result)
        for arm in arms:
            directory=output/arm
            run=campaign(directory,tasks,"fresh",max_episodes=len(tasks),
                         cohort_size=cohort_size, seed=seed, config=config, arm=arm,
                         prior_campaign=prior_campaign, skill_library=skill_library,
                         prepared_by_task=prepared_by_task)
            result["arms"][arm]=run
            atomic_json(output/"comparison.json",result)
        pairing={}
        for i,task in enumerate(tasks,1):
            users, inputs, trained, executed, history, decisions = {}, {}, {}, {}, {}, {}
            for arm in arms:
                episode=output/arm/f"episode-{i:03d}"
                freeze=episode/"cohort_frozen.json"
                info=json.loads(freeze.read_text(encoding="utf8")) if freeze.is_file() else {}
                users[arm]=info.get("users")
                inputs[arm]=info.get("scientific_input_sha256")
                training_file=episode/"agentcf_training.json"
                trained[arm]=(json.loads(training_file.read_text(encoding="utf8"))
                              .get("trained_state_sha256") if training_file.is_file() else None)
                evidence_file, plan_file=episode/"evidence.json",episode/"frozen_plan.json"
                evidence=json.loads(evidence_file.read_text(encoding="utf8")) if evidence_file.is_file() else {}
                history[arm]=len(evidence.get("relevant_prior_numeric_evidence", []))
                decisions[arm]=(json.loads(plan_file.read_text(encoding="utf8"))
                                .get("proposal") if plan_file.is_file() else None)
                audit_path, execution_path=episode/"audit.json",episode/"execution.json"
                if audit_path.is_file() and execution_path.is_file():
                    audit=json.loads(audit_path.read_text(encoding="utf8"))
                    execution=json.loads(execution_path.read_text(encoding="utf8"))
                    executed[arm]=(audit.get("task")==task and
                        audit.get("action")==execution.get("action") and
                        audit.get("independently_confirmed_discoveries")==0 and
                        not execution.get("status","").startswith("ABSTAIN_") and
                        execution.get("status") != "ARCHIVED_DEVELOPMENT_REPLAY")
                else:
                    executed[arm]=False
            all_executed=all(executed.values())
            if task.startswith("agentcf_"):
                same_users=all(users.values()) and len({tuple(u) for u in users.values()})==1
                same_inputs=all(inputs.values()) and len(set(inputs.values()))==1
                matched=bool(same_users and same_inputs)
                if all_executed and not matched:
                    raise ValueError("experiment inputs differ across policy arms")
            else:
                same_users, same_inputs, matched=None, None, all_executed
            same_trained=(all(trained.values()) and len(set(trained.values()))==1
                          ) if task.startswith("agentcf_") else None
            pairing[task]={"identical_frozen_user_cohorts":same_users,
                           "identical_frozen_experiment_inputs":same_inputs,
                           "identical_trained_agentcf_states":same_trained,
                           "matched_training_state_for_policy_comparison":bool(
                               all_executed and matched and same_trained),
                           "matched_for_development_comparison":matched,
                           "all_arms_ran_experiment":all_executed,
                           "execution_by_arm":executed,
                           "history_records_seen_by_arm":history,
                           "frozen_scientific_decisions_by_arm":{
                               arm:({k:plan.get(k) for k in ("action","metric","direction",
                                                             "minimum_abs_effect")}
                                    if plan else None)
                               for arm,plan in decisions.items()},
                           "users_by_arm":users,
                           "not_an_independent_confirmation":True}
        completed=all(p["all_arms_ran_experiment"] and p["matched_for_development_comparison"]
                      for p in pairing.values())
        usage={arm:result["arms"][arm].get("model_usage")
               for arm in arms}
        result.update(status=("COMPLETED_DEVELOPMENT" if completed
                              else "INCOMPLETE_DEVELOPMENT_COMPARISON"),
                      cohort_pairing=pairing,
                      memory_comparison_interpretable=bool(
                          completed and prior_campaign and
                          all(row["matched_training_state_for_policy_comparison"] and
                              row["history_records_seen_by_arm"].get("qwen_audited_history",0)>0 and
                              row["history_records_seen_by_arm"].get("qwen_no_history",0)==0
                              for row in pairing.values()) and
                          {"qwen_no_history","qwen_audited_history"}.issubset(arms)),
                      policy_arms_share_trained_state=bool(
                          completed and all(row["matched_training_state_for_policy_comparison"]
                                            for row in pairing.values())),
                      equal_budget_ceiling=True,
                      measured_usage_by_arm=usage,
                      independently_confirmed_scientific_discoveries=0,
                      no_evidence_of_rsi_self_improvement_yet=True)
    except Exception as exc:
        result.update(status="FAILED_CLOSED",error_type=type(exc).__name__,
                      error=str(exc)[:400])
        atomic_json(output/"comparison.json",result)
        raise
    atomic_json(output/"comparison.json",result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--tasks",nargs="+",choices=sorted(source.CATEGORIES),
                   default=["agentcf_popularity"])
    p.add_argument("--arms",nargs="+",choices=[*DEFAULT_ARMS,"qwen_promoted_skill"],
                   default=list(DEFAULT_ARMS))
    p.add_argument("--cohort-size",type=int,default=6)
    p.add_argument("--seed",type=int,default=20261017)
    p.add_argument("--prior-campaign")
    p.add_argument("--skill-library")
    args=p.parse_args(argv)
    config=json.loads(Path(args.config).read_text(encoding="utf8"))
    if config.get("llm",{}).get("provider")!="vllm":
        p.error("Qwen vLLM required; no silent external API fallback")
    # Preflight only; don't start, stop or reuse another user's process.
    health=create_model(args.output,config["budget"]["max_requests"],None,config)
    health.health()
    result=compare_development(args.output,args.tasks,config,arms=args.arms,
        cohort_size=args.cohort_size,seed=args.seed,
        prior_campaign=args.prior_campaign,skill_library=args.skill_library)
    print(json.dumps({"status":result["status"],
                      "arms":list(result["arms"]),
                      "independent_discoveries":0},ensure_ascii=False))


if __name__=="__main__":
    main()
