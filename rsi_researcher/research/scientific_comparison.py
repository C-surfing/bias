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
from rsi_researcher.runtime.client import atomic_json
from rsi_researcher.runtime.models import create_model, model_description


DEFAULT_ARMS=("strong_rule","qwen_no_history",
              "qwen_audited_history","qwen_shuffled_history")


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
        # Each arm is a different research-policy run, but reuses the same
        # precisely frozen development user cohort for matched comparison.
        for arm in arms:
            directory=output/arm
            run=campaign(directory,tasks,"fresh",max_episodes=len(tasks),
                         cohort_size=cohort_size, seed=seed, config=config, arm=arm,
                         prior_campaign=prior_campaign, skill_library=skill_library)
            result["arms"][arm]=run
            atomic_json(output/"comparison.json",result)
        pairing={}
        for i,task in enumerate(tasks,1):
            episode=f"episode-{i:03d}"
            users={}
            for arm in arms:
                f=output/arm/episode/"cohort_frozen.json"
                users[arm]=json.loads(f.read_text(encoding="utf8"))["users"] if f.is_file() else None
            completed=[tuple(u) for u in users.values() if u is not None]
            matching=bool(completed) and len(set(completed))==1
            if len(completed)==len(arms) and not matching:
                raise ValueError("completed comparison arms used different user cohorts")
            pairing[task]={"identical_frozen_user_cohorts":matching,
                           "all_arms_ran_experiment":len(completed)==len(arms),
                           "users_by_arm":users,
                           "not_an_independent_confirmation":True}
        # A zero-execution Qwen arm is a meaningful failure: do not fill in
        # fake metrics or silently substitute a fixed-rule outcome.
        result.update(status="COMPLETED_DEVELOPMENT",cohort_pairing=pairing,
                      memory_comparison_interpretable=bool(prior_campaign),
                      equal_budget_ceiling=True,
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
