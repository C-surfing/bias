"""Bounded, persistent multi-agent research campaign on Qwen + AgentCF.

Each role is a separate model invocation. Python freezes the plan, reviews
registered actions, executes the experiment, and scores its numeric forecast.
This is a DEVELOPMENT workflow; no holdout access or self-certified discoveries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from rsi_researcher.paths import ROOT
from rsi_researcher.research import crossbias_discovery as source
from rsi_researcher.research.scientific_agents import (
    DiagnosisAgent, DesignerAgent, CriticAgent, ReflectionAgent,
    authoritative_evidence,
)
from rsi_researcher.research import scientific_experiments as experiments
from rsi_researcher.research import scientific_skills as skills
from rsi_researcher.runtime.client import atomic_json, usage_summary
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.models import create_model, model_description


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def store_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Frozen artifact already exists: " + str(path))
    atomic_json(path, value)


def verified_experience(run_dir, prior_campaign=None):
    """Read only hashed prior numeric development audits. No reflection prose."""
    records = []
    sources=[]
    if prior_campaign is not None:
        prior=Path(prior_campaign).resolve()
        if prior.resolve()==Path(run_dir).resolve() or not (prior/"protocol.json").is_file():
            raise ValueError("prior campaign must be a distinct completed research run")
        expected=json.loads((prior/"protocol.sha256.json").read_text(encoding="utf8"))
        if expected["sha256"]!=digest(prior/"protocol.json"):
            raise ValueError("prior campaign protocol digest changed")
        sources.append(prior)
    sources.append(Path(run_dir))
    # One source of numeric research memory across linear and tree searches.
    # Node-index text and Reflection prose are NOT evidence; only completed
    # independent Python audit artifacts may be included.
    for sub in (ep for root in sources for ep in sorted(
            [*root.glob("episode-*"),*root.glob("nodes/N[0-9]*"),
             *root.glob("cycles/cycle-*/nodes/N[0-9]*")])):
        path = sub / "audit.json"
        if not path.is_file():
            continue
        audit = json.loads(path.read_text(encoding="utf8"))
        if audit.get("independently_confirmed_discoveries") != 0:
            raise ValueError("unsupported confirmation claim in development journal")
        records.append({
            "task": audit["task"], "action": audit["action"],
            "grade": audit["evidence_grade"],
            "measurements": audit["panel_measurements"],
            "source_sha256": digest(path),
            "limitation": audit["reason"],
        })
    return records[-4:]


def public_evidence(task, experience, arm="qwen_audited_history", skill_library=None):
    source.verify()
    path = source.RUN / "public_evidence.json"
    raw = json.loads(path.read_text(encoding="utf8"))
    if arm in ("qwen_no_history", "strong_rule", "qwen_promoted_skill"):
        relevant = []
    elif arm == "qwen_shuffled_history":
        relevant = [e for e in experience
                    if e["task"].startswith("coat_") != task.startswith("coat_")][-2:]
    else:
        relevant = [e for e in experience
                    if e["task"].startswith("coat_") == task.startswith("coat_")][-2:]
    promoted = (skills.select_promoted(skill_library, task)
                if arm == "qwen_promoted_skill" and skill_library else [])
    if arm == "qwen_promoted_skill" and not promoted:
        raise ValueError("promoted-skill arm requires a validated eligible skill")
    return {
        "question": source.CATEGORIES[task]["question"],
        "formal_five_bias_target": experiments.EXPERIMENT_BIAS_TARGET[task],
        "five_bias_registry_scope": [
            {"id":t["bias_id"],"maturity":t["maturity"],
             "current_experiment_adapters":t["existing_experimental_tasks"]}
            for t in experiments.five_bias_coverage()],
        "auxiliary_mechanism_not_debias_goal": task=="agentcf_memory",
        "registered_actions": list(experiments.task_spec(task)["actions"]),
        "registered_metrics": list(experiments.task_metrics(task)),
        "action_metric_contracts": experiments.metrics_by_action(task),
        "minimum_development_effect_by_metric": experiments.DEVELOPMENT_EFFECT_FLOOR,
        "original_mechanism_falsifier": source.CATEGORIES[task]["falsifier"],
        "old_development_observations": raw["coat" if task.startswith("coat_") else "agentcf"],
        "relevant_prior_numeric_evidence": relevant,
        "promoted_research_skills": promoted,
        "domain_priors": [
            "Not chosen is not equivalent to explicit dislike or negative feedback.",
            "Keep user memory, training history and candidate items paired under one-variable interventions.",
            "Measure stochastic repeated ranking before attributing changes to an intervention.",
            "Popularity exposure alone is not an identified causal bias.",
            "Never promote development observations or candidate research skills as independent findings."
        ],
        "evidence_sha256": digest(path),
        "experiment_protocol": {
            "agentcf": ("hold trained User/Item Memory and query candidates fixed; "
                        "compare original, unchanged repeat, and one prompt or "
                        "item-memory intervention; evaluate only conditional "
                        "ranking effects, not general causal popularity bias"),
            "coat": ("use same train and original random development users; "
                     "perturb estimated propensity but do not touch holdout"),
            "delta_sign": "all delta metrics are AFTER minus BEFORE, except brier_gain which is BEFORE minus AFTER",
            "quality_guard": "candidate NDCG@5 >= 99% of baseline on both development panels",
        },
        "limits": ("development observations only; no unseen test labels, "
                   "no causal conclusion, no access to executor from Qwen"),
    }


def review_plan(task, plan, critique):
    if plan["action"] == "abstain":
        return "ABSTAIN_NO_DISCRIMINATING_EXPERIMENT"
    if plan["action"] not in experiments.task_spec(task)["actions"]:
        return "REJECTED_UNREGISTERED_ACTION"
    if plan["metric"] not in experiments.action_metrics(task, plan["action"]):
        return "REJECTED_ACTION_METRIC_MISMATCH"
    if plan.get("minimum_abs_effect",experiments.DEVELOPMENT_EFFECT_FLOOR[plan["metric"]]) < experiments.DEVELOPMENT_EFFECT_FLOOR[plan["metric"]]:
        return "REJECTED_BELOW_PREDECLARED_EFFECT_FLOOR"
    if (task, plan["action"], plan["metric"]) == ("agentcf_popularity", "popularity_prompt", "head_excess_delta") and plan.get("direction") != "decrease":
        return "REJECTED_PREDECLARED_DIRECTION_MISMATCH"
    if critique["decision"] != "approve":
        return "REJECTED_BY_CRITIC"
    return "ACCEPTED_DESIGN_ONLY"


def fixed_rule_proposal(task):
    """One predeclared rule with the SAME metric floor as Qwen proposals."""
    action = source.RULE[task]
    metric = experiments.action_metrics(task, action)[0]
    return {
        "action": action,
        "metric": metric,
        "direction": "decrease" if task == "agentcf_popularity" else "increase",
        "minimum_abs_effect": experiments.DEVELOPMENT_EFFECT_FLOOR[metric],
        "hypothesis": "Predeclared rule tests one registered mechanism",
        "competing_explanation": "Ranking noise, propensity uncertainty or candidate mix",
        "falsifier": "No paired effect beyond the fixed metric floor, or quality harm",
        "budget_justification": "Rule and prediction threshold fixed before inference",
    }


def episode(directory, task, model, mode, excluded_users, seed, cohort_size,
            allowed_next_tasks=(), rule_fallback=False,
            arm="qwen_audited_history", skill_library=None, prior_campaign=None,
            prepared_state=None, tree_proposal=None, tree_diagnosis=None,
            tree_parent=None):
    directory.mkdir(parents=True, exist_ok=False)
    begin = perf_counter()
    calls_before = model.requests
    status = {"status": "RUNNING", "task": task, "mode": mode,
              "research_arm": arm, "started": utc()}
    atomic_json(directory / "status.json", status)
    try:
        ctx = public_evidence(task, verified_experience(directory.parent, prior_campaign),
                              arm, skill_library)
        cohort = None
        if mode == "fresh" and task.startswith("agentcf_"):
            cohort = experiments.prepare_cohort(excluded_users, seed, cohort_size)
            frozen = {"users": sorted(cohort[0]["users"]), "events":len(cohort[1]),
                      "queries":len(cohort[2]), "seed":seed,
                      "no_validation_targets_in_research_prompt":True,
                      "scientific_input_sha256":experiments.scientific_input_fingerprint(cohort)}
            store_new(directory / "cohort_frozen.json", frozen)
            if prepared_state is None:
                # Standalone science run: observe a genuine trained AgentCF
                # baseline BEFORE generating a hypothesis or experiment plan.
                prepared = experiments.train_and_observe_agentcf(cohort, model, directory)
                prepared_state = directory / "prepared_agentcf.json"
                shared_cost = False
            else:
                prepared = experiments.load_prepared_agentcf(prepared_state, cohort)
                shared_cost = True
            ctx["new_agentcf_input_observations"] = prepared["observation"]
            ctx["predecision_baseline_fingerprint"] = prepared["training"]["baseline_rankings_sha256"]
            ctx["shared_preparation_cost_accounted_separately"] = shared_cost
        if tree_parent is not None:
            # Only already-audited ancestry, never a sibling's future outcome.
            ctx["tree_parent_development_evidence"] = tree_parent
        store_new(directory / "evidence.json", ctx)
        eid = directory.name

        selected_by = ("tree_frozen_designer_proposal" if tree_proposal is not None
                       else "qwen_accepted_design")
        if arm == "strong_rule":
            # Strong fixed rule does not pay or receive credit for scientific-agent calls.
            plan = fixed_rule_proposal(task)
            selected_by = "fixed_rule_baseline"
            decision = "PREDECLARED_RULE_BASELINE"
            store_new(directory / "frozen_plan.json", {
                "task": task, "proposal": plan, "decision": decision, "frozen_at": utc()})
        else:
            diag = (tree_diagnosis if tree_diagnosis is not None else
                    DiagnosisAgent(model).act(ctx, eid))
            store_new(directory / "diagnosis.json", diag)
            designer = DesignerAgent(model)
            context = {"evidence": ctx, "diagnosis": diag,
                       "constraint": "single registered action; numeric forecast frozen before execution"}
            # Tree Designer has already frozen one distinct proposal. Running
            # another Designer call here would silently replace the selected node.
            plan = (tree_proposal if tree_proposal is not None else
                    designer.propose(context, eid, experiments.task_spec(task)["actions"],
                                     experiments.task_metrics(task)))
            if plan["action"] == "abstain":
                store_new(directory / "frozen_plan.json", {
                    "task":task,"proposal":plan,
                    "decision":"ABSTAIN_NO_DISCRIMINATING_EXPERIMENT",
                    "frozen_at":utc()})
                status.update(status="ABSTAIN_NO_DISCRIMINATING_EXPERIMENT",
                              reason=plan["budget_justification"],
                              model_calls=model.requests-calls_before)
                atomic_json(directory / "status.json",status)
                return status
            review = CriticAgent(model).act(
                {"evidence": ctx, "proposal": plan, "diagnosis": diag}, eid)
            revisions = []
            if (tree_proposal is None and
                    review_plan(task, plan, review) != "ACCEPTED_DESIGN_ONLY"):
                revisions.append({"plan": plan, "review": review})
                plan = designer.propose(
                    {**context, "critic_feedback": review,
                     "action_metric_contracts": ctx["action_metric_contracts"]}, eid,
                    experiments.task_spec(task)["actions"],
                    experiments.task_metrics(task))
                if plan["action"] == "abstain":
                    store_new(directory / "design_review.json", {
                        "revisions": revisions, "final_review": review})
                    store_new(directory / "frozen_plan.json", {
                        "task":task, "proposal":plan,
                        "decision":"ABSTAIN_NO_DISCRIMINATING_EXPERIMENT",
                        "frozen_at":utc()})
                    status.update(status="ABSTAIN_NO_DISCRIMINATING_EXPERIMENT",
                                  reason=plan["budget_justification"],
                                  model_calls=model.requests-calls_before)
                    atomic_json(directory / "status.json",status)
                    return status
                review = CriticAgent(model).act(
                    {"evidence": ctx, "proposal": plan, "diagnosis": diag,
                     "previous_rejection": revisions[0]}, eid)
            store_new(directory / "design_review.json",
                      {"revisions": revisions, "final_review": review})
            decision = review_plan(task, plan, review)
            store_new(directory / "frozen_plan.json",
                      {"task": task, "proposal": plan,
                       "decision": decision, "frozen_at": utc()})
            if decision != "ACCEPTED_DESIGN_ONLY":
                if not rule_fallback:
                    status.update(status="DESIGN_REJECTED", reason=decision,
                                  model_calls=model.requests-calls_before)
                    atomic_json(directory / "status.json", status)
                    return status
                plan = fixed_rule_proposal(task)
                selected_by = "fixed_rule_control_after_qwen_rejection"
                store_new(directory / "rule_control_plan.json", {
                    "origin":selected_by, "rejected_qwen_plan":decision,
                    "predeclared_rule":source.RULE[task], "control_plan":plan})

        # The fixed-rule fallback must obey the same registry and metric contract.
        if plan["metric"] not in experiments.action_metrics(task, plan["action"]):
            raise ValueError("refused experiment with mismatched action/metric")
        # Any interrupted real experiment is fail-closed: rerun with a NEW output.
        execute_start_calls = model.requests
        execute_start_time = perf_counter()
        result = experiments.execute(task, plan["action"], mode, model, directory, cohort,
                                     prepared_state)
        store_new(directory / "execution.json", result)
        audit = experiments.independent_verdict(task, plan["action"], plan, result)
        audit.update(execution_kind=result.get("execution_kind", result.get("status")),
                     proposal_origin=selected_by,
                     critic_disagreed=bool(selected_by.startswith("fixed_rule")),
                     experiment_seconds=round(perf_counter()-execute_start_time, 3),
                     episode_elapsed_seconds=round(perf_counter()-begin, 3),
                     experiment_model_calls=model.requests-execute_start_calls,
                     total_model_calls_to_audit=model.requests-calls_before)
        audit["authoritative_evidence"]=authoritative_evidence(audit)
        store_new(directory / "audit.json", audit)

        if arm == "strong_rule":
            reflection = {"suggested_next_task":"stop",
                          "reason":"Strong fixed rule has no reflection Agent"}
        else:
            reflection = ReflectionAgent(model).reflect(
                {"frozen_plan": plan, "independent_numeric_verdict": audit,
                 "no_independent_confirmation": True}, eid, list(allowed_next_tasks))
            store_new(directory / "reflection.json", {
                **reflection,
                "model_text_can_only_suggest_hypotheses": True,
                "authoritative_evidence": audit["authoritative_evidence"],
                "do_not_cite_model_numeric_interpretation_as_scientific_result":True})
            # Skill is NOT a model-generated fact. Link to verified numeric source
            # and freeze as a candidate that cannot promote itself.
            skills.proposal(directory, task, reflection["candidate_research_skill"],
                            directory / "audit.json")

        if result.get("status","").startswith("ABSTAIN_"):
            end_state = "ABSTAIN_INSUFFICIENT_MECHANISM_SUPPORT"
        else:
            end_state = ("COMPLETED_INCONCLUSIVE" if audit["evidence_grade"] == "inconclusive"
                         else "COMPLETED_DEVELOPMENT")
            if selected_by.startswith("fixed_rule"):
                end_state = ("COMPLETED_RULE_INCONCLUSIVE" if end_state == "COMPLETED_INCONCLUSIVE"
                             else "COMPLETED_RULE_CONTROL")
        status.update(status=end_state, model_calls=model.requests-calls_before,
                      proposal_origin=selected_by,
                      suggested_next_task=reflection["suggested_next_task"],
                      development_forecast_pass=bool(audit["forecast_passes_both_panels"]),
                      mechanism_forecast_pass=bool(
                          audit.get("mechanism_forecast_passes_both_panels")),
                      debias_utility_state=audit.get("debias_utility_state"),
                      independently_confirmed_discoveries=0)
    except Exception as exc:
        status.update(status="FAILED", error_type=type(exc).__name__,
                      error=str(exc)[:350], model_calls=model.requests-calls_before,
                      failed_closed=True)
        atomic_json(directory / "status.json", status)
        raise
    atomic_json(directory / "status.json", status)
    return status


def campaign(output, tasks, mode, model=None, max_episodes=4, cohort_size=6, seed=20261017,
             config=None, auto=False, rule_fallback=False,
             arm="qwen_audited_history", skill_library=None, prior_campaign=None,
             prepared_by_task=None):
    """One finite campaign; no self-authorized infinite loop or GPU provisioning."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Campaigns must use a fresh directory, never overwrite history")
    if arm not in ("qwen_no_history", "qwen_audited_history",
                   "qwen_shuffled_history", "qwen_promoted_skill", "strong_rule"):
        raise ValueError("unknown research comparison arm")
    if mode not in ("fresh", "archive") or not tasks or len(tasks) > max_episodes:
        raise ValueError("invalid mode or episode ceiling")
    if len(set(tasks)) != len(tasks) and (mode != "fresh" or any(t.startswith("coat_") for t in tasks)):
        raise ValueError("repeating previously used development tasks cannot produce new evidence")
    for task in tasks:
        experiments.task_spec(task)
    if prior_campaign is not None:
        # Verify the old run before creating any expensive model call.
        verified_experience(Path(output).parent, prior_campaign)
    output.mkdir(parents=True)
    source.verify()
    if model is None:
        if config is None or config.get("llm", {}).get("provider") != "vllm":
            raise ValueError("explicit Qwen vLLM configuration required")
        limits = RunLimits(output, **config["budget"])
        model = create_model(output, limits.max_requests, limits, config)
    protocol = {"version":"rsi-multiagent-development-v1", "mode":mode, "tasks":tasks,
                "autonomous_next_task_selection": auto,
                "fixed_rule_fallback_after_rejection": rule_fallback,
                "research_arm": arm,
                "skill_library":str(skill_library) if skill_library else None,
                "prior_campaign":str(prior_campaign) if prior_campaign else None,
                "agents":["diagnosis","design","critic","reflection"],
                "limits":{"max_episodes":max_episodes, "cohort_size":cohort_size},
                "model_requests_are_real":True,
                "no_arbitrary_shell_or_gpu_launch":True,
                "no_confirmatory_dataset_access":True,
                "numerical_auditor_independent_of_agents":True,
                "source_protocol_sha256":digest(source.RUN / "protocol.json"),
                "scientific_source_sha256": {
                    str(p.relative_to(Path(__file__).resolve().parents[1])):digest(p)
                    for p in (
                        Path(__file__).resolve(),
                        Path(__file__).with_name("scientific_agents.py"),
                        Path(__file__).with_name("scientific_experiments.py"),
                        Path(__file__).with_name("scientific_comparison.py"),
                        Path(__file__).parents[1] / "recommenders" / "agentcf.py",
                        Path(__file__).parents[1] / "datasets" / "author.py",
                        Path(__file__).parents[1] / "evaluation" / "author.py")},
                "base_mainline_commit":"4264151"}
    store_new(output / "protocol.json", protocol)
    store_new(output / "protocol.sha256.json", {"sha256":digest(output / "protocol.json")})
    excluded = set()
    statuses = []
    worklist = list(tasks) if not auto else [tasks[0]]
    for i in range(max_episodes if auto else len(worklist)):
        if i >= len(worklist):
            break
        task = worklist[i]
        if task.startswith("coat_") and task in worklist[:i]:
            raise ValueError("Coat development panel already consumed in this campaign")
        run_dir = output / f"episode-{i+1:03d}"
        try:
            status = episode(run_dir, task, model, mode, excluded, seed+i,
                             cohort_size, list(dict.fromkeys(tasks)) if auto else [],
                             rule_fallback, arm, skill_library, prior_campaign,
                             (prepared_by_task or {}).get(task))
        except Exception:
            statuses.append(json.loads((run_dir / "status.json").read_text(encoding="utf8")))
            atomic_json(output / "campaign.json", {"status":"FAILED_CLOSED","episodes":statuses,
                        "independently_confirmed_discoveries":0})
            raise
        statuses.append(status)
        frozen = run_dir / "cohort_frozen.json"
        if frozen.exists():
            excluded.update(json.loads(frozen.read_text(encoding="utf8"))["users"])
        if auto:
            candidate = status.get("suggested_next_task", "stop")
            if candidate == "stop" or status["status"] == "DESIGN_REJECTED":
                break
            if candidate not in tasks:
                raise ValueError("Agent proposed a task outside the preapproved campaign scope")
            if candidate.startswith("coat_") and candidate in worklist:
                break
            if mode == "archive" and candidate in worklist:
                break
            worklist.append(candidate)
    executed = sum(s["status"] in ("COMPLETED_DEVELOPMENT", "COMPLETED_INCONCLUSIVE",
                                  "COMPLETED_RULE_CONTROL", "COMPLETED_RULE_INCONCLUSIVE") for s in statuses)
    overall = ("COMPLETED_NO_EXECUTION" if executed == 0
               else "COMPLETED_DEVELOPMENT" if executed == len(statuses)
               else "PARTIAL_DEVELOPMENT")
    result = {"status":overall, "episodes":statuses,
              "executed_episodes":executed,
              "total_model_requests":model.requests,
              "independently_confirmed_discoveries":0,
              "no_claim_of_rsi_self_improvement":True}
    if (output/"calls.jsonl").is_file():
        result["model_usage"]=usage_summary(output)
    else:
        result["model_usage"]={"requests":model.requests,
                               "usage_measured":False,
                               "reason":"test model with no vLLM request log"}
    result["model_budget_ceiling"]=config.get("budget") if config else None
    atomic_json(output / "campaign.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--mode",choices=["fresh","archive"],default="fresh")
    parser.add_argument("--tasks",nargs="+",choices=sorted(source.CATEGORIES),
                        default=["agentcf_memory","agentcf_popularity"])
    parser.add_argument("--arm", choices=["qwen_no_history","qwen_audited_history",
                        "qwen_shuffled_history","qwen_promoted_skill","strong_rule"],
                        default="qwen_audited_history")
    parser.add_argument("--skill-library", default=None)
    parser.add_argument("--prior-campaign", default=None)
    parser.add_argument("--fallback-fixed-rule",action="store_true",
                        help="After Qwen design rejection, execute a labeled strong-rule control only")
    parser.add_argument("--auto",action="store_true",
                        help="Let Reflection propose the next registered task, subject to budget and data guards")
    parser.add_argument("--cohort-size",type=int,default=6)
    parser.add_argument("--seed",type=int,default=20261017)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    if config.get("llm", {}).get("provider") != "vllm":
        parser.error("Qwen vLLM is the only allowed scientific-agent backend")
    if not 4 <= args.cohort_size <= 12 or args.cohort_size % 2:
        parser.error("cohort-size must be an even number between 4 and 12")
    if len(args.tasks) > config.get("max_episodes", 4):
        parser.error("requested tasks exceed campaign episode budget")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("output directory already exists (immutable campaign)")
    # Check the already-authorized vLLM service; do not start/reuse a shared GPU.
    description = model_description(config)
    health_probe = create_model(output, config["budget"]["max_requests"], None, config)
    health_probe.health()
    result = campaign(output, args.tasks, args.mode,
                      max_episodes=config.get("max_episodes", 4),
                      cohort_size=args.cohort_size, seed=args.seed,
                      config=config, auto=args.auto,
                      rule_fallback=args.fallback_fixed_rule,
                      arm=args.arm, skill_library=args.skill_library,
                      prior_campaign=args.prior_campaign)
    atomic_json(output / "model_config.json",description)
    print(json.dumps({"status":result["status"],"episodes":len(result["episodes"]),
                      "independent_discoveries":0},ensure_ascii=False))


if __name__ == "__main__":
    main()
