"""Registered, bounded scientific interventions: Coat CPU or fresh Qwen-AgentCF.

Never reads author test labels, scores a Coat test partition, or grants an LLM
arbitrary tools. Fresh training/rank inference requires a supplied Qwen model.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from statistics import mean
import json

from rsi_researcher.datasets.author import author_data, author_profile
from rsi_researcher.recommenders.agentcf import AgentCF
from rsi_researcher.research.author_experiment import author_intervention
from rsi_researcher.research import crossbias_discovery as reference
from rsi_researcher.evaluation.author import score_author_ranking
from rsi_researcher.runtime.client import atomic_json

# Historical development records remain tied to their frozen original source.
# New AgentCF experiments read tracked data from the current Git worktree.
# Mixing mutable mainline data into fresh experiments is not reproducible.
SOURCE = Path("/data/cjh/rsi/project")
LOCAL_DATA = Path(__file__).resolve().parents[2] / "data"


def scientific_input_fingerprint(cohort):
    """Same initial memories, events, queries and prompts across policy arms."""
    raw = json.dumps([*cohort, source_digest(LOCAL_DATA / "author_prompts.json")],
                     ensure_ascii=False, sort_keys=True).encode("utf8")
    return sha256(raw).hexdigest()

# A registered action MUST expose the metrics that actually measure that action.
# No generic Brier delta can substitute for a gamma or placebo falsifier.
ACTION_METRICS = {
    "gamma_stress": ("gamma_reference_margin",),
    "propensity_shuffle": ("propensity_placebo_margin",),
    "split_utility": ("brier_gain", "ndcg_delta", "tail_delta"),
    "item_memory_ablation": ("rank_change_excess", "ndcg_delta", "head_excess_delta"),
    "popularity_prompt": ("head_excess_delta", "ndcg_delta", "rank_change_excess"),
}


def action_metrics(task, action):
    if task not in reference.CATEGORIES or action not in reference.CATEGORIES[task]["actions"]:
        raise ValueError("action is not registered for this task")
    return ACTION_METRICS[action]


def metrics_by_action(task):
    if task not in reference.CATEGORIES:
        raise ValueError("unknown task")
    return {a: list(action_metrics(task, a)) for a in reference.CATEGORIES[task]["actions"]}


def task_metrics(task):
    return tuple(dict.fromkeys(metric for metrics in metrics_by_action(task).values()
                               for metric in metrics))


def task_spec(task):
    if task not in reference.CATEGORIES:
        raise ValueError("unregistered research task")
    return reference.CATEGORIES[task]


def known_prior_users():
    prior = SOURCE / "runs/qwen-author-dev20-baseline-20261015/input.json"
    fresh = SOURCE / "runs/agentcf-newstate-forecast-20261017/protocol.json"
    old = json.loads(prior.read_text(encoding="utf8"))["initial"]["users"]
    newer = json.loads(fresh.read_text(encoding="utf8"))["cohort_users"]
    return set(old) | set(newer)


def prepare_cohort(excluded_users, seed, count=6, events_per_user=2):
    if count < 4 or count % 2:
        raise ValueError("fresh cohort must contain at least four users and split equally")
    state, events, queries = author_data(
        LOCAL_DATA, count, events_per_user, seed, "exclude_known",
        excluded_users=known_prior_users() | set(excluded_users),
    )
    if len(queries) != count or len(events) != count * events_per_user:
        raise ValueError("incomplete or unbalanced author cohort")
    if set(state["users"]) & (known_prior_users() | set(excluded_users)):
        raise ValueError("cohort overlaps historical research")
    return state, events, queries


def source_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def record_row(model, kernel, state, query, profile, label):
    model.scope = {"experiment_stage": "ranking", "condition": label, "query": query["query_id"]}
    before = model.requests
    ranking = kernel.rank(state, query)
    return {"query_id": query["query_id"], "ranking": ranking,
            "calls": model.requests - before,
            **score_author_ranking(ranking, query, profile)}


def run_fresh_agentcf(task, action, model, output, cohort):
    """Train a new, disjoint cohort and rerank in paired counterfactual conditions."""
    if task not in ("agentcf_memory", "agentcf_popularity"):
        raise ValueError("not an AgentCF research task")
    if action not in task_spec(task)["actions"]:
        raise ValueError("unregistered experiment action")
    if action not in ("item_memory_ablation", "popularity_prompt"):
        raise ValueError("unsupported fresh AgentCF action")
    initial, events, queries = cohort
    output = Path(output)
    prompts_path = LOCAL_DATA / "author_prompts.json"
    prompts = json.loads(prompts_path.read_text(encoding="utf8"))
    profile = author_profile(LOCAL_DATA)
    model.scope = {"experiment_stage": "training", "task": task}
    kernel = AgentCF(model, prompts)
    state = deepcopy(initial)
    training = []
    for event in events:
        before = model.requests
        state = kernel.event(state, event, "original_update")
        training.append({"event": event["event_id"], "calls": model.requests - before})
    atomic_json(output / "agentcf_training.json", {
        "training": training, "cohort_users": sorted(initial["users"]),
        "completed_events": len(state["completed"]),
        "prompts_sha256": source_digest(prompts_path),
        "data_protocol": "author_training_only_and_one_known_validation_target",
    })
    modified = deepcopy(state)
    candidates = {item for q in queries for item in q["candidates"]}
    changed = sum(modified["items"][i]["memory"] != initial["items"][i]["memory"]
                  for i in candidates)
    for item in candidates:
        modified["items"][item]["memory"] = initial["items"][item]["memory"]
    if action == "item_memory_ablation" and changed == 0:
        return {"action":action, "status":"ABSTAIN_NO_UPDATED_CANDIDATE_MEMORY",
                "training_events":len(events), "users":sorted(initial["users"]),
                "item_memories_changed":0, "panels":{},
                "independent_confirmation":False,
                "reason":"No updated Item Memory appears in the new candidate panels"}
    intervention_prompts = (author_intervention(prompts, "neutralize_rank_popularity")
                            if action == "popularity_prompt" else prompts)
    subsets = {"A": sorted(queries, key=lambda q: q["query_id"])[:len(queries)//2],
               "B": sorted(queries, key=lambda q: q["query_id"])[len(queries)//2:]}
    setups = {
        "original": (state, prompts),
        "repeat": (state, prompts),
        action: (modified if action == "item_memory_ablation" else state,
                 intervention_prompts),
    }
    panels = {}
    for group, queries_in_group in subsets.items():
        panels[group] = {}
        for label, (current_state, current_prompts) in setups.items():
            ranks = AgentCF(model, current_prompts)
            panels[group][label] = [record_row(model, ranks, current_state, q, profile, label)
                                    for q in queries_in_group]
    atomic_json(output / "agentcf_ranking_panels.json", panels)
    return {"action": action, "status": "FRESH_AGENTCF_DEVELOPMENT",
            "source": "newly trained disjoint users from the same author CDs dataset",
            "training_events": len(events), "users": sorted(initial["users"]),
            "item_memories_changed": changed, "panels": panels,
            "independent_confirmation": False,
            "limitations": [
                "same author CDs data and previously studied mechanism families",
                "only one known positive validation item per query; other relevance unknown",
                "new users do not establish new mechanisms or independent confirmation",
            ]}


def archive_replay(task, action):
    """Previously saved results, never a NEW scientific experiment."""
    if action not in task_spec(task)["actions"]:
        raise ValueError("unregistered experiment action")
    archive = json.loads((reference.RUN / "executions.json").read_text(encoding="utf8"))
    if task.startswith("agentcf_"):
        record = archive["agentcf_shared_rank_panels"]
        if action not in record["panels"]["a"]:
            raise ValueError("AgentCF action missing in frozen panels")
        selected = {"panels": record["panels"], "action": action,
                    "status": "ARCHIVED_DEVELOPMENT_REPLAY"}
    else:
        if action not in archive:
            raise ValueError("archived action unavailable")
        selected = archive[action]
    return selected | {
        "execution_kind": "ARCHIVED_DEVELOPMENT_REPLAY",
        "new_experimental_observations": 0,
        "independent_confirmation": False,
    }


def execute(task, action, mode, model, output, cohort=None):
    if mode == "archive":
        return archive_replay(task, action)
    if mode != "fresh":
        raise ValueError("unsupported execution mode")
    if task.startswith("coat_"):
        return reference.coat_observation(action) | {
            "execution_kind": "NEW_CPU_CALCULATION_OLD_DEVELOPMENT_PANEL",
            "independent_confirmation": False,
        }
    if cohort is None:
        raise ValueError("fresh AgentCF requires preregistered cohort")
    return run_fresh_agentcf(task, action, model, output, cohort)



def panel_measurements(task, action, result):
    """Per-panel numbers are independently derived from the CHOSEN action."""
    if task.startswith("coat_"):
        records = result["records"]
        baseline = records["unweighted"]["panels"]
        if action == "gamma_stress":
            refs = ["gamma1_cap20", "gamma1_cap100"]
            rivals = ["gamma0.75_cap20", "gamma0.75_cap100",
                      "gamma1.25_cap20", "gamma1.25_cap100"]
            # Positive margin: a gamma=1 weighting beats every perturbed gamma.
            labels = refs + rivals
            if any(k not in records or len(records[k]["panels"]) != len(baseline)
                   for k in labels):
                raise ValueError("gamma-stress response missing controlled configurations")
            results = []
            for i, base in enumerate(baseline):
                winner = min(refs, key=lambda k: records[k]["panels"][i]["Brier"])
                selected = records[winner]["panels"][i]
                margin = min(records[k]["panels"][i]["Brier"] for k in rivals) - selected["Brier"]
                results.append({"gamma_reference_margin": margin,
                    "gamma1_choice": winner,
                    "quality_gate": selected["NDCG5"] >= .99*base["NDCG5"]})
            return results
        if action == "propensity_shuffle":
            matched = records["matched_propensity"]["panels"]
            placebos = [value["panels"] for k,value in records.items()
                        if k.startswith("user_permuted_propensity_")]
            if len(placebos) < 2 or any(len(p) != len(baseline)
                                        for p in [matched,*placebos]):
                raise ValueError("propensity negative controls missing or unpaired")
            return [{"propensity_placebo_margin":
                      min(p[i]["Brier"] for p in placebos) - matched[i]["Brier"],
                     "placebo_count": len(placebos),
                     "quality_gate": matched[i]["NDCG5"] >= .99*base["NDCG5"]}
                    for i,base in enumerate(baseline)]
        if action != "split_utility":
            raise ValueError("unknown Coat scientific metric contract")
        weighted = records["gamma1_cap100"]["panels"]
        if len(weighted) != len(baseline):
            raise ValueError("Coat panels not aligned")
        return [{"brier_gain": base["Brier"] - row["Brier"],
                 "ndcg_delta": row["NDCG5"] - base["NDCG5"],
                 "tail_delta": (None if row["tail_relevant_recall"] is None or
                               base["tail_relevant_recall"] is None
                               else row["tail_relevant_recall"]-
                                    base["tail_relevant_recall"]),
                 "quality_gate": row["NDCG5"] >= .99*base["NDCG5"]}
                for base,row in zip(baseline,weighted)]

    result_panels = result["panels"]
    if set(result_panels) != {"A", "B"} and set(result_panels) != {"a", "b"}:
        raise ValueError("requires exactly two disjoint AgentCF development panels")
    per = []
    for conditions in result_panels.values():
        baseline, repeat, changed = (conditions[k] for k in ("original","repeat",action))
        if not (len(baseline) == len(repeat) == len(changed) and len(baseline) > 0):
            raise ValueError("unpaired or incomplete rankings")
        ids = [r["query_id"] for r in baseline]
        if ids != [r["query_id"] for r in repeat] or ids != [r["query_id"] for r in changed]:
            raise ValueError("query IDs do not align")
        noisy = mean(a["ranking"] != b["ranking"] for a,b in zip(baseline,repeat))
        shifted = mean(a["ranking"] != b["ranking"] for a,b in zip(baseline,changed))
        pre_ndcg = mean(r["ndcg_at_5"] for r in baseline)
        post_ndcg = mean(r["ndcg_at_5"] for r in changed)
        per.append({"rank_change_excess": shifted-noisy,
                    "ndcg_delta": post_ndcg-pre_ndcg,
                    "head_excess_delta": mean(r["excess_popularity_exposure"] for r in changed) -
                                         mean(r["excess_popularity_exposure"] for r in baseline),
                    "quality_gate": pre_ndcg > 0 and post_ndcg >= .99*pre_ndcg,
                    "baseline_repeat_change_rate": noisy})
    return per


def independent_verdict(task, action, plan, result):
    if action != plan["action"] or plan["metric"] not in action_metrics(task, action):
        raise ValueError("frozen plan and actual result disagree")
    if result.get("status") == "ABSTAIN_NO_UPDATED_CANDIDATE_MEMORY":
        return {"task":task, "action":action, "forecast_metric":plan["metric"],
                "panel_measurements":[], "forecast_passes_both_panels":False,
                "mechanism_development_check":None, "evidence_grade":"inconclusive",
                "independently_confirmed_discoveries":0, "reason":result["reason"]}
    panels = panel_measurements(task, action, result)
    metric, threshold, direction = (plan["metric"], plan["minimum_abs_effect"], plan["direction"])
    if len(panels) != 2:
        raise ValueError("prospective replication requires two development panels")
    def conforms(value):
        if value is None:
            return False
        if direction == "increase":
            return value >= threshold and value > 0
        if direction == "decrease":
            return value <= -threshold and value < 0
        return abs(value) <= max(0.02, threshold)
    predicts = all(conforms(p[metric]) and p["quality_gate"] for p in panels)
    # Development forecast and action-specific falsification are distinct.
    # Both MUST remain independent of any language-model reflection.
    mechanism = (reference.assess(task, action, result)
                 if task.startswith("coat_") and result.get("status") == "EXECUTED_DEVELOPMENT"
                 else None)
    # Existing reference.assess understands Coat; fresh AgentCF evidence
    # has the same panel schema but is not upgraded to a confirmed finding.
    return {"task": task, "action": action, "forecast_metric": metric,
            "predeclared_direction": direction, "predeclared_minimum_effect": threshold,
            "panel_measurements": panels, "forecast_passes_both_panels": predicts,
            "mechanism_development_check": mechanism,
            "evidence_grade": "previous_development_only" if result.get("execution_kind") == "ARCHIVED_DEVELOPMENT_REPLAY"
                              else "new_unconfirmed_development",
            "independently_confirmed_discoveries": 0,
            "reason": "no untouched independent confirmation panel in this episode"}
