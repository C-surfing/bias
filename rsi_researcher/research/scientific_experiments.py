"""Registered, bounded scientific interventions: Coat CPU or fresh Qwen-AgentCF.

Never reads author test labels, scores a Coat test partition, or grants an LLM
arbitrary tools. Fresh training/rank inference requires a supplied Qwen model.
"""
from __future__ import annotations

from copy import deepcopy
from csv import DictReader
from random import Random
from hashlib import sha256
from pathlib import Path
from statistics import mean
import json

from rsi_researcher.datasets.author import author_data, author_profile
from rsi_researcher.recommenders.agentcf import AgentCF
from rsi_researcher.research.author_experiment import author_intervention
from rsi_researcher.research import crossbias_discovery as reference
from rsi_researcher.research import bias_registry
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
# Minimum development signals fixed by the evaluator, never chosen by Qwen.
# These are practical-effect thresholds, NOT power analyses or p-values.
# Legacy Coat margins use the frozen 1e-4 sensitivity convention; for the
# small AgentCF panels no tiny positive change counts as a discovery signal.
DEVELOPMENT_EFFECT_FLOOR = {
    "gamma_reference_margin": 1e-4,
    "propensity_placebo_margin": 1e-4,
    "brier_gain": 1e-4,
    "ndcg_delta": 0.02,
    "tail_delta": 0.02,
    "rank_change_excess": 0.2,
    "linked_rank_change_excess": 0.2,
    "feedback_semantic_excess_over_training_repeat": 0.2,
    "head_excess_delta": 0.02,
}

ACTION_METRICS = {
    "gamma_stress": ("gamma_reference_margin",),
    "propensity_shuffle": ("propensity_placebo_margin",),
    "split_utility": ("brier_gain", "ndcg_delta", "tail_delta"),
    "item_memory_ablation": ("rank_change_excess", "ndcg_delta", "head_excess_delta"),
    "cross_user_memory_ablation": ("linked_rank_change_excess", "ndcg_delta", "head_excess_delta"),
    "neutral_feedback_memory_propagation": ("feedback_semantic_excess_over_training_repeat",),
    "popularity_prompt": ("head_excess_delta", "ndcg_delta", "rank_change_excess"),
}


# Extend only the new science task, never mutate the code-hashed older
# Cross-Bias task definitions or their frozen development records.
EXTRA_SCIENCE_ACTIONS = {"agentcf_memory": (
    "cross_user_memory_ablation", "neutral_feedback_memory_propagation")}


def action_metrics(task, action):
    if task not in reference.CATEGORIES or action not in task_spec(task)["actions"]:
        raise ValueError("action is not registered for this task")
    return ACTION_METRICS[action]


def metrics_by_action(task):
    return {a: list(action_metrics(task, a)) for a in task_spec(task)["actions"]}


def task_metrics(task):
    return tuple(dict.fromkeys(metric for metrics in metrics_by_action(task).values()
                               for metric in metrics))


def task_spec(task):
    if task not in reference.CATEGORIES:
        raise ValueError("unregistered research task")
    spec = reference.CATEGORIES[task]
    return {**spec, "actions":[*spec["actions"], *EXTRA_SCIENCE_ACTIONS.get(task,())]}


def known_prior_users():
    prior = SOURCE / "runs/qwen-author-dev20-baseline-20261015/input.json"
    fresh = SOURCE / "runs/agentcf-newstate-forecast-20261017/protocol.json"
    old = json.loads(prior.read_text(encoding="utf8"))["initial"]["users"]
    newer = json.loads(fresh.read_text(encoding="utf8"))["cohort_users"]
    # Frozen October 17 paired pilot; the four users are development evidence.
    # This specific archived run predates the scientific comparison protocol.
    paired = Path(__file__).resolve().parents[2] / (
        "runs/live-science-policy-comparison-20261017-01/"
        "strong_rule/episode-001/cohort_frozen.json")
    paired_users = json.loads(paired.read_text(encoding="utf8"))["users"] if paired.is_file() else []
    return set(old) | set(newer) | set(paired_users)


def prepare_cohort(excluded_users, seed, count=6, events_per_user=2):
    if count < 4 or count % 2:
        raise ValueError("fresh cohort must contain at least four users and split equally")
    excluded = known_prior_users() | set(excluded_users)
    # Apply seeded sampling in the NEW scientific adapter only: author.py
    # belongs to a historical code-hashed forecast protocol.
    with (LOCAL_DATA / "CDs-100-user-dense.train.inter").open(encoding="utf8") as stream:
        ordered = list(dict.fromkeys(row["user_id:token"]
                    for row in DictReader(stream, delimiter="\t")))
    eligible = [u for u in ordered if u not in excluded]
    if len(eligible) < count:
        raise ValueError("insufficient fresh AgentCF training users")
    chosen = set(Random(seed).sample(eligible, count))
    selected_exclusions = excluded | (set(eligible) - chosen)
    state, events, queries = author_data(
        LOCAL_DATA, count, events_per_user, seed, "exclude_known",
        excluded_users=selected_exclusions,
    )
    if len(queries) != count or len(events) != count * events_per_user:
        raise ValueError("incomplete or unbalanced author cohort")
    if set(state["users"]) & excluded:
        raise ValueError("cohort overlaps historical research")
    return state, events, queries


def source_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def structural_memory_support(cohort):
    """Cheap PRE-TRAINING support bound: possible cross-user memory reads.

    Positive training events and unlabeled candidate IDs only. No rankings,
    inferred dislike, model output, validation labels, or seed selection.
    Even positive support does NOT guarantee a changed learned Item Memory.
    """
    _,events,queries=cohort
    writers={}
    for e in events:
        writers.setdefault(e["positive"],set()).add(e["user"])
    ordered=sorted(queries,key=lambda q:q["query_id"])
    halfway=len(ordered)//2
    panels={"A":ordered[:halfway],"B":ordered[halfway:]}
    links={}
    readers={}
    for name,qs in panels.items():
        links[name]=sum(bool(writers.get(item,set())-{q["user"]})
                        for q in qs for item in q["candidates"])
        readers[name]=sum(any(writers.get(item,set())-{q["user"]}
                              for item in q["candidates"]) for q in qs)
    return {
        "phase":"PRETRAINING_STRUCTURAL_SUPPORT",
        "potential_cross_user_links_by_panel":links,
        "potential_distinct_readers_by_panel":readers,
        "passes":all(readers[name]>=1 for name in ("A","B")),
        "used_validation_labels":False,
        "meaning":"Potential writer-to-reader overlap only; learned memory "
                  "changes and causal effect remain unidentified.",
    }


def preintervention_observations(cohort):
    """Measured input support, not a learned-effect or held-out-label oracle."""
    initial, events, queries = cohort
    trained_items = {event["positive"] for event in events}
    candidate_items = {item for q in queries for item in q["candidates"]}
    return {
        "source": "real AgentCF training events and unlabeled candidate IDs",
        "training_events": len(events),
        "training_users": len(initial["users"]),
        "distinct_training_positive_items": len(trained_items),
        "distinct_query_candidates": len(candidate_items),
        "trained_items_in_query_candidates": len(trained_items & candidate_items),
        "queries_with_training_positive_candidate": sum(
            bool(trained_items & set(q["candidates"])) for q in queries),
        "query_count": len(queries),
        "interpretation": "Observational support only, before Qwen training; "
                          "no ranking behavior, intervention outcome, or relevance label observed.",
    }


def record_row(model, kernel, state, query, profile, label):
    model.scope = {"experiment_stage": "ranking", "condition": label, "query": query["query_id"]}
    before = model.requests
    ranking = kernel.rank(state, query)
    return {"query_id": query["query_id"], "ranking": ranking,
            "calls": model.requests - before,
            **score_author_ranking(ranking, query, profile)}


def train_and_observe_agentcf(cohort, model, output):
    """One frozen training state and pre-decision ranking observations.

    Queries include a known validation target, but only candidate IDs,
    rankings and aggregate popularity diagnostics reach the research agents.
    No validation quality or intervention outcomes are in their evidence.
    """
    initial, events, queries = cohort
    output = Path(output)
    prompts_path = LOCAL_DATA / "author_prompts.json"
    prompts = json.loads(prompts_path.read_text(encoding="utf8"))
    kernel = AgentCF(model, prompts)
    profile = author_profile(LOCAL_DATA)
    state = deepcopy(initial)
    training, memory_examples, item_writers = [], [], {}
    for event in events:
        model.scope = {"experiment_stage": "shared_training"}
        before = model.requests
        previous_user = state["users"][event["user"]]
        previous_item = state["items"][event["positive"]]["memory"]
        state = kernel.event(state, event, "original_update")
        training.append({"event": event["event_id"], "calls": model.requests-before})
        if state["items"][event["positive"]]["memory"] != previous_item:
            item_writers.setdefault(event["positive"], set()).add(event["user"])
        if len(memory_examples) < 3:
            memory_examples.append({
                "event_id": event["event_id"], "observed_training_user": event["user"],
                "observed_positive_item": event["positive"],
                "user_before": previous_user[:350],
                "user_after": state["users"][event["user"]][:350],
                "item_before": previous_item[:350],
                "item_after": state["items"][event["positive"]]["memory"][:350],
            })

    rankings = {}
    for query in queries:
        model.scope = {"experiment_stage": "predecision_baseline", "query": query["query_id"]}
        rankings[query["query_id"]] = kernel.rank(state, query)

    user_changed = sum(initial["users"][u] != memory for u,memory in state["users"].items())
    item_changed = {i for i in state["items"]
                    if state["items"][i]["memory"] != initial["items"][i]["memory"]}
    candidates = {i for q in queries for i in q["candidates"]}
    # True shared-memory support requires a training writer DIFFERENT from
    # the later ranking user, not just any text mutation.
    cross_user_links = [
        {"query_id":q["query_id"],"reading_user":q["user"],
         "item_id":i,"other_writers":sorted(item_writers[i])}
        for q in queries for i in q["candidates"]
        if i in item_changed and item_writers.get(i,set()) and
           q["user"] not in item_writers[i]
    ]
    ordered_queries = sorted(queries,key=lambda q:q["query_id"])
    half = len(ordered_queries)//2
    cross_user_by_panel = {
        "A":sum(q["query_id"]==x["query_id"] for q in ordered_queries[:half]
                for x in cross_user_links),
        "B":sum(q["query_id"]==x["query_id"] for q in ordered_queries[half:]
                for x in cross_user_links),
    }
    from rsi_researcher.datasets.author import popularity_group
    def is_head(i):
        return popularity_group(i, profile["training_item_counts"],
                                profile["head_min_count"], profile["tail_max_count"]) == "head"
    head_deltas = [
        sum(map(is_head, rankings[q["query_id"]][:5]))/5 -
        sum(map(is_head, q["candidates"]))/len(q["candidates"])
        for q in queries
    ]
    observation = {
        **preintervention_observations(cohort),
        "source": "trained AgentCF state and predecision baseline rankings, without relevance labels",
        "user_memories_changed_after_training": user_changed,
        "item_memories_changed_after_training": len(item_changed),
        "changed_item_memories_in_query_candidates": len(item_changed & candidates),
        "cross_user_candidate_memory_links": len(cross_user_links),
        "cross_user_candidate_links_by_panel": cross_user_by_panel,
        "queries_with_cross_user_memory_links": len({x["query_id"] for x in cross_user_links}),
        "cross_user_memory_examples": cross_user_links[:3],
        "mean_baseline_head_excess_vs_candidate_mix": mean(head_deltas),
        "observed_training_memory_changes": memory_examples,
        "observed_example_rankings": [
            {"query_id":q["query_id"], "top5_ids":rankings[q["query_id"]][:5]}
            for q in queries[:min(2,len(queries))]],
        "interpretation": "Conditional memory/ranking observation, not proof of "
                          "causal popularity bias or debias effectiveness. "
                          "Unknown relevance and quality scores remain withheld.",
    }
    training_info = {
        "training": training, "cohort_users": sorted(initial["users"]),
        "completed_events": len(state["completed"]),
        "prompts_sha256": source_digest(prompts_path),
        "trained_state_sha256": sha256(json.dumps(
            state, sort_keys=True, ensure_ascii=False).encode("utf8")).hexdigest(),
        "baseline_rankings_sha256": sha256(json.dumps(
            rankings, sort_keys=True, ensure_ascii=False).encode("utf8")).hexdigest(),
        "input_sha256": scientific_input_fingerprint(cohort),
        "item_memory_writer_users": {item:sorted(users)
                                     for item,users in item_writers.items()},
        "cross_user_candidate_memory_links": len(cross_user_links),
        "cross_user_candidate_links_by_panel": cross_user_by_panel,
        "data_protocol": "author_training_only_and_one_known_validation_target",
    }
    prepared = {"state":state, "baseline_rankings":rankings,
                "training":training_info, "observation":observation}
    atomic_json(output / "prepared_agentcf.json", prepared)
    return prepared


def load_prepared_agentcf(path, cohort):
    """Check that the frozen state and rankings belong to this input cohort."""
    prepared = json.loads(Path(path).read_text(encoding="utf8"))
    meta = prepared["training"]
    if meta["input_sha256"] != scientific_input_fingerprint(cohort):
        raise ValueError("prepared state was trained on different inputs")
    state, ranks = prepared["state"], prepared["baseline_rankings"]
    if meta["trained_state_sha256"] != sha256(json.dumps(
            state, sort_keys=True, ensure_ascii=False).encode("utf8")).hexdigest():
        raise ValueError("frozen trained state changed")
    if meta["baseline_rankings_sha256"] != sha256(json.dumps(
            ranks, sort_keys=True, ensure_ascii=False).encode("utf8")).hexdigest():
        raise ValueError("frozen baseline rankings changed")
    if set(ranks) != {q["query_id"] for q in cohort[2]}:
        raise ValueError("prepared baseline queries differ")
    return prepared


def neutral_feedback_prompts(prompts):
    """Correct one false-negative *semantics* at both memory-writing stages.

    Keep simulation selection, observed positive feedback, candidate order and
    ranking instructions byte-identical. The older author experiment only
    neutralizes USER reflection; Item Update has its own false-dislike claim.
    """
    changed = author_intervention(prompts,"unseen_neutral_user_reflection")
    replacements = {
        "item_prompt_template": (
            ("However, the user found that he actually doesn't enjoy the CD that he previously chosen. Conversely, he finds the CD that he didn't choose before, titled '$pos_item_title', more appealing. This indicates that he made an unsuitable choice. This may be because the user was misled by the descriptions of these two candidate CDs.",
             "The only observed positive feedback concerns '$pos_item_title'. The other CD ('$neg_item_title') was sampled for comparison; its exposure and listening history are unknown, so no dislike should be inferred."),
            ("while he dislikes the other CD ('$neg_item_title') because it possesses features that he disliked.",
             "while the sampled comparison CD ('$neg_item_title') has no observed dislike or rejection signal."),
            ("Explore the user disliked CD's new features that aligns with the user's dislikes and goes against with his preferences. In this case, the differences between the descriptions of these two items should be amplified.",
             "For the sampled comparison CD, preserve existing factual descriptions and infer no new dislike, since exposure and preference are unknown."),
        ),
        "item_prompt_template_true": (
            ("while he rejected the other CD ('$neg_item_title') because it possesses features that he disliked.",
             "while the sampled comparison CD ('$neg_item_title') has unknown exposure and no observed dislike."),
            ("Explore the rejected CD's new features that aligns with the user's dislikes and goes against with his preferences. In this case, the differences between the descriptions of these two items should be amplified.",
             "Do not infer dislike-related new features for the sampled comparison CD; preserve its factual descriptions."),
        ),
    }
    for key,pairs in replacements.items():
        for old,new in pairs:
            if changed[key].count(old) != 1:
                raise ValueError("unmatched feedback semantic anchor: "+key)
            changed[key] = changed[key].replace(old,new)
    expected={"user_prompt_template","user_prompt_template_true",
              "item_prompt_template","item_prompt_template_true"}
    if {k for k in changed if changed[k]!=prompts[k]}!=expected:
        raise ValueError("feedback semantics changed an unrelated prompt")
    return changed


def run_feedback_semantic_propagation(model, output, cohort, prepared, prompts, profile):
    """Test complete sampled-negative semantics through changed shared Item Memory.

    Both training branches start at exactly the same initial state and use the
    same observed events/candidates, but fresh LLM draws can diverge. An
    unchanged-prompt retraining is a necessary training-variance comparator.
    Ranking readers always keep the ORIGINAL trained User Memory.
    """
    initial, events, queries = cohort
    output = Path(output)
    original = prepared["state"]
    writers = {i:set(users) for i,users in
               prepared["training"]["item_memory_writer_users"].items()}
    neutral_prompts = neutral_feedback_prompts(prompts)
    changed_keys = sorted(k for k in prompts if neutral_prompts[k] != prompts[k])

    def retrain(condition, templates):
        kernel = AgentCF(model, templates)
        state = deepcopy(initial)
        event_calls = []
        for event in events:
            model.scope = {"experiment_stage":"feedback_semantics_training",
                           "condition":condition,"event":event["event_id"]}
            before = model.requests
            prior_user = state["users"][event["user"]]
            prior_item = state["items"][event["positive"]]["memory"]
            state = kernel.event(state, event, "original_update")
            event_calls.append({
                "event":event["event_id"],"writer_user":event["user"],
                "observed_positive_item":event["positive"],
                "sampled_comparator_not_observed_negative":event["negative"],
                "user_memory_changed":prior_user!=state["users"][event["user"]],
                "positive_item_memory_changed":(
                    prior_item!=state["items"][event["positive"]]["memory"]),
                "calls":model.requests-before})
        return state, event_calls

    # One extra original trajectory captures training-time nondeterminism,
    # which a same-state repeat ranking alone could never reveal.
    repeated, control_calls = retrain("unchanged_training_repeat",prompts)
    neutral, neutral_calls = retrain("unknown_exposure_feedback_semantics",neutral_prompts)

    subsets = {"A":sorted(queries,key=lambda q:q["query_id"])[:len(queries)//2],
               "B":sorted(queries,key=lambda q:q["query_id"])[len(queries)//2:]}
    exposures, changes, panels = {}, {}, {}
    for name, panel in subsets.items():
        panels[name] = {}
        for label in ("original","repeat","training_repeat",
                      "neutral_feedback_memory_propagation"):
            rows = []
            kernel = AgentCF(model,prompts)
            for q in panel:
                if label == "original":
                    ranking = prepared["baseline_rankings"][q["query_id"]]
                    row = {"query_id":q["query_id"],"ranking":ranking,"calls":0,
                           "origin":"frozen_predecision_baseline",
                           **score_author_ranking(ranking,q,profile)}
                else:
                    state = original
                    if label in ("training_repeat","neutral_feedback_memory_propagation"):
                        source = (repeated if label=="training_repeat" else neutral)
                        state = deepcopy(original)
                        eligible = [item for item in q["candidates"]
                                    if writers.get(item) and q["user"] not in writers[item]]
                        if label == "neutral_feedback_memory_propagation":
                            exposures[q["query_id"]] = eligible
                            changes[q["query_id"]] = [
                                item for item in eligible
                                if source["items"][item]["memory"] !=
                                   original["items"][item]["memory"]]
                        for item in eligible:
                            # Hold the reader's learned user preferences fixed.
                            # Transfer only a memory written by OTHER users.
                            state["items"][item]["memory"] = source["items"][item]["memory"]
                    row = record_row(model,kernel,state,q,profile,label)
                rows.append(row)
            panels[name][label] = rows

    counts = {name:sum(bool(exposures.get(q["query_id"])) for q in panel)
              for name,panel in subsets.items()}
    changed_counts = {name:sum(bool(changes.get(q["query_id"])) for q in panel)
                      for name,panel in subsets.items()}
    protocol = {
        "action":"neutral_feedback_memory_propagation",
        "training_difference":"user and item memory-writing prompts consistently treat sampled comparator as unobserved, never disliked",
        "original_training_state_sha256":prepared["training"]["trained_state_sha256"],
        "original_prompts_sha256":source_digest(LOCAL_DATA/"author_prompts.json"),
        "intervened_memory_writing_template_keys":changed_keys,
        "unchanged_training_repeat_calls":control_calls,
        "neutral_feedback_training_calls":neutral_calls,
        "training_states_sha256":{
            "unchanged_repeat":sha256(json.dumps(repeated,sort_keys=True,
                       ensure_ascii=False).encode("utf8")).hexdigest(),
            "unknown_exposure":sha256(json.dumps(neutral,sort_keys=True,
                       ensure_ascii=False).encode("utf8")).hexdigest()},
        "eligible_cross_user_candidate_items_by_query":exposures,
        "semantically_changed_cross_user_items_by_query":changes,
        "linked_readers_by_panel":counts,
        "changed_readers_by_panel":changed_counts,
        "explanation":"Selection/update calls are fresh and may differ between "
                      "training branches; a single retraining control is not "
                      "a statistical guarantee. Relevance labels are not "
                      "used to decide which memory to transfer.",
    }
    atomic_json(output/"feedback_semantics_protocol.json",protocol)
    if any(n==0 for n in counts.values()):
        return {"action":protocol["action"],
                "status":"ABSTAIN_NO_SHARED_MEMORY_READER_PANEL",
                "reason":"One panel lacks a different-user memory reader",
                "panels":{},"independent_confirmation":False}
    if not any(changed_counts.values()):
        return {"action":protocol["action"],
                "status":"ABSTAIN_NO_FEEDBACK_SEMANTIC_MEMORY_CHANGE",
                "reason":"Neutral feedback semantics produced no changed cross-user candidate Item Memory",
                "panels":{},"independent_confirmation":False}
    atomic_json(output/"agentcf_ranking_panels.json",panels)
    return {"action":protocol["action"],"status":"FRESH_AGENTCF_DEVELOPMENT",
            "source":"paired same-input feedback-semantics developmental comparison",
            "training_events":len(events),"users":sorted(initial["users"]),
            "panels":panels,"eligible_readers_by_query":exposures,
            "changed_readers_by_query":changes,
            "training_repeat_control_used":True,
            "independent_confirmation":False,
            "limitations":[
                "only sampled comparison negatives, no exposure/dislike truth",
                "same initial events but independent stochastic model training",
                "one unchanged-prompt retraining and one same-state ranking repeat are weak noise controls",
                "the reader User Memory is original in every ranking condition",
                "one known positive target does not fully measure recommendation utility",
            ]}


def run_fresh_agentcf(task, action, model, output, cohort, prepared=None):
    """Apply the chosen intervention on the frozen, pre-decision trained state."""
    if task not in ("agentcf_memory", "agentcf_popularity"):
        raise ValueError("not an AgentCF research task")
    if action not in task_spec(task)["actions"]:
        raise ValueError("unregistered experiment action")
    if action not in ("item_memory_ablation", "cross_user_memory_ablation",
                      "neutral_feedback_memory_propagation","popularity_prompt"):
        raise ValueError("unsupported fresh AgentCF action")
    initial, events, queries = cohort
    output = Path(output)
    if prepared is None:
        prepared = train_and_observe_agentcf(cohort, model, output)
    else:
        prepared = load_prepared_agentcf(prepared, cohort) if isinstance(prepared,(str,Path)) else prepared
    # Audit the same training state across all arms; separate the one-time
    # shared training/probing cost from research-arm calls.
    atomic_json(output / "agentcf_training.json", prepared["training"])
    state = prepared["state"]
    originals = prepared["baseline_rankings"]
    prompts = json.loads((LOCAL_DATA / "author_prompts.json").read_text(encoding="utf8"))
    profile = author_profile(LOCAL_DATA)
    if action == "neutral_feedback_memory_propagation":
        return run_feedback_semantic_propagation(
            model, output, cohort, prepared, prompts, profile)
    modified = deepcopy(state)
    candidates = {item for q in queries for item in q["candidates"]}
    changed = sum(modified["items"][i]["memory"] != initial["items"][i]["memory"]
                  for i in candidates)
    for item in candidates:
        modified["items"][item]["memory"] = initial["items"][item]["memory"]
    if task == "agentcf_memory" and action in (
            "item_memory_ablation","cross_user_memory_ablation") and not (
            prepared["training"]["cross_user_candidate_memory_links"]):
        return {"action":action, "status":"ABSTAIN_NO_CROSS_USER_MEMORY_PATH",
                "training_events":len(events), "users":sorted(initial["users"]),
                "item_memories_changed":changed, "panels":{},
                "independent_confirmation":False,
                "reason":"No observed item memory written by another user appears "
                         "in the evaluation candidate panels"}
    if task == "agentcf_memory" and action in (
            "item_memory_ablation","cross_user_memory_ablation") and any(
            n==0 for n in prepared["training"]["cross_user_candidate_links_by_panel"].values()):
        return {"action":action,"status":"ABSTAIN_UNSUPPORTED_CROSS_USER_REPLICATION",
                "training_events":len(events),"users":sorted(initial["users"]),
                "item_memories_changed":changed,"panels":{},
                "independent_confirmation":False,
                "reason":"At least one preregistered development panel has no "
                         "cross-user written candidate memory to test"}
    if action in ("item_memory_ablation","cross_user_memory_ablation") and changed == 0:
        return {"action":action, "status":"ABSTAIN_NO_UPDATED_CANDIDATE_MEMORY",
                "training_events":len(events), "users":sorted(initial["users"]),
                "item_memories_changed":0, "panels":{},
                "independent_confirmation":False,
                "reason":"No updated Item Memory appears in the new candidate panels"}
    intervention_prompts = (author_intervention(prompts, "neutralize_rank_popularity")
                            if action == "popularity_prompt" else prompts)
    subsets = {"A": sorted(queries, key=lambda q:q["query_id"])[:len(queries)//2],
               "B": sorted(queries, key=lambda q:q["query_id"])[len(queries)//2:]}
    panels = {}
    targeted_removals = {}
    writers = {i:set(users) for i,users in
               prepared["training"].get("item_memory_writer_users",{}).items()}
    for group, group_queries in subsets.items():
        panels[group] = {}
        for label in ("original", "repeat", action):
            if label == "original":
                panels[group][label] = [
                    {"query_id":q["query_id"],"ranking":originals[q["query_id"]],
                     "calls":0,"origin":"predecision_frozen_baseline",
                     **score_author_ranking(originals[q["query_id"]],q,profile)}
                    for q in group_queries]
            else:
                current_prompts = intervention_prompts if label == action else prompts
                kernel = AgentCF(model, current_prompts)
                rows = []
                for q in group_queries:
                    if label == "cross_user_memory_ablation":
                        # Only items written by OTHER users, never by this
                        # reader, are reverted in this reader's query.
                        affected = [i for i in q["candidates"]
                                    if writers.get(i) and q["user"] not in writers[i]
                                    and state["items"][i]["memory"] != initial["items"][i]["memory"]]
                        targeted_removals[q["query_id"]] = affected
                        current_state = deepcopy(state)
                        for i in affected:
                            current_state["items"][i]["memory"] = initial["items"][i]["memory"]
                    else:
                        current_state = modified if label == "item_memory_ablation" else state
                    rows.append(record_row(model,kernel,current_state,q,profile,label))
                panels[group][label] = rows
    atomic_json(output / "agentcf_ranking_panels.json", panels)
    return {"action": action, "status": "FRESH_AGENTCF_DEVELOPMENT",
            "source": "one frozen trained AgentCF state and predecision baseline",
            "training_events": len(events), "users": sorted(initial["users"]),
            "item_memories_changed": changed, "panels": panels,
            "targeted_cross_user_memory_removals": targeted_removals,
            "independent_confirmation": False,
            "limitations": [
                "author CDs data and previously studied mechanism families",
                "only one known positive validation item; other relevance unknown",
                "shared training removes training-state confounding, not ranking sampling noise",
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


def execute(task, action, mode, model, output, cohort=None, prepared=None):
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
    return run_fresh_agentcf(task, action, model, output, cohort, prepared)



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
        linked_metrics = {}
        if action == "neutral_feedback_memory_propagation":
            exposure = result.get("eligible_readers_by_query",{})
            training_repeat = conditions["training_repeat"]
            if ids != [r["query_id"] for r in training_repeat]:
                raise ValueError("training repeat is not aligned with paired queries")
            linked = [i for i,qid in enumerate(ids) if exposure.get(qid)]
            unlinked = [i for i,qid in enumerate(ids) if not exposure.get(qid)]
            if not linked:
                raise ValueError("no cross-user readers in feedback panel")
            def change_rate(indices,other):
                return mean(baseline[i]["ranking"] != other[i]["ranking"] for i in indices)
            semantic = change_rate(linked,changed)
            sampling = change_rate(linked,repeat)
            training_noise = change_rate(linked,training_repeat)
            linked_metrics = {
                "linked_query_count":len(linked),
                "unlinked_query_count":len(unlinked),
                "semantic_rank_change_rate":semantic,
                "unchanged_rank_repeat_change_rate":sampling,
                "unchanged_training_repeat_change_rate":training_noise,
                "feedback_semantic_excess_over_training_repeat":
                    semantic-max(sampling,training_noise),
                "linked_known_target_ndcg_delta":
                    mean(changed[i]["ndcg_at_5"]-baseline[i]["ndcg_at_5"]
                         for i in linked),
                "unlinked_rank_change_rate":(
                    change_rate(unlinked,changed) if unlinked else None),
            }
        if action == "cross_user_memory_ablation":
            exposed = result.get("targeted_cross_user_memory_removals",{})
            if set(exposed) != {row["query_id"] for panel in result_panels.values()
                               for row in panel["original"]}:
                raise ValueError("targeted ablation lacks complete query exposure mapping")
            linked = [i for i,qid in enumerate(ids) if exposed[qid]]
            unlinked = [i for i,qid in enumerate(ids) if not exposed[qid]]
            if not linked:
                raise ValueError("targeted ablation has no cross-user memory readers in panel")
            def change_excess(indices):
                return (mean(baseline[i]["ranking"] != changed[i]["ranking"]
                             for i in indices)
                        -mean(baseline[i]["ranking"] != repeat[i]["ranking"]
                              for i in indices))
            linked_metrics = {
                "linked_query_count":len(linked),
                "unlinked_query_count":len(unlinked),
                "linked_rank_change_excess":change_excess(linked),
                "unlinked_rank_change_excess":(
                    change_excess(unlinked) if unlinked else None),
            }
        per.append({"rank_change_excess": shifted-noisy,
                    **linked_metrics,
                    "ndcg_delta": post_ndcg-pre_ndcg,
                    "head_excess_delta": mean(r["excess_popularity_exposure"] for r in changed) -
                                         mean(r["excess_popularity_exposure"] for r in baseline),
                    "quality_gate": post_ndcg >= .99*pre_ndcg,
                    "baseline_repeat_change_rate": noisy})
    return per


# The five BIAS targets are not equivalent to the four experiment families.
# Only a demonstrated Debias intervention may feed an RSI Debias acceptance.
EXPERIMENT_BIAS_TARGET = {
    "coat_propensity": "selection",
    "coat_utility": "selection",
    "agentcf_popularity": "popularity",
    "agentcf_memory": None,  # Mechanism-only, NOT a sixth bias category.
}


def classify_debias_evidence(audit):
    """One owner for diagnostic vs proxy vs actual Debias-candidate semantics.

    Called by the independent Python audit, never by Qwen/Director. A proxy
    or observed rank change is not a qualified correction of a causal bias.
    """
    task, action = audit["task"], audit["action"]
    bias = EXPERIMENT_BIAS_TARGET[task]
    registered = audit.get("registered_action_falsifier_passes")
    prediction = audit.get("mechanism_forecast_passes_both_panels") is True
    quality = audit.get("known_target_quality_guard_passes_both_panels") is True
    if audit.get("evidence_grade") == "previous_development_only":
        stage = "ARCHIVED_DEVELOPMENT_REPLAY"
    elif audit.get("mechanism_forecast_state") == "NOT_EXECUTED":
        stage = "NO_EXECUTED_INTERVENTION"
    elif bias is None:
        stage = "AUXILIARY_MEMORY_MECHANISM_ONLY"
    elif bias == "popularity":
        stage = ("POPULARITY_EXPOSURE_PROXY_ONLY"
                 if audit.get("debias_utility_state","").startswith("PROXY_GAIN")
                 else "POPULARITY_MECHANISM_NOT_DEBIAS")
    elif action in ("gamma_stress", "propensity_shuffle"):
        stage = "SELECTION_FALSIFICATION_CONTROL_ONLY"
    elif (task == "coat_utility" and action == "split_utility" and
          audit.get("forecast_metric") == "brier_gain" and
          audit.get("predeclared_direction") == "increase" and
          prediction and quality and registered is True):
        stage = "SELECTION_DEVELOPMENT_DEBIAS_CANDIDATE"
    else:
        stage = "SELECTION_DEBIAS_NOT_ESTABLISHED"
    eligible = stage == "SELECTION_DEVELOPMENT_DEBIAS_CANDIDATE"
    return {
        "bias_task_id": bias,
        "debias_evidence_stage": stage,
        "debias_candidate_eligible": eligible,
        "registered_falsifier_status": (
            "PASSED_DEVELOPMENT" if registered is True else
            "FAILED_DEVELOPMENT" if registered is False else
            "NOT_IDENTIFIED"),
        "scientific_claim": (
            "DEVELOPMENT_ONLY_PENDING_INDEPENDENT_CONFIRMATION" if eligible
            else "NOT_A_VALIDATED_DEBIAS_RESULT"),
        "independently_confirmed_discoveries": 0,
    }


def five_bias_coverage():
    """Read the existing frozen five-type registry; no parallel task registry."""
    implementations = {t.id:[] for t in bias_registry.TASKS}
    for experiment, bias in EXPERIMENT_BIAS_TARGET.items():
        if bias is not None:
            implementations[bias].append(experiment)
    return [{
        "bias_id":t.id,
        "definition":t.definition,
        "dataset":t.primary_dataset,
        "planned_methods":list(t.allowable_interventions),
        "existing_experimental_tasks":implementations[t.id],
        "maturity":t.current_status,
        "method_compared_on_real_development_data":t.id=="selection",
        "independently_validated_debias":False,
        "explicit_limitation":t.caveat,
    } for t in bias_registry.TASKS]


def independent_verdict(task, action, plan, result):
    if action != plan["action"] or plan["metric"] not in action_metrics(task, action):
        raise ValueError("frozen plan and actual result disagree")
    if result.get("status","").startswith("ABSTAIN_"):
        audit = {"task":task, "action":action, "forecast_metric":plan["metric"],
                "panel_measurements":[], "forecast_passes_both_panels":False,
                "mechanism_forecast_passes_both_panels":False,
                "mechanism_forecast_state":"NOT_EXECUTED",
                "known_target_quality_guard_passes_both_panels":None,
                "debias_utility_state":"NOT_EVALUATED",
                "falsification_state":"NOT_EXECUTED",
                "mechanism_development_check":None, "evidence_grade":"inconclusive",
                "independently_confirmed_discoveries":0, "reason":result["reason"]}
        audit.update(classify_debias_evidence(audit))
        return audit
    panels = panel_measurements(task, action, result)
    metric, direction = plan["metric"], plan["direction"]
    threshold = max(plan["minimum_abs_effect"], DEVELOPMENT_EFFECT_FLOOR[metric])
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
    # Mechanism / forecast evidence and application utility are separate.
    # A real effect can make known-target quality worse; do not erase it.
    predictions = [conforms(p[metric]) for p in panels]
    predicts = direction != "approximately_zero" and all(predictions)
    quality = all(p["quality_gate"] for p in panels)
    # The two tiny panels cannot establish equivalence or causal attribution.
    if direction == "approximately_zero":
        forecast_state = "EQUIVALENCE_UNIDENTIFIED"
    elif predicts:
        forecast_state = "DIRECTION_REPLICATED_DEVELOPMENT"
    elif any(predictions):
        forecast_state = "PANEL_HETEROGENEITY_OR_NOISE"
    else:
        forecast_state = "PREDICTION_NOT_SUPPORTED"
    # Head-exposure reduction is a proxy only, not independently confirmed
    # improvement to the user's utility or causal population popularity bias.
    proxy_improved = (
        task == "agentcf_popularity" and action == "popularity_prompt" and
        metric == "head_excess_delta" and direction == "decrease" and predicts)
    debias_state = ("PROXY_GAIN_WITH_QUALITY_GUARD_DEVELOPMENT_ONLY"
                    if proxy_improved and quality else
                    "PROXY_GAIN_WITH_UTILITY_COST" if proxy_improved else
                    "NO_REGISTERED_DEBIAS_GAIN")
    # Development forecast and action-specific falsification are distinct.
    # Both MUST remain independent of any language-model reflection.
    mechanism = (reference.assess(task, action, result)
                 if task.startswith("coat_") and result.get("status") == "EXECUTED_DEVELOPMENT"
                 else None)
    # A directional forecast is NOT a passed action-specific falsifier.
    # Coat's frozen assessor may reject gamma stability even if both scalar
    # margins exceed the prospective magnitude threshold.
    registered_pass=(mechanism.get("replicated_development_signal")
                     if isinstance(mechanism,dict) else None)
    falsification_state=(
        "REGISTERED_ACTION_FALSIFIER_FAILED" if registered_pass is False else
        "ACTION_FALSIFIER_NOT_IDENTIFIED" if registered_pass is None else
        "DEVELOPMENT_PATTERN_ONLY_ALTERNATIVE_NOT_EXCLUDED" if predicts else
        "COMPETING_EXPLANATION_UNRESOLVED")
    # Existing reference.assess understands Coat; fresh AgentCF evidence
    # has the same panel schema but is not upgraded to a confirmed finding.
    audit = {"task": task, "action": action, "forecast_metric": metric,
            "predeclared_direction": direction,
            "model_predicted_minimum_effect": plan["minimum_abs_effect"],
            "evaluation_minimum_effect": threshold,
            "panel_measurements": panels,
            "forecast_passes_both_panels": bool(
                predicts and quality and registered_pass is not False),
            "registered_action_falsifier_passes": registered_pass,
            "mechanism_forecast_passes_both_panels": bool(predicts),
            "mechanism_forecast_state": forecast_state,
            "panel_forecast_matches": predictions,
            "known_target_quality_guard_passes_both_panels": quality,
            "debias_utility_state": debias_state,
            "falsification_state": falsification_state,
            "mechanism_development_check": mechanism,
            "evidence_grade": "previous_development_only" if result.get("execution_kind") == "ARCHIVED_DEVELOPMENT_REPLAY"
                              else "new_unconfirmed_development",
            "independently_confirmed_discoveries": 0,
            "reason": "no untouched independent confirmation panel in this episode"}
    audit.update(classify_debias_evidence(audit))
    return audit
