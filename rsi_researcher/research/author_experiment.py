"""Author-data intervention operators and a training-free ranking re-run.

H1 (``unseen_neutral_user_reflection``) changes *only* the user-reflection
prompt so a sampled, unobserved comparison item is treated as having no known
exposure — never as disliked.  Item reflection, popularity fallback and the
ranking rule stay byte-identical to the baseline.

H2 (``neutralize_rank_popularity``) changes *only* the ranking prompt: it
drops reliance on popularity / mainstream reputation and ranks on the user's
stated preferences alone.  It re-ranks the already-trained state, so it needs
no training re-run.

Both are prompt-only edits; every replacement is verified to have actually
matched so a future prompt edit cannot silently turn an intervention into a
no-op.
"""
from copy import deepcopy
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from rsi_researcher.recommenders.agentcf import AgentCF

AUTHOR_OPERATORS = {
    "retain": "Baseline prompts unchanged (no-op control).",
    "unseen_neutral_user_reflection": "User reflection treats a sampled comparison item as unknown exposure, not disliked (targets exposure bias).",
    "neutralize_select_popularity": "Selection prompt drops the popularity/mainstream prior (targets conformity bias).",
    "neutralize_rank_popularity": "Ranking adds an instruction to ignore popularity/reputation (targets a popularity proxy, not an original explicit ranking fallback).",
    "neutral_candidate_order": "Ranking is told the candidate list order is arbitrary (targets position bias).",
    "balanced_observed_replay": "Reweight real training events by inverse item frequency with a fixed event budget and user coverage; tests selection-related learning imbalance, not missing exposure correction or PAAC reproduction.",
}

RANK_ONLY_OPERATORS = {"neutralize_rank_popularity", "neutral_candidate_order"}


def training_intervention(events, operator, seed):
    """Change the learning schedule, never feedback labels or validation panels.

    This is weighted replay, not new evidence: repeated events retain their
    source ID. One event per user keeps user coverage; remaining slots sample
    observed events with inverse within-cohort item frequency. Sort by source
    order so a user's later observed event never precedes an earlier one.
    """
    if operator not in AUTHOR_OPERATORS:
        raise ValueError("unknown author operator: " + operator)
    if operator != "balanced_observed_replay":
        return deepcopy(events), {"policy": "original", "events": len(events)}
    if not events:
        raise ValueError("balanced replay requires observed training events")
    counts = Counter(event["positive"] for event in events)
    by_user = defaultdict(list)
    for index, event in enumerate(events):
        by_user[event["user"]].append(index)
    rng = random.Random(seed)
    chosen = [rng.choice(indices) for indices in by_user.values()]
    chosen += rng.choices(range(len(events)),
                          weights=[1 / counts[e["positive"]] for e in events],
                          k=len(events) - len(chosen))
    replay = []
    for index in sorted(chosen):
        event = deepcopy(events[index])
        event["source_event_id"] = event["event_id"]
        event["event_id"] = f"replay-{len(replay):04d}"
        replay.append(event)
    return replay, {
        "policy": operator, "seed": seed, "events": len(replay),
        "source_events": len(events), "unique_source_events": len(set(chosen)),
        "users": len(by_user), "weight": "inverse within-cohort positive-item frequency",
        "source_item_counts": dict(counts),
        "replay_item_counts": dict(Counter(e["positive"] for e in replay)),
        "interpretation": "Matched event budget; repeated observations are reweighting, not independent evidence. Exposure propensities are unknown.",
    }

# Distinctive original phrases and their neutral replacements.  Each tuple is
# (old_phrase, new_phrase); the intervention asserts every old_phrase was found.
_TRUE_ASSERTION = (
    "And you indeed dislike the CD that you did not choose before ('$neg_item_title').",
    "The other CD ('$neg_item_title') was only sampled as a comparison; its exposure and listening "
    "history are unknown. There is no evidence that you dislike it, so record no dislike about it.",
)
_TRUE_STEP2 = (
    "determine your dislikes based on the features of CDs you really don't enjoy ('$neg_item_title').",
    "Do not record any preference or dislike about the comparison CD ('$neg_item_title'), because "
    "its exposure and listening history are unknown.",
)
_WRONG_ASSERTION = (
    "you discovered that you don't like the CD that you initially chose ('$neg_item_title').",
    "the observed interaction concerns '$pos_item_title'. The comparison CD ('$neg_item_title') was "
    "only sampled; its exposure and listening history are unknown, so record no dislike about it.",
)
_WRONG_STEP2 = (
    "determine your dislikes based on the features of CDs you truly don't enjoy ('$neg_item_title').",
    "Do not record any preference or dislike about the comparison CD ('$neg_item_title'), because "
    "its exposure and listening history are unknown.",
)

_SELECT_POPULARITY = (
    "use common knowledge to inform your decision, such as the popularity of the CDs.",
    "state that you have no specific preference for either CD and choose on neutral grounds, without "
    "relying on popularity or mainstream reputation.",
)

_NEUTRAL_RANK_NOTE = (" Ignore any popularity, commercial success, or mainstream reputation implied "
                      "by a CD's description. Rank only on the user's stated preferences and dislikes.")
_NEUTRAL_ORDER_NOTE = (" The order in which the candidates are listed is arbitrary and carries no "
                       "information; do not let the list order influence your ranking.")


def _replace_verified(text, old, new):
    if old not in text:
        raise ValueError("intervention anchor not found; prompts may have changed: " + old[:60])
    return text.replace(old, new)


def author_intervention(prompts, operator):
    """Return a modified prompt dict for a single author operator."""
    if operator not in AUTHOR_OPERATORS:
        raise ValueError("unknown author operator: " + operator)
    changed = dict(prompts)
    if operator == "unseen_neutral_user_reflection":
        for template in ("user_prompt_template", "user_prompt_template_true"):
            text = changed[template]
            text = _replace_verified(text, *_TRUE_ASSERTION) if template.endswith("_true") else \
                   _replace_verified(text, *_WRONG_ASSERTION)
            step2 = _TRUE_STEP2 if template.endswith("_true") else _WRONG_STEP2
            text = _replace_verified(text, *step2)
            changed[template] = text
    elif operator == "neutralize_select_popularity":
        changed["system_prompt_template"] = _replace_verified(
            prompts["system_prompt_template"], *_SELECT_POPULARITY)
    elif operator == "neutralize_rank_popularity":
        changed["system_prompt_template_evaluation_basic"] = \
            prompts["system_prompt_template_evaluation_basic"] + _NEUTRAL_RANK_NOTE
    elif operator == "neutral_candidate_order":
        changed["system_prompt_template_evaluation_basic"] = \
            prompts["system_prompt_template_evaluation_basic"] + _NEUTRAL_ORDER_NOTE
    return changed  # balanced_observed_replay changes the schedule, not prompts


def rank_only(directory, model, state, queries, prompts, profile, operator="neutralize_rank_popularity", variant="primary"):
    """Re-rank an already-trained state with an intervened ranking prompt.

    Zero training calls: it reads a frozen ``state`` and issues one ranking
    request per query.  Used by H2 to isolate ranking-time popularity priors.
    """
    from rsi_researcher.runtime.client import atomic_json, InvalidOutput
    from rsi_researcher.evaluation.author import score_author_ranking

    directory = Path(directory)
    intervened = author_intervention(prompts, operator)
    atomic_json(directory / "ranking_operator.json", {"operator": operator,
                "ranking_prompt": intervened["system_prompt_template_evaluation_basic"]})
    kernel = AgentCF(model, intervened)
    model.scope = {"arm": "rank_only", "operator": operator, "diagnostic_variant": variant}
    rows = []
    for query in queries:
        try:
            ranking = kernel.rank(state, query)
            rows.append({"query": query, "ranking": ranking, "valid": True,
                         **score_author_ranking(ranking, query, profile)})
        except InvalidOutput as exc:
            rows.append({"query": query, "ranking": [], "valid": False,
                         "error": type(exc).__name__, **score_author_ranking([], query, profile)})
    atomic_json(directory / "ranking.json", rows)
    return rows


def controlled_order_rankings(directory, model, state, queries, prompts, profile, operator, primary_rows):
    """Six matched ranking panels, including same-order repeats for both prompts.

    The primary candidate panel is reused, so total calls are six per query,
    not seven. Training state is never modified or regenerated.
    """
    from rsi_researcher.evaluation.author_patterns import order_control_effect, paired_quality
    from rsi_researcher.evaluation.author import summarize_author_rankings
    from rsi_researcher.runtime.client import atomic_json
    if operator != "neutral_candidate_order":
        raise ValueError("order controls require the position intervention")
    panels = {}
    for label, action in (("baseline", "retain"), ("candidate", operator)):
        panels[label] = {}
        for variant in ("original", "repeat", "reversed"):
            if label == "candidate" and variant == "original":
                panels[label][variant] = primary_rows
                continue
            probe = deepcopy(queries)
            if variant == "reversed":
                for query in probe:
                    query["candidates"].reverse()
            destination = Path(directory)/"order-controls"/label/variant
            destination.mkdir(parents=True, exist_ok=False)
            panels[label][variant] = rank_only(destination, model, state, probe, prompts, profile,
                                               action, variant=f"{label}/{variant}")
    result = {label: {"order_effect": order_control_effect(p["original"],p["repeat"],p["reversed"]),
                      "quality": summarize_author_rankings(p["original"])}
              for label, p in panels.items()}
    result["quality_comparison"] = paired_quality(panels["baseline"]["original"],panels["candidate"]["original"])
    result["protocol"] = "same_state_order_and_noise_v1"
    result["scope"] = "Six matched panels on one trained state and model config; no training calls; primary candidate panel reused."
    atomic_json(Path(directory)/"order_controls.json",result)
    return result


def load_run_artifacts(run_dir):
    """Read the committed artifacts a rank-only re-run needs."""
    run_dir = Path(run_dir)
    state = json.loads((run_dir / "original_update" / "state.json").read_text(encoding="utf-8"))
    inputs = json.loads((run_dir / "input.json").read_text(encoding="utf-8"))
    profile = json.loads((run_dir / "data_profile.json").read_text(encoding="utf-8"))
    return state, inputs["queries"], inputs["prompts"], profile


def rank_only_cli(args):
    """H2 runner: re-rank a frozen state with an intervened ranking prompt."""
    from rsi_researcher.runtime.models import create_model, model_description
    from rsi_researcher.runtime.limits import RunLimits
    from rsi_researcher.runtime.client import atomic_json, usage_summary
    from rsi_researcher.evaluation.author import summarize_author_rankings

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    state, queries, prompts, profile = load_run_artifacts(args.source)
    if args.operator not in RANK_ONLY_OPERATORS:
        raise ValueError("rank-only requires an operator that changes only ranking")
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "rank_only_config.json", {"source": str(Path(args.source).resolve()),
                "operator": args.operator, "model_config": model_description(config)})
    controls_requested = getattr(args, "order_controls", False)
    default_requests = len(queries)*(6 if controls_requested else 1)+max(6,len(queries))
    limits = RunLimits(directory, args.max_requests or default_requests, args.max_tokens, args.max_seconds)
    model = create_model(directory, limits.max_requests, limits, config)
    rows = rank_only(directory, model, state, queries, prompts, profile, args.operator)
    if controls_requested:
        controlled_order_rankings(directory,model,state,queries,prompts,profile,args.operator,rows)
    summary = summarize_author_rankings(rows)
    summary.update(operator=args.operator, source=str(Path(args.source).resolve()),
                   status="completed" if not summary["invalid_rankings"] else "failed",
                   purpose="ranking-time prior isolation; zero training calls")
    atomic_json(directory / "summary.json", summary)
    if (directory / "calls.jsonl").exists():
        atomic_json(directory / "usage.json", usage_summary(directory))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Author-data intervention operators.")
    sub = parser.add_subparsers(dest="command", required=True)
    rank = sub.add_parser("rank-only", help="re-rank a frozen state with an intervened ranking prompt (H2)")
    rank.add_argument("--source", required=True, help="a committed baseline run directory")
    rank.add_argument("--output", required=True)
    rank.add_argument("--config", required=True, help="config with the vllm llm block")
    rank.add_argument("--operator", choices=sorted(RANK_ONLY_OPERATORS), default="neutralize_rank_popularity")
    rank.add_argument("--order-controls", action="store_true", help="same-state original/repeat/reverse panels for both prompts; six rankings per query")
    rank.add_argument("--max-requests", type=int, help="default: panel count plus bounded ranking-repair allowance")
    rank.add_argument("--max-tokens", type=int, default=500000)
    rank.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.command == "rank-only":
        rank_only_cli(args)
