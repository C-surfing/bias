#!/usr/bin/env python3
"""CPU-only evidence-gated Coat research episodes; no new LLM calls or holdout scoring.

Reuses the frozen task registry, the existing Coat trainer/evaluator, Planner V2
proposal review, and Phase 4 provenance-checked Research Memory. Never launches
subprocesses, training services or AgentCF operations.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import math
import re
import sys

ROOT = Path("/data/cjh/rsi")
HERE = ROOT / "project"
CLASSIC = ROOT / "experiments/classic-five-20261016"
PLANNER = ROOT / "outputs/research-planner-v2-20261016"
INTEGRATION = ROOT / "outputs/research-planner-integration-20261016"
PROJECT = ROOT / "project"

from rsi_researcher.research.planner import ResearchProposal, ResearchSituation, review, source_ref
from rsi_researcher.research.memory import ResearchMemory, build_prior_cards, retrieve_history, validate_card
from rsi_researcher.research.coat_selection import DATA, SEED, estimated_p_matrix, fit_item_mean, metrics
from rsi_researcher.datasets.coat import load_coat

REGISTRY = PROJECT / "configs/bias_tasks.json"
DEV_REFERENCE = CLASSIC / "runs/coat_dev_02/coat_selection_development.json"
QWEN_RECORDS = CLASSIC / "runs/coat_qwen_multiround_01/decisions.json"
QWEN_PROTOCOL = CLASSIC / "runs/coat_qwen_multiround_01/frozen_protocol.json"
POLICIES = ("unweighted", "estimated_propensity_clip_5",
            "estimated_propensity_clip_20", "estimated_propensity_clip_100")
TASK_ID = "coat-selection-development-unified-v1"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_new(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def frozen_sources() -> dict:
    if digest(REGISTRY) != REGISTRY.with_suffix(".sha256").read_text().strip():
        raise ValueError("Frozen bias task registry hash changed")
    registry = json.loads(REGISTRY.read_text(encoding="utf8"))
    tasks = {t["id"]: t for t in registry["tasks"]}
    task = tasks["selection"]
    if task["current_status"] != "BENCHMARK_READY_FOR_DEVELOPMENT":
        raise ValueError("Selection task has no qualified development benchmark")
    if digest(QWEN_PROTOCOL) != (QWEN_PROTOCOL.parent / "protocol.sha256").read_text().strip():
        raise ValueError("Archived Qwen protocol hash changed")
    return {
        "registry_sha256": digest(REGISTRY),
        "coat_method_sha256": digest(PROJECT / "rsi_researcher/research/coat_selection.py"),
        "coat_data_adapter_sha256": digest(PROJECT / "rsi_researcher/datasets/coat.py"),
        "planner_sha256": digest(PROJECT / "rsi_researcher/research/planner.py"),
        "qwen_decisions_sha256": digest(QWEN_RECORDS),
        "qwen_protocol_sha256": digest(QWEN_PROTOCOL),
        "reference_development_sha256": digest(DEV_REFERENCE),
    }


def situation() -> ResearchSituation:
    refs = {
        "TASK_REGISTRY": source_ref("TASK_REGISTRY", str(REGISTRY), "Frozen five-bias task definitions"),
        "COAT_ADAPTER": source_ref("COAT_ADAPTER",
            str(PROJECT / "rsi_researcher/datasets/coat.py"),
            "Observed ratings and pre-existing development/holdout split"),
        "PROPENSITY_SOURCE": source_ref("PROPENSITY_SOURCE",
            str(DATA / "propensities.ascii"),
            "Author-estimated propensities; not true logging probabilities"),
    }
    ctx = ResearchSituation(
        situation_id=TASK_ID, mechanism="selection propensity weighting",
        question="Can bounded propensity weighting improve Coat random-development Brier while preserving NDCG?",
        conditions={"mechanism_grounded": True, "intervention_registry_ready": True},
        evidence=refs, stage="development")
    ctx.validate(verify_files=True)
    return ctx


def proposal(decision: dict) -> ResearchProposal:
    action = decision.get("action")
    if action not in ("evaluate", "measure_weight_support", "stop"):
        raise ValueError("Unregistered Coat research action")
    policy = decision.get("policy", "unweighted")
    if action == "evaluate" and policy not in POLICIES:
        raise ValueError("Unregistered Coat candidate policy")
    if action != "evaluate" and policy != "unweighted":
        raise ValueError("Non-evaluation action cannot carry a candidate")
    fields = ("hypothesis", "alternative_explanation", "falsifier", "reasoning")
    if any(not isinstance(decision.get(k), str) or not decision[k].strip() for k in fields):
        raise ValueError("Research proposal must include substantive hypotheses and falsifier")
    return ResearchProposal(
        action="propose_bounded_intervention" if action == "evaluate" else "abstain_and_measure",
        hypothesis=decision["hypothesis"], competing_hypothesis=decision["alternative_explanation"],
        falsifier=decision["falsifier"], rationale=decision["reasoning"],
        evidence_ids=["TASK_REGISTRY", "COAT_ADAPTER", "PROPENSITY_SOURCE"],
        expected_outcome="Development-only Brier and NDCG measurement or an explicit stop",
        budget_reason="One bounded CPU development evaluation; no holdout or GPU use",
        parameters={"policy": policy} if action == "evaluate" else {},
    )


def gate(ctx: ResearchSituation, decision: dict) -> dict:
    # Reject unexpected tool/parameter injection before the generic evidence review.
    if set(decision) - {"action", "policy", "hypothesis", "alternative_explanation",
                        "falsifier", "reasoning"}:
        return {"status": "REJECTED", "issues": ["unexpected_proposal_field"]}
    try:
        p = proposal(decision)
    except (ValueError, TypeError) as exc:
        return {"status": "REJECTED", "issues": [str(exc)]}
    result = review(ctx, p)
    if result["status"] != "DESIGN_ACCEPTED" or result["execution_authorized"]:
        return {"status": "REJECTED", "issues": result["issues"]}
    if p.action == "propose_bounded_intervention" and set(p.parameters) != {"policy"}:
        return {"status": "REJECTED", "issues": ["invalid_policy_parameters"]}
    return {"status": "DEVELOPMENT_CPU_ONLY", "issues": [],
            "generic_review_status": result["status"], "execution_authorized": False}


def fixed_choice(policy: str) -> dict:
    return {
        "action": "evaluate", "policy": policy,
        "hypothesis": "Bounded propensity weighting may improve random-development calibration.",
        "alternative_explanation": "Changes may result from noise or estimated propensity error.",
        "falsifier": "No Brier improvement or NDCG below the predeclared 99% safeguard.",
        "reasoning": "Evaluate exactly one predeclared policy using the same model and panel.",
    }


def replay_arms() -> dict:
    archive = json.loads(QWEN_RECORDS.read_text(encoding="utf8"))
    if len(archive) != 6:
        raise ValueError("Unexpected archived Qwen decision count")
    arms = {"fixed_grid": [fixed_choice(x) for x in POLICIES[1:]],
            "qwen_no_history_replay": [], "qwen_with_history_replay": []}
    for arm_name, target in (("no_history", "qwen_no_history_replay"),
                             ("with_history", "qwen_with_history_replay")):
        rows = sorted((r for r in archive if r["arm"] == arm_name), key=lambda r: r["round"])
        if [r["round"] for r in rows] != [1, 2, 3]:
            raise ValueError("Invalid archived Qwen turn numbering")
        for r in rows:
            raw = json.loads(r["llm_meta"]["raw"])
            if raw != r["decision"]:
                raise ValueError("Archived Qwen decision differs from raw response")
            arms[target].append(raw)
    return arms


def evaluate_candidate(ds: dict, props: list, train_counts: dict, policy: str) -> dict:
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    cap = None if policy == "unweighted" else float(policy.rsplit("_", 1)[1])
    pred, fit = fit_item_mean(ds["train"], ds["item_ids"],
                              props if cap is not None else None, cap)
    scored = metrics(ds["development"], pred, train_counts, ds["item_ids"])
    result = {
        "Brier": scored["Brier_random_development"],
        "NDCG5": scored["NDCG5_random_development"],
        "Hit5": scored["Hit5_random_development"],
        "tail_relevant_recall": scored["tail_relevant_recall"],
        "ESS": fit["effective_train_rows"],
    }
    if not all(isinstance(v, (int, float)) and math.isfinite(v)
               for v in result.values() if v is not None):
        raise ValueError("Nonfinite CPU evaluation metric")
    return result


def select(rows: list, base: dict) -> str:
    valid = [r for r in rows if r["status"] == "EVALUATED"
             and r["evaluation"]["Brier"] < base["Brier"]
             and r["evaluation"]["NDCG5"] >= .99 * base["NDCG5"]]
    return min(valid, key=lambda x: x["evaluation"]["Brier"])["policy"] if valid else "unweighted"


def arm_choices(external_file: Path | None) -> tuple[dict, str, str | None]:
    if external_file is None:
        return replay_arms(), "archived_qwen_replay", None
    path = external_file.resolve()
    if not path.is_relative_to(HERE) or not path.is_file():
        raise ValueError("External research proposals must be a file within this owned directory")
    payload = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(payload, dict) or set(payload) != {"arm_name", "decisions"}:
        raise ValueError("Expected only arm_name and decisions in the proposal file")
    name = payload["arm_name"]
    decisions = payload["decisions"]
    if not isinstance(name, str) or not re.fullmatch(r"external_[a-z0-9_]{1,40}", name):
        raise ValueError("Untrusted external arm name")
    if not isinstance(decisions, list) or not 1 <= len(decisions) <= 3:
        raise ValueError("Per-arm decision budget must be between one and three")
    if not all(isinstance(d, dict) for d in decisions):
        raise ValueError("Every external decision must be a JSON object")
    # Fixed-rule control is always retained, and the gate reviews each proposal.
    return {"fixed_grid": [fixed_choice(x) for x in POLICIES[1:]],
            name: decisions}, "provided_proposals_cpu_only", digest(path)


def run(output: Path, external_file: Path | None = None) -> dict:
    output = output.resolve()
    if not output.is_relative_to(HERE / "runs"):
        raise ValueError("All outputs must stay within this chat-owned runs directory")
    source_hashes = frozen_sources()
    ctx = situation()
    # Existing loader parses the physical random-rating file to define its fixed
    # development/holdout partition. Only ds["development"] is ever evaluated.
    ds = load_coat(DATA, split_seed=SEED)
    if (len(ds["train"]), len(ds["development"]), ds["statistics"]["test_rows"]) != (6960, 2065, 2209):
        raise ValueError("Coat data partitions unexpectedly changed")
    reference = json.loads(DEV_REFERENCE.read_text(encoding="utf8"))
    if reference["source_dataset_fingerprint"] != ds["fingerprint"]:
        raise ValueError("Development benchmark fingerprint changed")
    if reference["test_evaluation_done"] or reference["independent_confirmation"]:
        raise ValueError("Expected development-only reference")
    output.mkdir(parents=True, exist_ok=False)
    props = estimated_p_matrix()
    cnt = Counter(r["item"] for r in ds["train"])
    counts = {k: cnt.get(k, 0) for k in ds["item_ids"]}
    cache: dict = {}
    baseline = evaluate_candidate(ds, props, counts, "unweighted")
    old = reference["policies"]["unweighted"]["development_metrics"]["Brier_random_development"]
    if not math.isclose(baseline["Brier"], old, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Baseline numerical regression")
    cache["unweighted"] = baseline
    context_started = timestamp()
    prior = build_prior_cards()
    history_refs = retrieve_history(
        task_id=TASK_ID, mechanism=ctx.mechanism,
        started_utc=context_started, cards=prior,
        excluded_source_hashes=[ref.sha256 for ref in ctx.evidence.values()])
    choices_by_arm, proposal_mode, external_sha = arm_choices(external_file)
    output_arms = {}
    for name, choices in choices_by_arm.items():
        rows = []
        for number, decision in enumerate(choices, 1):
            verdict = gate(ctx, decision)
            row = {"round": number, "action": decision["action"],
                   "policy": decision.get("policy", "unweighted"),
                   "proposal": decision, "gate": verdict}
            if verdict["status"] != "DEVELOPMENT_CPU_ONLY":
                row["status"] = "REJECTED"
            elif decision["action"] == "stop":
                row["status"] = "STOPPED"
            elif decision["action"] == "measure_weight_support":
                row["status"] = "MEASURE_ONLY"
                row["effective_rows_by_cap"] = {
                    str(int(cap)): fit_item_mean(ds["train"], ds["item_ids"], props, cap)[1]["effective_train_rows"]
                    for cap in (5, 20, 100)}
            else:
                policy = decision["policy"]
                hit = policy in cache
                if not hit:
                    cache[policy] = evaluate_candidate(ds, props, counts, policy)
                row["status"] = "EVALUATED"
                row["evaluation"] = cache[policy]
                row["quality_gate"] = cache[policy]["NDCG5"] >= .99 * baseline["NDCG5"]
                row["brier_improved"] = cache[policy]["Brier"] < baseline["Brier"]
                row["cached_cpu_evaluation"] = hit
            rows.append(row)
            if row["status"] == "STOPPED":
                break
        output_arms[name] = {"episodes": rows, "selected_development_policy": select(rows, baseline)}
    summary = {
        "scope": "Coat random-rated development ONLY",
        "status": "CPU_DEVELOPMENT_REPLAY_COMPLETE",
        "no_model_called_by_this_runner": True,
        "proposal_mode": proposal_mode,
        "external_proposal_sha256": external_sha,
        "no_causal_or_RSI_confirmation": True,
        "new_llm_requests": 0,
        "gpu_runs": 0,
        "independent_confirmation": False,
        "holdout_metrics_calculated": False,
        "holdout_rows_withheld": ds["statistics"]["test_rows"],
        "source_fingerprints": source_hashes,
        "dataset_fingerprint": ds["fingerprint"],
        "task_id": TASK_ID,
        "created_utc": timestamp(),
        "baseline": baseline,
        "research_memory_prior_cards_eligible": [x["card_id"] for x in history_refs],
        "memory_retrieval_used_to_generate_decisions": False,
        "arms": output_arms,
        "scientific_limitations": [
            "Archive mode replays 2026-10-16 Qwen; provided mode consumes external choices but makes no model calls.",
            "Selection estimates use learned Coat propensity scores, not known assignment probabilities.",
            "The original loader constructs both split objects; this run scores only development.",
            "All arms reuse one development panel and candidate cache; no new independent evidence.",
            "Research Memory records conditional lessons, not proven RSI policy learning.",
        ],
    }
    save_new(output / "research_episodes.json", summary)
    cards = []
    for name, arm in output_arms.items():
        card = ResearchMemory(
            card_id="unified-" + name, task_id=TASK_ID,
            created_utc=summary["created_utc"],
            source_file=str(output / "research_episodes.json"),
            source_sha256=digest(output / "research_episodes.json"),
            mechanism_tags=["selection", "propensity", "recommender bias"],
            decision_action="bounded_development_evaluation",
            conditional_lesson=f"{name}: development selected {arm['selected_development_policy']}",
            limitation="Same Coat development panel; no causal effect or cross-task RSI proof",
            evidence_grade="development_only", split="development", confirmed=False)
        validate_card(card)
        from dataclasses import asdict
        cards.append(asdict(card))
    save_new(output / "research_memory_cards.json", cards)
    report = [
        "# Unified RSI Coat development evidence loop",
        "", "Existing code reused: classic-five Coat evaluator + Planner V2 review + Phase 4 ResearchMemory.",
        "No Qwen or GPU requested; input mode: " + proposal_mode + ".",
        "", "| Arm | Selected development policy | Episodes |",
        "|---|---|---:|",
    ]
    for name, arm in output_arms.items():
        report.append(f"| {name} | {arm['selected_development_policy']} | {len(arm['episodes'])} |")
    report += ["", f"Baseline Brier: {baseline['Brier']:.8f}.",
               "All outcomes are development-only, unconfirmed and do not establish RSI.",
               "Inputs include the original Coat random-rating file because the existing loader performs the predeclared split; only the development subset is scored.",
               "The planner's design approval authorizes only these CPU measurements, never GPU or AgentCF training.",
               "Memory cards preserve traceable conditional conclusions; archived decisions did not consume newly retrieved cards."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposals-file", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.output, args.proposals_file)
    print(json.dumps({"status": result["status"],
        "selected": {a: r["selected_development_policy"] for a, r in result["arms"].items()},
        "new_llm_requests": 0, "gpu_runs": 0, "test_evaluated": False}, ensure_ascii=False))
