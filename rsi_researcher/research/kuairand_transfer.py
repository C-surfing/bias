#!/usr/bin/env python3
"""Second-task smoke test: reuse the same evidence and action gates on KuaiRand.

Read-only exposure logging diagnostics, NOT a debias benchmark or causal claim.
"""
from pathlib import Path
import argparse
import json

from rsi_researcher.research.unified import HERE, digest, save_new, timestamp
from rsi_researcher.research.planner_adapters import load_real_situations
from rsi_researcher.research.planner_integration import minimal_proposal, execute_read_only
from rsi_researcher.research.memory import ResearchMemory, validate_card
from dataclasses import asdict


def run(output: Path) -> dict:
    output = output.resolve()
    if not output.is_relative_to(HERE / "runs"):
        raise ValueError("Outputs must remain inside this chat-owned directory")
    context = next(x for x in load_real_situations()
                   if x.situation_id == "logging-support")
    context.validate()
    rows = []
    for action in ("audit_common_support", "collect_exposure_logging_evidence"):
        proposal = minimal_proposal(context, action)
        result = execute_read_only(context, proposal)
        if result["status"] != "CPU_COMPLETED":
            raise ValueError("KuaiRand audit did not clear review")
        rows.append({"action": action, "pre_review": result["review"],
                     "observation": result["result"]})
    support = rows[0]["observation"]
    propensity = rows[1]["observation"]
    if support["same_useritem_pairs"] != 0:
        raise ValueError("Existing zero-exact-pair identification caveat changed; revise protocol")
    if propensity["status"] != "missing_logging_probability":
        raise ValueError("Propensity identification status changed; revise protocol")
    blocked = execute_read_only(
        context, minimal_proposal(context, "run_independent_replication"))
    if blocked["status"] != "BLOCKED":
        raise ValueError("Unauthorized replication escaped review gate")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "EXPOSURE_IDENTIFICATION_NOT_YET_POSSIBLE",
        "task": "kuairand-exposure-logging",
        "created_utc": timestamp(),
        "new_llm_requests": 0, "gpu_runs": 0,
        "dataset": "KuaiRand existing random and standard development logs",
        "evidence": {key: {"source_file": ref.source_file, "sha256": ref.sha256}
                     for key, ref in context.evidence.items()},
        "actions": rows,
        "replication_rejected": blocked,
        "independent_confirmation": False,
        "next_required": "Identify logging propensities/positivity or use controlled reference; no causal debias claim",
    }
    save_new(output / "exposure_audit.json", result)
    card = ResearchMemory(
        card_id="unified-kuai-exposure", task_id="kuairand-exposure-logging",
        created_utc=result["created_utc"],
        source_file=str(output / "exposure_audit.json"),
        source_sha256=digest(output / "exposure_audit.json"),
        mechanism_tags=["exposure", "logging", "propensity"],
        decision_action="audit_common_support",
        conditional_lesson=result["next_required"],
        limitation="Development logging audit only, no identified counterfactual effect",
        evidence_grade="development_only", confirmed=False)
    validate_card(card)
    save_new(output / "research_memory_card.json", asdict(card))
    (output / "REPORT.md").write_text(
        "# KuaiRand exposure task transfer check\n\n"
        "Planner V2 evidence review and Phase 4 read-only action dispatch were reused.\n\n"
        f"Common item count: {support['common_item_count']}; "
        f"item Jaccard: {support['item_set_jaccard']}.\n\n"
        "Missing reliable logging propensities and zero exact user-item overlap prevent "
        "a naive causal comparison. Confirmation actions remain blocked.\n"
        "No Qwen requests, no model training, no held-out test evaluation.\n",
        encoding="utf8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    result = run(parser.parse_args().output)
    print(json.dumps({"status": result["status"], "actions": len(result["actions"]),
        "new_llm_requests": 0, "gpu_runs": 0}, ensure_ascii=False))
