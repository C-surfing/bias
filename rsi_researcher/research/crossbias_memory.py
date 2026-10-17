"""Create only evidence-checked, development-grade Research Memory cards.

No LLM-generated 'lessons' are admitted. Both positive and negative mechanism
outcomes are recorded, and all cards point to the immutable numeric assessment.
"""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import json

from rsi_researcher.research import crossbias_discovery as b
from rsi_researcher.research.memory import ResearchMemory,validate_card
from rsi_researcher.research.crossbias_audit import audit


def factual_lesson(task,action,outcome):
    assert outcome["action"]==action
    verdict=outcome["falsifier"]
    qualified=outcome["replicated_development_signal"]
    if action=="gamma_stress":
        text=("Coat development gamma sensitivity selected different best cap/gamma "
              f"on the two disjoint user panels: {verdict['best_weighting_by_panel']}.")
    elif action=="propensity_shuffle":
        deltas=[round(x["delta_matched_minus_placebo"],7) for x in verdict["cohorts"]]
        text=("Coat correct propensity mapping yielded lower Brier than the better "
              f"of two user-shuffled mappings on each development panel: {deltas}.")
    elif action=="split_utility":
        parts=verdict["cohorts"]
        text=("Coat cap100 vs unweighted on two development panels: Brier changes "
              f"{[round(x['brier_delta'],6) for x in parts]}, NDCG ratios "
              f"{[round(x['ndcg_ratio'],5) for x in parts]}, "
              f"tail-recall changes {[round(x['tail_delta'],5) for x in parts]}.")
    else:
        changes=[x["candidate_ranking_change_rate"] for x in verdict["cohorts"]]
        head=[round(x["head_excess_delta"],4) for x in verdict["cohorts"]]
        quality=[x["NDCG5_quality_pass"] for x in verdict["cohorts"]]
        text=(f"AgentCF {action} on 2 panels of 5 author-development users: "
              f"ranking changed fractions {changes}, head-excess changes {head}, "
              f"NDCG quality gates {quality}.")
    text+=(" Registered within-development falsifier PASS."
           if qualified else " Registered within-development falsifier FAIL/INCONCLUSIVE.")
    return text


def build_cards():
    audit_result=audit()
    if audit_result["status"]!="CROSSBIAS_INDEPENDENT_AUDIT_COMPLETE":
        raise ValueError("Independent evidence review not complete")
    plans=json.loads((b.RUN/"plans.json").read_text(encoding="utf8"))
    assessed=json.loads((b.RUN/"assessments.json").read_text(encoding="utf8"))
    source=b.RUN/"assessments.json"
    digest=b.sha_file(source)
    created=datetime.now(timezone.utc).isoformat()
    by_action={}
    for arm in b.ARMS:
        for task,plan in plans[arm].items():
            action=plan["action"]
            row=assessed[arm][task]
            key=(task,action)
            if key in by_action:
                if row!=by_action[key]:
                    raise ValueError("Arms disagreed about shared physical evidence")
                continue
            by_action[key]=row
    cards=[]
    for task,action in sorted(by_action):
        observation=by_action[(task,action)]
        c=ResearchMemory(
            card_id="crossbias-20261017-"+task+"-"+action,
            task_id="crossbias-development-20261017-"+task,
            created_utc=created,
            source_file=str(source),
            source_sha256=digest,
            mechanism_tags=(
                ["coat","selection","propensity"] if task.startswith("coat") else
                ["agentcf","popularity"] if "popularity" in task else
                ["agentcf","item","memory"]),
            decision_action=action,
            conditional_lesson=factual_lesson(task,action,observation),
            limitation=("Development-only; same previously studied Coat panel or shared "
               "author state. Outcome proxy does not prove causal debiasing. "
               "Do not infer the original Qwen hypothesis was true without a "
               "formalized directional prediction and independent replication."),
            evidence_grade="development_only",split="development",confirmed=False)
        validate_card(c)
        cards.append(asdict(c))
    return cards


def main():
    path=b.RUN/"SOURCE_CHECKED_RESEARCH_MEMORY.json"
    if path.exists():raise FileExistsError(path)
    cards=build_cards()
    b.save_new(path,{"status":"FACTUAL_MEMORY_DEVELOPMENT_ONLY",
                    "llm_written_claims_admitted":0,
                    "independently_confirmed":0,
                    "cards":cards})
    print(json.dumps({"status":"FACTUAL_MEMORY_DEVELOPMENT_ONLY",
        "cards":len(cards),"llm_claims_admitted":0,
        "independently_confirmed":0},ensure_ascii=False))


if __name__=="__main__":main()
