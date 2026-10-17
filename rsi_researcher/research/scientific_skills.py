"""Research strategies are NOT facts: skill changes need held-out matched evidence.

Development reflection creates only an immutable candidate. A promoted skill
must beat strong rules, no-history Qwen and shuffled-history Qwen on previously
unseen independently audited tasks under the same predeclared budgets.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from datetime import datetime, timezone
import json
from rsi_researcher.runtime.client import atomic_json

ARMS = ("evolved_skill", "strong_rule", "qwen_no_history", "qwen_shuffled_history")


def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def proposal(episode, task, text, audit):
    """Write a candidate alongside its source audit. Cannot promote itself."""
    episode, audit = Path(episode), Path(audit)
    if not audit.is_file() or audit.parent != episode:
        raise ValueError("candidate must point to its own independently computed episode audit")
    source = json.loads(audit.read_text(encoding="utf8"))
    if source.get("task") != task or source.get("independently_confirmed_discoveries") != 0:
        raise ValueError("invalid development source audit")
    if not isinstance(text,str) or len(text.strip()) < 12:
        raise ValueError("candidate skill is empty")
    identity = sha256((task+"\n"+text.strip()+"\n"+fingerprint(audit)).encode()).hexdigest()[:20]
    card = {
        "skill_id": identity, "version": 1, "scope_task":task,
        "text":text.strip(), "origin_episode":episode.name,
        "source_audit_sha256":fingerprint(audit),
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"unverified_candidate",
        "evidence_grade":"development_only",
        "validated_skill":False,
        "requires_unseen_task_evaluation":True,
        "known_limitations":source.get("reason","Unconfirmed development evidence"),
    }
    path = episode/"candidate_skill.json"
    if path.exists():
        raise FileExistsError("candidate was already frozen")
    atomic_json(path,card)
    return card


def score_matched_trial(trial):
    """Ground every arm in a true independent numeric audit, not LLM prose."""
    if set(trial["arms"]) != set(ARMS):
        raise ValueError("each trial requires all four predeclared comparison arms")
    seen=[]
    budget=trial["budget"]
    if not all(isinstance(budget[k],int) and budget[k]>0
               for k in ("max_requests","max_tokens")):
        raise ValueError("invalid comparison budget")
    cohort=trial["cohort_sha256"]
    if not isinstance(cohort,str) or len(cohort)!=64:
        raise ValueError("missing paired cohort fingerprint")
    for arm in ARMS:
        record=trial["arms"][arm]
        path=Path(record["audit_file"])
        if not path.is_file() or fingerprint(path)!=record["audit_sha256"]:
            raise ValueError("missing or modified independent numeric audit")
        audit=json.loads(path.read_text(encoding="utf8"))
        if (audit.get("cohort_sha256")!=cohort or
            audit.get("research_task")!=trial["task_id"] or
            audit.get("evaluation_split")!="independent" or
            not audit.get("pre_registered_confirmation")):
            raise ValueError("unequal cohort, unknown task, or nonindependent experiment")
        if audit.get("arm")!=arm:
            raise ValueError("audit arm identity mismatch")
        confirmations=audit.get("independently_confirmed_discoveries")
        requests=audit.get("model_requests")
        tokens=audit.get("total_tokens")
        if (type(confirmations)is not int or confirmations<0 or
            type(requests)is not int or requests<0 or requests>budget["max_requests"] or
            type(tokens)is not int or tokens<0 or tokens>budget["max_tokens"]):
            raise ValueError("invalid discovery or cost accounting")
        seen.append((arm,confirmations,tokens))
    return dict((arm,{"discoveries":d,"tokens":t}) for arm,d,t in seen)


def evaluate_candidate(candidate, trials):
    """A conservative programmatic eligibility gate, not a statistical proof.

    Requires >=8 disjoint unseen, independently tested and budget-matched tasks,
    strictly more discoveries than each comparator, and no larger total token
    consumption than 110% of the *cheapest* comparator.
    """
    if candidate.get("status")!="unverified_candidate" or candidate.get("validated_skill") is not False:
        raise ValueError("only unverified skill candidates may be compared")
    if len(trials)<8:
        return {"eligible":False,"reason":"fewer than eight independent matched tasks"}
    task_ids=[t["task_id"] for t in trials]
    if len(task_ids)!=len(set(task_ids)) or candidate["scope_task"] in task_ids:
        raise ValueError("task leakage, repeated tasks or source task reused")
    cohorts=[t["cohort_sha256"] for t in trials]
    if len(cohorts)!=len(set(cohorts)):
        raise ValueError("reused independent evaluation cohort")
    scored=[score_matched_trial(t) for t in trials]
    total={arm:{"discoveries":sum(row[arm]["discoveries"] for row in scored),
                "tokens":sum(row[arm]["tokens"] for row in scored)}
           for arm in ARMS}
    proposed=total["evolved_skill"]
    comparators=[total[arm] for arm in ARMS if arm!="evolved_skill"]
    wins=sum(row["evolved_skill"]["discoveries"] >
             max(row[arm]["discoveries"] for arm in ARMS if arm!="evolved_skill")
             for row in scored)
    # Require gains from independently-confirmed discoveries, not proxy metrics.
    eligible=(wins>=7 and
              proposed["discoveries"]>max(a["discoveries"] for a in comparators) and
              proposed["tokens"] <= 1.1*min(a["tokens"] for a in comparators))
    return {
        "eligible":eligible,
        "status":"eligible_for_separate_confirmation" if eligible else "rejected_by_predeclared_gate",
        "total_by_arm":total, "paired_task_wins":wins, "task_count":len(trials),
        "reason":("Passes mechanical prescreen; does not itself establish statistical "
                  "significance or permit Qwen self-promotion") if eligible else
                  "Insufficient paired independent discovery improvement at matched cost",
    }


def select_promoted(skills_path, task):
    """Read only externally authorized, versioned promoted skills, never draft text."""
    path=Path(skills_path)
    if not path.is_file():
        return []
    cards=json.loads(path.read_text(encoding="utf8"))
    if not isinstance(cards,list):
        raise ValueError("skill library must be a list")
    result=[]
    for card in cards:
        if card.get("status")!="promoted" or card.get("scope_task")!=task:
            continue
        audit_path=Path(card["independent_evaluation_file"])
        if not audit_path.is_file() or fingerprint(audit_path)!=card["evaluation_sha256"]:
            raise ValueError("promoted skill lost its evaluation provenance")
        evidence=json.loads(audit_path.read_text(encoding="utf8"))
        if evidence.get("status")!="eligible_for_separate_confirmation" or not evidence.get("eligible"):
            raise ValueError("promoted skill lacks evaluation gate")
        if not card.get("external_approval_reference"):
            raise ValueError("promoted skill lacks separate approval")
        result.append({"skill_id":card["skill_id"],"version":card["version"],
                       "text":card["text"],"scope_task":task,
                       "source_evidence":card["evaluation_sha256"]})
    return result
