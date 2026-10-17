"""Read development research artifacts into typed evidence situations, no label leakage."""
import json
from pathlib import Path
from rsi_researcher.research.planner import ResearchSituation, source_ref
ROOT=Path("/data/cjh/rsi")
RUN=ROOT/"project/runs/qwen-five-bias-verification-20261016"
AUDIT=ROOT/"outputs/evidence-audit-20261016"
DATA=ROOT/"outputs/bias-grounding-pilot-20261016/stage2-20261016"
def read(p):return json.loads(p.read_text(encoding="utf8"))
def context(sid,mechanism,question,conditions,refs,stage="development"):
    return ResearchSituation(sid,mechanism,question,conditions,refs,stage)

def load_real_situations():
    pattern_path=RUN/"research/baseline-patterns.json"
    manual_path=DATA/"annotation_manifest_private.json"
    pattern=read(pattern_path)
    manual=read(manual_path)
    marked=pattern["patterns"]["exposure"]["new_explicit_claim_events"]
    assert marked>0 and not manual["truth_labels_created"],"Need validated claim probe before context"
    negative=context("feedback-evidence","unsupported_negative_inference",
        "Which evidence must be measured before accepting a reduction in unsupported negative claims?",
        {"text_probe_uncalibrated":True,"mechanism_grounded":False},
        {"NEGATIVE_SCREEN":source_ref("NEGATIVE_SCREEN",str(pattern_path),
         "Text-only screen, not verified dislike"),
         "BLIND_STATUS":source_ref("BLIND_STATUS",str(manual_path),"Manual truth labels still absent")})
    replay_input=RUN/"research/trial-0001/input.json"
    cases_path=AUDIT/"case_details.json"
    prior_input=RUN/"baseline/input.json"
    rp=read(replay_input);base=read(prior_input);cases=read(cases_path)
    assert rp["initial"]==base["initial"] and rp["users"]==base["users"]
    assert len(set(r["source_event_id"] for r in rp["events"]))<len(base["events"])
    tails=[c for c in cases if c["group"]=="tail"]
    assert tails and any(c["replay_delta"]<0 for c in tails) and any(c["replay_delta"]>0 for c in tails)
    replay=context("replay-heterogeneity","learning_opportunity_replay",
      "How do replay omissions and shared memory influence conflicting long-tail outcomes?",
      {"replay_source_changed":True,"subgroup_divergence":True,"mechanism_grounded":False},
      {"REPLAY_SCHEDULE":source_ref("REPLAY_SCHEDULE",str(replay_input),"Replayed source IDs"),
       "TAIL_CASES":source_ref("TAIL_CASES",str(cases_path),"Paired ranks and user/item memories")})
    archive_path=RUN/"research/archive.json"
    archive=read(archive_path)
    position_trial=next(t for t in archive["trials"] if t["operator"]=="neutral_candidate_order")
    assert position_trial["comparison"]["measurement_gate"] is False
    position=context("order-control","candidate_order_sensitivity",
      "What is needed before interpreting input-order metric improvement as real mechanism evidence?",
      {"order_measurement_gate_failed":True},
      {"POSITION_CONTROL":source_ref("POSITION_CONTROL",str(archive_path),"Same-order control gate failed")})
    kuai_path=DATA/"kuairand_dev_common_support.json"
    kuai=read(kuai_path)
    assert kuai["standard_rows"]>0 and kuai["random_rows"]>0
    kuai_ctx=context("logging-support","exposure_logging_selection",
      "What should we check before inferring causality from ordinary vs random log click rates?",
      {"logging_policies_compared":True,
       "causal_exposure_question":True,"logging_propensities_missing":True},
      {"KUAI_SUPPORT":source_ref("KUAI_SUPPORT",str(kuai_path),
            "Development policy log item overlap and missing exact user/item matches")})
    protocol_path=AUDIT/"independent_validation_protocol.json"
    protocol=read(protocol_path)
    assert protocol["status"]=="DESIGN_FROZEN_EXECUTION_BLOCKED"
    confirmation=context("confirmation-blocked","confirmation_integrity",
      "Can unknown-feedback independent confirmation begin?",
      {"confirmation_requested":True,"holdout_uncleared":True,
       "holdout_cleared":False,"shared_memory_isolated":False,
       "frozen_protocol":True,"budget_approved":False},
      {"FROZEN_PROTOCOL":source_ref("FROZEN_PROTOCOL",str(protocol_path),
                 "Protocol frozen but confirmation execution blocked")},
      stage="confirmation_design")
    audit_path=AUDIT/"evidence_audit.json"
    audit=read(audit_path)
    assert any(any(x["code"]=="TARGET_REDUCTION_MISSTATED" for x in row["issues"]) for row in audit["reviews"])
    reflection=context("narrative-conflict","research_reflection_integrity",
      "What must be corrected before the Qwen explanation can be stored as trustworthy Research Memory?",
      {"reflection_numerical_conflict":True},
      {"MACHINE_GATES":source_ref("MACHINE_GATES",str(audit_path),
          "Python numbers contradict free-text researcher interpretation")})
    return [negative,replay,position,kuai_ctx,confirmation,reflection]

def relevant_memory_fixtures(situations):
    """Demonstrate context-specific retrieval without importing unrelated task answers."""
    cards=[]
    for s in situations:
        first=next(iter(s.evidence))
        cards.append({"card_id":"prior-"+s.situation_id,
          "mechanism_tags":s.mechanism.replace("_"," ").split(),
          "source":first,"grade":"development_only",
          "conditional_lesson":"Before advancing this mechanism, evaluate its specific evidence prerequisites and report unresolved alternatives.",
          "limitation":"Development observation, no independent experiment-level confirmation."})
    return cards
