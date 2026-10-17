"""Controlled, read-only developer evidence measurements. No dynamic tool dispatch/import."""
import json
from pathlib import Path
from rsi_researcher.research.planner import ResearchSituation
ROOT=Path("/data/cjh/rsi")
RUN=ROOT/"project/runs/qwen-five-bias-verification-20261016"
AUDIT=ROOT/"outputs/evidence-audit-20261016"
DATA=ROOT/"outputs/bias-grounding-pilot-20261016/stage2-20261016"

def load(p):return json.loads(p.read_text(encoding="utf8"))
def run_registered_cpu_action(ctx:ResearchSituation,action:str,design_accepted:bool)->dict:
    if not design_accepted:raise PermissionError("review must accept design")
    # Tool IDs and status are explicit; never executes arbitrary code, shell or Qwen.
    if action=="annotate_claim_evidence":
        manifest=load(DATA/"annotation_manifest_private.json")
        return {"status":"await_human_annotation","samples":manifest["blind_records"],
            "manual_labels_created":manifest["truth_labels_created"],
            "scientific_conclusion":"text screen uncalibrated; no ground-truth claim correction established"}
    if action=="diagnose_replay_coverage":
        base=load(RUN/"baseline/input.json")
        trial=load(RUN/"research/trial-0001/input.json")
        cases=load(AUDIT/"case_details.json")
        original={e["event_id"] for e in base["events"]}
        used={e["source_event_id"] for e in trial["events"]}
        tail=[c for c in cases if c["group"]=="tail"]
        return {"status":"mechanism_discrimination_needed","original_event_count":len(original),
          "replay_updates":len(trial["events"]),"unique_source_events":len(used),
          "omitted_events":len(original-used),
          "tail_worsened":[c["user"] for c in tail if c["replay_delta"]<0],
          "tail_improved":[c["user"] for c in tail if c["replay_delta"]>0],
          "scientific_conclusion":"different user trajectories; omission and shared-memory mediation not yet isolated"}
    if action=="audit_order_noise_control":
        archive=load(RUN/"research/archive.json")
        trial=next(t for t in archive["trials"] if t["operator"]=="neutral_candidate_order")
        c=trial["comparison"]
        return {"status":"measurement_control_failed","measurement_gate":c["measurement_gate"],
          "numeric_target_reduction":c["target"]["relative_reduction"],
          "quality_gate":c["quality_gate"],
          "scientific_conclusion":"position input prompt not confirmed by paired noise controls"}
    if action in ("audit_common_support","collect_exposure_logging_evidence"):
        audit=load(DATA/"kuairand_dev_common_support.json")
        if action=="collect_exposure_logging_evidence":
            return {"status":"missing_logging_probability","propensities_assumed":audit["propensities_assumed"],
               "same_useritem_pairs":audit["same_user_same_item_pairs"],
               "scientific_conclusion":"cannot estimate intervention effect without identified support and logging policy"}
        return {"status":"descriptive_logging_comparison",
          "item_set_jaccard":audit["item_jaccard"],"common_item_count":audit["common_item_count"],
          "same_useritem_pairs":audit["same_user_same_item_pairs"],
          "scientific_conclusion":"descriptive logging-policy shift; comparison is not a causal debias gain"}
    if action=="audit_confirmation_isolation":
        protocol=load(AUDIT/"independent_validation_protocol.json")
        return {"status":"independent_replication_blocked","protocol_status":protocol["status"],
           "provisional_cohort_only":True,"shared_memory_verified":False,
           "scientific_conclusion":"cannot call provisional users independent until source histories and item memory are verified"}
    if action=="correct_reflection_from_numbers":
        audit=load(AUDIT/"evidence_audit.json")
        t=next(t for t in audit["reviews"] if
             any(i["code"]=="TARGET_REDUCTION_MISSTATED" for i in t["issues"]))
        return {"status":"canonical_numerical_reflection",
         "target_reduction":t["relative_target_reduction"],
         "target_numeric_gate":t["numeric_target_gate"],
         "quality_gate":t["quality_gate"],
         "overall_development_proxy_success":t["proxy_goal_pass"],
         "scientific_conclusion":"numeric target passes, quality fails; old Qwen narrative contradictory"}
    if action=="abstain_and_measure":
        return {"status":"no_action_insufficient_evidence","scientific_conclusion":"abstain without inventing outcomes"}
    if action in ("propose_bounded_intervention","run_independent_replication"):
        return {"status":"not_authorized","scientific_conclusion":"CPU pilot cannot execute interventions or confirmation"}
    raise ValueError("unknown or unsafe research action")
