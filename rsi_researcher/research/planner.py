"""Versioned evidence-gated research planning: zero Qwen calls and no arbitrary execution."""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
from pathlib import Path
from hashlib import sha256
from typing import Mapping
import json
from datetime import datetime,timezone

SCHEMA="rsi_evidence_action_v2"
ACTION_REQUIREMENTS={
 "annotate_claim_evidence":("text_probe_uncalibrated",),
 "diagnose_replay_coverage":("replay_source_changed","subgroup_divergence"),
 "audit_order_noise_control":("order_measurement_gate_failed",),
 "audit_common_support":("logging_policies_compared",),
 "audit_confirmation_isolation":("confirmation_requested","holdout_uncleared"),
 "correct_reflection_from_numbers":("reflection_numerical_conflict",),
 "collect_exposure_logging_evidence":("causal_exposure_question","logging_propensities_missing"),
 "propose_bounded_intervention":("mechanism_grounded","intervention_registry_ready"),
 "run_independent_replication":("confirmation_requested","holdout_cleared","shared_memory_isolated",
                                "frozen_protocol","budget_approved"),
 "abstain_and_measure":(),
}
ACTION_KIND={
 "annotate_claim_evidence":"measure",
 "diagnose_replay_coverage":"discriminate",
 "audit_order_noise_control":"measure",
 "audit_common_support":"measure",
 "audit_confirmation_isolation":"audit",
 "correct_reflection_from_numbers":"correct",
 "collect_exposure_logging_evidence":"measure",
 "propose_bounded_intervention":"intervene",
 "run_independent_replication":"replicate",
 "abstain_and_measure":"abstain"
}
# Compare requirements, not task IDs; this is a strong interpretable rule baseline.
PRIORITY=[
 "correct_reflection_from_numbers","annotate_claim_evidence",
 "collect_exposure_logging_evidence","audit_common_support",
 "audit_order_noise_control","audit_confirmation_isolation",
 "diagnose_replay_coverage","propose_bounded_intervention",
 "run_independent_replication","abstain_and_measure"]
SAFE_SOURCE_ROOT=Path("/data/cjh/rsi").resolve()

@dataclass(frozen=True)
class EvidenceRef:
    evidence_id:str
    source_file:str
    sha256:str
    note:str

@dataclass
class ResearchSituation:
    situation_id:str
    mechanism:str
    question:str
    conditions:dict[str,bool]
    evidence:dict[str,EvidenceRef]
    stage:str="development"
    prior_notes:list[str]=field(default_factory=list)
    def validate(self,verify_files:bool=True)->None:
        if not self.situation_id or not self.mechanism or not self.question:raise ValueError("missing context metadata")
        if not self.evidence:raise ValueError("no evidence")
        if any(type(v) is not bool for v in self.conditions.values()):raise ValueError("nonboolean prereq")
        if self.stage not in ("development","confirmation_design"):raise ValueError("unknown stage")
        if verify_files:
            for eid,ref in self.evidence.items():
                if eid!=ref.evidence_id:raise ValueError("mismatched evidence ID")
                f=Path(ref.source_file).resolve()
                if not f.is_relative_to(SAFE_SOURCE_ROOT) or not f.is_file():
                    raise ValueError("evidence outside verified own RSI source directory")
                if sha256(f.read_bytes()).hexdigest()!=ref.sha256:
                    raise ValueError("evidence source hash mismatch: "+eid)

@dataclass
class ResearchProposal:
    action:str
    hypothesis:str
    competing_hypothesis:str
    falsifier:str
    rationale:str
    evidence_ids:list[str]
    expected_outcome:str
    budget_reason:str
    parameters:dict=field(default_factory=dict)

def admissible_actions(ctx:ResearchSituation)->list[str]:
    return [a for a in PRIORITY if all(ctx.conditions.get(k,False) for k in ACTION_REQUIREMENTS[a])]

def default_policy(ctx:ResearchSituation)->dict:
    ctx.validate()
    eligible=admissible_actions(ctx)
    choice=eligible[0] if eligible else "abstain_and_measure"
    return {"policy":"deterministic_evidence_prerequisite_v2",
      "next_action":choice,"action_type":ACTION_KIND[choice],
      "eligible_actions":eligible,
      "reason":"First evidence-admissible action in frozen order; not proof of scientifically optimal information gain.",
      "execution_authorized":False}

def review(ctx:ResearchSituation,proposal:ResearchProposal)->dict:
    ctx.validate()
    problems=[]
    if proposal.action not in ACTION_REQUIREMENTS:
        problems.append("unregistered_action")
    else:
        for req in ACTION_REQUIREMENTS[proposal.action]:
            if not ctx.conditions.get(req,False):
                problems.append("unmet_prerequisite:"+req)
    text_fields=["hypothesis","competing_hypothesis","falsifier",
                 "rationale","expected_outcome","budget_reason"]
    for name in text_fields:
        if not isinstance(getattr(proposal,name),str) or not getattr(proposal,name).strip():
            problems.append("missing_text:"+name)
    if not proposal.evidence_ids:
        problems.append("no_evidence_cited")
    elif any(not isinstance(x,str) or x not in ctx.evidence for x in proposal.evidence_ids):
        problems.append("ungrounded_evidence_id")
    if proposal.action in ("propose_bounded_intervention","run_independent_replication"):
        if ctx.stage=="confirmation_design" and proposal.action=="propose_bounded_intervention":
            problems.append("confirmation_set_must_not_tune_intervention")
        if proposal.action=="run_independent_replication" and ctx.stage!="confirmation_design":
            problems.append("confirmation_run_requires_confirmation_design_stage")
    if proposal.parameters and proposal.action not in ("propose_bounded_intervention",):
        problems.append("unregistered_parameterized_action")
    status="REQUEST_REVISION" if problems else "DESIGN_ACCEPTED"
    return {"status":status,"issues":problems,
      "requested_action":proposal.action,
      "eligible_actions":admissible_actions(ctx),
      "execution_authorized":False,
      "no_model_success_claim":True,
      "next":"revise_or_abstain" if problems else "design_further_or_seek_run_approval"}

def _tags_overlap(task_mechanism,card):
    tags=card.get("mechanism_tags",[])
    if not isinstance(tags,list):return 0
    tokens=set(task_mechanism.lower().replace("_"," ").split())
    expanded=set(" ".join(tags).lower().replace("_"," ").split())
    return len(tokens & expanded)

def retrieve_relevant_memory(ctx:ResearchSituation,cards:list[dict],limit:int=3)->list[dict]:
    """Filter explicitly tagged cards, reject unevaluated/scenario-specific numerical leakage."""
    selected=[]
    for c in cards:
        if c.get("grade") not in ("development_only","conditional","independently_confirmed"):
            continue
        if c.get("source") not in ctx.evidence:continue
        overlap=_tags_overlap(ctx.mechanism,c)
        if not overlap:continue
        score=overlap*10+(3 if c.get("grade")=="independently_confirmed" else 0)
        selected.append((score,str(c.get("card_id","")),c))
    selected.sort(key=lambda v:(-v[0],v[1]))
    return [{"card_id":c["card_id"],"mechanism_tags":c["mechanism_tags"],
       "conditional_lesson":c.get("conditional_lesson"),
       "limitation":c.get("limitation"),
       "evidence_id":c["source"],"grade":c["grade"]} for _,_,c in selected[:limit]]

def canonical_result(result:dict)->dict:
    """Never let free-text reflections override independently calculated numeric gates."""
    target=result.get("target",{})
    comparison=result.get("comparison",result)
    quality=comparison.get("quality_gate")
    proxy=comparison.get("proxy_goal_pass")
    if quality not in (True,False) or proxy not in (True,False):
        raise ValueError("missing Python-computed gate")
    numeric=target.get("relative_reduction")
    if numeric is None:
        target=comparison.get("target") or {}
        numeric=target.get("relative_reduction")
    return {"numeric_target_reduction":numeric,
      "reported_quality_gate":quality,"reported_proxy_gate":proxy,
      "quality_uncertainty":comparison.get("quality_uncertainty"),
      "independent_confirmation":False,
      "status":"development_screen_only",
      "source_of_truth":"Python metric record; narrative does not override values"}

def source_ref(ref_id:str,path:str,note:str)->EvidenceRef:
    p=Path(path).resolve()
    if not p.is_relative_to(SAFE_SOURCE_ROOT) or not p.is_file():raise ValueError("untrusted source")
    return EvidenceRef(ref_id,str(p),sha256(p.read_bytes()).hexdigest(),note)

def write_json_atomic(p:Path,value:dict)->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError("append-only/new file output required")
    tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    tmp.rename(p)
