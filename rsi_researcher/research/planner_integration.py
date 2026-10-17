"""Safe integration boundary: V2 preflight before frozen AgentCF run_arm.

Execution requires separately supplied approval and a trusted executor callback.
The default CLI below has NO path that launches Qwen, Python subprocesses or GPU.
"""
import ast
from dataclasses import asdict
from datetime import datetime,timezone
from hashlib import sha256
import hmac
import os
import json
from pathlib import Path
import sys
from typing import Callable

BASE=Path("/data/cjh/rsi")
V2=BASE/"outputs/research-planner-v2-20261016"
from rsi_researcher.research.planner import ResearchProposal, ResearchSituation, default_policy, review, source_ref
from rsi_researcher.research.planner_adapters import load_real_situations
from rsi_researcher.research.planner_tools import run_registered_cpu_action

FROZEN=BASE/"project/releases/20261016-five-bias-validation"
FROZEN_OPERATORS=FROZEN/"rsi_researcher/research/author_experiment.py"
DEFAULT_DEV_BASELINE=BASE/"project/runs/qwen-five-bias-verification-20261016/baseline"
CPU_ACTIONS=frozenset({
 "annotate_claim_evidence","diagnose_replay_coverage","audit_order_noise_control",
 "audit_common_support","audit_confirmation_isolation","correct_reflection_from_numbers",
 "collect_exposure_logging_evidence","abstain_and_measure"
})
TRAIN_ACTIONS=frozenset({"propose_bounded_intervention","run_independent_replication"})

def _hash(p):
    return sha256(Path(p).read_bytes()).hexdigest()

def source_operators():
    """Read the actual, frozen Python literal registry without importing/run side effects."""
    tree=ast.parse(FROZEN_OPERATORS.read_text(encoding="utf8"),filename=str(FROZEN_OPERATORS))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id=="AUTHOR_OPERATORS" for t in node.targets):
            obj=ast.literal_eval(node.value)
            assert isinstance(obj,dict) and obj
            return obj
    raise ValueError("Frozen AUTHOR_OPERATORS not found")

def prepare_trial(ctx,proposal,baseline=DEFAULT_DEV_BASELINE,config=None):
    """No execution: validate provenance and build a traceable proposal only."""
    verdict=review(ctx,proposal)
    if verdict["status"]!="DESIGN_ACCEPTED":
        return {"status":"REJECTED_BEFORE_RUN_ARM","review":verdict,"run_arm_invocations":0}
    if proposal.action not in TRAIN_ACTIONS:
        return {"status":"NOT_A_MODEL_RUN_ACTION","review":verdict,"run_arm_invocations":0}
    operator=proposal.parameters.get("operator") if isinstance(proposal.parameters,dict) else None
    registry=source_operators()
    if operator not in registry or operator=="retain":
        return {"status":"REJECTED_UNREGISTERED_FROZEN_OPERATOR","review":verdict,
                "run_arm_invocations":0}
    baseline=Path(baseline).resolve()
    if not baseline.is_relative_to(BASE) or not (baseline/"input.json").is_file():
        raise ValueError("Missing owned baseline inputs")
    inp=json.loads((baseline/"input.json").read_text(encoding="utf8"))
    if inp.get("operator","retain")!="retain":
        raise ValueError("Baseline must be the unmodified operator")
    if config is not None:
        cfgp=Path(config).resolve()
        if not cfgp.is_relative_to(BASE) or not cfgp.is_file():raise ValueError("Untrusted config")
        cfg=json.loads(cfgp.read_text(encoding="utf8"))
        for k in ("seed","users","events_per_user"):
            if cfg.get(k)!=inp.get(k):raise ValueError("candidate/baseline mismatch: "+k)
        cfg_sha=_hash(cfgp)
    else:cfg_sha=None
    return {"status":"APPROVED_DESIGN_ONLY","review":verdict,
       "run_arm_invocations":0,"operator":operator,
       "registered_description":registry[operator],
       "frozen_operator_registry_hash":_hash(FROZEN_OPERATORS),
       "baseline_input_sha256":_hash(baseline/"input.json"),
       "config_sha256":cfg_sha,
       "baseline_dir":str(baseline),
       "source_release":str(FROZEN),
       "source_release_kind":"frozen_20261016_five_bias",
       "approval_required":"external budget, private cohort check, model version and GPU allocation",
       "execution_authorized":False,
       "claim_level":"design_only_not_debiasing"}

def release_run_arm_gate(ctx,proposal,executor:Callable,
                         approval:dict|None=None,**prepare_kw):
    """Actual integration point for a NEW future runner, not an executable CLI.

    Approval is an explicit externally created, trial-specific certificate. The
    simulator cannot grant its own approval; it never imports the frozen run_arm.
    """
    design=prepare_trial(ctx,proposal,**prepare_kw)
    if design.get("status")!="APPROVED_DESIGN_ONLY":
        return {"status":"BLOCKED","reason":design,"called_executor":False}
    approved=(isinstance(approval,dict) and approval.get("approved") is True
      and approval.get("operator")==design["operator"]
      and approval.get("baseline_input_sha256")==design["baseline_input_sha256"]
      and approval.get("registry_sha256")==design["frozen_operator_registry_hash"]
      and approval.get("execution_budget_verified") is True
      and approval.get("independent_cohort_gate_verified") is True
      and approval.get("shared_memory_isolation_verified") is True
      and approval.get("model_version_locked") is True)
    # A plain JSON boolean is NOT authorization. Require a separate HMAC
    # generated outside the LLM-visible project using a deployment secret.
    # The CLI does not offer any way to mint this signature.
    secret=os.environ.get("RSI_EXTERNAL_APPROVAL_KEY","")
    signed_fields={k:v for k,v in (approval or {}).items() if k!="signature"} if isinstance(approval,dict) else {}
    payload=json.dumps(signed_fields,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf8")
    signed=(len(secret)>=32 and isinstance(approval,dict) and
      isinstance(approval.get("signature"),str) and
      hmac.compare_digest(hmac.new(secret.encode("utf8"),payload,"sha256").hexdigest(),approval["signature"]))
    if not (approved and signed):
        return {"status":"BLOCKED_AWAITING_SIGNED_EXTERNAL_APPROVAL","called_executor":False,
                "prepared_design":design}
    return {"status":"HANDOFF_TO_REGISTERED_EXECUTOR","called_executor":True,
        "result":executor(design["operator"],design)}

def execute_read_only(ctx,proposal):
    verdict=review(ctx,proposal)
    if verdict["status"]!="DESIGN_ACCEPTED":
        return {"status":"BLOCKED","review":verdict}
    if proposal.action in TRAIN_ACTIONS:
        return prepare_trial(ctx,proposal)
    if proposal.action not in CPU_ACTIONS:
        return {"status":"BLOCKED_UNREGISTERED_TOOL"}
    return {"status":"CPU_COMPLETED","review":verdict,
            "result":run_registered_cpu_action(ctx,proposal.action,True)}

def minimal_proposal(ctx,action,operator=None):
    return ResearchProposal(
        action=action,
        hypothesis="Available evidence might reflect a remediable mechanism.",
        competing_hypothesis="Measurement artefacts, confounding, or stochastic memory updates explain the pattern.",
        falsifier="A matched control or independent reference contradicts the proposed mechanism.",
        rationale="Select the lowest-cost action whose evidence preconditions are demonstrably met.",
        evidence_ids=list(ctx.evidence),
        expected_outcome="Either obtain a clearer evidence state or record that the question is unresolved.",
        budget_reason="Design and read-only measurements; training is separately authorized.",
        parameters={"operator":operator} if operator else {})

def run_dry_integration(out_dir):
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=False)
    contexts=load_real_situations()
    records=[]
    for ctx in contexts:
        action=default_policy(ctx)["next_action"]
        proposal=minimal_proposal(ctx,action)
        result=execute_read_only(ctx,proposal)
        records.append({"context":ctx.situation_id,"action":action,
            "result":result,"evidence_files_sha256":{k:v.sha256 for k,v in ctx.evidence.items()}})
    # Demonstrate that the proposed experiment isn't allowed to bypass the reviewer,
    # even if an LLM emits a valid operator name.
    ctx=contexts[0]
    blocked=release_run_arm_gate(ctx,minimal_proposal(ctx,"propose_bounded_intervention",
                "unseen_neutral_user_reflection"),
                executor=lambda *_: (_ for _ in ()).throw(AssertionError("executor called")))
    payload={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
      "status":"DRY_INTEGRATION_COMPLETE",
      "reviewed_real_development_situations":len(contexts),
      "rows":records,"attempted_frozen_training_handoff":blocked,
      "model_calls":0,"training_calls":0,"test_labels_read":False,
      "release_modified":False,"integration_scope":"new versioned caller only"}
    (out_dir/"run.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":payload["status"],"situations":len(contexts),
      "training_handoff":blocked["status"],"gpu_requests":0},ensure_ascii=False))
    return payload

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    run_dry_integration(args.out)
