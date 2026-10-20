"""Bounded evidence-guided search over the EXISTING scientific episode executor.

Only this scheduler owns tree topology and frontier. Plans, executions,
audits and reflections remain node-local artifacts of scientific_loop.episode.
All outcomes are development or archive replay, never confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsi_researcher.research import crossbias_discovery as source
from rsi_researcher.research import scientific_experiments as experiments
from rsi_researcher.research.scientific_agents import DiagnosisAgent, DesignerAgent
from rsi_researcher.research.scientific_loop import (
    digest, episode, public_evidence, review_plan, store_new, utc,
    verified_experience,
)
from rsi_researcher.runtime.client import atomic_json, usage_summary
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.models import create_model, model_description


def search(output, task, *, model=None, config=None, mode="fresh",
           arm="qwen_no_history", cohort_size=12, seed=20261017,
           width=2, max_depth=2, max_nodes=4, max_proposals=6,
           max_requests=180, prior_campaign=None):
    """Build competing branches, execute selected nodes, and backtrack.

    Parentage records hypothesis refinement, NEVER a mutable model-state
    dependency. AgentCF siblings always reload one frozen trained state.
    max_nodes caps attempted node executions; runtime limits cap real LLM usage.
    """
    output = Path(output).resolve()
    if (output.exists() or mode not in ("fresh", "archive") or
        arm not in ("qwen_no_history", "qwen_audited_history",
                    "qwen_shuffled_history") or
        not 1 <= width <= 3 or not 1 <= max_depth <= 3 or
        not 1 <= max_nodes <= 8 or not 2 <= max_proposals <= 12 or
        max_requests < 1 or
        cohort_size not in (4, 6, 8, 10, 12)):
        raise ValueError("invalid, unbounded or existing scientific tree")
    spec = experiments.task_spec(task)
    if mode == "archive" and task.startswith("agentcf_"):
        allowed = [a for a in spec["actions"]
                   if a in ("item_memory_ablation", "popularity_prompt")]
    else:
        allowed = list(spec["actions"])
    if not allowed:
        raise ValueError("no executable experimental action")
    if model is None and (config is None or
                          config.get("llm", {}).get("provider") != "vllm"):
        raise ValueError("explicit authorized Qwen vLLM configuration required")
    source.verify()
    experience = (verified_experience(output, prior_campaign)
                  if prior_campaign else [])
    output.mkdir(parents=True)
    (output / "nodes").mkdir()
    (output / "plans").mkdir()
    if model is None:
        limits = RunLimits(output, **config["budget"])
        model = create_model(output, limits.max_requests, limits, config)

    protocol = {
        "version": "bounded-scientific-tree-development-v1",
        "task":task, "mode":mode, "arm":arm, "width":width,
        "max_depth":max_depth, "max_executed_nodes":max_nodes,
        "max_generated_proposals":max_proposals,
        "max_model_requests_advisory":max_requests,
        "cohort_size":cohort_size, "seed":seed,
        "independent_confirmation_used":False,
        "parent_id_is_hypothesis_lineage_not_mutable_state":True,
        "source_protocol_sha256":digest(source.RUN/"protocol.json"),
        "source_sha256":{
            name:digest(Path(__file__).with_name(name))
            for name in ("scientific_tree.py","scientific_loop.py",
                         "scientific_agents.py","scientific_experiments.py")},
        "predeclared_actions":allowed,
        "model":model_description(config) if config else {"kind":"offline_test_double"},
        "state_reused_across_all_nodes":task.startswith("agentcf_") and mode=="fresh",
        "prior_campaign":str(prior_campaign) if prior_campaign else None,
    }
    store_new(output/"protocol.json", protocol)
    store_new(output/"protocol.sha256.json",{"sha256":digest(output/"protocol.json")})

    prepared_path = None
    prepared = None
    if task.startswith("agentcf_") and mode=="fresh":
        shared = output/"shared_states"/"state-001"
        shared.mkdir(parents=True)
        cohort = experiments.prepare_cohort(set(), seed, cohort_size)
        prepared = experiments.train_and_observe_agentcf(cohort, model, shared)
        prepared_path = shared/"prepared_agentcf.json"

    context = public_evidence(task, experience, arm)
    context["tree_search_scope"] = (
        "prospective DEVELOPMENT experiment search, no confirmatory panel; "
        "one fixed trained state across branches; numeric audits from already "
        "evaluated ancestor nodes are development feedback only")
    if prepared is not None:
        context["new_agentcf_input_observations"] = prepared["observation"]
        context["predecision_baseline_fingerprint"] = (
            prepared["training"]["baseline_rankings_sha256"])
        context["shared_preparation_cost_accounted_separately"] = True

    tree = {"status":"RUNNING","root":"ROOT","task":task,
            "state_ref":(str(prepared_path.relative_to(output))
                         if prepared_path else None),
            "training_sha256":(prepared["training"]["trained_state_sha256"]
                               if prepared else None),
            "nodes":{}, "frontier":[], "selection_history":[],
            "confirmed_discoveries":0}
    path = output/"tree_index.json"
    log = output/"search_decisions.jsonl"
    def save():
        atomic_json(path,tree)

    def record(kind, **fields):
        with log.open("a",encoding="utf8") as stream:
            stream.write(json.dumps({"event":kind,**fields},ensure_ascii=False,
                                    sort_keys=True)+"\n")

    def eligible_to_propose():
        return len(tree["nodes"]) < max_proposals and model.requests < max_requests

    def known_actions():
        return {n["action"] for n in tree["nodes"].values()
                if n.get("action")!="abstain"}

    def expand(parent_id, depth, parent_evidence, how_many):
        # No validity/quality "reward" can silently change the independent
        # evaluator. A parent audit can only guide the NEXT search question.
        if not eligible_to_propose():
            return
        diagnosis = DiagnosisAgent(model).act(
            {"observations":context,
             "completed_parent_development_audit":parent_evidence,
             "search_depth":depth,
             "unused_legal_actions":[a for a in allowed
                                     if a not in known_actions()]},
            f"{parent_id}-expansion")
        for _ in range(how_many):
            available=[a for a in allowed if a not in known_actions()]
            if not available or not eligible_to_propose():
                break
            proposal = DesignerAgent(model).propose(
                {"evidence":context,"diagnosis":diagnosis,
                 "completed_parent_development_audit":parent_evidence,
                 "parent_id":parent_id, "depth":depth,
                 "already_registered_branches":sorted(known_actions()),
                 "constraint":"choose a DISTINCT legal discriminating action; "
                              "no intervention stacking, never assume a result"},
                f"{parent_id}-child-{len(tree['nodes'])+1:03d}",
                available, experiments.task_metrics(task))
            if proposal["action"]=="abstain":
                record("designer_abstained_expansion",parent_id=parent_id,
                       explanation=proposal["budget_justification"])
                break
            node_id=f"N{len(tree['nodes'])+1:03d}"
            proposed_path=output/"plans"/f"{node_id}.json"
            store_new(proposed_path,{
                "parent_id":parent_id,"task":task,
                "proposal":proposal,"diagnosis":diagnosis,
                "parent_development_audit":parent_evidence,
                "frozen_at":utc()})
            verdict=review_plan(task,proposal,{"decision":"approve"})
            node={"node_id":node_id,"parent_id":parent_id,"depth":depth,
                  "task":task,"action":proposal["action"],
                  "metric":proposal["metric"],"status":"FRONTIER",
                  "state_ref":tree["state_ref"],
                  "plan_ref":str(proposed_path.relative_to(output)),
                  "audit_ref":None,"admission":verdict}
            if verdict!="ACCEPTED_DESIGN_ONLY":
                node["status"]="REJECTED_CONTRACT_PRE_EXECUTION"
            else:
                tree["frontier"].append(node_id)
            tree["nodes"][node_id]=node
            record("branch_proposed",node_id=node_id,parent_id=parent_id,
                   action=proposal["action"],admission=verdict)
            save()

    def parent_brief(node):
        node_dir=output/"nodes"/node["node_id"]
        audit_file=node_dir/"audit.json"
        if not audit_file.is_file():
            return None
        audit=json.loads(audit_file.read_text(encoding="utf8"))
        # Development numeric evidence is visible only AFTER it exists.
        return {"parent_id":node["node_id"],
                "action":audit["action"],"metric":audit["forecast_metric"],
                "mechanism_forecast_state":audit.get("mechanism_forecast_state"),
                "panel_measurements":audit["panel_measurements"],
                "known_target_quality_guard":
                    audit.get("known_target_quality_guard_passes_both_panels"),
                "evidence_grade":audit["evidence_grade"],
                "independently_confirmed_discoveries":0}

    def priority(node_id):
        n=tree["nodes"][node_id]
        ancestor=tree["nodes"].get(n["parent_id"])
        uncertain=(ancestor and ancestor.get("mechanism_forecast_state") in (
            "PANEL_HETEROGENEITY_OR_NOISE","PREDICTION_NOT_SUPPORTED",
            "EQUIVALENCE_UNIDENTIFIED"))
        # Inspect anomalies first; never rank candidates by observed NDCG
        # or count re-used development results as discoveries.
        return (0 if uncertain else 1, n["depth"],
                int(node_id[1:]))

    try:
        expand("ROOT",1,None,width)
        attempted=0
        last_selected=None
        while tree["frontier"] and attempted < max_nodes and model.requests < max_requests:
            node_id=min(tree["frontier"],key=priority)
            tree["frontier"].remove(node_id)
            node=tree["nodes"][node_id]
            backtrack=bool(last_selected and node["parent_id"]!=last_selected)
            reason=("unresolved_parent_falsification" if priority(node_id)[0]==0
                    else "bounded_best_first_untried_branch")
            tree["selection_history"].append({
                "node_id":node_id,"parent_id":node["parent_id"],
                "reason":reason,"backtracked_from":last_selected if backtrack else None})
            record("select_frontier",node_id=node_id,reason=reason,
                   backtracked_from=last_selected if backtrack else None)
            last_selected=node_id
            attempted+=1
            node["status"]="EXECUTING"
            save()
            frozen=json.loads((output/node["plan_ref"]).read_text(encoding="utf8"))
            parent=frozen["parent_development_audit"]
            try:
                status=episode(output/"nodes"/node_id,task,model,mode,set(),seed,
                               cohort_size,arm=arm,prior_campaign=prior_campaign,
                               prepared_state=prepared_path,
                               tree_proposal=frozen["proposal"],
                               tree_diagnosis=frozen["diagnosis"],
                               tree_parent=parent)
            except Exception as exc:
                node["status"]="FAILED_CLOSED"
                node["error_type"]=type(exc).__name__
                tree["status"]="FAILED_CLOSED"
                record("node_failed",node_id=node_id,
                       error_type=type(exc).__name__)
                save()
                raise
            node["status"]=status["status"]
            node["actual_model_requests"]=status.get("model_calls",0)
            execution_path=output/"nodes"/node_id/"execution.json"
            if execution_path.is_file():
                execution=json.loads(execution_path.read_text(encoding="utf8"))
                node["execution_kind"]=execution.get("execution_kind",
                                                    execution.get("status"))
            audit_path=output/"nodes"/node_id/"audit.json"
            if audit_path.is_file():
                node["audit_ref"]=str(audit_path.relative_to(output))
                audit=json.loads(audit_path.read_text(encoding="utf8"))
                node["evidence_grade"]=audit["evidence_grade"]
                node["mechanism_forecast_state"]=audit.get("mechanism_forecast_state")
                if task.startswith("agentcf_") and mode=="fresh":
                    training=json.loads((audit_path.parent/"agentcf_training.json").read_text())
                    if training["trained_state_sha256"]!=tree["training_sha256"]:
                        raise ValueError("tree node did not use the frozen training state")
            record("node_evaluated",node_id=node_id,status=node["status"],
                   audit_ref=node["audit_ref"])
            save()
            if (node["audit_ref"] and node["depth"]<max_depth and
                mode=="fresh" and eligible_to_propose()):
                expand(node_id,node["depth"]+1,parent_brief(node),1)

        completed={"COMPLETED_DEVELOPMENT","COMPLETED_INCONCLUSIVE",
                   "COMPLETED_RULE_CONTROL","COMPLETED_RULE_INCONCLUSIVE"}
        executed=sum(n["status"] in completed and
                     n.get("execution_kind")!="ARCHIVED_DEVELOPMENT_REPLAY"
                     for n in tree["nodes"].values())
        replayed=sum(n.get("execution_kind")=="ARCHIVED_DEVELOPMENT_REPLAY"
                     for n in tree["nodes"].values())
        tree["status"]=("COMPLETED_ARCHIVE_REPLAY_SEARCH" if mode=="archive"
                        and replayed else "COMPLETED_DEVELOPMENT_SEARCH"
                        if executed else "COMPLETED_NO_EXECUTION")
        tree["termination_reason"]=(
            "max_executed_nodes" if attempted==max_nodes else
            "request_ceiling" if model.requests>=max_requests else
            "frontier_exhausted")
        tree["attempted_nodes"]=attempted
        tree["executed_development_nodes"]=executed
        tree["archived_replay_nodes"]=replayed
        tree["total_model_requests"]=model.requests
        tree["unselected_frontier_nodes"]=list(tree["frontier"])
        tree["model_usage"]=(usage_summary(output)
             if (output/"calls.jsonl").is_file() else
             {"requests":model.requests,"usage_measured":False,
              "reason":"offline test model, not Qwen usage"})
        tree["independently_confirmed_discoveries"]=0
        tree["search_is_not_confirmatory_evaluation"]=True
        record("search_terminated",reason=tree["termination_reason"],
               remaining_frontier=tree["frontier"])
    except Exception:
        tree["status"]="FAILED_CLOSED"
        save()
        raise
    save()
    atomic_json(output/"campaign.json",{
        "status":tree["status"],
        "executed_episodes":tree.get("executed_development_nodes",0),
        "independently_confirmed_discoveries":0,
        "no_claim_of_rsi_self_improvement":True})
    return tree


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--task",required=True,choices=sorted(source.CATEGORIES))
    parser.add_argument("--mode",choices=["fresh","archive"],default="fresh")
    parser.add_argument("--arm",choices=["qwen_no_history","qwen_audited_history",
                                        "qwen_shuffled_history"],default="qwen_no_history")
    parser.add_argument("--cohort-size",type=int,default=12)
    parser.add_argument("--seed",type=int,default=20261017)
    parser.add_argument("--width",type=int,default=2)
    parser.add_argument("--max-depth",type=int,default=2)
    parser.add_argument("--max-nodes",type=int,default=4)
    parser.add_argument("--max-proposals",type=int,default=6)
    parser.add_argument("--max-requests",type=int,default=180)
    parser.add_argument("--prior-campaign")
    args=parser.parse_args(argv)
    config=json.loads(Path(args.config).read_text(encoding="utf8"))
    if config.get("llm",{}).get("provider")!="vllm":
        parser.error("Qwen vLLM required, never fall back to an external API")
    # No launching or modifying an existing shared GPU service.
    health=create_model(args.output,config["budget"]["max_requests"],None,config)
    health.health()
    result=search(args.output,args.task,config=config,mode=args.mode,arm=args.arm,
                  cohort_size=args.cohort_size,seed=args.seed,width=args.width,
                  max_depth=args.max_depth,max_nodes=args.max_nodes,
                  max_proposals=args.max_proposals,max_requests=args.max_requests,
                  prior_campaign=args.prior_campaign)
    print(json.dumps({"status":result["status"],
                      "attempted_nodes":result["attempted_nodes"],
                      "independent_discoveries":0},ensure_ascii=False))


if __name__=="__main__":
    main()
