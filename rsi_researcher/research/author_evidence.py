"""Author-data evidence cards shared across runs, plus retrospective review.

Uses the existing experiment journal. Old absolute scores, complete memories
and user IDs are not copied into the next researcher's historical context.
"""
import json
from pathlib import Path

from rsi_researcher.runtime.client import atomic_json
from rsi_researcher.research.journal import ExperimentJournal
from rsi_researcher.evaluation.author_patterns import (
    describe_run, render_patterns, pattern_brief, compare_saved_quality,
)


def trial_conditions(inputs):
    return {"dataset":"CDs-100-user-dense", "users":inputs["users"],
            "events_per_user":inputs["events_per_user"], "seed":inputs["seed"],
            "model":inputs["model_config"].get("model_parameters",{}).get("model"),
            "author_commit":inputs["author_commit"]}


def record_author_trial(journal, trial_dir, plan, comparison, learn, conditions, origin="prospective_development"):
    return journal.record_trial(trial_dir, {
        "workflow":"agentcf_author_bias", "dataset":"CDs-100-user-dense",
        "proposal":plan, "comparison":comparison, "model_lesson":learn.get("lesson"),
        "conditions":conditions, "origin":origin,
        "evidence_level":"development_only_not_independent_confirmation",
    })


def compact_author_history(journal, limit=5):
    entries = [r for r in journal.research_history() if r.get("workflow")=="agentcf_author_bias"]
    latest_by_action={}
    for entry in entries:
        latest_by_action[entry["proposal"]["operator"]]=entry
    result = []
    for entry in list(latest_by_action.values())[-limit:] if limit > 0 else []:
        c = entry["comparison"]
        quality = c.get("quality_uncertainty",{})
        qchange = quality.get("relative_change")
        reduction = (c.get("target") or {}).get("relative_reduction")
        related=[r for r in entries if r["proposal"]["operator"]==entry["proposal"]["operator"]]
        feedback=[]
        for previous in related:
            for validation in previous.get("validation_feedback",[]):
                measured=validation.get("comparison",{})
                feedback.append({"candidate_conditions":previous["conditions"],
                                 "status":validation.get("status","not recorded"),
                                 "quality_gate":measured.get("quality_gate"),
                                 "target_gate":measured.get("target_gate"),
                                 "quality_evidence":measured.get("quality_uncertainty",{}).get("quality_evidence","NOT_EVALUATED")})
        result.append({
            "candidate_id":entry["candidate_id"], "operator":entry["proposal"]["operator"],
            "conditions":entry["conditions"], "origin":entry["origin"],
            "target_screen":"PASS" if c["target_gate"] else "FAIL_OR_NOT_MEASURED",
            "quality_screen":"PASS" if c["quality_gate"] else "FAIL",
            "quality_change_label":f"{qchange:+.1%}" if qchange is not None else "not comparable",
            "target_reduction_label":f"{reduction:.1%}" if reduction is not None else "not evaluable",
            "quality_evidence":quality.get("quality_evidence","NOT_EVALUATED"),
            "observed_conditions":len(related),
            "external_validation_records":len(feedback),
            "recent_validation_feedback":feedback[-2:],
            "evidence_level":entry["evidence_level"],
            "interpretation":"Historical condition-specific observation. Different cohort/seed is not a current metric; counts of validation records do not mean they passed.",
        })
    return result


def review_campaign(campaign_dir, baseline_dir, output_dir, journal_path):
    """Reanalyze completed trials without invoking or restarting a model."""
    from rsi_researcher.research.agentcf_loop import bias_summary, compare_bias
    read = lambda p:json.loads(Path(p).read_text(encoding="utf-8"))
    campaign_dir,baseline_dir,output_dir = map(Path,(campaign_dir,baseline_dir,output_dir))
    archive=read(campaign_dir/"archive.json")
    if archive.get("status")!="completed":
        raise ValueError("retrospective review requires a completed campaign")
    inputs=read(baseline_dir/"input.json")
    baseline=bias_summary(baseline_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    journal=ExperimentJournal(journal_path)
    patterns=describe_run(baseline_dir)
    atomic_json(output_dir/"baseline-patterns.json",patterns)
    (output_dir/"baseline-patterns.md").write_text(render_patterns(patterns),encoding="utf-8")
    result={"status":"completed", "model_calls":0, "origin":"retrospective_diagnostic_review",
            "baseline":str(baseline_dir.resolve()), "trials":[]}
    for trial in archive["trials"]:
        if trial.get("execution_status")!="completed":
            continue
        source=campaign_dir/f"trial-{trial['round']:04d}"
        report=describe_run(source)
        atomic_json(output_dir/f"patterns-{trial['round']:04d}.json",report)
        (output_dir/f"patterns-{trial['round']:04d}.md").write_text(render_patterns(report),encoding="utf-8")
        comparison=compare_bias(baseline,bias_summary(source),trial["plan"]["targeted_bias"])
        comparison["quality_uncertainty"]=compare_saved_quality(baseline_dir,source)
        identity=record_author_trial(journal,source,trial["plan"],comparison,trial.get("learn",{}),
                                     trial_conditions(inputs),origin="retrospective_diagnostic_review")
        result["trials"].append({"round":trial["round"], "operator":trial["operator"],
                                "comparison":comparison, "candidate_id":identity,
                                "pattern_brief":pattern_brief(report)})
    result["next_researcher_history"]=compact_author_history(journal)
    atomic_json(output_dir/"review.json",result)
    lines=["# 作者数据：偏差模式与已完成实验复查", "", "没有模型调用；这是事后诊断，不是新实验或独立确认。", "",
           "| 动作 | 目标代理门槛 | 质量均值门槛 | 质量证据 |", "|---|---|---|---|"]
    for t in result["trials"]:
        c=t["comparison"]
        lines.append(f"| {t['operator']} | {c['target_gate']} | {c['quality_gate']} | {c['quality_uncertainty']['quality_evidence']} |")
    lines += ["", "模式原文见 baseline-patterns.md 和 patterns-000*.md。跨任务历史只传条件、相对变化和程序标签，不把旧分数当成当前分数。",
              "位置候选没有同序重复/倒序对照时，不通过位置机制门槛。各组样本量和记忆读写路径一并报告。"]
    (output_dir/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",required=True)
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--journal",required=True)
    args=parser.parse_args()
    result=review_campaign(args.campaign,args.baseline,args.output,args.journal)
    print(json.dumps({"status":result["status"],"model_calls":0,"reviewed_trials":len(result["trials"])}))
