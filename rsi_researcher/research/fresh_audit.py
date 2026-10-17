"""Independent post-run audit for fresh AgentCF structured hypothesis trial.

Only reads recorded frozen forecasts and Python evidence. Never grants Qwen
self-evaluation scientific authority or imports its free-text reflections.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from datetime import datetime,timezone
import json

from rsi_researcher.research import fresh_predictions as f
from rsi_researcher.runtime.client import atomic_json


def read(name):
    return json.loads((f.NEW/name).read_text(encoding="utf8"))


def stage_cost():
    items=[json.loads(v) for v in (f.NEW/"calls.jsonl").read_text(encoding="utf8").splitlines()]
    started={v["call_id"]:v for v in items if v["status"]=="started"}
    received={v["call_id"]:v for v in items if v["status"]=="received"}
    if started.keys()!=received.keys():raise ValueError("Incomplete model request ledger")
    d=defaultdict(lambda:{"calls":0,"prompt_tokens":0,"completion_tokens":0})
    for id,entry in started.items():
        phase=entry["context"].get("phase","unknown")
        key=phase
        usage=received[id]["usage"]
        d[key]["calls"]+=1
        d[key]["prompt_tokens"]+=usage["prompt_tokens"]
        d[key]["completion_tokens"]+=usage["completion_tokens"]
    return dict(d)


def audit():
    protocol=f.verify()
    status=read("status.json")
    result=read("comparison.json")
    frozen=read("forecasts_frozen_at.json")
    plans=read("predictions.json")
    scores=read("machine_scored_forecasts.json")
    measured=read("measured_numeric_evidence.json")
    if status["status"]!="COMPLETED" or result["status"]!="NEW_COHORT_STRUCTURED_FORECAST_COMPLETED":
        raise ValueError("Cannot promote an incomplete run")
    if frozen["hash_sha256"]!=f.digest(f.NEW/"predictions.json"):
        raise ValueError("Hypothesis predictions were edited after freezing")
    if result["protocol_sha256"]!=f.digest(f.NEW/"protocol.json"):
        raise ValueError("Source protocol mismatch")
    if not frozen["training_started"] is False:
        raise ValueError("Forecasts were not demonstrably frozen before training")
    checked=f.retrospective_evaluate(plans,measured)
    if checked!=scores:
        raise ValueError("Numerical forecast scoring has changed")
    stage=stage_cost()
    total={"calls":sum(x["calls"] for x in stage.values()),
           "prompt_tokens":sum(x["prompt_tokens"] for x in stage.values()),
           "completion_tokens":sum(x["completion_tokens"] for x in stage.values())}
    if total["calls"]!=status["model_requests"]:
        raise ValueError("Model request total differs from source records")
    if total["prompt_tokens"]!=result["overall_model_usage"]["prompt_tokens"]:
        raise ValueError("Token use does not reconcile")
    by_arm={}
    for arm in f.ARMS:
        rows=scores[arm]
        by_arm[arm]={
          "prediction_contracts":len(rows),
          "matching_direction_and_effect_and_quality_in_both_panels":
              sum(x["both_panels_pass"] for x in rows.values()),
          "independent_scientific_discoveries":0,
          "validated_hypotheses":0,
          "research_plan_and_reflection_calls":0 if arm=="rule" else 4}
    return {"status":"INDEPENDENT_NEW_STATE_FORECAST_AUDIT_COMPLETE",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "protocol_sha256":result["protocol_sha256"],
        "forecast_precedes_training":True,
        "disjoint_user_count":protocol["queries"],
        "by_arm":by_arm,"stage_cost":stage,"total_cost":total,
        "numeric_measured":measured,
        "machine_forecast_scores":scores,
        "reflection_text_promoted":0,
        "source_checked_cards_promoted_to_confirmed":0,
        "strict_independently_confirmed_scientific_discoveries":0,
        "limits":[
            "Six new author users, same training data source and same global item metadata",
            "Three queries per panel are underpowered for hypothesis confirmation",
            "Single trained item/user memory state; no second random-seed validation",
            "Known-target NDCG is not relevance completeness and has high variance",
            "One run cannot establish longitudinal RSI policy improvement",
            "LLM post-hoc reflections are unaudited text and never count as findings",
            "Forecast result count is metric/quality contract agreement, not scientific discovery count"]}


def main():
    report=audit()
    out=f.NEW/"INDEPENDENT_AUDIT.json"
    if out.exists():raise FileExistsError(out)
    with out.open("x",encoding="utf8") as file:
        json.dump(report,file,ensure_ascii=False,sort_keys=True,indent=2)
        file.write("\n")
    lines=["# 新用户 AgentCF 数值预测独立复核","",
       "该报告只复核预冻结的方向/效应幅度/质量约束，模型自评不参与判分。",
       "用户不与上轮 AgentCF 20 人 cohort 重叠，但来源数据集和机制任务已被研究过。",
       "","| 策略 | 数值预测数 | 两个面板都吻合的预测 | 独立科学发现 | 规划+复盘 Qwen 请求 |",
       "|---|---:|---:|---:|---:|"]
    for arm,v in report["by_arm"].items():
        lines.append(f"| {arm} | {v['prediction_contracts']} | "
         f"{v['matching_direction_and_effect_and_quality_in_both_panels']} | "
         f"{v['independent_scientific_discoveries']} | "
         f"{v['research_plan_and_reflection_calls']} |")
    lines+=["",
      f"实际完整模型请求：{report['total_cost']['calls']}；"
      f"输入 Tokens {report['total_cost']['prompt_tokens']}，"
      f"输出 Tokens {report['total_cost']['completion_tokens']}。", "",
      "## 各项实验测量"]
    for task,parts in report["numeric_measured"].items():
        lines.append("### "+task)
        for cohort,v in parts.items():
            lines.append(f"- {cohort}: excess rank change={v['excess_rank_change']:+.3f}; "
                         f"head excess Δ={v['head_excess_delta']:+.3f}; "
                         f"NDCG Δ={v['ndcg_delta']:+.3f}; "
                         f"repeat change={v['repeat_change_rate']:.3f}; quality gate={v['quality_pass']}")
    lines+=["","## 研究解释",
      "结构化假设预测是否符合观测必须与‘发现机制具有独立因果证据’分开计数。",
      "一个新用户状态、每组 3 个已知目标，尚不足以宣称真实科研发现。",
      "要证明跨任务 RSI，需预注册新的未见任务机制、训练状态与真实评估面板，并重复模型训练种子。",
      "新发现卡片只能按开发条件事实保留，不允许把 Qwen 反思自动提升为 verified。"]
    (f.NEW/"INDEPENDENT_AUDIT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    return report


if __name__=="__main__":
    x=main()
    print(json.dumps({"status":x["status"],"total_requests":x["total_cost"]["calls"],
                      "accepted_forecasts":{a:v["matching_direction_and_effect_and_quality_in_both_panels"]
                            for a,v in x["by_arm"].items()}},ensure_ascii=False))
