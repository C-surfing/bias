"""Read-only synthesis of completed, frozen Qwen synthetic research pilot.

Cluster resamples by mechanism family (7 clusters, 2 variants each). With so
few families intervals are exploratory, not frequentist proof of RSI.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
from random import Random
from statistics import mean
from hashlib import sha256
import json

from rsi_researcher.research.decision_benchmark import ROOT,MODES,read_frozen
from rsi_researcher.research.decision_qwen_pilot import ARMS,RUN,DIR


def mean_diff_ci(a,b,seed=20261017,draws=2000):
    if set(a)!=set(b):raise ValueError("Mismatched task identifiers")
    by_mode=defaultdict(list)
    for tid in a:
        mode=next((m for m in MODES if "-"+m+"-" in tid),None)
        if mode is None:raise ValueError("Unrecognized task mechanism")
        by_mode[mode].append(a[tid]-b[tid])
    if set(by_mode)!=set(MODES) or any(len(rows)!=2 for rows in by_mode.values()):
        raise ValueError("Expected exactly 2 task variants per mechanism")
    vals=[mean(by_mode[m]) for m in MODES]
    observed=mean(vals)
    rng=Random(seed)
    bootstrap=sorted(mean(vals[rng.randrange(len(vals))] for _ in vals)
                     for _ in range(draws))
    return {"mean_excess_regret":observed,
            "cluster_bootstrap_95": [bootstrap[round(0.025*(draws-1))],
                                     bootstrap[round(0.975*(draws-1))]],
            "mechanism_clusters":len(MODES),"draws":draws,
            "limitation":"Seven fixed mechanism families; resampling gives exploratory uncertainty only"}


def main():
    p=read_frozen(RUN)
    status=json.loads((DIR/"status.json").read_text(encoding="utf8"))
    results=json.loads((DIR/"comparison.json").read_text(encoding="utf8"))
    decisions=json.loads((DIR/"model_decisions.json").read_text(encoding="utf8"))
    if status["status"]!="COMPLETED" or results["status"]!="MODEL_PILOT_COMPLETE":
        raise ValueError("Model pilot not complete")
    if len(decisions)!=4*len(p["evaluation_tasks"]):
        raise ValueError("Missing Qwen decision records")
    digest=(RUN/"protocol.sha256").read_text().strip()
    if digest!=results["frozen_protocol_sha256"]:
        raise ValueError("Scoring protocol differs from the original freeze")
    scored=results["scores"]
    mapping={arm:{r["task_id"]:r["regret"] for r in v["rows"]}
             for arm,v in scored.items()}
    comparison={}
    for arm in ARMS:
        comparison[arm+"_vs_fixed"]=mean_diff_ci(mapping[arm],mapping["fixed_rule"])
    comparison["related_vs_shuffled"]=mean_diff_ci(mapping["related_history"],mapping["shuffled_history"])
    comparison["revised_vs_no_history"]=mean_diff_ci(mapping["strategy_revision"],mapping["no_history"])
    decisions_by_arm={}
    for arm in ARMS:
        group=[d for d in decisions if d["arm"]==arm]
        decisions_by_arm[arm]={
           "actions":dict(Counter(d["chosen_action"] for d in group)),
           "safety_gate_overrides":sum(bool(d["validation_rejection"]) for d in group),
           "rejected_reasons":dict(Counter(d["validation_rejection"]
                               for d in group if d["validation_rejection"])),
           "no_history_on_eval_feedback":True
        }
    out={"status":"INDEPENDENT_RESEARCH_REVIEW_COMPLETE",
       "synthetic_only":True,"protocol_sha256":digest,
       "model_call_count":status["requests_sent"],
       "calibration_calls_included":True,
       "gpu_memory_released":status["gpu_memory_released"],
       "comparison_mechanism_cluster_ci":comparison,
       "policy_decisions":decisions_by_arm,
       "token_usage":results["usage"],
       "scored_means":{k:{"mean_regret":v["mean_regret"],
                           "action_agreement":v["action_agreement"],
                           "invalid_causal_decisions":v["invalid_causal_decisions"],
                           "unnecessary_abstentions":v["unnecessary_abstentions"]}
                        for k,v in scored.items()},
       "interpretation":"No Qwen method beat the fixed-rule baseline in this pilot; history and revision not demonstrated helpful",
       "constraints":[
           "14 held-out configurations drawn from the same seven mechanism families as seven development tasks",
           "Empirical action oracle is 24 simulation-repetition risk minimum, not analytic Bayes action",
           "All Qwen arms have 15 calls, but not equal token usage",
           "Rule baseline uses no Qwen calls, so equal-call comparison is only among Qwen arms",
           "One ordered pilot on one local Qwen model; uncertainty intervals cannot demonstrate statistical significance",
           "Gate-filtered invalid decisions are not evidence that model always recognized missing assumptions"
       ]}
    path=DIR/"ANALYSIS.json"
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    lines=["# Qwen3-8B / Research Memory：冻结合成科研决策实验复盘","",
           f"冻结协议 SHA-256: {digest}","",
           "仅 7 种机制、每种 2 个不同配置；模型未见评价真值或后续评分。模型各组调用 15 次，总计 60 次。",
           "",
           "| 策略 | 模型调用 | Prompt tokens | 平均 regret（越低越好） | 相对规则的成对额外 regret | 机制聚类 bootstrap 95% 区间 |",
           "|---|---:|---:|---:|---:|---|"]
    names=[("fixed_rule","固定规则"),("no_history","Qwen 无历史"),
           ("related_history","Qwen 相关历史"),("strategy_revision","Qwen 策略修订"),
           ("shuffled_history","Qwen 打乱历史")]
    for arm,name in names:
        v=out["scored_means"][arm]
        usage=out["token_usage"].get(arm,{})
        if arm=="fixed_rule":extra=0;interval="—"
        else:
            d=comparison[arm+"_vs_fixed"]
            extra=d["mean_excess_regret"]
            interval="["+", ".join(f"{v:.6f}" for v in d["cluster_bootstrap_95"])+"]"
        lines.append(f"| {name} | {usage.get('calls',0)} | {usage.get('prompt_tokens',0)} | "
                     f"{v['mean_regret']:.6f} | {extra:+.6f} | {interval} |")
    lines+=["","## 研究解释",
            "- 固定规则是当前最优方法；本次实验未显示 LLM、相关历史或策略修订收益。",
            "- 打乱历史引发更多不必要弃权，说明检索质量可能影响科研动作，但不能用这一次验证因果。",
            "- 风险和 best action 来自保留仿真种子的 24 次 Monte Carlo；并不是实际用户的去偏真值。",
            "- 机制 bootstrap 只有 7 类，统计力有限。不能用其范围当作正式显著性检验。",
            "- 所有 Qwen 组请求数相同但 prompt tokens 不同，应同时呈现成本。",
            "",
            "## 下一轮预注册建议",
            "1. 不再调这批 14 道题的提示词或选择赢家；把它们作为已使用过的开发结果。",
            "2. 新开独立机制/种子/任务类型；随机化测试题和 Arm 执行顺序以防顺序效应。",
            "3. 用更细粒度的检索充分性/错误记忆匹配检测和停用记忆基线，隔离错误弃权机制。",
            "4. 比较研究策略修订后在未见任务上的**发现数量/成本**，而不只是 OPE 动作选择。",
            "5. 将 AgentCF 真实偏差机制与独立人工真值加入外部测试集，保留失败结果。"]
    report=DIR/"ANALYSIS.md"
    report.write_text("\n".join(lines)+"\n",encoding="utf8")
    return out


if __name__=="__main__":
    result=main()
    print(json.dumps({"status":result["status"],"model_calls":result["model_call_count"],
         "fixed_regret":result["scored_means"]["fixed_rule"]["mean_regret"],
         "related_regret":result["scored_means"]["related_history"]["mean_regret"]},
         ensure_ascii=False))
