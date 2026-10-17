"""Independent source-grounded audit of frozen cross-Bias pilot evidence.

Read-only: does not rerun any model, mutate historical proposals or rescore test
data. Distinguishes test selection from hypothesis confirmation, and flags
literal, measurable contradictions in agent reflection.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import json

from rsi_researcher.research import crossbias_discovery as b


def load(name):
    return json.loads((b.RUN/name).read_text(encoding="utf8"))


def call_costs():
    records=[json.loads(line) for line in (b.RUN/"calls.jsonl").read_text(encoding="utf8").splitlines()]
    started={r["call_id"]:r for r in records if r["status"]=="started"}
    received={r["call_id"]:r for r in records if r["status"]=="received"}
    if len(started)!=60 or len(received)!=60 or started.keys()!=received.keys():
        raise ValueError("Pilot request log incomplete or duplicate")
    count=defaultdict(lambda:{"calls":0,"prompt_tokens":0,"completion_tokens":0})
    for cid,s in started.items():
        ctx=s["context"]
        phase=ctx["phase"]
        arm=ctx.get("arm","shared_rank_executor")
        action=ctx.get("action","")
        key=(phase,arm,action)
        n=count[key]
        n["calls"]+=1
        u=received[cid]["usage"]
        n["prompt_tokens"]+=u["prompt_tokens"]
        n["completion_tokens"]+=u["completion_tokens"]
    out={}
    for (phase,arm,action),v in count.items():
        key=phase+"/"+arm+(("/"+action) if action else "")
        out[key]=v
    return out


def audit():
    protocol=b.verify()
    plans=load("plans.json")
    assessments=load("assessments.json")
    reflections=load("reflections.json")
    official=load("comparison.json")
    s=load("status.json")
    if official["status"]!="PROSPECTIVE_CROSSBIAS_PILOT_COMPLETED" or s["status"]!="COMPLETED":
        raise ValueError("Pilot did not complete")
    if official["protocol_sha256"]!=b.sha_file(b.RUN/"protocol.json"):
        raise ValueError("Pilot protocol hash changed")
    phase_cost=call_costs()
    totals={"calls":sum(x["calls"] for x in phase_cost.values()),
            "prompt_tokens":sum(x["prompt_tokens"] for x in phase_cost.values()),
            "completion_tokens":sum(x["completion_tokens"] for x in phase_cost.values())}
    if totals["calls"]!=official["model_requests_total"]:
        raise ValueError("Request totals mismatch")
    if totals["prompt_tokens"]!=official["model_usage"]["prompt_tokens"]:
        raise ValueError("Prompt token usage mismatch")
    if totals["completion_tokens"]!=official["model_usage"]["completion_tokens"]:
        raise ValueError("Completion token usage mismatch")
    by_arm={}
    for arm in b.ARMS:
        plan_actions={k:plans[arm][k]["action"] for k in b.CATEGORIES}
        findings=[]
        for task in b.CATEGORIES:
            outcome=assessments[arm][task]
            findings.append({"task":task,"falsifier":plan_actions[task],
                "executed":True,
                "development_signal":outcome["replicated_development_signal"],
                "independent_scientific_confirmation":False,
                "hypothesis_semantic_truth_adjudicated":False})
        arm_total=sum(phase_cost.get(f"proposal/{arm}",{}).get("calls",0) for _ in [0])
        arm_total+=phase_cost.get(f"reflection/{arm}",{}).get("calls",0)
        by_arm[arm]={
          "findings":findings,
          "registered_proposals":len(findings),
          "model_research_calls":arm_total,
          "conditional_development_signals":sum(x["development_signal"] for x in findings),
          "independent_scientific_discoveries":0,
          "qualified_hypothesis_confirmations":0,
          "hypothesis_confirmation_note":"Free-text claims are not machine-verifiable forecast contracts",
          "distinct_actions":len(set(plan_actions.values())),
          "development_signals_per_model_call":(
              sum(x["development_signal"] for x in findings)/arm_total if arm_total else None)}
    for arm in ("qwen_no_history","qwen_related_memory"):
        if by_arm[arm]["model_research_calls"]!=8:raise ValueError("Planner/reflection budget changed")
    if by_arm["rule"]["model_research_calls"]!=0:
        raise ValueError("Rule unexpectedly called Qwen")
    coat=assessments["rule"]["coat_utility"]["falsifier"]["cohorts"]
    assert all(c["tail_delta"]<0 and c["ndcg_ratio"]>=1 for c in coat)
    issues=[]
    first=reflections["qwen_no_history"]["coat_utility"]["lesson"].lower()
    second=reflections["qwen_related_memory"]["coat_utility"]["lesson"].lower()
    if "decrease in ranking-quality" in first:
        issues.append({"arm":"qwen_no_history","task":"coat_utility",
         "type":"NDCG_DIRECTION_CONTRADICTION",
         "observed":"NDCG ratios >= 1.0 in both panels",
         "quoted_phrase":"decrease in ranking-quality",
         "severity":"incorrect_numerical_reflection"})
    if "without compromising ranking-quality or long-tail recall" in second:
        issues.append({"arm":"qwen_related_memory","task":"coat_utility",
         "type":"TAIL_RECALL_CONTRADICTION",
         "observed":"Tail relevant recall decreases in both panels",
         "quoted_phrase":"without compromising ... long-tail recall",
         "severity":"incorrect_numerical_reflection"})
    plans_agree=all(plans["qwen_no_history"][task]["action"]==
                   plans["qwen_related_memory"][task]["action"] for task in b.CATEGORIES)
    return {"status":"CROSSBIAS_INDEPENDENT_AUDIT_COMPLETE",
       "audited_utc":datetime.now(timezone.utc).isoformat(),
       "protocol_hash":official["protocol_sha256"],
       "no_confirmation_labels_scored":True,
       "all_plans_before_falsification":True,
       "qwen_history_did_not_change_actions":plans_agree,
       "by_arm":by_arm,
       "model_usage_totals":totals,
       "stage_and_action_usage":phase_cost,
       "shared_executor_queries_not_fairly_attributable_by_arm":True,
       "independent_reflection_contradictions":issues,
       "observations":{
         "coat_utility_split":coat,
         "author_item_ablation":assessments["rule"]["agentcf_memory"]["falsifier"]["cohorts"],
         "author_popularity_prompt":assessments["rule"]["agentcf_popularity"]["falsifier"]["cohorts"]},
       "caveats":[
         "Four research tasks already familiar by domain, no independent human-labeled gold hypothesis validity",
         "Qwen arms proposed same action, so no evidence Research Memory improved action selection",
         "Development signals are action-outcome checks, not proof the generated hypothesis was correct",
         "Coat quality gate checks NDCG but long-tail declines remain material adverse results",
         "AgentCF memory ablation changes rankings but loses known-target NDCG on both small panels",
         "AgentCF 10 author users and one shared training state, not cross-cohort independent replication",
         "Qwen reflection can contradict numeric evidence; keep it out of future Research Memory until validated",
         "Shared rank calls physically executed once; compare marginal model/planner and logical actions separately"
       ]}


def main():
    result=audit()
    p=b.RUN/"INDEPENDENT_AUDIT.json"
    if p.exists():raise FileExistsError(p)
    b.save_new(p,result)
    text=["# Cross-Bias 科研发现与成本独立审计","",
       "## 实验与评价口径",
       "四个任务的实验均为 Qwen 提案先行、随后受控执行、独立 Python 审查。"
       "仅发展期两组内部复核，尚未产生任何独立确认的科研发现。",
       "",
       "| 策略 | 完成实验 | 开发信号 | 独立发现 | 研究决策/复盘模型调用 |",
       "|---|---:|---:|---:|---:|"]
    for arm,value in result["by_arm"].items():
        text.append(f"| {arm} | {len(value['findings'])} | "
          f"{value['conditional_development_signals']} | "
          f"{value['independent_scientific_discoveries']} | "
          f"{value['model_research_calls']} |")
    costs=result["model_usage_totals"]
    text+=["",f"总 GPU 调用：{costs['calls']} 次，输入 {costs['prompt_tokens']} tokens，输出 {costs['completion_tokens']} tokens。"
        "其中包含共用的 AgentCF 排序执行，而不全是 Qwen 规划调用。",
        "",
        "## 真实实验事实",
        "- Coat 正确倾向映射相较两种打乱倾向负对照在两组开发用户 Brier 都较低；它不代表倾向是真实分配概率。",
        "- Coat 截断 100 的 Brier 两组改善，NDCG 不降，但长尾召回在两组均下降；不能声称完全无效用损失。",
        "- AgentCF 去除 Item Memory 改变大量排名，也降低两组 NDCG，不属于通过质量约束的改善。",
        "- AgentCF 去掉流行度提示只在一组减少热门过量，另一组无效；没有跨组稳定证据。",
        "",
        "## Research Memory 与复盘风险",
        f"- Qwen 无历史与带相关历史四项行动是否完全相同：{result['qwen_history_did_not_change_actions']}。"
        "没有发现历史信息改变研究动作的证据。",
        f"- 发现 {len(result['independent_reflection_contradictions'])} 处可由 Python 观测反驳的复盘措辞。",
        "不能未经验证就把 Qwen 的自然语言总结写成高等级 Research Memory。",
        "- 当前报告中的开发信号是“选中并执行的反证实验满足预设数值门槛”，"
        "不是自动证明 Qwen 文字假设成立。自由文本假设尚未形式化为可独立评分的预测。",
        "",
        "## 下一轮必须修改的科研协议",
        "1. 将 hypothesis 转成注册的 metric / sign / min_effect / competing_prediction / failure_boundary 结构化预测。",
        "2. Research Memory 只由独立 Evidence Reviewer 根据数值和边界生成；Qwen 的自评不能自动写为确认。",
        "3. 加入 AgentCF 多随机种子与独立用户-物品状态；重测 shared item memory 传播及流行度效果。",
        "4. 另建未见机制/真实数据确认任务；本次使用的开发用户不能重复算作新测试。",
        "5. 模型 token、rank 调用与 CPU 成本分开统计；比较受控发现/成本，而非未经确认的“成功次数”。"]
    (b.RUN/"INDEPENDENT_AUDIT.md").write_text("\n".join(text)+"\n",encoding="utf8")
    return result


if __name__=="__main__":
    r=main()
    print(json.dumps({"status":r["status"],"calls":r["model_usage_totals"]["calls"],
         "reflection_contradictions":len(r["independent_reflection_contradictions"]),
         "history_actions_identical":r["qwen_history_did_not_change_actions"]},ensure_ascii=False))
