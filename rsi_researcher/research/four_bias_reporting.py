"""Report actual fixed/random/agent comparisons and evidence limitations."""
from pathlib import Path
import json,argparse
ROOT=Path("/data/cjh/rsi/project/runs")
PATHS={
"selection":"four-bias-selection-agent-workflow-20261022-v3",
"exposure":"four-bias-exposure-agent-workflow-20261022-v3",
"position":"four-bias-amazon-sports-position-agent-workflow-20261022-v1",
"popularity":"four-bias-amazon-beauty-pop-agent-workflow-20261022-v2"}
def get(path):
    return json.loads(path.read_text()) if path.is_file() else None
def science(kind,seed):
    r=ROOT/PATHS[kind]
    data=get(r/f"seed-{seed}/result.json")
    rnd=get(r/"independent_random_control/result.json")
    aux=get(r/"reviewer_auxiliary_metrics.json")
    if data is None: return {"state":"still_running"}
    dev=data["development"]
    base=dev["baseline"]["5"]
    fixed=dev["fixed"][0] if dev["fixed"] else None
    agent=next((t for t in dev["tree"] if t["id"]=="N001"),None)
    rand=rnd["random_search"][0] if rnd else None
    def normalize(row,key):
        m=row if key=="baseline" else (row.get("metrics",{}).get("5",{}) if row else {})
        audit=row.get("audit") if row and key!="baseline" else None
        return {"NDCG@5":m.get("NDCG@5"),"HR@5":m.get("HR@5"),
          "PopREO@5":m.get("PopREO@5"),
          "CPB@5":m.get("CandidatePositionBias_NDCG@5"),
          "eligible":audit.get("eligible_development_only") if audit else False,
          "failure":audit.get("failure_mode") if audit else None,
          "requests":(row.get("requests",0) if row else 0)}
    return {"state":"complete","dataset":"Sports" if kind=="position" else "Beauty",
        "baseline":normalize(base,"baseline"),"fixed":normalize(fixed,"fixed"),
        "random":normalize(rand,"random"),"agent":normalize(agent,"agent"),
        "usage":data["usage"],"selected":dev["selected_rsi"]["id"],
        "random_control_ready":rnd is not None,"holdout_opened":data["holdout"] is not None,
        "auxiliary_metrics":aux}

def write(path):
    o=Path(path);o.mkdir(exist_ok=True,parents=True)
    all={}
    for task in ("selection","exposure"):
        result=get(ROOT/PATHS[task]/"result.json")
        assert result is not None
        all[task]={"state":"complete","baseline":result["baseline"],
           "fixed":result["fixed"],"random":result["random"],"agent":result["agent"],
           "cost":result["cost"],"independent_holdout_opened":False}
    all["exposure"]["paired_uncertainty"]=get(ROOT/PATHS["exposure"]/"paired_uncertainty.json")
    all["position"]=science("position",20261027)
    all["popularity"]=science("popularity",20261022)
    lines=["# RSI Debias 四类任务 Fixed / Random / Agent 比较","",
           "四任务使用同一评价契约与现有科研 Agent。所有结果为 Development 探索，不存在独立确认的 Debias 发现。",
           "此次没有调用历史 Beauty Holdout；Fairness / Filter Bubble 暂停。",
           "固定/随机和 Agent 的实际 Qwen 请求不相等，因此不能宣称严格同预算的发现效率比较。",""]
    for task in ("selection","exposure"):
        v=all[task]
        key="Brier" if task=="selection" else "SNIPS"
        lines.extend(["## "+task+" ("+key+")","",
           "| Arm | Policy | "+key+" | Development pass |","|---|---|---:|---|"])
        for arm in ("baseline","fixed","random","agent"):
            x=v[arm]
            value=x.get("metrics",{}).get(key) if x.get("metrics") else None
            lines.append("| "+arm+" | "+str(x.get("arm") or "rejected")+" | "+
                (f"{value:.6f}" if value is not None else "-")+" | "+
                str(x.get("audit",{}).get("development_eligible",False))+" |")
        lines.append("")
        lines.append("Scientist Qwen requests: "+str(v["cost"]["agent_scientist_requests"])+
            "; fixed/random CPU evaluation Qwen requests: 0.")
        if task=="exposure":
            lines.append("SNIPS = off-policy utility estimate; not direct proof of Exposure Bias reduction. 10k random events have only 38 clicks.")
        else:
            lines.append("Artificial extra MNAR masking is identifiable; naturally missing KuaiRec feedback is not.")
        lines.append("")
    paired=all["exposure"].get("paired_uncertainty")
    if paired is not None:
        lines.extend(["## Exposure 开发集配对不确定性",
          "Agent/Random head20 - Fixed mix policy SNIPS差值 "+str(paired["observed_delta"]["head20_minus_fixed"])+
          "，配对 Bootstrap 95% 区间 "+str(paired["CI95"]["head20_minus_fixed"])+
          "，跨 0，不能认为 Agent 在相同真实随机日志上明显优于 Fixed。",
          "Head20 - Uniform 改善区间 "+str(paired["CI95"]["head20_minus_uniform"])+
          "，同样跨 0。",""])
    for kind in ("position","popularity"):
        v=all[kind];lines.append("## "+kind+" (Amazon)")
        if v["state"]!="complete":
            lines.extend(["当前研究树仍在执行；未发表不完整样本的去偏结论。",""])
            continue
        lines.extend(["| Arm | NDCG@5 | HR@5 | PopREO@5 | CPB@5 | Pass | Failure |",
                      "|---|---:|---:|---:|---:|---|---|"])
        for arm in ("baseline","fixed","random","agent"):
            a=v[arm]
            def f(k):
                x=a.get(k)
                return "-" if x is None else f"{x:.4f}"
            lines.append("| "+arm+" | "+" | ".join(f(k) for k in ("NDCG@5","HR@5","PopREO@5","CPB@5"))+
                " | "+str(a["eligible"])+" | "+str(a["failure"] or "reference")+" |")
        lines += ["","Agent tree Qwen requests: "+str(v["usage"]["agent_tree_total_requests"])+
            "; baseline Qwen: "+str(v["usage"]["baseline_dev_requests"])+
            "; fixed Qwen: "+str(v["usage"]["fixed_search_requests"])+
            "; selected RSI node: "+v["selected"]+".",
            "Random comparable arm available: "+str(v["random_control_ready"])+
            "; independent holdout opened: "+str(v["holdout_opened"])+".",""]
    lines.extend(["## 小样本跨 Amazon 域固定惩罚机制回顾",
       "此前相同离线流行度惩罚强度 (lambda=1) 在三个 Amazon 品类固定 56 用户 Qwen 排名上的开发集结果；本实验用于检验趋势是否一致，部分流行度组不足 5 个正例。",
       "| 数据 | Base NDCG | Penalty NDCG | Base PopREO | Penalty PopREO |",
       "|---|---:|---:|---:|---:|"])
    for ds in ("Beauty","Sports","CDs"):
        prior=get(ROOT/"amazon-position-popularity-qwen-dev-20261022-v1"/ds/"results.json")
        if prior:
            b=prior["baseline_random"]
            a=prior["popularity_rerank_offline"]["1.0"]
            lines.append(f"| {ds} | {b['NDCG@5']:.4f} | {a['NDCG@5']:.4f} | {b['PopREO@5']:.4f} | {a['PopREO@5']:.4f} |")
    lines.extend(["跨域表现方向不一致，尤其小组正例支持不足；不可宣称同一惩罚政策跨域有效。",
        "","## Selection known MNAR 机制的跨数据参考",
        "KuaiRec 本轮与以前 MovieLens-1M 的受控额外选择机制均观察到 IPW 降低 Brier 的趋势，但掩蔽参数与推荐模型不同，不代表实际自然选择偏差的识别或同一 Agent 策略迁移。",""])
    beauty_ci=get(ROOT/PATHS["popularity"]/"beauty_agent_paired_uncertainty.json")
    if beauty_ci:
        x=beauty_ci["paired_popreo_reduction"]
        lines.extend(["## Beauty Agent 目标偏差配对不确定性",
          "Agent 观察性 PopREO 降幅的配对 Bootstrap CI95: "+
          str(x["ci95"])+
          "；跨 0，不能确认方向。格式未修复双臂配对仅 38/120 位用户，"
          "低于预定义 clean 支持门槛，不能晋升。",
          ""])
    lines.extend(["## 真实模型调用成本",
       "从各隔离运行目录 calls.jsonl 的 status=started / received 重新核算，"
       "包含初次中断、续跑、随机对照缓存冲突尝试及科学 Agent 推理。",
       "总尝试调用 1460，收到回复 1458，因停止本会话自有无效重复进程有 2 条未完成。",
       "独立确认发现数 0，因此每有效发现消耗的模型请求数：不可定义。"
       " Fixed/Random/Agent 实际成本不匹配，禁止宣传严格同预算优势。",
       "另有部分 Qwen 相同哈希请求生成不同回答，已按缓存冲突拒绝合并，"
       "保留中断的研究目录用于复查而未覆盖原始回答。",""])
    lines.extend(["## 科学解释与后续",
      "- Position Sports 基线 |CPB|≈0.0107 小于最小绝对改善门槛0.02，必须在下一轮冻结协议前做可改进空间预筛查。",
      "- Popularity 必须报告五组正例与命中数、格式 clean 子集和配对不确定性，不能只看 ARP。",
      "- Selection 当前 Agent 是已知 MNAR 下的有界 IPW 截断实验，非真实选择偏差因果确认。",
      "- Exposure SNIPS 与随机策略支持度是 OPE 而不是偏差减少率；需独立策略价值确认和不确定性。",
      "- 非相同 Qwen 预算不能当作严格 Agent 搜索优势；每有效发现调用数目前不可定义（确认发现0）。",
      "- KuaiRec 的 Position 与 Popularity 参考实验、KuaiRand 随机曝光转移是不同协议的补充，尚未构成同策略跨域复现。",
      "",
      "## 复现输入",
      *["- "+task+": "+str(ROOT/folder) for task,folder in PATHS.items()],""])
    (o/"RESEARCH_REPORT.md").write_text("\n".join(lines))
    (o/"COMPARISON.json").write_text(json.dumps({"model_requests_started":1460,
       "model_requests_received":1458,
       "model_requests_unfinished_own_stopped":2,
       "current_session_independent_confirmed_discoveries":0,
       "model_requests_per_confirmed_discovery":None,
       "popularity_agent_paired_uncertainty":beauty_ci,
       "status":"development_only",
       "four_bias_contract":"rsi_researcher.research.four_bias_contracts",
       "confirmed_discoveries":0,"cost_per_confirmed_discovery":None,
       "strict_equal_model_compute":False,"results":all},indent=2,ensure_ascii=False))
    print("FINAL_REPORT",o/"RESEARCH_REPORT.md",
         "position",all["position"]["state"],"popularity",all["popularity"]["state"])
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True)
    a=p.parse_args();write(a.output)
