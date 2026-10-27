"""Reproducible aggregate report from frozen four-bias tree artifacts only."""
import argparse,hashlib,json,statistics
from pathlib import Path
ROOT=Path("/data/cjh/rsi/project")
RUN=ROOT/"runs"
PATHS={
 "selection":["selection-rsi-tree-20261022-round1",
              "selection-rsi-tree-20261022-round2-confirm"],
 "exposure":["exposure-rsi-tree-20261022-round1",
             "exposure-rsi-tree-20261022-round2-confirm"]}

def load(path):return json.loads(path.read_text())
def summarise(task):
    dirs=[RUN/n for n in PATHS[task]]
    first=load(dirs[0]/"summary.json")
    last=load(dirs[1]/"summary.json")
    conf=load(dirs[1]/("confirmation.json" if task=="selection" else
                      "recovered_confirmation.json"))
    freeze=load(dirs[1]/"frozen_before_confirmation.json")
    old_nodes=[n for n in first["tree"] if n["id"]!="ROOT"]
    assert last["holdout_opened"] and freeze["confirmation_allowed"]
    assert all(n["id"].startswith("P") for n in last.get("tree",last.get("nodes",[])) if n.get("inherited_from"))
    assert conf["confirmed_bias_causal_claims"]==0
    metrics={}
    key="Brier" if task=="selection" else "SNIPS"
    for label,body in conf["summary"].items():
        if label=="baseline":
            vals=[m[key] for m in body["by_seed"]]
            metrics[label]={"value":statistics.mean(vals),"values":vals}
        else:
            samples=body["by_seed"]
            vals=[m["candidate"][key] for m in samples]
            deltas=[m["delta_primary"] for m in samples]
            metrics[label]={"value":statistics.mean(vals),
               "per_seed_values":vals,"mean_gain":statistics.mean(deltas),
               "per_seed_gain":deltas,
               "CI95_by_seed":[m["CI95_gain"] for m in samples],
               "passes_locked_criteria":conf["independent_confirmation_of_specific_estimand"][label],
               "strategy":body["strategy"]}
    return {
      "task":task,"source_dirs":[str(d) for d in dirs],
      "development_round_1":first,"development_round_2":last,
      "confirmation":conf,"frozen":freeze,"comparison":metrics,
      "new_qwen_requests":first["qwen_requests"]+last["qwen_requests"],
      "independent_confirmation_scope":
        "KNOWN_EXTRA_MNAR_ONLY" if task=="selection"
        else "STATIC_OPE_REWARD_ONLY",
      "generalized_bias_discovery_confirmed":False}

def report(out):
    o=Path(out);o.mkdir(parents=True,exist_ok=True)
    all={k:summarise(k) for k in PATHS}
    lines=["# Selection / Exposure 多轮科学树与独立确认报告",
           "",
           "日期：2026-10-22。执行基于现有科学 Agent + 独立 Python Evaluator。",
           "数据是 KuaiRec 2.0 与官方完整 Open Bandit（随机动作日志）。",
           "**重要**：两类任务都已经冻结并执行确认，但两个主指标的科学含义不同。",
           "Selection 仅确认 *额外人为已知 MNAR* 的校正；Exposure 仅确认 *给定动作概率下静态策略的离线价值提升*。",
           "二者都不能据此声称一般意义的因果 Bias 消除；Agent 尚未超过固定强基线。",""]
    for task,x in all.items():
        metric="Brier ↓" if task=="selection" else "SNIPS ↑"
        lines.extend(["## "+task+" — "+metric,"",
         "| Method | Independent-panel metric | Improvement over baseline | Frozen method strategy |",
         "|---|---:|---:|---|"])
        for label in ("baseline","fixed","random","agent"):
            t=x["comparison"][label]
            gain="-" if label=="baseline" else f"{t['mean_gain']:+.6f}"
            strat="baseline frozen" if label=="baseline" else str(t["strategy"])
            lines.append(f"| {label} | {t['value']:.6f} | {gain} | {strat} |")
        lines+=["",
            "### 每种子的冻结确认差值与用户/时序分组置信区间",
            "| Arm | Seed 20261029 | Seed 20261030 | Seed 20261031 | Locked criterion |",
            "|---|---|---|---|---|"]
        for label in ("fixed","random","agent"):
            xarm=x["comparison"][label]
            parts=[]
            for gain,ci in zip(xarm["per_seed_gain"],xarm["CI95_by_seed"]):
                parts.append(f"{gain:+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}]")
            lines.append("| "+label+" | "+" | ".join(parts)+
                         " | "+str(xarm["passes_locked_criteria"])+" |")
        lines.extend(["",
          "### RSI 搜索轨迹",
          "| Round | Node | Parent | Numeric strategy | Dev assessment |",
          "|---|---|---|---|---|"])
        for rnd,record in (("R1",x["development_round_1"]),
                           ("R2",x["development_round_2"])):
            for n in record.get("tree",record.get("nodes",[])):
                if n["id"]=="ROOT" or n.get("inherited_from"):continue
                lines.append(f"| {rnd} | {n['id']} | {n['parent']} | "+
                    str(n["strategy"])+" | "+n["audit"]["failure_mode"]+" |")
        calls=x["new_qwen_requests"]
        lines.extend(["",
            f"Qwen Scientist 实际请求：{calls}（两轮）。随机基线数值执行不需要额外 Qwen；"
            "因此实际计算量不相等，不能用这个结果宣称同预算搜索优胜。",
            "确认时没有 Qwen 请求，策略、数据分割和超参数在结果计算前已经冻结。",
            ""])
    lines+=["## 数据支持与不确定性解释",
      "- Selection: train 3,251,569 事件，dev 1,020,633，confirmation 404,368（122 个独立留出用户），三个种子只改变已知额外 MNAR 掩码；不是三个独立用户数据集。",
      "- Exposure: official random/all（排除此前示例前 10k 条），dev 1,091,462 条，confirmation 272,865 条，按 500 行相邻时序区块切为三个互斥评估分组；随机动作 pscore=1/80。不能按用户聚类，因为此面板未提供可使用的唯一用户 ID。",
      "- Exposure 的零假设是固定目标静态策略相比 uniform 随机策略的 OPE 点击收益差是否为正；不是曝光偏差测度。更严谨的 Exposure Debias 需要与偏置记录下的策略学习/评价误差相结合。",
      "- Selection 独立性：本轮开发/确认用户隔离，但 KuaiRec 先前开展过探索性全数据报告，不能声称这些用户在整个项目历史中从未出现过。",
      "- Exposure 确认结果发生一次 JSON int64 导出失败，记录了无 Agent、未调参的同策略技术重算；详见 CONFIRMATION_RECOVERY_AUDIT.json。此前 10k OBD 示例的已观察行被排除。",
      "- 严格多重试验、序贯搜索调整未建立；Bootstrap 区间是面板条件下的描述性不确定性证据。",
      "- 用户需求的四类主目标系统中 Popularity/Position 仍沿用 Amazon LLMRank science-tree，Selection/Exposure 使用新科学树共享研究 Agent 和数值契约；两者 frontier 实现尚未完全抽取为同一基础调度器。",
      "","## 科研判断",
      "Fixed Oracle IPW 对 Selection Brier 的改善大于 Agent RSI；Exposure 的 Fixed 也有更高的 SNIPS 增益。"
      "Agent RSI 已可跨轮反证及独立确认，但 **没有超越固定强基线**。",
      "一轮有效基础机制的确认不能归功于 Agent 的自主发现；目前可宣称的 *Agent 超越 Fixed 的独立 Debias Discovery*=0，故每发现模型请求数不可定义。",
      "跨数据集算法复现：尚未完成；不能因为一个任务使用 KuaiRec、另一个使用 OBD 就称相同策略跨域复现。",
      "","## 文件",
      "代码主入口：python -m rsi_researcher four-bias-tree",
      "证据契约：rsi_researcher/research/four_bias_contracts.py",
      "任务适配与数据隔离：rsi_researcher/research/four_bias_benchmarks.py",
      "节点/反证/跨轮：rsi_researcher/research/four_bias_tree.py",
      "模型成本以每个独立运行目录 calls.jsonl 为准，不把缓存命中当作新请求。",
      ""]
    (o/"RESEARCH_REPORT.md").write_text("\n".join(lines),encoding="utf8")
    simple={}
    for k,v in all.items():
        simple[k]={z:v[z] for z in ("task","source_dirs","comparison",
                    "new_qwen_requests","independent_confirmation_scope",
                    "generalized_bias_discovery_confirmed")}
        simple[k]["confirmation_recovery"]=k=="exposure"
        simple[k]["confirmed_method_specific_estimand"]={
            key:z["passes_locked_criteria"] for key,z in v["comparison"].items() if key!="baseline"}
    summary={
      "status":"COMPLETED_FROZEN_DEVELOPMENT_AND_CONFIRMATION",
      "dataset_confidential_holdout_observed_only_after_freeze":True,
      "tasks":simple,"qwen_scientist_requests_total":sum(
          v["new_qwen_requests"] for v in all.values()),
      "general_causal_bias_discoveries":0,
      "agent_outperforms_fixed":False,
      "strict_equal_compute":False,
      "documentation":"RESEARCH_REPORT.md"}
    (o/"SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print("FOUR_BIAS_TREE_REPORT",json.dumps({
        "output":str(o),"requests":summary["qwen_scientist_requests_total"],
        "selection":{k:(v["value"],v.get("mean_gain")) for k,v in all["selection"]["comparison"].items()},
        "exposure":{k:(v["value"],v.get("mean_gain")) for k,v in all["exposure"]["comparison"].items()}},ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    a=p.parse_args()
    report(a.output)
