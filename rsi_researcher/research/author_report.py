"""Assemble the Stage-2 author report (Table 1/2/3 + cases) from committed runs."""
import json
from pathlib import Path

from rsi_researcher.datasets.author import author_profile
from rsi_researcher.evaluation.author_probes import run_author_probes
from rsi_researcher.research.author_researcher import compare_arms
from rsi_researcher.paths import ROOT


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def _arm_metrics(summary, arm="original_update"):
    a = summary["arms"][arm]
    top5, cand = a["mean_top5_head_share"], a["mean_candidate_head_share"]
    return {"ndcg_at_5": a["mean_ndcg_at_5"], "ndcg_at_10": a["mean_ndcg_at_10"],
            "hit_at_5": a["mean_hit_at_5"], "top5_head_share": top5,
            "excess_popularity_exposure": top5 - cand}


def _fmt(v, digits=4):
    return "—" if v is None else f"{v:.{digits}f}"


def build_report(data_dir, baseline_dir, h1_dir, h2_dir, researcher_dir, output_dir):
    baseline_dir, h1_dir, h2_dir, researcher_dir, output_dir = (
        Path(baseline_dir), Path(h1_dir), Path(h2_dir), Path(researcher_dir), Path(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = author_profile(data_dir)
    b_probes = run_author_probes(baseline_dir)
    h1_probes = run_author_probes(h1_dir) if (h1_dir / "original_update" / "state.json").exists() else None
    b_summary = _read(baseline_dir / "summary.json")
    h1_summary = _read(h1_dir / "summary.json")
    h2_summary = _read(h2_dir / "summary.json")
    researcher = _read(researcher_dir / "researcher.json")

    b_metrics = _arm_metrics(b_summary) if b_summary else {}
    h1_metrics = _arm_metrics(h1_summary) if h1_summary else {}
    h2_metrics = h2_summary and {k: h2_summary.get("mean_" + k) for k in
                                 ("ndcg_at_5", "ndcg_at_10", "hit_at_5", "top5_head_share",
                                  "candidate_head_share", "excess_popularity_exposure")} or {}
    comparison = researcher.get("comparison") if researcher else None

    # --- Table 1: data distribution ---
    counts = profile["training_item_counts"]
    head_min, tail_max = profile["head_min_count"], profile["tail_max_count"]
    ranks = _read(baseline_dir / "original_update" / "ranking.json") or []
    from collections import Counter
    target_groups = Counter(r["target_group"] for r in ranks)
    table1 = [
        "## 表 1：20 人开发集数据分布",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 训练集（全量）| {profile['training_users']} 用户 / {profile['training_events']} 事件 / {profile['training_items']} 物品 |",
        f"| 热度分组定义 | head≥{head_min} 次；tail≤{tail_max} 次；其余 mid；0 次 unseen |",
        f"| 训练物品热度分布 | head {profile['training_group_sizes'].get('head',0)} / mid {profile['training_group_sizes'].get('mid',0)} / tail {profile['training_group_sizes'].get('tail',0)} |",
        f"| 开发用户 | 20 人（按训练热度四分位分层抽取，seed 20261015）|",
        f"| 训练事件 | 100 条（每人 5 条）|",
        f"| 验证目标分组 | head {target_groups.get('head',0)} / mid {target_groups.get('mid',0)} / tail {target_groups.get('tail',0)} / unseen {target_groups.get('unseen',0)}（共 {len(ranks)}）|",
        "",
    ]

    # --- Table 2: probe comparison ---
    lines = ["## 表 2：五类偏差诊断探针 + 质量/热门偏差（基线 vs H1 vs H2）", ""]
    lines += ["| 指标 | 基线 | H1（未知反馈修正）| 相对变化 | H2（去热门兜底）| 相对变化 |",
              "|---|---:|---:|---:|---:|---:|"]
    def rel(new, old):
        return (new - old) / old if old else None

    # memory probes (H1 only)
    rows2 = []
    b_rate = b_probes["exposure_bias"]["false_dislike_memory_rate"]
    if h1_probes:
        h1_rate = h1_probes["exposure_bias"]["false_dislike_memory_rate"]
        rows2.append(("曝光偏差 false_dislike_memory_rate", _fmt(b_rate), _fmt(h1_rate),
                      _fmt(rel(h1_rate, b_rate)), "—", "—"))
        for key, label in (("select_reason_popularity_rate", "从众偏差 选品理由热门先验率"),
                           ("user_memory_popularity_rate", "从众偏差 用户记忆热门先验率")):
            bv = b_probes["conformity_bias"][key]
            hv = h1_probes["conformity_bias"][key]
            rows2.append((label, _fmt(bv), _fmt(hv), _fmt(rel(hv, bv)), "—", "—"))
        for g, glabel in (("head", "选择/PAAC head 平均更新次数"), ("unseen", "选择/PAAC unseen 平均更新次数")):
            bv = b_probes["selection_paac"]["groups"][g]["mean_update_count"]
            hv = h1_probes["selection_paac"]["groups"][g]["mean_update_count"]
            rows2.append((glabel, _fmt(bv), _fmt(hv), _fmt(rel(hv, bv)), "—", "—"))
        bp = b_probes["position_bias"]["input_order_following_kendall_tau"]
        hp = h1_probes["position_bias"]["input_order_following_kendall_tau"]
        rows2.append(("位置偏差 input-order τ", _fmt(bp), _fmt(hp), _fmt(rel(hp, bp)), "—", "—"))
    # quality/popularity (baseline vs H1 vs H2)
    for key, label in (("ndcg_at_5", "质量 NDCG@5"), ("ndcg_at_10", "质量 NDCG@10"),
                       ("hit_at_5", "质量 Hit@5"), ("excess_popularity_exposure", "偏差 额外热门倾斜")):
        bv = b_metrics.get(key)
        h1v = h1_metrics.get(key)
        h2v = h2_metrics.get(key)
        rows2.append((label, _fmt(bv), _fmt(h1v) if h1v is not None else "—",
                      _fmt(rel(h1v, bv)) if h1v is not None else "—",
                      _fmt(h2v) if h2v is not None else "—",
                      _fmt(rel(h2v, bv)) if h2v is not None else "—"))
    for row in rows2:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines += ["", "> H2 是排序期干预（无训练重放），故记忆探针（曝光/从众/选择）不适用，仅质量与热门偏差可比。", ""]

    # --- Table 3: researcher audit ---
    lines += ["## 表 3：研究员 预测登记 → 实际结果 → 假说修订 审计", ""]
    if researcher:
        pred, rev = researcher.get("prediction", {}), researcher.get("revision", {})
        lines += ["| 阶段 | 内容 |", "|---|---|"]
        lines += [f"| 主要假说 | {pred.get('primary_hypothesis', '')} |",
                  f"| 竞争假说 | {pred.get('competing_hypothesis', '')} |",
                  f"| 选定算子 | `{pred.get('selected_operator', '')}` |",
                  f"| 预期效果 | {json.dumps(pred.get('expected_effects', {}), ensure_ascii=False)} |",
                  f"| 证伪条件 | {pred.get('falsification_condition', '')} |"]
        if comparison:
            gates = comparison.get("gates_pass", {})
            lines += [f"| 实际门槛 | false_dislike={gates.get('false_dislike')} quality={gates.get('quality')} bias={gates.get('bias')}（Python 计算）|"]
        lines += [f"| 适用条件 | {rev.get('applicability_conditions', '')} |",
                  f"| 已证实机制 | {rev.get('confirmed_mechanisms', '')} |",
                  f"| 被推翻的预期 | {rev.get('falsified_expectations', '')} |",
                  f"| 未排除的竞争解释 | {rev.get('unresolved_competing_explanations', '')} |",
                  f"| 下一优先动作 | `{rev.get('next_priority_action', '')}` |",
                  f"| 诚实评估 | {rev.get('honest_assessment', '')} |", ""]
    else:
        lines += ["（研究员闭环尚未运行）", ""]

    # --- representative cases ---
    lines += ["## 三类真实案例（记忆文本差异）", ""]
    if h1_probes and comparison:
        b_cases = b_probes["exposure_bias"]["representative_cases"]
        h_cases = h1_probes["exposure_bias"]["representative_cases"]
        if b_cases and h_cases:
            lines += ["**基线（脑补 dislike）案例**：", ""]
            c = b_cases[0]
            lines += [f"- 事件 `{c['event_id']}` 用户 `{c['user']}`，未交互对比物品 `{c['negative_title']}`：",
                      f"  - 更新前：{c['user_memory_before']}", f"  - 更新后：{c['user_memory_after']}", ""]
            lines += ["**H1（未知曝光修正）案例**：", ""]
            c = h_cases[0]
            lines += [f"- 事件 `{c['event_id']}` 用户 `{c['user']}`：",
                      f"  - 更新前：{c['user_memory_before']}", f"  - 更新后：{c['user_memory_after']}", ""]

    report = "\n".join(table1 + lines) + "\n"
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    summary = {"dataset": "CDs-100-user-dense", "baseline_dir": str(baseline_dir),
               "h1_dir": str(h1_dir), "h2_dir": str(h2_dir), "researcher_dir": str(researcher_dir),
               "table1_groups": dict(target_groups),
               "baseline_metrics": b_metrics, "h1_metrics": h1_metrics, "h2_metrics": h2_metrics,
               "exposure_bias": {"baseline": b_probes["exposure_bias"]["false_dislike_memory_rate"],
                                 "h1": h1_probes["exposure_bias"]["false_dislike_memory_rate"] if h1_probes else None},
               "comparison": comparison,
               "scope": "honest Stage-2 audit; gates are Python-computed, failures are not spun"}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--h1", required=True)
    parser.add_argument("--h2", required=True)
    parser.add_argument("--researcher", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_report(args.data_dir, args.baseline, args.h1, args.h2, args.researcher, args.output))
