"""Bounded five-mechanism research loop on the AgentCF author data.

Qwen proposes a registered intervention, Python executes and screens it, and
Qwen revises its hypothesis from measured results. Training interventions replay
observed data; ranking interventions reuse the baseline's trained state. Passing
development proxy thresholds is not validated debiasing or proof of RSI.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from rsi_researcher.paths import ROOT
from rsi_researcher.runtime.client import TEXT_SCHEMA, object_schema, atomic_json, InvalidOutput
from rsi_researcher.evaluation.author_probes import run_author_probes, probe_position_bias
from rsi_researcher.research.author_experiment import AUTHOR_OPERATORS, RANK_ONLY_OPERATORS
from rsi_researcher.evaluation.author_patterns import describe_run, pattern_brief, render_patterns, compare_saved_quality
from rsi_researcher.research.author_evidence import record_author_trial, compact_author_history, trial_conditions
from rsi_researcher.research.journal import ExperimentJournal

BIAS_TARGETS = ["exposure", "conformity", "popularity", "position", "selection_paac", "none"]

# Each operator fixes one bias; the pair is fixed so the target gate always
# judges an operator on the metric it actually moves.
OPERATOR_TARGETS = {
    "retain": "none",
    "unseen_neutral_user_reflection": "exposure",
    "neutralize_select_popularity": "conformity",
    "neutralize_rank_popularity": "popularity",
    "neutral_candidate_order": "position",
    "balanced_observed_replay": "selection_paac",
}

# Operators Qwen may actually propose. "retain" is the no-op fallback used by
# _select when nothing wins, not a real intervention to try.
PROPOSABLE_OPERATORS = [op for op in AUTHOR_OPERATORS if op != "retain"]

# bias target -> (section, key, label) into bias_summary(); every metric here is
# "lower is better".
TARGET_METRICS = {
    "exposure": ("exposure_bias", "false_dislike_memory_rate", "false_dislike_rate"),
    "conformity": ("conformity_bias", "select_reason_popularity_rate", "select_reason_popularity_rate"),
    "popularity": ("quality_popularity", "excess_popularity_exposure", "excess_popularity_exposure"),
    "position": ("position_bias", "input_order_following_tau", "position_tau"),
    "selection_paac": None,  # special: head vs tail memory-update gap
}


def _plan_schema(allowed_operators):
    """Proposal schema with the operator enum restricted to the operators still
    allowed this round (already-tried operators are excluded so the model cannot
    structurally re-pick an already-run intervention)."""
    return object_schema({
        "operator": {"type": "string", "enum": sorted(allowed_operators)},
        "hypothesis": TEXT_SCHEMA,
        "competing_hypothesis": TEXT_SCHEMA,
        "expected_effects": object_schema({
            "target_metric": {"type": "string", "enum": ["decrease", "increase", "no_change", "unknown"]},
            "quality_ndcg": {"type": "string", "enum": ["maintain_or_improve", "decrease", "unknown"]},
        }),
        "falsifier": TEXT_SCHEMA,
        "reason": TEXT_SCHEMA,
    })


def _next_allowed_operators(tried_operators, registered_operators=None):
    """Stop when the registered action space is exhausted; never invent combinations."""
    return [op for op in (PROPOSABLE_OPERATORS if registered_operators is None else registered_operators) if op not in tried_operators]


LEARN_SCHEMA = object_schema({
    "lesson": TEXT_SCHEMA,
    "next_question": TEXT_SCHEMA,
    "next_operator": {"type": "string", "enum": sorted(PROPOSABLE_OPERATORS)},
    "evidence_state": {"type": "string", "enum": ["supported", "weakened", "inconclusive"]},
})


def bias_summary(run_dir):
    """Compact the probes + quality into a bounded evidence dict."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if "arms" in summary:
        probes = run_author_probes(run_dir)
        arm = summary["arms"]["original_update"]
    else:
        # Ranking-only trials explicitly inherit training diagnostics from their
        # source. No synthetic event records or independent training claims.
        probes = run_author_probes(summary["source"])
        probes["position_bias"] = probe_position_bias(json.loads((run_dir / "ranking.json").read_text(encoding="utf-8")))
        arm = summary
    result = {
        "exposure_bias": {"false_dislike_memory_rate": probes["exposure_bias"]["false_dislike_memory_rate"]},
        "conformity_bias": {"select_reason_popularity_rate": probes["conformity_bias"]["select_reason_popularity_rate"],
                            "user_memory_popularity_rate": probes["conformity_bias"]["user_memory_popularity_rate"]},
        "selection_paac": {group: {"mean_update_count": v["mean_update_count"],
                                   "mean_growth_ratio": v["mean_growth_ratio"]}
                           for group, v in probes["selection_paac"]["groups"].items()},
        "position_bias": {"input_order_following_tau": probes["position_bias"]["input_order_following_kendall_tau"]},
        "quality_popularity": {"ndcg_at_5": arm["mean_ndcg_at_5"], "hit_at_5": arm["mean_hit_at_5"],
                               "top5_head_share": arm["mean_top5_head_share"],
                               "candidate_head_share": arm["mean_candidate_head_share"],
                               "excess_popularity_exposure": arm["mean_top5_head_share"] - arm["mean_candidate_head_share"]},
    }
    if (run_dir/"order_controls.json").exists():
        result["position_bias"]["controls"] = json.loads((run_dir/"order_controls.json").read_text(encoding="utf-8"))
    return result


def _head_tail_gap(summary):
    """Head vs tail learning-opportunity gap, not memory quality or MNAR bias."""
    groups = summary["selection_paac"]
    return groups["head"]["mean_update_count"] - groups["tail"]["mean_update_count"]


def _magnitude(value, bias):
    if value is None:
        return None
    if bias in ("position", "selection_paac"):
        return abs(value)
    if bias == "popularity":
        return max(0.0, value)
    return value


def _target_delta(baseline, candidate, targeted_bias):
    """Delta dict for the operator's target bias, or None for a no-op target."""
    if not targeted_bias or targeted_bias == "none":
        return None
    if targeted_bias == "selection_paac":
        b = _head_tail_gap(baseline)
        c = _head_tail_gap(candidate)
        label = "head_tail_update_gap"
    elif targeted_bias == "position" and candidate["position_bias"].get("controls"):
        controls=candidate["position_bias"]["controls"]
        b=controls["baseline"]["order_effect"]["excess_order_instability"]
        c=controls["candidate"]["order_effect"]["excess_order_instability"]
        label="noise_adjusted_order_instability"
    else:
        section, key, label = TARGET_METRICS[targeted_bias]
        b = baseline[section][key]
        c = candidate[section][key]
    if label=="noise_adjusted_order_instability":
        b_m,c_m=(max(0,v) if v is not None else None for v in (b,c))
    else:
        b_m, c_m = _magnitude(b, targeted_bias), _magnitude(c, targeted_bias)
    comparable = b_m is not None and c_m is not None
    return {"bias": targeted_bias, "label": label, "baseline": b, "candidate": c,
            "change": c - b if comparable else None,
            "baseline_magnitude": b_m, "candidate_magnitude": c_m,
            "relative_reduction": (b_m - c_m) / b_m if comparable and b_m > 0 else None,
            "improved": comparable and c_m < b_m}


def compare_bias(baseline, candidate, targeted_bias=None, quality_floor=0.99, minimum_reduction=0.10):
    """Python-precomputed deltas and gates for one arm vs baseline.

    Success = the operator improved ITS OWN target bias AND quality stayed within
    the floor. Every other bias dimension is reported as a collateral effect, not
    a hard gate (single-variable interventions are expected to have trade-offs).
    """
    controls=candidate["position_bias"].get("controls") if targeted_bias=="position" else None
    if controls:
        # Fresh matched panels provide the quality reference for the controlled
        # position experiment. Historical baseline rankings remain context only.
        baseline,candidate=deepcopy(baseline),deepcopy(candidate)
        for label,summary in (("baseline",baseline),("candidate",candidate)):
            q=controls[label]["quality"]
            summary["quality_popularity"]={"ndcg_at_5":q["mean_ndcg_at_5"],"hit_at_5":q["mean_hit_at_5"],
                                           "top5_head_share":q["mean_top5_head_share"],"candidate_head_share":q["mean_candidate_head_share"],
                                           "excess_popularity_exposure":q["mean_top5_head_share"]-q["mean_candidate_head_share"]}
    b_ndcg = baseline["quality_popularity"]["ndcg_at_5"]
    c_ndcg = candidate["quality_popularity"]["ndcg_at_5"]
    b_excess = baseline["quality_popularity"]["excess_popularity_exposure"]
    c_excess = candidate["quality_popularity"]["excess_popularity_exposure"]
    b_dislike = baseline["exposure_bias"]["false_dislike_memory_rate"]
    c_dislike = candidate["exposure_bias"]["false_dislike_memory_rate"]
    b_conf = baseline["conformity_bias"]["select_reason_popularity_rate"]
    c_conf = candidate["conformity_bias"]["select_reason_popularity_rate"]
    b_tau = baseline["position_bias"]["input_order_following_tau"]
    c_tau = candidate["position_bias"]["input_order_following_tau"]
    b_gap = _head_tail_gap(baseline)
    c_gap = _head_tail_gap(candidate)

    quality_gate = c_ndcg >= quality_floor * b_ndcg

    def m(b, c, improved):
        return {"baseline": b, "candidate": c, "change": c - b, "improved": improved}

    metrics = {
        "false_dislike_rate": m(b_dislike, c_dislike, c_dislike < b_dislike),
        "select_reason_popularity_rate": m(b_conf, c_conf, c_conf < b_conf),
        "excess_popularity_exposure": m(b_excess, c_excess, max(0, c_excess) < max(0, b_excess)),
        "position_tau": m(b_tau, c_tau, abs(c_tau) < abs(b_tau)),
        "head_tail_update_gap": m(b_gap, c_gap, abs(c_gap) < abs(b_gap)),
        "ndcg_at_5": m(b_ndcg, c_ndcg, quality_gate),
    }

    target = _target_delta(baseline, candidate, targeted_bias)
    reduction = target["relative_reduction"] if target else None
    target_gate = reduction is not None and reduction >= minimum_reduction
    control_gate=True
    if targeted_bias=="position":
        control_gate=False
        if controls:
            b,c=(controls[label]["order_effect"] for label in ("baseline","candidate"))
            complete=all(v["queries"]>0 and v["valid_queries"]==v["queries"] for v in (b,c))
            control_gate=(complete and c["same_order_distance"]<=b["same_order_distance"]+.02
                          and c["reversed_order_distance"]<b["reversed_order_distance"])
        target_gate=target_gate and control_gate

    return {
        "metrics": metrics,
        "quality_gate": quality_gate,
        "target": target,
        "target_gate": target_gate,
        "debiasing_win": target_gate and quality_gate,
        "proxy_goal_pass": target_gate and quality_gate,
        "thresholds": {"quality_floor_ratio": quality_floor, "minimum_target_reduction": minimum_reduction,
                       "position_noise_tolerance":.02},
        "evidence_level": "development_proxy_only",
        "measurement_gate":control_gate,
        "measurement_status":"paired_order_controls" if controls else "requires_order_controls" if targeted_bias=="position" else "descriptive_pattern_screen",
        "interpretation": "Point-estimate screening only, not validated debiasing. Legacy debiasing_win aliases proxy_goal_pass. Position requires paired order/noise controls; legacy tau is descriptive. Selection update gap is action compliance, not evidence quality. No causal or RSI claim.",
    }


def run_arm(config_path, output_dir, operator, data_dir=None, baseline_dir=None):
    """Run one AgentCF arm (train + rank) via the author-check entry point."""
    output_dir = Path(output_dir)
    if (output_dir / "summary.json").exists():
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        return 0 if summary.get("status") == "completed" else 1
    if operator in RANK_ONLY_OPERATORS and baseline_dir:
        cmd = [sys.executable, "-m", "rsi_researcher.research.author_experiment", "rank-only",
               "--source", str(baseline_dir), "--output", str(output_dir),
               "--config", str(config_path), "--operator", operator]
        if operator=="neutral_candidate_order":
            cmd.append("--order-controls")
        return subprocess.run(cmd, cwd=str(ROOT)).returncode
    cmd = [sys.executable, "-m", "rsi_researcher", "author-check",
           "--config", str(config_path), "--output", str(output_dir), "--operator", operator]
    if data_dir:
        cmd += ["--data", str(data_dir)]
    completed = subprocess.run(cmd, cwd=str(ROOT))
    return completed.returncode


def _propose(model, baseline_summary, history, round_index, rounds, allowed_operators, suggested_next=None, patterns=None, prior_cards=None):
    operators = {name: desc for name, desc in AUTHOR_OPERATORS.items() if name in allowed_operators}
    tried = sorted({t["operator"] for t in history})
    payload = {
        "baseline_probes": baseline_summary,
        "current_pattern_evidence":patterns or {},
        "historical_tasks":prior_cards or [],
        "available_operators": operators,
        "operator_targets": OPERATOR_TARGETS,
        "target_metrics": {bias: (f"abs({TARGET_METRICS[bias][2]}) (接近零)" if bias == "position"
                                   else f"{TARGET_METRICS[bias][2]} (越低越好)")
                           for bias in TARGET_METRICS if TARGET_METRICS[bias]},
        "already_tried_operators": tried,
        "suggested_next_operator": suggested_next,
        "prior_trials": [{"round": t["round"], "operator": t["operator"], "comparison": t["comparison"],
                          "lesson": t["learn"].get("lesson")} for t in history],
        "budget": {"round": round_index + 1, "max_rounds": rounds},
    }
    prompt = ("你是推荐系统研究员。下面是当前开发集的偏差探针（Python 预计算）与可用干预算子清单。"
              "每个算子对应一类机制代理指标（见 operator_targets）；指标下降本身不证明真实去偏。"
              "系统已移除此前试过的算子，你只能从 available_operators 中选择，且不要重复已试过的算子。"
              "suggested_next_operator 来自上一轮的实证学习，优先考虑但不必盲从。"
              "选一个算子，提出可证伪的假设和竞争解释，并预测目标指标与质量的方向。"
              "selection_paac 是历史字段名：现在只测学习机会差距，不是 PAAC 复现。均衡重放只改变已有观察的权重，不产生新反馈。"
              "位置相关性需要接近零，负相关也可能是倒序偏好。不能宣称因果；热门下降不等于去偏；未观察反馈保持未知。用中文。\n"
              "historical_tasks 是其他任务的条件性记录，不是当前数据。不得把历史任务的相对变化抄成当前分数。"
              "先依据 current_pattern_evidence 选择要检验的机制；训练重放只验证学习机会，位置须对比同序重复与倒序。\n"
              + json.dumps(payload, ensure_ascii=False))
    return model.ask(prompt, _plan_schema(allowed_operators), {"stage": "discover_bias", "round": round_index + 1})


def _learn(model, plan, comparison, history):
    payload = {"registered_plan": plan, "actual_comparison": comparison,
               "prior_lessons": [t["learn"].get("lesson") for t in history[-3:]]}
    prompt = ("把登记的预测与实际 Python 计算的结果对比。诚实判断机制是被支持、削弱还是无法判断，"
              "给出下一问题和下一算子。不要美化失败；未通过开发均值门槛只说明本次候选不达标，不能将一次结果写成普遍定律或因果证伪。用中文。\n"
              + json.dumps(payload, ensure_ascii=False))
    return model.ask(prompt, LEARN_SCHEMA, {"stage": "learn", "round": len(history) + 1})


def _select(trials):
    """Build a per-bias debiasing map: for each bias, the winning operator (the
    one that best reduced that bias without hurting quality)."""
    wins = [t for t in trials if t.get("comparison") and t["comparison"]["debiasing_win"]]
    best_by_bias = {}
    for t in wins:
        target = t["comparison"]["target"]
        if not target:
            continue  # no-op targets (retain) are not counted as wins
        bias = target["bias"]
        if bias not in best_by_bias or target["change"] < best_by_bias[bias]["comparison"]["target"]["change"]:
            best_by_bias[bias] = t
    if not best_by_bias:
        return {"selected": False, "best_by_bias": {},
                "reason": "没有候选同时通过目标代理指标与质量的开发均值门槛，保留原版。"}
    summary = {bias: {"operator": t["operator"], "round": t["round"],
                      "target_change": t["comparison"]["target"]["change"],
                      "target_candidate": t["comparison"]["target"]["candidate"],
                      "ndcg_change": t["comparison"]["metrics"]["ndcg_at_5"]["change"]}
               for bias, t in best_by_bias.items()}
    return {"selected": True, "best_by_bias": summary,
            "reason": "按目标代理指标选择开发候选；质量均值损失在门槛内，尚未独立验证。"}


def run_loop(config_path, baseline_dir, output_dir, rounds=5, data_dir=None, max_requests=12, max_tokens=60000, max_seconds=3600, journal_path=None, operators=None):
    from rsi_researcher.runtime.models import create_model, model_description
    from rsi_researcher.runtime.limits import RunLimits
    from rsi_researcher.runtime.client import usage_summary

    output_dir = Path(output_dir).resolve()
    baseline_dir = Path(baseline_dir).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    registered=list(PROPOSABLE_OPERATORS if operators is None else operators)
    if not registered or len(set(registered))!=len(registered) or not set(registered)<=set(PROPOSABLE_OPERATORS):
        raise ValueError("registered operators must be a nonempty subset of author interventions")
    inputs = json.loads((baseline_dir / "input.json").read_text(encoding="utf-8"))
    # Compare experiments on the registered cohort and sampling seed, rather
    # than accidentally comparing unrelated people or model settings.
    for key in ("users", "events_per_user", "seed"):
        if config[key] != inputs[key]:
            raise ValueError("baseline and candidate config differ: " + key)
    if (config.get("user_selection", "first") != inputs.get("user_selection", "first")
            or model_description(config) != inputs["model_config"]
            or inputs.get("operator", "retain") != "retain"):
        raise ValueError("loop requires the matching original baseline cohort and model")
    output_dir.mkdir(parents=True, exist_ok=False)
    researcher_dir = output_dir / "researcher"
    researcher_dir.mkdir(parents=True, exist_ok=False)
    limits = RunLimits(researcher_dir, max_requests, max_tokens, max_seconds)
    model = create_model(researcher_dir, limits.max_requests, limits, config)

    baseline_summary = bias_summary(baseline_dir)
    journal=ExperimentJournal(journal_path or ROOT/"runs/agentcf-research-journal.json")
    prior_cards=compact_author_history(journal)
    patterns=describe_run(baseline_dir)
    atomic_json(output_dir/"baseline-patterns.json",patterns)
    (output_dir/"baseline-patterns.md").write_text(render_patterns(patterns),encoding="utf-8")
    archive = {"status": "running", "baseline_dir": str(baseline_dir), "rounds": rounds,
               "baseline_probes": baseline_summary, "trials": [], "prior_research_cards":prior_cards,
               "journal":str(journal.path), "registered_operators":registered}
    atomic_json(output_dir / "archive.json", archive)

    history = []
    tried_operators = set()
    suggested_next = None
    for index in range(rounds):
        allowed = _next_allowed_operators(tried_operators,registered)
        if not allowed:
            archive["stop_reason"] = "registered_operators_exhausted"
            break
        plan = _propose(model, baseline_summary, history, index, rounds, allowed, suggested_next,
                        pattern_brief(patterns),prior_cards)
        operator = plan["operator"]
        if operator not in PROPOSABLE_OPERATORS or operator not in allowed:
            raise InvalidOutput("proposed operator is not permitted")
        targeted_bias = OPERATOR_TARGETS[operator]
        plan["targeted_bias"] = targeted_bias
        tried_operators.add(operator)
        atomic_json(output_dir / f"plan-{index + 1:04d}.json", plan)
        trial_dir = output_dir / f"trial-{index + 1:04d}"
        returncode = run_arm(config_path, trial_dir, operator, data_dir, baseline_dir)
        if returncode != 0 or not (trial_dir / "summary.json").exists():
            record = {"round": index + 1, "operator": operator, "plan": plan,
                      "allowed_operators": allowed, "suggested_next_operator": suggested_next,
                      "execution_status": "failed", "returncode": returncode}
            history.append({"round": index + 1, "operator": operator, "comparison": None,
                            "learn": {"lesson": "execution failed"}, "plan": plan})
            archive["trials"].append(record)
            atomic_json(output_dir / "archive.json", archive)
            continue
        result = json.loads((trial_dir / "summary.json").read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            archive["trials"].append({"round": index + 1, "operator": operator,
                                      "execution_status": "failed", "reason": "incomplete or invalid rankings"})
            atomic_json(output_dir / "archive.json", archive)
            continue
        trial_summary = bias_summary(trial_dir)
        comparison = compare_bias(baseline_summary, trial_summary, targeted_bias)
        comparison["quality_uncertainty"]=compare_saved_quality(baseline_dir,trial_dir)
        trial_patterns=describe_run(trial_dir)
        atomic_json(output_dir/f"patterns-{index+1:04d}.json",trial_patterns)
        (output_dir/f"patterns-{index+1:04d}.md").write_text(render_patterns(trial_patterns),encoding="utf-8")
        learn = _learn(model, plan, comparison, history)
        candidate_id=record_author_trial(journal,trial_dir,plan,comparison,learn,trial_conditions(inputs))
        suggested_next = learn.get("next_operator")
        record = {"round": index + 1, "operator": operator, "plan": plan,
                  "allowed_operators": allowed, "suggested_next_operator": suggested_next,
                  "execution_status": "completed", "candidate_probes": trial_summary,
                  "comparison": comparison, "learn": learn}
        record["candidate_id"]=candidate_id
        record["pattern_brief"]=pattern_brief(trial_patterns)
        history.append({"round": index + 1, "operator": operator,
                        "comparison": comparison, "learn": learn, "plan": plan})
        archive["trials"].append(record)
        atomic_json(output_dir / "archive.json", archive)

    selection = _select(archive["trials"])
    archive["selection"] = selection
    archive["status"] = "completed"
    archive["model_requests"] = model.requests
    archive["researcher_requests"] = model.requests
    archive["total_new_model_requests"] = model.requests + sum(
        usage_summary(output_dir / f"trial-{t['round']:04d}")["requests"]
        for t in archive["trials"]
        if (output_dir / f"trial-{t['round']:04d}" / "calls.jsonl").exists())
    atomic_json(output_dir / "archive.json", archive)
    _render_report(output_dir, archive)
    return archive


def _render_report(output_dir, archive):
    value=lambda number:"未测得" if number is None else f"{number:.3f}"
    lines = ["# AgentCF 自主研究迭代：开发代理指标筛选", "",
             f"状态：{archive['status']}；已执行 {len(archive['trials'])}/{archive['rounds']} 轮；"
             f"研究员请求 {archive.get('researcher_requests', 0)}；含实验的新请求总计 {archive.get('total_new_model_requests', 0)}。", "",
             "每轮：读偏差探针 → Qwen 选算子提干预 → 执行 → Python 按「目标偏差是否下降 + 质量是否保住」判胜负 → Qwen 学习并选下一步。", "",
             "开发筛选要求目标代理指标至少下降 10%，NDCG@5 相对损失不超过 1%；这不是科学定律或最终验证。", "",
             "五类机制均可尝试。选择类只测已有训练记录的重放失衡；没有真实曝光概率，不能声称校正现实中的选择偏差。", ""]
    b = archive["baseline_probes"]
    lines += ["## 基线偏差", "", "| 探针 | 值 |", "|---|---:|",
              f"| 曝光 false_dislike | {b['exposure_bias']['false_dislike_memory_rate']:.4f} |",
              f"| 从众 选品热门先验 | {b['conformity_bias']['select_reason_popularity_rate']:.4f} |",
              f"| 位置 input-order τ | {b['position_bias']['input_order_following_tau']:.4f} |",
              f"| 质量 NDCG@5 | {b['quality_popularity']['ndcg_at_5']:.4f} |",
              f"| 热门倾斜 excess | {b['quality_popularity']['excess_popularity_exposure']:.4f} |", ""]
    lines += ["## 迭代轨迹", ""]
    for t in archive["trials"]:
        c = t.get("comparison")
        if not c:
            lines.append(f"- 第 {t['round']} 轮 `{t['operator']}`：执行失败")
            continue
        tg = c["target"]
        if tg:
            target_str = f"{tg['label']} {value(tg['baseline'])}→{value(tg['candidate'])}"
        else:
            target_str = "无目标（no-op）"
        lines.append(f"- 第 {t['round']} 轮 `{t['operator']}`（目标偏差 {t['plan'].get('targeted_bias')}）："
                     f"{target_str}，NDCG {c['metrics']['ndcg_at_5']['candidate']:.3f} → 开发代理门槛通过 {c['proxy_goal_pass']}")
        collateral = []
        for key, label in [("false_dislike_rate", "false_dislike"),
                           ("select_reason_popularity_rate", "从众"),
                           ("excess_popularity_exposure", "热门倾斜"),
                           ("position_tau", "位置τ"),
                           ("head_tail_update_gap", "head-tail更新差")]:
            mm = c["metrics"][key]
            collateral.append(f"{label} {mm['baseline']:.3f}→{mm['candidate']:.3f}")
        lines.append("  - 全维度：" + "，".join(collateral))
        quality=c.get("quality_uncertainty",{})
        evidence={"INCONCLUSIVE":"尚不能确认质量损失在门槛内",
                  "SUPPORTED_CONDITIONALLY":"当前训练状态下的用户重抽样支持质量门槛",
                  "QUALITY_LOSS_SUPPORTED_CONDITIONALLY":"当前训练状态下的用户重抽样支持超额质量损失",
                  "INVALID_EXECUTION":"执行不完整，不能评价方法效果",
                  "NO_REFERENCE_QUALITY":"没有可用的相对质量参照"}.get(quality.get("quality_evidence"),"未计算")
        lines.append(f"  - 质量证据：{evidence}；尚非独立验证。")
        if c.get("measurement_status")=="requires_order_controls":
            lines.append("  - 缺少同序重复/倒序对照，本轮不判位置机制通过。")
        lines.append(f"  - 学习：{t['learn'].get('lesson', '')}")
    lines += ["", "## 开发候选图谱（未独立验证）", ""]
    sel = archive["selection"]
    if sel.get("selected"):
        lines += ["| 偏差 | 最优算子 | 轮 | 目标指标变化 | NDCG 变化 |", "|---|---|---|---|---|"]
        for bias, info in sel["best_by_bias"].items():
            lines.append(f"| {bias} | `{info['operator']}` | {info['round']} | {info['target_change']:+.3f} | {info['ndcg_change']:+.3f} |")
    else:
        lines.append(f"未找到达标算子。{sel['reason']}")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True, help="a committed baseline run directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--data", help="override the author data directory")
    parser.add_argument("--max-requests", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=60000)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--journal",help="shared author-data research journal; defaults to runs/agentcf-research-journal.json")
    parser.add_argument("--operators",nargs="+",choices=PROPOSABLE_OPERATORS,help="pre-register the mechanisms to test in this budget; default: all five")
    args = parser.parse_args()
    run_loop(args.config, args.baseline, args.output, args.rounds, args.data,
             args.max_requests, args.max_tokens, args.max_seconds,args.journal,args.operators)
