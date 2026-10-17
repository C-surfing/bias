"""Freeze a development candidate and test it in new controlled environments.

This is development validation, not live A/B testing or an untouched final test.
The workflow never asks a researcher to redesign a candidate during validation.
"""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import statistics

from rsi_researcher.datasets.controlled import build_world, ordered_events
from rsi_researcher.recommenders.execution import execute_arm, score_arm
from rsi_researcher.research.evidence import build_candidate_evidence
from rsi_researcher.research.operators import OPERATORS, execute_candidate
from rsi_researcher.runtime.client import BudgetExceeded, ModelFailure, atomic_json


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_trial(source_batch, trial_id):
    batch = Path(source_batch).resolve()
    trials = batch / "trials"
    choices = [trials / trial_id] if trial_id else sorted(trials.iterdir())
    for directory in reversed(choices):
        if not (directory / "archive.json").exists():
            continue
        archive = _read(directory / "archive.json")
        completed = archive.get("execution_status") == "completed"
        if completed and archive.get("proposal") and (directory / "candidate/state.json").exists():
            return directory, archive
    raise ValueError("no completed development trial with an executed candidate was found")


def _environments(protocol, strategy, source_seed):
    dimensions = {key: strategy["world"][key] for key in ("users", "items", "events_per_user")}
    dimensions.update(protocol.get("world", {}))
    if "environments" in protocol:
        environments = [dimensions | entry for entry in protocol["environments"]]
    else:
        environments = [dimensions | {"seed": seed, "stratum": stratum}
                        for seed in protocol["seeds"] for stratum in protocol["strata"]]
    if not environments:
        raise ValueError("validation requires at least one predeclared environment")
    identities = [(entry["seed"], entry["stratum"]) for entry in environments]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate validation environments do not constitute independent repetitions")
    if any(seed == source_seed for seed, _ in identities):
        raise ValueError("validation seeds must differ from the development seed")
    return environments


def _summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seed"]].append(row)
    by_seed = []
    for seed, conditions in grouped.items():
        complete = all(row["status"] == "completed" for row in conditions)
        by_seed.append({"seed": seed, "environment_count": len(conditions),
                        "all_conditions_completed": complete,
                        "quality_change": statistics.mean(r["comparison"]["quality_change"] for r in conditions) if complete else None,
                        "bias_change": statistics.mean(r["comparison"]["bias_change"] for r in conditions) if complete else None,
                        "failed_conditions": [r["environment_id"] for r in conditions if r["status"] != "completed"]})
    completed = [row for row in rows if row["status"] == "completed"]
    all_complete = len(completed) == len(rows)
    return {"planned_environments": len(rows), "completed_environments": len(completed),
            "failed_or_unrun_environments": len(rows) - len(completed), "by_seed": by_seed,
            "quality_change": statistics.mean(r["comparison"]["quality_change"] for r in rows) if all_complete else None,
            "bias_change": statistics.mean(r["comparison"]["bias_change"] for r in rows) if all_complete else None,
            "all_quality_gates_passed": all(r["comparison"]["quality_gate"] for r in rows) if all_complete else False,
            "all_bias_gates_passed": all(r["comparison"]["bias_gate"] is True for r in rows) if all_complete else False,
            "invalid_rankings": sum(sum(score["invalid_rankings"] for score in row["metrics"].values()) for row in completed),
            "statistical_support": "not_tested", "promotion": "none"}


def _save_result(directory, result):
    atomic_json(directory / "result.json", result)
    lines = ["# 冻结候选的独立环境验证", "", "这是开发验证；不是线上 A/B，也不是未见的最终测试。", "",
             "候选选择：" + result["selection_reason"], "", "执行状态：" + result["status"], "",
             "| 新环境 | 执行状态 | 质量变化 | 偏差变化 |", "|---|---|---:|---:|"]
    for row in result.get("environments", []):
        comparison = row.get("comparison")
        quality = f"{comparison['quality_change']:+.4f}" if comparison else "未完成"
        bias = f"{comparison['bias_change']:+.4f}" if comparison else "未完成"
        lines.append(f"| {row['environment_id']} | {row['status']} | {quality} | {bias} |")
    if result.get("reason"):
        lines += ["", result["reason"]]
    lines += ["", "协议要求每个新环境重新生成输入并执行原版与候选；不复用开发排名。实际完成与失败条件逐项保留。",
              "", "验证结果回写研究日志后，会成为下一轮开发历史；不能再称作未见最终测试。"]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconcile_validation(directory, terminal, journal=None):
    """Finish a running validation record when its worker was interrupted."""
    directory = Path(directory).resolve()
    path = directory / "result.json"
    if not path.exists():
        return None
    result = _read(path)
    if result["status"] != "running":
        return result
    result["status"] = terminal["status"] if terminal["status"] in ("failed", "timed_out") else "incomplete"
    result["reason"] = "父进程已结束验证工作进程；未完成的条件保留，不作为已验证结果。"
    for row in result["environments"]:
        if row["status"] == "running":
            row["status"] = "interrupted"
        elif row["status"] == "pending":
            row["status"] = "not_run"
    result["summary"] = _summarize(result["environments"])
    _save_result(directory, result)
    if journal is not None and result.get("candidate_id"):
        journal.record_validation(result["candidate_id"], result)
    return result


def run_validation(directory, model, source_batch, protocol, journal=None, trial_id=None, prepared=False):
    """Execute a fixed candidate under a predeclared independent-world protocol."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=prepared)
    source, archive = _source_trial(source_batch, trial_id)
    strategy, source_world = _read(source / "strategy.json"), _read(source / "world.json")
    proposal, prompts = deepcopy(archive["proposal"]), _read(source / "baseline_prompts.json")
    environments = _environments(protocol, strategy, source_world["protocol"]["seed"])
    strength = strategy["rerank_strength"]
    frozen = {"proposal": proposal, "prompts": prompts, "rerank_strength": strength,
              "source_model_execution": _read(source / "protocol.json"),
              "validation_model_execution": deepcopy(getattr(model, "metadata", {}))}
    atomic_json(directory / "frozen_candidate.json", frozen)
    atomic_json(directory / "protocol.json", deepcopy(protocol) | {
        "environments": environments, "evidence_scope": "development_validation_not_final_test",
        "candidate_redesign": False, "online_experiment": False, "statistical_promotion": False})
    candidate_id = journal.record_trial(source, archive) if journal is not None else None
    result = {"status": "running", "directory": str(directory), "source_trial": str(source),
              "candidate_id": candidate_id, "operator": proposal["operator"], "purpose": proposal["purpose"],
              "selection_reason": "指定开发候选，尚未宣称最优。" if trial_id else "最近完成且已实际执行的开发候选，尚未宣称最优。",
              "evidence_scope": "development_validation_not_final_test", "online_experiment": False,
              "statistical_support": "not_tested", "promotion": "none", "environments": []}
    if proposal["operator"] == "dependency_replay":
        result.update(status="needs_new_root_rule", reason="源事件身份只属于原开发环境；需要可转移的根事件选择规则，不能把旧事件编号映射到新环境。")
    elif not OPERATORS[proposal["operator"]]["executable"]:
        result.update(status="candidate_not_executable", reason="该提案没有可直接迁移执行的候选动作。")
    else:
        result["environments"] = [{"environment_id": f"{index:03d}-seed-{environment['seed']}-{environment['stratum']}",
                                    "seed": environment["seed"], "stratum": environment["stratum"], "status": "pending"}
                                   for index, environment in enumerate(environments, 1)]
        _save_result(directory, result)
        stop_status = None
        for index, environment in enumerate(environments, 1):
            row = result["environments"][index - 1]
            environment_id = row["environment_id"]
            row["status"] = "not_run_" + stop_status if stop_status else "running"
            if stop_status:
                continue
            _save_result(directory, result)
            target = directory / "environments" / environment_id
            target.mkdir(parents=True)
            try:
                world = build_world(**environment)
                atomic_json(target / "world.json", world)
                events = ordered_events(world["public"]["events"]["high"], world["private"]["popular"], "pair_0")
                pairs = {}
                for arm, arm_events in (("initial", []), ("B0", events)):
                    model.scope = {"validation_environment": environment_id, "research_arm": arm}
                    pairs[arm] = execute_arm(target / arm, model, prompts, world["public"], arm_events)
                model.scope = {"validation_environment": environment_id, "research_arm": "candidate", "operator": proposal["operator"]}
                pairs["candidate"] = execute_candidate(target / "candidate", model, prompts, world["public"],
                                                       target / "B0", proposal, strength)
                metrics = {arm: score_arm(world, pair) for arm, pair in pairs.items()}
                for arm, score in metrics.items():
                    atomic_json(target / arm / "score.json", score)
                baseline, candidate = metrics["B0"], metrics["candidate"]
                quality_ratio, bias_ratio = protocol.get("quality_floor_ratio", .99), protocol.get("bias_ceiling_ratio", .90)
                comparison = {"quality_change": candidate["utility"] - baseline["utility"],
                              "bias_change": candidate["bias"] - baseline["bias"],
                              "quality_gate": candidate["utility"] >= quality_ratio * baseline["utility"],
                              "bias_gate": candidate["bias"] <= bias_ratio * baseline["bias"] if baseline["bias"] >= .001 else None,
                              "minimum_reference_bias": .001, "quality_floor_ratio": quality_ratio, "bias_ceiling_ratio": bias_ratio}
                comparison["interpretation"] = ("mechanism_probe_descriptive_not_adoption_test"
                                                  if proposal["purpose"] == "mechanism_probe" else "debias_validation_no_promotion")
                evidence = build_candidate_evidence(world, target / "B0", target / "candidate", pairs["B0"],
                                                    pairs["candidate"], proposal["operator"])
                atomic_json(target / "candidate_evidence.json", evidence)
                atomic_json(target / "comparison.json", comparison)
                row.update(status="completed", metrics=metrics, comparison=comparison,
                           evidence_path=(Path("environments") / environment_id / "candidate_evidence.json").as_posix())
            except Exception as exc:
                if isinstance(exc, BudgetExceeded):
                    stop_status = "budget"
                elif isinstance(exc, ModelFailure) and exc.http_status in (401, 402, 403):
                    stop_status = "model_unavailable"
                row.update(status="budget_exhausted" if stop_status == "budget" else stop_status or "failed",
                           error=type(exc).__name__)
                if isinstance(exc, ModelFailure):
                    row["http_status"] = exc.http_status
                atomic_json(target / "failure.json", {key: row[key] for key in ("status", "error", "http_status") if key in row})
            _save_result(directory, result)
        result["summary"] = _summarize(result["environments"])
        result["status"] = "completed" if all(r["status"] == "completed" for r in result["environments"]) else "incomplete"
    _save_result(directory, result)
    if journal is not None:
        journal.record_validation(candidate_id, result)
    return result
