"""E2 paired memory mechanism matrix; no candidate search or statistical promotion."""

import argparse
from copy import deepcopy
import csv
from itertools import product
import json
from pathlib import Path
import random
import statistics
import sys

from rsi_researcher.runtime.client import API_PROTOCOL, MEMORY_CONTROL_PROTOCOL, MODEL_PARAMETERS, RANKING_PROTOCOL, InvalidOutput, StrictDeepSeek, atomic_json, usage_summary
from rsi_researcher.paths import ROOT
from rsi_researcher.datasets.controlled import EVALUATION_PANEL_SIZE, WORLD_PROTOCOL, build_world, ordered_events
from rsi_researcher.recommenders.execution import execute_arm, score_arm
from rsi_researcher.evaluation.learning import audit_learning
from rsi_researcher.evaluation.pilot_report import summarize as summarize_pilot
from rsi_researcher.experiments.pilot import supervise
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.source_snapshot import snapshot_package, verify_snapshot, module_command


def matrix_schedule(protocol, matrix_path=ROOT / "configs/mechanism_matrix.csv"):
    with Path(matrix_path).open(encoding="utf-8-sig", newline="") as stream:
        cells = list(csv.DictReader(stream))
    factors = [(r["exposure_A"], r["prompt_position_P"], r["memory_M"]) for r in cells]
    expected = set(product(("low", "high"), ("paired_balanced", "head_first"), ("frozen", "original_update")))
    if len(factors) != 8 or set(factors) != expected or len({r["cell_id"] for r in cells}) != 8:
        raise ValueError("matrix must contain the eight unique planned cells")
    if any(not r["cell_id"].startswith("E2-") or not r["cell_id"][3:].isdigit() for r in cells):
        raise ValueError("unsafe matrix cell ID")
    seeds, strata = protocol["seeds"], protocol["strata"]
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty unique integers")
    if len(strata) != 2 or set(strata) != {"unrelated", "aligned"} or type(protocol["order_seed"]) is not int:
        raise ValueError("both fixed popularity strata and an integer order seed are required")
    schedule = []
    for seed, stratum in product(seeds, strata):
        for cell in cells:
            positions = ("pair_0", "pair_1") if cell["prompt_position_P"] == "paired_balanced" else ("head_first",)
            for position in positions:
                schedule.append({"seed": seed, "stratum": stratum, "cell_id": cell["cell_id"],
                                 "exposure": cell["exposure_A"], "position": position, "memory_policy": cell["memory_M"]})
        for policy, position in product(("user_only", "item_only", "no_sharing"), ("pair_0", "pair_1")):
            schedule.append({"seed": seed, "stratum": stratum, "cell_id": "ablation-" + policy,
                             "exposure": "high", "position": position, "memory_policy": policy})
    random.Random(protocol["order_seed"]).shuffle(schedule)
    return schedule


def case_id(spec):
    return f"{spec['seed']}-{spec['stratum']}/{spec['cell_id']}-{spec['position']}"


def summarize_matrix(protocol, schedule, rows, failures, terminal_status=None):
    expected = {case_id(s): s for s in schedule}
    observed = {r["case"]: r for r in rows}
    failed = {r["case"]: r for r in failures}
    if (len(observed) != len(rows) or len(failed) != len(failures) or observed.keys() & failed.keys()
            or not observed.keys() <= expected.keys() or not failed.keys() <= expected.keys()):
        raise ValueError("duplicate, overlapping or unexpected matrix results")
    for row in rows + failures:
        if row["spec"] != expected[row["case"]]:
            raise ValueError("matrix result factors do not match the frozen schedule")
    worlds = []
    for seed, stratum in product(protocol["seeds"], protocol["strata"]):
        cells = {}
        for spec in schedule:
            if spec["seed"] == seed and spec["stratum"] == stratum:
                cells.setdefault(spec["cell_id"], []).append(case_id(spec))
        means = {}
        for cell, ids in cells.items():
            means[cell] = ({key: statistics.mean(observed[i]["score"][key] for i in ids)
                           for key in ("utility", "bias", "signed_excess", "position_top1_disagreement")}
                          if all(i in observed for i in ids) else None)
        interactions = {}
        for position in ("paired_balanced", "head_first"):
            matched = [s for s in schedule if s["seed"] == seed and s["stratum"] == stratum
                       and not s["cell_id"].startswith("ablation-")
                       and (s["position"] == "head_first") == (position == "head_first")]
            table = {(s["exposure"], s["memory_policy"]): means[s["cell_id"]] for s in matched}
            interactions[position] = (table["high", "original_update"]["bias"] - table["low", "original_update"]["bias"]
                                      - table["high", "frozen"]["bias"] + table["low", "frozen"]["bias"]
                                      if all(table.values()) else None)
        worlds.append({"seed": seed, "stratum": stratum, "cells": means, "bias_interactions": interactions})
    seed_effects = []
    for seed in protocol["seeds"]:
        paired = [w for w in worlds if w["seed"] == seed]
        if all(all(v is not None for v in w["bias_interactions"].values()) for w in paired):
            values = {p: statistics.mean(w["bias_interactions"][p] for w in paired)
                      for p in ("paired_balanced", "head_first")}
            seed_effects.append({"seed": seed, **values, "position_interaction_difference": values["head_first"] - values["paired_balanced"]})
    all_recorded = len(observed) == len(expected) and not failed
    return {"complete": all_recorded and terminal_status == "completed", "all_pools_recorded": all_recorded,
            "terminal_status": terminal_status,
            "completed_pools": len(observed), "expected_pools": len(expected), "failed_pools": list(failed.values()),
            "missing_pools": sorted(expected.keys() - observed.keys()), "worlds": worlds, "paired_seed_effects": seed_effects,
            "independent_seed_count": len(seed_effects), "statistical_support": "not_tested", "candidate_promotion": "not_permitted",
            "scope": "Training position factor only; all arms use identical forward/reverse evaluation panels. "
                     "Complementary position pools and popularity strata are averaged within seed, not counted as independent repetitions."}


def run_matrix(directory, protocol, prompts, model, resume=False):
    directory = Path(directory)
    schedule = (json.loads((directory / "schedule.json").read_text(encoding="utf-8"))
                if resume else matrix_schedule(protocol))
    for name, value in (("protocol.json", protocol), ("prompts.json", prompts), ("schedule.json", schedule)):
        path = directory / name
        if path.exists():
            if not resume or json.loads(path.read_text(encoding="utf-8")) != value:
                raise ValueError("matrix inputs differ from the frozen batch")
        else:
            if resume:
                raise ValueError("cannot resume a matrix without frozen inputs")
            atomic_json(path, value)
    rows_path, failures_path = directory / "completed_cells.json", directory / "failed_cells.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8")) if resume and rows_path.exists() else []
    failures = json.loads(failures_path.read_text(encoding="utf-8")) if resume and failures_path.exists() else []
    summarize_matrix(protocol, schedule, rows, failures)
    completed = {r["case"]: r for r in rows}
    failed = {r["case"] for r in failures}
    worlds = {}
    for seed, stratum in product(protocol["seeds"], protocol["strata"]):
        world_id = f"{seed}-{stratum}"
        world = build_world(seed, protocol["users"], protocol["items"], protocol["events_per_user"], stratum)
        world_dir = directory / world_id
        world_dir.mkdir(exist_ok=resume)
        path = world_dir / "world.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != world:
            raise ValueError("matrix world differs from the frozen generator")
        atomic_json(path, world)
        worlds[world_id] = world
    for spec in schedule:
        identifier = case_id(spec)
        if identifier in failed:
            continue
        world = worlds[f"{spec['seed']}-{spec['stratum']}"]
        events = ordered_events(world["public"]["events"][spec["exposure"]], world["private"]["popular"], spec["position"])
        model.scope = {"case": identifier, "world": f"{spec['seed']}-{spec['stratum']}", "memory_policy": spec["memory_policy"]}
        atomic_json(directory / "progress.json", {"case": identifier, "completed_pools": len(rows), "failed_pools": len(failures)})
        arm_dir = directory / identifier
        if identifier in completed and any(not (arm_dir / name).exists() for name in ("rankings.json", "state.json", "evaluation.json")):
            raise ValueError("completed pool is missing its recorded artifacts")
        try:
            pair = execute_arm(arm_dir, model, prompts, world["public"], events, resume=resume and arm_dir.exists(),
                               memory_policy=spec["memory_policy"])
        except InvalidOutput as exc:
            failure = {"case": identifier, "spec": spec, "error_type": type(exc).__name__, "last_request": model.requests}
            failures.append(failure)
            atomic_json(arm_dir / "failure.json", failure)
            atomic_json(failures_path, failures)
            continue
        row = {"case": identifier, "spec": spec, "score": score_arm(world, pair),
               "learning_audit": audit_learning(world, events, pair)}
        if identifier in completed:
            if row != completed[identifier] or json.loads((arm_dir / "evaluation.json").read_text(encoding="utf-8")) != row:
                raise ValueError("completed pool score disagrees with its saved records")
            continue
        atomic_json(arm_dir / "evaluation.json", row)
        rows.append(row)
        atomic_json(rows_path, rows)
    return write_matrix_report(directory)


def write_matrix_report(directory, terminal_status=None):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    protocol, schedule = read("protocol.json"), read("schedule.json")
    rows = read("completed_cells.json") if (directory / "completed_cells.json").exists() else []
    failures = read("failed_cells.json") if (directory / "failed_cells.json").exists() else []
    summary = summarize_matrix(protocol, schedule, rows, failures, terminal_status)
    atomic_json(directory / "matrix_summary.json", summary)
    labels = {"low": "低曝光", "high": "高曝光", "pair_0": "正反位置配对", "pair_1": "正反位置配对", "head_first": "热门在前",
              "original_update": "正常更新", "frozen": "不保存更新", "user_only": "只留用户笔记", "item_only": "只留物品笔记",
              "no_sharing": "每人自己的物品笔记"}
    number = lambda value: "未完成" if value is None else f"{value:.4f}"
    lines = ["# 机制对照记录", "", "批次已完整执行。" if summary["complete"] else "批次未确认完整执行。",
             f"保存完整结果的笔记池 {summary['completed_pools']}/{summary['expected_pools']}；完成全部因素配对的独立随机编号 {summary['independent_seed_count']} 个。",
             "", "| 世界与热门关系 | 对照 | 推荐质量 | 额外热门倾斜 |", "|---|---|---:|---:|"]
    for world in summary["worlds"]:
        for cell, values in sorted(world["cells"].items()):
            spec = next(s for s in schedule if s["seed"] == world["seed"] and s["stratum"] == world["stratum"] and s["cell_id"] == cell)
            name = " / ".join(labels[spec[key]] for key in ("exposure", "position", "memory_policy"))
            lines.append(f"| {world['seed']} / {world['stratum']} | {name} | {number(values['utility'] if values else None)} | {number(values['bias'] if values else None)} |")
    lines += ["", "| 世界与热门关系 | 位置配对下的记忆交互 | 热门在前时的记忆交互 |", "|---|---:|---:|"]
    for world in summary["worlds"]:
        interactions = world["bias_interactions"]
        lines.append(f"| {world['seed']} / {world['stratum']} | {number(interactions['paired_balanced'])} | {number(interactions['head_first'])} |")
    lines += ["", "记忆交互 =（高曝光正常更新 − 低曝光正常更新）−（高曝光不保存更新 − 低曝光不保存更新），这里比较额外热门倾斜。",
              "正反位置分别用独立笔记池运行，再合为同一个对照；同一随机编号的两个热门关系等权。它们和池内用户不能充当额外独立重复。",
              "位置因素只改变训练物品对的排列，所有组的最终评价都使用相同的正反候选面板。unrelated 表示热门与偏好无关的生成层，aligned 表示两者有关的层。",
              "缺失对照保持未完成，不补成零、不删除。这里只提供描述性结果，没有统计支持、候选晋升或 RSI 结论。"]
    if failures:
        lines += ["", "格式失败的笔记池仍保留在原计划中："] + [f"- {f['case']}：{f['error_type']}。" for f in failures]
    if terminal_status == "failed" and (directory / "worker_failure.json").exists():
        failure = read("worker_failure.json")
        lines += ["", f"运行停止：{failure['error_type']}；HTTP 状态 {failure.get('http_status')}。详细记录保留在 calls.jsonl。"]
    (directory / "matrix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def validate_protocol(protocol):
    if protocol.get("status") != "frozen_exploratory" or protocol.get("purpose") != "E2_mechanism_only":
        raise ValueError("E2 requires a frozen exploratory protocol, not a draft or a promotion experiment")
    fixed = {"model_parameters": MODEL_PARAMETERS, "api_protocol": API_PROTOCOL, "ranking_protocol": RANKING_PROTOCOL,
             "memory_control_protocol": MEMORY_CONTROL_PROTOCOL, "position_scope": "training_pairs_only",
             "primary_contrast": "paired_balanced_bias_interaction", "candidate_promotion_allowed": False,
             "world_protocol": WORLD_PROTOCOL}
    if any(protocol.get(k) != v for k, v in fixed.items()) or protocol.get("candidate_promotion_allowed") is not False:
        raise ValueError("protocol disagrees with the common model, position, or evaluation rules")
    if any(type(protocol.get(k)) is not int or protocol[k] <= 0 for k in
           ("users", "items", "events_per_user", "max_requests", "max_tokens", "max_seconds")):
        raise ValueError("world sizes and global budgets must be positive integers")
    if protocol["items"] < 12 or protocol["events_per_user"] >= protocol["items"]:
        raise ValueError("need at least 12 items and fewer events per user than items")
    if protocol.get("evaluation_panel_size") != min(protocol["items"], EVALUATION_PANEL_SIZE):
        raise ValueError("evaluation panels must use the fixed twenty-item policy")
    matrix_schedule(protocol)
    pilot = Path(protocol["pilot_run"])
    if not pilot.is_absolute():
        raise ValueError("pilot_run must be an absolute path")
    read = lambda name: json.loads((pilot / name).read_text(encoding="utf-8"))
    pilot_protocol = read("protocol.json")
    if (len(pilot_protocol["seeds"]) != 3 or set(pilot_protocol["strata"]) != {"unrelated", "aligned"}
            or any(pilot_protocol.get(k) != v for k, v in {"users": 12, "items": 60, "events_per_user": 4,
                   "api_protocol": API_PROTOCOL, "ranking_protocol": RANKING_PROTOCOL,
                   "model_parameters": MODEL_PARAMETERS,
                   "world_protocol": WORLD_PROTOCOL, "evaluation_panel_size": EVALUATION_PANEL_SIZE}.items())):
        raise ValueError("pilot must use the planned six worlds and common inference protocol")
    if not summarize_pilot(pilot_protocol, read("completed_worlds.json"), read("run_result.json")["status"])["complete"]:
        raise ValueError("pilot is incomplete; E2 paid execution is not ready")
    if set(protocol["seeds"]) & set(pilot_protocol["seeds"]):
        raise ValueError("E2 seeds must be independent of the pilot seeds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", help="frozen E2 exploratory protocol; required for a new batch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    if args.worker:
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        try:
            limits = RunLimits(directory, protocol["max_requests"], protocol["max_tokens"], protocol["max_seconds"])
            prompts = json.loads((directory / "prompts.json").read_text(encoding="utf-8"))
            run_matrix(directory, protocol, prompts, StrictDeepSeek(directory, protocol["max_requests"], limits), resume=True)
        except Exception as exc:
            atomic_json(directory / "worker_failure.json", {"error_type": type(exc).__name__, "http_status": getattr(exc, "http_status", None)})
            write_matrix_report(directory, "failed")
            sys.exit(1)
        sys.exit(0)
    if args.resume:
        if args.protocol:
            parser.error("resume uses the saved protocol; --protocol cannot replace it")
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        prior = json.loads((directory / "run_result.json").read_text(encoding="utf-8"))
        history_path = directory / "continuations.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        if prior["status"] != "failed" or len(history) >= 2:
            raise ValueError("only a failed matrix can continue, at most twice")
        history.append(prior)
    else:
        if not args.protocol:
            parser.error("a new batch needs --protocol")
        protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
        history = []
    validate_protocol(protocol)
    if args.resume:
        verify_snapshot(directory)
        limits = RunLimits(directory, protocol["max_requests"], protocol["max_tokens"], protocol["max_seconds"])
        limits.remaining_seconds()
        atomic_json(directory / "continuations.json", history)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / "protocol.json", protocol)
        atomic_json(directory / "prompts.json", json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8")))
        atomic_json(directory / "schedule.json", matrix_schedule(protocol))
        snapshot_package(directory)
        limits = RunLimits(directory, protocol["max_requests"], protocol["max_tokens"], protocol["max_seconds"])
    atomic_json(directory / "run_result.json", {"status": "running"})
    result = supervise(module_command(directory, "rsi_researcher.experiments.matrix", ["--worker", "--output", str(directory)]),
                       limits.remaining_seconds(), directory / f"worker-{len(history)}.log")
    if result["status"] == "completed" and not json.loads((directory / "matrix_summary.json").read_text(encoding="utf-8"))["all_pools_recorded"]:
        result["status"] = "incomplete"
    atomic_json(directory / "run_result.json", result)
    write_matrix_report(directory, result["status"])
    if (directory / "calls.jsonl").exists():
        try:
            atomic_json(directory / "usage.json", usage_summary(directory))
        except json.JSONDecodeError:
            atomic_json(directory / "usage_incomplete.json", {"reason": "interrupted log record; raw log retained"})
    print(json.dumps(result))
    sys.exit(0 if result["status"] == "completed" else 1)
