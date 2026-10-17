"""Engineering smoke check on the pinned author data, not a bias experiment."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from rsi_researcher.paths import ROOT
from rsi_researcher.runtime.client import (AUTHOR_COMMIT, RANKING_PROTOCOL, InvalidOutput, ModelFailure, BudgetExceeded, atomic_json, usage_summary)
from rsi_researcher.datasets.author import AUTHOR_DATA_PROTOCOL, author_data, author_profile
from rsi_researcher.evaluation.author import ndcg, score_author_ranking, summarize_author_rankings
from rsi_researcher.recommenders.agentcf import AgentCF
from rsi_researcher.runtime.models import create_model, model_description
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.source_snapshot import snapshot_package, verify_snapshot, module_command
from rsi_researcher.research.author_experiment import author_intervention, training_intervention, AUTHOR_OPERATORS

def run(args):
    # shortcut: one process writes each run directory; add locking before parallel writers.
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    description = model_description(config)
    if description["provider"] != "vllm" or config.get("dataset") != "CDs-100-user-dense":
        raise ValueError("author-check requires the local vLLM backend and author CDs data")
    args.users = args.users if args.users is not None else config["users"]
    args.events_per_user = args.events_per_user if args.events_per_user is not None else config["events_per_user"]
    operator = getattr(args, "operator", "retain")
    if operator not in AUTHOR_OPERATORS:
        raise ValueError("unknown author operator")
    budget = dict(config["budget"])
    if args.max_requests is not None:
        budget["max_requests"] = args.max_requests
    if min(args.users, args.events_per_user) < 1:
        raise ValueError("user and event counts must be positive")
    data = Path(args.data or config["data_dir"])
    data = (ROOT / data if not data.is_absolute() else data).resolve()
    config.update(data_dir=str(data), users=args.users, events_per_user=args.events_per_user, budget=budget)
    directory = Path(args.output).resolve()
    if args.resume:
        inputs = json.loads((directory / "input.json").read_text(encoding="utf-8"))
        operator = inputs.get("operator", "retain")
        if any(inputs[key] != value for key, value in
               (("users", args.users), ("events_per_user", args.events_per_user),
                ("budget", budget), ("author_commit", AUTHOR_COMMIT), ("model_config", description),
                ("data_protocol", AUTHOR_DATA_PROTOCOL), ("seed", config["seed"]),
                ("user_selection", config.get("user_selection", "first")),
                ("ranking_protocol", RANKING_PROTOCOL))):
            raise ValueError("resume inputs differ from the original run")
        verify_snapshot(directory)
        initial, events, queries, prompts = (inputs[key] for key in ("initial", "events", "queries", "prompts"))
        profile = json.loads((directory / "data_profile.json").read_text(encoding="utf-8"))
    else:
        initial, events, queries = author_data(data, args.users, args.events_per_user, config["seed"], config.get("user_selection", "first"))
        source_events = deepcopy(events)
        events, training_schedule = training_intervention(events, operator, config["seed"])
        profile = author_profile(data)
        prompts = json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8"))
        if operator != "retain":
            prompts = author_intervention(prompts, operator)
        inputs = {"author_commit": AUTHOR_COMMIT, "initial": initial, "events": events, "queries": queries,
                  "prompts": prompts, "operator": operator, "training_schedule": training_schedule,
                  "source_events": source_events,
                  "users": args.users, "events_per_user": args.events_per_user,
                  "budget": budget, "ranking_protocol": RANKING_PROTOCOL, "model_config": description,
                  "data_protocol": AUTHOR_DATA_PROTOCOL, "seed": config["seed"],
                  "user_selection": config.get("user_selection", "first"),
                  "purpose": "D1 engineering smoke, not a bias experiment"}
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / "input.json", inputs)
        atomic_json(directory / "config.json", config)
        atomic_json(directory / "model_config.json", description)
        atomic_json(directory / "data_profile.json", profile)
        snapshot_package(directory)
    limits = RunLimits(directory, **budget)
    model = create_model(directory, limits.max_requests, limits, config)
    try:
        return execute(directory, model, initial, events, queries, prompts, profile, args.resume)
    finally:
        if (directory / "calls.jsonl").exists():
            atomic_json(directory / "usage.json", usage_summary(directory))


def execute(directory, model, initial, events, queries, prompts, profile, resume=False):
    kernel = AgentCF(model, prompts)
    for arm in ("frozen", "original_update"):
        model.scope = {"arm": arm}
        arm_dir = directory / arm
        arm_dir.mkdir(exist_ok=resume)
        state = deepcopy(initial)
        if not (arm_dir / "state.json").exists():
            atomic_json(arm_dir / "state.json", state)
        # Frozen means initial memory: generating discarded reflections has no effect.
        for event in events if arm == "original_update" else ():
            committed = arm_dir / (event["event_id"] + ".json")
            if committed.exists():
                record = json.loads(committed.read_text(encoding="utf-8"))
                if record["event"] != event or record["before"] != state:
                    raise ValueError("committed event chain is inconsistent")
                state = record["after"]
                atomic_json(arm_dir / "state.json", state)
                continue
            try:
                after = kernel.event(state, event, arm)
            except (InvalidOutput, ModelFailure, BudgetExceeded) as exc:
                atomic_json(arm_dir / (event["event_id"] + ".failed.json"), {"event": event, "error": type(exc).__name__})
                raise
            atomic_json(committed, {"event": event, "before": state, "after": after, "memory_policy": arm})
            atomic_json(arm_dir / "state.json", after)
            state = after
            print(f"{arm}: committed {event['event_id']}", flush=True)
        results = []
        if (arm_dir / "ranking.json").exists():
            continue
        for query in queries:
            try:
                ranking = kernel.rank(state, query)
                result = {"ranking": ranking, "valid": True, **score_author_ranking(ranking, query, profile)}
            except InvalidOutput as exc:
                result = {"ranking": [], "valid": False, "error": type(exc).__name__,
                          **score_author_ranking([], query, profile)}
            results.append({"query": query, **result})
            print(f"{arm}: ranked {query['query_id']} valid={result['valid']}", flush=True)
        atomic_json(arm_dir / "ranking.json", results)
    summary = usage_summary(directory)
    summary["arms"] = {}
    for arm in ("frozen", "original_update"):
        results = json.loads((directory / arm / "ranking.json").read_text(encoding="utf-8"))
        summary["arms"][arm] = summarize_author_rankings(results)
    summary["status"] = "completed" if not any(a["invalid_rankings"] for a in summary["arms"].values()) else "failed"
    summary["frozen_training_requests"] = 0
    summary["purpose"] = "engineering smoke only; no scientific acceptance decision"
    atomic_json(directory / "summary.json", summary)
    write_report(directory, summary, profile, initial, events)
    return summary


def write_report(directory, summary, profile, initial, events):
    lines = ["# Qwen 与作者 AgentCF 数据：真实流程检查", "",
             "这是接口与执行检查；少量用户的分数不能证明去偏有效或 RSI 成立。", "",
             f"作者数据共 {profile['training_users']} 人、{profile['training_events']} 条训练目标事件；作者候选池每人 100 条，实际排序为 9 个陪选加 1 个目标。",
             "`.random` 没有评分或真实曝光记录；其他候选的偏好未知。本次只用训练与 valid，不读取 test。", "",
             "| 记忆设置 | NDCG@5 | Hit@1 | Hit@5 | 前5热门占比 | 无效排序 |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, row in summary["arms"].items():
        lines.append(f"| {arm} | {row['mean_ndcg_at_5']:.4f} | {row['mean_hit_at_1']:.4f} | {row['mean_hit_at_5']:.4f} | {row['mean_top5_head_share']:.4f} | {row['invalid_rankings']} |")
    lines += ["", "冻结对照直接使用初始记忆，没有生成随后丢弃的训练反思。更新组按两轮选择与反思更新用户和正物品；负物品的生成描述不写入共享记忆。",
              "热门占比是描述性观察，尚未证明额外偏差；训练频次来自训练目标事件，历史前缀不重复累加。", "",
              f"实际请求 {summary['requests']} 次；完整证据见 input.json、data_profile.json、calls.jsonl 和 original_update 下的事件文件。"]
    if events:
        event = events[0]
        record = json.loads((directory / "original_update" / (event["event_id"] + ".json")).read_text(encoding="utf-8"))
        lines += ["", "## 第一条真实事件", "", f"用户 `{event['user']}`，已观察物品：{initial['items'][event['positive']]['title']}。抽样比较物品不是明确差评。", "",
                  "更新前用户记忆：", "", record["before"]["users"][event["user"]], "",
                  "更新后用户记忆：", "", record["after"]["users"][event["user"]], "",
                  "正反馈物品更新后描述：", "", record["after"]["items"][event["positive"]]["memory"]]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_arguments(parser):
    parser.add_argument("--config", default=str(ROOT / "configs/qwen_agentcf_author.json"))
    parser.add_argument("--data", help="override the author data directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--users", type=int)
    parser.add_argument("--events-per-user", type=int)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--operator", choices=sorted(AUTHOR_OPERATORS), default="retain",
                        help="registered prompt or training-schedule intervention; retain = baseline unchanged")
    parser.add_argument("--resume", action="store_true", help="resume the same inputs and request budget from committed events")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)


def dispatch(options):
    if options.resume and not options.worker:
        verify_snapshot(options.output)
        saved = Path(options.output).resolve()
        command = module_command(options.output, "rsi_researcher.diagnostics.agentcf_smoke",
                                 ["--output", str(saved), "--config", str(saved / "config.json"), "--resume", "--worker"])
        result = subprocess.run(command, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return result.returncode
    try:
        result = run(options)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "completed" else 1
    except (InvalidOutput, ModelFailure, BudgetExceeded, ValueError) as exc:
        print(f"Stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    sys.exit(dispatch(parser.parse_args()))
