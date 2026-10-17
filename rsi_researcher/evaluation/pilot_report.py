"""Summarize completed pilot worlds without treating users as independent trials."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


def request_diagnostics(records):
    starts, received, failures = {}, {}, {}
    for row in records:
        if row["status"] == "started":
            if row["call_id"] in starts:
                raise ValueError("duplicate request ID")
            starts[row["call_id"]] = row
        elif row["status"] in ("received", "failed"):
            target = received if row["status"] == "received" else failures
            if row["call_id"] not in starts or row["call_id"] in target:
                raise ValueError("response without unique preceding request")
            target[row["call_id"]] = row
    stages, latencies, start_times, end_times = {}, [], [], []
    unknown_usage, recorded_seconds = 0, 0.0
    for call_id, start in starts.items():
        stage = start["context"].get("stage", "unknown")
        counts = stages.setdefault(stage, {"requests": 0, "responses": 0, "transport_failures": 0})
        counts["requests"] += 1
        counts["responses"] += call_id in received
        counts["transport_failures"] += call_id in failures
        response = received.get(call_id)
        usage = (response or {}).get("usage")
        known = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens"))
        unknown_usage += not known
        timestamp = start.get("time_utc")
        if timestamp:
            epoch = datetime.fromisoformat(timestamp).timestamp()
            start_times.append(epoch)
            terminal = response or failures.get(call_id)
            if terminal and "seconds" in terminal:
                seconds = terminal["seconds"]
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("invalid request duration")
                recorded_seconds += seconds
                end_times.append(epoch + seconds)
                if response:
                    latencies.append(seconds)
    window = max(end_times) - min(start_times) if start_times and end_times else None
    ordered = sorted(latencies)
    return {"requests": len(starts), "responses": len(received), "transport_failures": len(failures),
            "unmatched_started": len(starts.keys() - received.keys() - failures.keys()),
            "requests_with_unknown_usage": unknown_usage,
            "retry_attempts": sum(row.get("attempt", 1) > 1 for row in starts.values()),
            "empty_responses": sum(r["status"] == "invalid_response" and r.get("reason") == "empty_content" for r in records),
            "schema_violations": sum(r["status"] == "invalid_response" and r.get("reason") == "schema_violation" for r in records),
            "ranking_corrections": {status: sum(r["status"] == status for r in records) for status in
                                    ("ranking_membership_invalid", "ranking_correction_succeeded", "ranking_correction_failed")},
            "response_median_seconds": statistics.median(ordered) if ordered else None,
            "response_p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None,
            "observed_request_window_seconds": window,
            "summed_recorded_request_seconds": recorded_seconds,
            "received_per_minute_in_observed_window": 60 * len(received) / window if window and window > 0 else None,
            "stages": stages,
            "interpretation": "Descriptive nearest-rank latency percentiles. Unmatched requests may be pending or interrupted; "
                              "request window includes recovery gaps but excludes setup and any unfinished tail; "
                              "summed recorded request durations exclude gaps and pending requests. Neither is total batch wall time."}


def request_markdown(diagnostics):
    d = diagnostics
    lines = ["", "请求记录快照：共 " + str(d["requests"]) + " 次尝试，" + str(d["responses"]) + " 次响应，"
             + str(d["retry_attempts"]) + " 次为单次接口调用内的重试；" + str(d["requests_with_unknown_usage"]) + " 次用量尚不完整。"]
    lines.append("批次恢复另记 " + str(d.get("batch_continuations", 0)) + " 次，未完成事件的重新执行也已计入总请求数。")
    lines.append(f"本地字段校验拒绝 {d['schema_violations']} 份响应；空白响应 {d['empty_responses']} 份。字段齐全不代表笔记内容正确。")
    if d["response_median_seconds"] is not None:
        lines.append(f"已记录响应中，一半约在 {d['response_median_seconds']:.2f} 秒内返回，约 95% 在 {d['response_p95_seconds']:.2f} 秒内返回。不保证后续速度相同。")
    lines.append(f"带时间记录的已结束请求，耗时合计约 {d['summed_recorded_request_seconds'] / 60:.1f} 分钟；不包括请求之间的充值等待或尚未返回的请求。请求记录的起止窗口则包含中间恢复等待。")
    lines += ["", "| 步骤 | 请求 | 响应 | 传输失败 |", "|---|---:|---:|---:|"]
    labels = {"select": "选择物品", "user_update": "用户笔记", "item_update": "物品笔记", "rank": "排名", "rank_missing_insertions": "遗漏项插入"}
    for stage, counts in sorted(d["stages"].items()):
        lines.append(f"| {labels.get(stage, stage)} | {counts['requests']} | {counts['responses']} | {counts['transport_failures']} |")
    lines.append("\n未匹配的已发起请求可能仍在等待，也可能被中断；不能仅凭日志判定进程存活。格式修正和重试均计入请求数。")
    return "\n".join(lines) + "\n"


def summarize(protocol, rows, terminal_status):
    expected = {(seed, layer) for seed in protocol["seeds"] for layer in protocol["strata"]}
    observed = [(row["seed"], row["stratum"]) for row in rows]
    if len(set(observed)) != len(observed) or not set(observed) <= expected:
        raise ValueError("duplicate or unexpected pilot world")
    seed_effects = []
    for seed in protocol["seeds"]:
        paired = [r for r in rows if r["seed"] == seed]
        if len(paired) == len(protocol["strata"]):
            seed_effects.append({"seed": seed,
                "utility_gain": statistics.mean(r["utility_gain"] for r in paired),
                "bias_change": statistics.mean(r["bias_change"] for r in paired)})
    return {"complete": terminal_status == "completed" and set(observed) == expected,
            "completed_worlds": len(rows), "expected_worlds": len(expected),
            "missing_worlds": [{"seed": s, "stratum": layer} for s, layer in sorted(expected - set(observed))],
            "paired_seed_effects": seed_effects,
            "independent_seed_count": len(seed_effects),
            "utility_gain_mean": statistics.mean(r["utility_gain"] for r in seed_effects) if seed_effects else None,
            "utility_gain_sd": statistics.stdev(r["utility_gain"] for r in seed_effects) if len(seed_effects) > 1 else None,
            "bias_change_mean": statistics.mean(r["bias_change"] for r in seed_effects) if seed_effects else None,
            "statistical_support": "not_tested",
            "scope": "high-exposure initial vs original-update pilot; not E2 interaction or candidate promotion"}


def markdown(summary, rows):
    lines = ["# 独立试运行记录", "", "完整批次已结束。" if summary["complete"] else "批次未完整完成；以下只列已完成世界。",
             f"已完成 {summary['completed_worlds']}/{summary['expected_worlds']} 个环境；两个热门关系都完成的独立随机编号数为 {summary['independent_seed_count']}。",
             "", "| 随机编号 | 热门关系 | 初始质量 | 学习后质量 | 质量变化 | 初始偏差 | 学习后偏差 |", "|---|---|---:|---:|---:|---:|---:|"]
    strata = {"unrelated": "与偏好无关", "aligned": "与偏好有关"}
    for row in rows:
        initial, learned = row["arms"]["initial"]["score"], row["arms"]["original_update"]["score"]
        lines.append(f"| {row['seed']} | {strata[row['stratum']]} | {initial['utility']:.4f} | {learned['utility']:.4f} | {row['utility_gain']:+.4f} | {initial['bias']:.4f} | {learned['bias']:.4f} |")
    lines += ["", "质量分越高，表示生成程序设定的偏好物品排得越靠前；偏差分越高，表示热门物品相对固定偏好参考占了更多靠前位置，只统计超出参考的部分。两项独立计算，不合成一个分数。",
              "随机编号固定一批用户、物品和生成起点。同一编号的两个热门关系等权合并；用户共享物品笔记，不能把他们当成独立重复。缺失环境不替换、不从计划中删除。",
              "", "这批只比较高曝光下的学习前后，用于检查运行成本和波动。没有低曝光对照，不能支持完整机制交互结论；没有候选比较，不进行晋升或 RSI 判定。只有三个独立随机编号，不足以稳定估计尾部风险，也不自动确定正式样本量。"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    directory = Path(args.run)
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    protocol = read("protocol.json")
    rows = read("completed_worlds.json") if (directory / "completed_worlds.json").exists() else []
    status = read("run_result.json")["status"] if (directory / "run_result.json").exists() else None
    summary = summarize(protocol, rows, status)
    summary["failed_worlds"] = read("failed_worlds.json") if (directory / "failed_worlds.json").exists() else []
    from rsi_researcher.runtime.client import atomic_json
    from rsi_researcher.evaluation.tie_sensitivity import audit_ties
    atomic_json(directory / "pilot_analysis.json", summary)
    sensitivity = {}
    for row in rows:
        world_dir = directory / row["world"]
        world = json.loads((world_dir / "world.json").read_text(encoding="utf-8"))
        sensitivity[row["world"]] = {}
        for arm in ("initial", "original_update"):
            rankings = json.loads((world_dir / arm / "rankings.json").read_text(encoding="utf-8"))
            sensitivity[row["world"]][arm] = audit_ties(world, [rankings["forward"], rankings["reverse"]])
    atomic_json(directory / "tie_sensitivity.json", sensitivity)
    text = markdown(summary, rows)
    if summary["failed_worlds"]:
        text += "\n未完成的格式失败世界（仍保留在原计划分母中）：\n\n"
        for failure in summary["failed_worlds"]:
            text += f"- {failure['world']}：{failure['error_type']}，最后请求 {failure['last_request']}。\n"
        text += "\n成功世界的分数不能代表全部预定世界；含失败的批次不标完整完成。\n"
    if (directory / "calls.jsonl").exists():
        log = [json.loads(line) for line in (directory / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        diagnostics = request_diagnostics(log)
        diagnostics["batch_continuations"] = len(read("continuations.json")) if (directory / "continuations.json").exists() else 0
        atomic_json(directory / "request_diagnostics.json", diagnostics)
        text += request_markdown(diagnostics)
    text += "\n同分参考敏感性见 `tie_sensitivity.json`：保留编号破同分的主指标，另报同效用参考的上下界及随机破同分期望。它不更换主指标或晋升规则。\n"
    (directory / "pilot_report.md").write_text(text, encoding="utf-8")
