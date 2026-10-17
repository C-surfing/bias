"""Fixed development pilot with an external deadline; no candidate selection."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rsi_researcher.runtime.client import API_PROTOCOL, MODEL_PARAMETERS, StrictDeepSeek, InvalidOutput, RANKING_PROTOCOL, atomic_json, usage_summary
from rsi_researcher.paths import ROOT
from rsi_researcher.datasets.controlled import EVALUATION_PANEL_SIZE, WORLD_PROTOCOL, build_world, ordered_events
from rsi_researcher.recommenders.execution import execute_arm, score_arm
from rsi_researcher.evaluation.learning import audit_learning
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.source_snapshot import snapshot_package, verify_snapshot, module_command


def run_pilot(directory, protocol, prompts, model, resume=False):
    rows = []
    failure_path = directory / "failed_worlds.json"
    failures = json.loads(failure_path.read_text(encoding="utf-8")) if resume and failure_path.exists() else []
    for seed in protocol["seeds"]:
        for stratum in protocol["strata"]:
            world_id = f"{seed}-{stratum}"
            world_dir = directory / world_id
            world_dir.mkdir(exist_ok=resume)
            world = build_world(seed, users=protocol["users"], items=protocol["items"],
                                events_per_user=protocol["events_per_user"], stratum=stratum)
            if resume and (world_dir / "world.json").exists():
                if json.loads((world_dir / "world.json").read_text(encoding="utf-8")) != world:
                    raise ValueError("resume world differs from frozen inputs")
            else:
                atomic_json(world_dir / "world.json", world)
            if any(f["world"] == world_id for f in failures):
                continue
            events = ordered_events(world["public"]["events"]["high"], world["private"]["popular"], "pair_0")
            results = {}
            try:
                for arm in ("initial", "original_update"):
                    model.scope = {"world": world_id, "pilot_arm": arm}
                    atomic_json(directory / "progress.json", {"world": world_id, "arm": arm, "completed_worlds": len(rows), "failed_worlds": len(failures)})
                    pair = execute_arm(world_dir / arm, model, prompts, world["public"], [] if arm == "initial" else events, resume=resume)
                    score = score_arm(world, pair)
                    # Use the same observed-item subset for before/after comparisons.
                    audit = audit_learning(world, events, pair)
                    results[arm] = {"score": score, "learning_audit": audit}
                    atomic_json(world_dir / arm / "evaluation.json", results[arm])
            except InvalidOutput as exc:
                failure = {"world": world_id, "seed": seed, "stratum": stratum, "arm": arm,
                           "error_type": type(exc).__name__, "last_request": model.requests}
                failures.append(failure)
                atomic_json(world_dir / "failure.json", failure)
                atomic_json(failure_path, failures)
                continue
            row = {"world": world_id, "seed": seed, "stratum": stratum, "arms": results,
                   "utility_gain": results["original_update"]["score"]["utility"] - results["initial"]["score"]["utility"],
                   "bias_change": results["original_update"]["score"]["bias"] - results["initial"]["score"]["bias"]}
            rows.append(row)
            atomic_json(directory / "completed_worlds.json", rows)
            print(f"completed {world_id}; requests={model.requests}", flush=True)
    atomic_json(directory / "pilot_summary.json", {"worlds": rows, "completed_worlds": len(rows),
                "attempted_worlds": len(rows) + len(failures), "failed_worlds": failures,
                "independent_seeds": len(protocol["seeds"]),
                "interpretation": "Pilot diagnostics only. Strata within each seed are paired; users are not independent replicates. "
                                  "No low-exposure contrast, candidate selection, or statistical promotion."})
    return rows


def supervise(command, seconds, log_path):
    if seconds <= 0:
        return {"status": "timed_out", "worker_started": False}
    with Path(log_path).open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            code = process.wait(timeout=seconds)
            return {"status": "completed" if code == 0 else "failed", "returncode": code,
                    "worker_pid": process.pid}
        except subprocess.TimeoutExpired:
            # Worker is deliberately single-process and never spawns descendants.
            process.kill()
            process.wait(timeout=10)
            return {"status": "timed_out", "worker_pid": process.pid, "returncode": process.returncode}


def worker(directory, resume=False):
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    prompts = json.loads((directory / "prompts.json").read_text(encoding="utf-8"))
    limits = RunLimits(directory, protocol["max_requests"], protocol["max_tokens"], protocol["max_seconds"])
    try:
        run_pilot(directory, protocol, prompts, StrictDeepSeek(directory, protocol["max_requests"], limits), resume=resume)
    except Exception as exc:
        atomic_json(directory / "worker_failure.json", {"error_type": type(exc).__name__})
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-requests", type=int, default=2400)
    parser.add_argument("--max-tokens", type=int, default=6000000)
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help="continue failed batch with original limits; at most two continuations")
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    if args.worker:
        sys.exit(worker(directory, resume=args.resume))
    if min(args.max_requests, args.max_tokens, args.max_seconds) <= 0:
        parser.error("limits must be positive")
    if not args.resume:
        directory.mkdir(parents=True, exist_ok=False)
    protocol = {"seeds": [2026101301, 2026101302, 2026101303], "strata": ["unrelated", "aligned"],
                "users": 12, "items": 60, "events_per_user": 4, "exposure": "high",
                "ranking_protocol": RANKING_PROTOCOL,
                "api_protocol": API_PROTOCOL,
                "model_parameters": MODEL_PARAMETERS,
                "world_protocol": WORLD_PROTOCOL, "evaluation_panel_size": EVALUATION_PANEL_SIZE,
                "max_requests": args.max_requests, "max_tokens": args.max_tokens, "max_seconds": args.max_seconds,
                "purpose": "development throughput and paired learning pilot, not full E2 or final-test",
                "selection": "none", "missing_comparisons": "retain and report; never replace seeds",
                "failure_handling": "retain training format failures and continue worlds; grade rank failures; stop on uncaught transport or budget errors"}
    resume_number = 0
    if args.resume:
        frozen = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        if frozen != protocol:
            raise ValueError("resume protocol or limits changed")
        prior = json.loads((directory / "run_result.json").read_text(encoding="utf-8"))
        if prior["status"] != "failed":
            raise ValueError("only a confirmed failed batch can continue")
        history_path = directory / "continuations.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        if len(history) >= 2:
            raise ValueError("batch continuation limit reached")
        resume_number = len(history) + 1
        verify_snapshot(directory)
        limits = RunLimits(directory, args.max_requests, args.max_tokens, args.max_seconds)
        limits.remaining_seconds()
        failure = json.loads((directory / "worker_failure.json").read_text(encoding="utf-8"))
        history.append({"prior_result": prior, "prior_failure": failure, "resume_number": resume_number,
                        "reason": "continue from complete event records; original quotas and deadline retained"})
        atomic_json(history_path, history)
    else:
        atomic_json(directory / "protocol.json", protocol)
        atomic_json(directory / "prompts.json", json.loads((ROOT / "data/author_prompts.json").read_text(encoding="utf-8")))
        snapshot_package(directory)
    limits = RunLimits(directory, args.max_requests, args.max_tokens, args.max_seconds)
    atomic_json(directory / "run_result.json", {"status": "running", "resume_number": resume_number})
    worker_args = ["--worker", "--output", str(directory)]
    if args.resume:
        worker_args.append("--resume")
    command = module_command(directory, "rsi_researcher.experiments.pilot", worker_args)
    result = supervise(command, limits.remaining_seconds(), directory / (f"worker-resume-{resume_number}.log" if args.resume else "worker.log"))
    if result["status"] == "completed":
        summary = json.loads((directory / "pilot_summary.json").read_text(encoding="utf-8"))
        if summary["completed_worlds"] != len(protocol["seeds"]) * len(protocol["strata"]):
            result["status"] = "incomplete"
    atomic_json(directory / "run_result.json", result)
    if (directory / "calls.jsonl").exists():
        try:
            atomic_json(directory / "usage.json", usage_summary(directory))
        except json.JSONDecodeError:
            atomic_json(directory / "usage_incomplete.json", {"reason": "interrupted log record; preserve raw log for audit"})
    print(json.dumps(result))
    sys.exit(0 if result["status"] == "completed" else 1)
