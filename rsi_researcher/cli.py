"""One entry point for discovery, independent validation, and research history."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from rsi_researcher.paths import ROOT
from rsi_researcher.datasets.controlled import build_world
from rsi_researcher.experiments.pilot import supervise
from rsi_researcher.research.baseline import source_prompts
from rsi_researcher.research.journal import ExperimentJournal
from rsi_researcher.research.validation import reconcile_validation, run_validation
from rsi_researcher.research.workflow import (inspect_existing, model_metadata, read, reconcile_execution, run_cycles)
from rsi_researcher.runtime.client import StrictDeepSeek, atomic_json, usage_summary
from rsi_researcher.runtime.limits import RunLimits
from rsi_researcher.runtime.models import create_model, model_description, check_local_model
from rsi_researcher.runtime.offline import OfflineResearchModel
from rsi_researcher.runtime.source_snapshot import module_command, snapshot_package


def _journal(args, directory):
    return ExperimentJournal(args.journal or directory / "journal.json")


def _demo(directory, config, validation_protocol):
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "offline_demo.json", {"kind": "offline_test_double", "paid_api_calls": 0,
                "network_calls": 0, "meaning": "固定回复只检查研究流程连接，不证明模型研究能力或去偏效果。"})
    atomic_json(directory / "config.json", config)
    limits = RunLimits(directory, **config["budget"])
    model = OfflineResearchModel(budget=limits.max_requests, limits=limits)
    journal = ExperimentJournal(directory / "journal.json")
    world = build_world(**config["world"])
    prompts = read(ROOT / "data/author_prompts.json")
    discovery = run_cycles(directory, model, world, prompts, config, journal=journal)
    if discovery["execution_status"] != "completed":
        return {"status": "failed", "paid_api_calls": 0, "network_calls": 0, "simulated_responses": model.requests}
    validation = run_validation(directory / "validation", model, directory, validation_protocol, journal=journal)
    atomic_json(directory / "context_after_validation.json", journal.research_history())
    followup_config = deepcopy(config)
    followup_config["candidate_limit"] = 1
    followup = run_cycles(directory / "next_discovery", model, world, prompts, followup_config,
                          source_arm=directory / "trials/0001/B0", journal=journal)
    success = validation["status"] == "completed" and followup["execution_status"] == "completed"
    discovery["loops"] = {"discovery": "completed", "independent_validation": validation["status"],
                          "feedback_discovery": followup["execution_status"]}
    discovery["execution_status"] = "completed" if success else "failed"
    discovery["validation_summary"] = validation.get("summary")
    discovery["request_count"] = model.requests
    atomic_json(directory / "archive.json", discovery)
    report = directory / "report.md"
    report.write_text(report.read_text(encoding="utf-8") +
        "\n## 论文双循环的本地实现\n\n开发候选 → 新环境独立验证 → 验证结果写回共享日志 → 下一轮开发。\n\n"
        "验证报告：[validation/report.md](validation/report.md)；反馈后的研究：[next_discovery/report.md](next_discovery/report.md)。\n\n"
        "四个验证条件来自两个新种子与两种热门/偏好关系；它们属于开发验证。全部回复固定，不能当作方法效果。\n", encoding="utf-8")
    return {"status": discovery["execution_status"], "paid_api_calls": 0, "network_calls": 0,
            "simulated_responses": model.requests, "loops": discovery["loops"]}


def main(argv=None):
    # New science subcommands share the ONE package entry point; historical
    # commands remain available while their frozen evidence is audited.
    import sys
    requested=list(sys.argv[1:] if argv is None else argv)
    if requested and requested[0]=="science":
        from rsi_researcher.research.scientific_loop import main as run_science
        return run_science(requested[1:])
    if requested and requested[0]=="science-compare":
        from rsi_researcher.research.scientific_comparison import main as compare_science
        return compare_science(requested[1:])
    if requested and requested[0]=="science-tree":
        from rsi_researcher.research.scientific_tree import main as run_scientific_tree
        return run_scientific_tree(requested[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("science", help="Qwen multi-agent bias research with independent development audit")
    sub.add_parser("science-compare", help="paired-budget comparison of research strategies")
    sub.add_parser("science-tree", help="bounded best-first development experiment search")
    demo = sub.add_parser("demo", help="完整离线双循环演示，不访问网络或读取密钥")
    demo.add_argument("--config", default=str(ROOT / "configs/research_workflow.json"))
    demo.add_argument("--validation-protocol", default=str(ROOT / "configs/validation.json"))
    demo.add_argument("--output", required=True)
    run = sub.add_parser("run", help="开发快环：会调用付费模型")
    run.add_argument("--config", default=str(ROOT / "configs/research_workflow.json"))
    run.add_argument("--output", required=True)
    run.add_argument("--history", action="append", default=[])
    run.add_argument("--source-arm")
    run.add_argument("--journal", help="跨批次共用的实验日志 JSON 文件")
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    validate = sub.add_parser("validate", help="冻结候选后在新环境验证，默认调用付费模型")
    validate.add_argument("--source", required=True)
    validate.add_argument("--protocol", default=str(ROOT / "configs/validation.json"))
    validate.add_argument("--output", required=True)
    validate.add_argument("--journal")
    validate.add_argument("--trial", help="指定待验证轮次；默认最近完成的候选")
    validate.add_argument("--offline", action="store_true", help="仅用固定回复检查验证流程")
    validate.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    inspect = sub.add_parser("inspect", help="回看已有结果，不调用模型")
    inspect.add_argument("--source", required=True)
    inspect.add_argument("--output", required=True)
    history = sub.add_parser("journal", help="查看共享研究日志，不调用模型")
    history.add_argument("--path", required=True)
    fetch = sub.add_parser("fetch-data", help="从作者官方来源下载并核验公开数据，不调用模型")
    fetch.add_argument("--dataset", choices=["coat", "kuairand-pure"], required=True)
    fetch.add_argument("--output", required=True)
    study = sub.add_parser("study", help="官方真实数据研究与策略迁移，调用配置的本地模型或付费接口")
    study.add_argument("--config", default=str(ROOT / "configs/qwen_research_v2.json"))
    study.add_argument("--output", required=True)
    study.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    check = sub.add_parser("check-model", help="检查自部署模型健康状态与结构化输出，不调用 DeepSeek")
    check.add_argument("--config", required=True)
    check.add_argument("--output", required=True)
    from rsi_researcher.diagnostics.agentcf_smoke import add_arguments, dispatch
    add_arguments(sub.add_parser("author-check", help="只用 AgentCF 作者 CDs 数据，调用本地 Qwen 检查真实记忆更新与排序"))
    bias_tasks = sub.add_parser("bias-tasks", help="展示冻结的五类 Bias 任务，无模型调用")
    bias_coat = sub.add_parser("bias-coat", help="统一 Coat 开发评估；CPU、无模型调用")
    bias_coat.add_argument("--output", required=True)
    bias_coat.add_argument("--proposals-file", help="已准备的提案 JSON；只允许本项目内文件")
    bias_exposure = sub.add_parser("bias-exposure", help="KuaiRand 日志支持集审计；CPU、无模型调用")
    bias_exposure.add_argument("--output", required=True)
    sensitivity = sub.add_parser("bias-coat-sensitivity",
        help="Coat 估计倾向稳健性审计（开发数据、CPU，无模型调用）")
    sensitivity.add_argument("--output",required=True)
    exposure_sim = sub.add_parser("bias-exposure-sim",
        help="已知曝光概率的受控模拟 OPE Benchmark，CPU、无模型调用")
    exposure_sim.add_argument("--output",required=True)
    exposure_sim.add_argument("--reps",type=int,default=32)
    exposure_sim.add_argument("--events",type=int,default=3000)
    decisions = sub.add_parser("bias-decision-freeze",
        help="冻结合成 Exposure 科研动作基准并运行强规则基线，仅 CPU")
    decisions.add_argument("--output",required=True)
    args = parser.parse_args(argv)
    if args.command == "bias-decision-freeze":
        from rsi_researcher.research.decision_benchmark import freeze
        r=freeze(Path(args.output))
        print(json.dumps({"status":r["status"],"evaluation_cases":r["evaluation_cases"],
                          "rule_regret":r["rule"]["mean_regret"],
                          "new_model_requests":0},ensure_ascii=False))
        return 0
    if args.command == "bias-exposure-sim":
        from rsi_researcher.research.exposure_synthetic import run as run_sim
        r=run_sim(Path(args.output),reps=args.reps,events=args.events)
        print(json.dumps({"status":r["status"],"targets":list(r["targets"]),
            "propensity_missing_gate":r["missing_propensity_control"]["status"],
            "independent_real_data_confirmation":False,"new_model_calls":0},
            ensure_ascii=False))
        return 0
    if args.command == "bias-coat-sensitivity":
        from rsi_researcher.research.coat_sensitivity import run as run_sensitivity
        r=run_sensitivity(Path(args.output))
        print(json.dumps({"status":r["status"],
                          "best_caps":[x["best_cap"] for x in r["sweep"]],
                          "stable":r["stable_best_cap_across_perturbations"],
                          "confirmation":False,"new_model_requests":0},
                         ensure_ascii=False))
        return 0
    if args.command == "bias-tasks":
        from hashlib import sha256
        p = ROOT / "configs/bias_tasks.json"
        data = p.read_bytes()
        if sha256(data).hexdigest() != p.with_suffix(".sha256").read_text().strip():
            raise ValueError("Frozen Bias Task Registry SHA-256 mismatch")
        print(data.decode("utf8"))
        return 0
    if args.command == "bias-coat":
        from rsi_researcher.research.unified import run as run_coat
        r = run_coat(Path(args.output), Path(args.proposals_file) if args.proposals_file else None)
        print(json.dumps({"status":r["status"],"task":r["task_id"],
                          "arms":{k:v["selected_development_policy"] for k,v in r["arms"].items()},
                          "new_model_requests":0,"confirmation":False},ensure_ascii=False))
        return 0
    if args.command == "bias-exposure":
        from rsi_researcher.research.kuairand_transfer import run as run_exposure
        r = run_exposure(Path(args.output))
        print(json.dumps({"status":r["status"],"task":r["task"],"model_requests":0},ensure_ascii=False))
        return 0
    if args.command == "author-check":
        return dispatch(args)
    if args.command == "check-model":
        result = check_local_model(args.output, read(args.config))
        print(json.dumps({"status": result["status"], "scope": result["scope"]}, ensure_ascii=False))
        return 0
    if args.command == "journal":
        print(json.dumps(ExperimentJournal(args.path).entries(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "inspect":
        result = inspect_existing(args.source, args.output)
        print(json.dumps({"status": result["status"], "model_calls": 0}, ensure_ascii=False))
        return 0
    if args.command == "fetch-data":
        from rsi_researcher.datasets.fetch import fetch_dataset
        manifest = fetch_dataset(args.dataset, args.output)
        print(json.dumps({"dataset": manifest["dataset"], "sha256": manifest["sha256"], "output": args.output}, ensure_ascii=False))
        return 0
    directory = Path(args.output).resolve()
    if args.command == "study":
        from rsi_researcher.datasets.coat import load_coat
        from rsi_researcher.datasets.kuairand import load_kuairand
        from rsi_researcher.research.benchmark import run_real_campaign, run_transfer_check
        config = read(args.config)
        if args.worker:
            transfer = load_kuairand(config["transfer_data_dir"], config["user_limit"], config["split_seed"],
                                     config.get("training_scope", "cohort"))
            limits = RunLimits(directory, **config["budget"])
            model = create_model(directory, limits.max_requests, limits, config)
            if config.get("mode") == "context_transfer_check":
                previous = read(Path(config["source_campaign"]) / "campaign.json")
                result = run_transfer_check(directory, model, transfer, previous["with_history"],
                                            previous["strategy_revision"]["strategy"])
            else:
                source = load_coat(config["source_data_dir"], config["split_seed"])
                result = run_real_campaign(directory, model, source, transfer, config)
            return 0 if result["status"] == "completed" else 1
        directory.mkdir(parents=True, exist_ok=False)
        for field in ("source_data_dir", "transfer_data_dir", "source_campaign"):
            if field in config:
                path = Path(config[field])
                config[field] = str((ROOT / path if not path.is_absolute() else path).resolve())
        if config.get("mode") == "context_transfer_check":
            previous_runs = {Path(config["source_campaign"])}
            previous_runs.update((ROOT / Path(path)).resolve() for path in config.get("prior_checks", []))
            config["prior_checks"] = [str(path) for path in sorted(previous_runs) if path != Path(config["source_campaign"])]
            previous_requests = sum(json.loads(line).get("status") == "started" for path in previous_runs
                                    for line in (path / "calls.jsonl").read_text(encoding="utf-8").splitlines())
            if previous_requests + config["budget"]["max_requests"] > config["shared_request_ceiling"]:
                raise ValueError("context check would exceed the shared campaign request ceiling")
            config["previous_requests"] = previous_requests
        atomic_json(directory / "config.json", config)
        atomic_json(directory / "model_config.json", model_description(config))
        snapshot_package(directory)
        limits = RunLimits(directory, **config["budget"])
        atomic_json(directory / "run_result.json", {"status": "running"})
        flags = ["study", "--config", str(directory / "config.json"), "--output", str(directory), "--worker"]
        terminal = supervise(module_command(directory, "rsi_researcher", flags), limits.remaining_seconds(), directory / "worker.log")
        atomic_json(directory / "run_result.json", terminal)
        if terminal["status"] != "completed" and (directory / "campaign.json").exists():
            campaign = read(directory / "campaign.json")
            campaign.update(status=terminal["status"], terminal=terminal)
            atomic_json(directory / "campaign.json", campaign)
            from rsi_researcher.research.benchmark import render_campaign_report
            render_campaign_report(directory, campaign)
        if (directory / "calls.jsonl").exists():
            atomic_json(directory / "usage.json", usage_summary(directory))
        print(json.dumps(terminal, ensure_ascii=False))
        return 0 if terminal["status"] == "completed" else 1
    if args.command == "demo":
        result = _demo(directory, read(args.config), read(args.validation_protocol))
        atomic_json(directory / "run_result.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "completed" else 1
    if args.command == "validate":
        source = Path(args.source).resolve()
        source_journal = source / "journal.json"
        journal = ExperimentJournal(args.journal or source_journal)
        if args.worker:
            config, protocol = read(directory / "config.json"), read(directory / "input_validation_protocol.json")
        else:
            if (source / "offline_demo.json").exists() and not args.offline:
                parser.error("离线演示候选请用 --offline 验证；其固定回复不应作为付费实验依据。")
            protocol = read(args.protocol)
            config = read(source / "config.json")
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(directory / "config.json", config)
            atomic_json(directory / "input_validation_protocol.json", protocol)
            if args.offline:
                atomic_json(directory / "offline_demo.json", {"kind": "offline_test_double", "paid_api_calls": 0, "network_calls": 0})
        limits = RunLimits(directory, **config["budget"])
        if args.offline or args.worker:
            model = OfflineResearchModel(budget=limits.max_requests, limits=limits) if args.offline else create_model(directory, limits.max_requests, limits, config)
            result = run_validation(directory, model, source, protocol, journal=journal, trial_id=args.trial, prepared=True)
            terminal = {"status": "completed" if result["status"] == "completed" else "failed",
                        "research_status": result["status"]}
            if args.offline:
                terminal.update(paid_api_calls=0, network_calls=0, simulated_responses=model.requests)
            atomic_json(directory / "run_result.json", terminal)
        else:
            snapshot_package(directory)
            flags = ["validate", "--source", str(source), "--output", str(directory), "--journal", str(journal.path), "--worker"]
            if args.trial:
                flags += ["--trial", args.trial]
            terminal = supervise(module_command(directory, "rsi_researcher", flags), limits.remaining_seconds(), directory / "worker.log")
            atomic_json(directory / "run_result.json", terminal)
            reconcile_validation(directory, terminal, journal=journal)
        if (directory / "calls.jsonl").exists():
            atomic_json(directory / "usage.json", usage_summary(directory))
        print(json.dumps(terminal, ensure_ascii=False))
        return 0 if terminal["status"] == "completed" else 1
    config = read(args.config)
    journal = _journal(args, directory)
    if args.worker:
        limits = RunLimits(directory, **config["budget"])
        result = run_cycles(directory, create_model(directory, limits.max_requests, limits, config), read(directory / "input_world.json"),
                            read(directory / "input_prompts.json"), config, read(directory / "input_history.json"),
                            args.source_arm, journal=journal)
        return 0 if result["execution_status"] == "completed" else 1
    world = read(Path(args.source_arm).parent / "world.json") if args.source_arm else build_world(**config["world"])
    prompts = source_prompts(args.source_arm) if args.source_arm else read(ROOT / "data/author_prompts.json")
    history = [read(Path(path) / "archive.json") for path in args.history]
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in (("config", config), ("input_world", world), ("input_prompts", prompts), ("input_history", history)):
        atomic_json(directory / (name + ".json"), value)
    atomic_json(directory / "model_config.json", model_description(config))
    snapshot_package(directory)
    limits = RunLimits(directory, **config["budget"])
    atomic_json(directory / "run_result.json", {"status": "running"})
    flags = ["run", "--config", str(directory / "config.json"), "--output", str(directory), "--journal", str(journal.path), "--worker"]
    if args.source_arm:
        flags += ["--source-arm", str(Path(args.source_arm).resolve())]
    terminal = supervise(module_command(directory, "rsi_researcher", flags), limits.remaining_seconds(), directory / "worker.log")
    atomic_json(directory / "run_result.json", terminal)
    reconcile_execution(directory, terminal)
    if (directory / "calls.jsonl").exists():
        atomic_json(directory / "usage.json", usage_summary(directory))
    print(json.dumps(terminal, ensure_ascii=False))
    return 0 if terminal["status"] == "completed" else 1
