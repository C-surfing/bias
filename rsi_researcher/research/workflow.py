"""A small recommendation researcher: observe, hypothesize, test, evaluate, remember."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import shutil

from rsi_researcher.runtime.client import API_PROTOCOL, MEMORY_CONTROL_PROTOCOL, MODEL_PARAMETERS, RANKING_PROTOCOL, InvalidOutput, ModelFailure, BudgetExceeded, TEXT_SCHEMA, atomic_json, object_schema
from rsi_researcher.datasets.controlled import ordered_events
from rsi_researcher.recommenders.execution import execute_arm, score_arm
from rsi_researcher.recommenders.memory_dependencies import dependency_graph
from rsi_researcher.research.operators import OPERATORS, describe_delta, execute_candidate, execution_key, rerank
from rsi_researcher.research.baseline import load_baseline, make_baseline_contract, source_prompts
from rsi_researcher.research.evidence import build_candidate_evidence, compact_candidate_evidence
from rsi_researcher.evaluation.metrics import point_decision


STAGES = {
    "observe": ("观察收益和代价", "把学习前后质量与偏差放在一起，提出值得研究的问题"),
    "diagnose": ("提出机制假设", "引用事件证据，同时列出竞争解释和缺失证据"),
    "design": ("设计最小对照", "把一个解释变成具体干预、预期方向和反证条件"),
    "review": ("反证复核", "检查这项对照能否区分解释，而非附和设计者"),
    "execute": ("执行候选", "在单独目录中改一件事，保留原版与原事件"),
    "evaluate": ("独立比较", "分别检查质量、偏差和已有守门条件"),
    "learn": ("更新研究判断", "根据实际结果修正解释，提出下一轮要区分的问题"),
    "archive": ("记录研究所得", "让下一轮看到做过什么、哪里失败、还缺什么证据"),
}
STATUS_LABELS = {"running": "进行中", "inconclusive": "证据不足，暂不下结论", "rejected": "未达到条件，保留原版",
                 "eligible": "达到开发区点门槛，仍待正式验证", "failed": "执行未完成", "design_rejected": "对照设计被复核否决",
                 "needs_new_exposure_data": "需要新的曝光与反馈", "no_change": "保留当前规则",
                 "probe_completed": "机制对照已执行，等待研究解释",
                 "research_question_prepared": "已整理待研究的问题"}
EXECUTION_LABELS = {"running": "进行中", "completed": "已完成", "failed": "未完成", "timed_out": "已到时间上限"}


def model_metadata(model):
    return getattr(model, "metadata", {"api_protocol": API_PROTOCOL, "ranking_protocol": RANKING_PROTOCOL,
                      "memory_control_protocol": MEMORY_CONTROL_PROTOCOL, "model_parameters": MODEL_PARAMETERS})


def experiment_fields(event_ids):
    """One executable plan format shared by the designer and the next-round planner."""
    return {"research_question": TEXT_SCHEMA,
            "purpose": {"type": "string", "enum": ["debias", "mechanism_probe"]},
            "operator": {"type": "string", "enum": sorted(OPERATORS)},
            "root_event_ids": {"type": "array", "items": {"type": "string", "enum": sorted(event_ids)}},
            "prediction_if_hypothesis": TEXT_SCHEMA, "prediction_if_alternative": TEXT_SCHEMA}


def research_agenda(history, default_question):
    for record in reversed(history):
        update = record.get("research_update") or {}
        plan = update.get("next_experiment")
        if plan:
            return {"research_question": plan["research_question"], "suggested_experiment": deepcopy(plan),
                    "source_trial": record.get("trial_id"), "previous_status": record.get("status")}
        question = update.get("next_question") or record.get("research_question")
        if question:
            return {"research_question": question, "suggested_experiment": None,
                    "legacy_next_test": update.get("next_test"), "previous_status": record.get("status")}
    return {"research_question": default_question, "suggested_experiment": None}


def validate_experiment_scope(plan, event_ids):
    roots = plan["root_event_ids"]
    if plan["operator"] not in OPERATORS or not set(roots) <= set(event_ids) or bool(roots) != (plan["operator"] == "dependency_replay"):
        raise InvalidOutput("repair scope does not match the planned operator")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def change_observation(before, after):
    quality = after["utility"] - before["utility"]
    bias = after["bias"] - before["bias"]
    return {"quality_before": before["utility"], "quality_after": after["utility"],
            "bias_before": before["bias"], "bias_after": after["bias"],
            "quality_change": quality, "bias_change": bias,
            "quality_bias_tradeoff_observed": quality > 0 and bias > 0,
            "meaning": "质量提高且额外热门倾斜上升，值得查明收益与代价的来源。" if quality > 0 and bias > 0
                       else "本次没有观察到质量与额外热门倾斜同时上升；仍按原值报告。",
            "evidence_level": "descriptive_observation_not_causal_identification"}


def visible_exposure(world):
    if "exposure" in world["public"]:
        return deepcopy(world["public"]["exposure"])
    return {"preperiod_counts": deepcopy(world["private"]["preperiod_counts"]),
            "source": "Legacy recorded preperiod counts; design probabilities were not recorded. No propensity is inferred."}


def visible_evidence(world, events, records, before, baseline, history):
    return {"question": "推荐质量改善是否伴随额外热门倾斜？是什么机制，如何区分替代解释？",
            "observation": change_observation(before, baseline), "development_metrics": baseline,
            "events": [{"event_id": e["event_id"], "user": e["user"],
                        "observed_consumption": e["positive"], "unobserved_comparison": e["negative"],
                        "candidate_order": e["candidates"],
                        "user_memory_before": r["before"]["users"][e["user"]],
                        "user_memory_after": r["after"]["users"][e["user"]],
                        "item_memory_before": r["before"]["items"][e["positive"]]["memory"],
                        "item_memory_after": r["after"]["items"][e["positive"]]["memory"]}
                       for e, r in zip(events, records)],
            "read_dependencies": dependency_graph(records),
            "visible_exposure": visible_exposure(world), "previous_research": history,
            "operators": OPERATORS,
            "interpretation": "Consumption is observed; sampled non-consumption is unknown. Read dependencies are not causal proof."}


def render_report(directory, archive, timeline):
    observation = archive.get("observation")
    lines = ["# 推荐系统研究员：本轮记录", "", archive["mode_note"], "",
             f"本轮状态：{STATUS_LABELS.get(archive['status'], archive['status'])}；保留的规则：原版（B0）。", ""]
    if "execution_status" in archive:
        lines += ["流程执行：" + EXECUTION_LABELS[archive["execution_status"]] + "。", ""]
    if observation:
        lines += [observation["meaning"], "", "| 指标 | 比较前 | 比较后 | 变化 |", "|---|---:|---:|---:|"]
        for title, key in (("推荐质量", "quality"), ("额外热门倾斜", "bias")):
            lines.append(f"| {title} | {observation[key + '_before']:.4f} | {observation[key + '_after']:.4f} | {observation[key + '_change']:+.4f} |")
        lines += ["", "这是需要解释的观察，不代表已确定偏差成因。", ""]
    lines += ["| 步骤 | 为什么做 | 实际产物 |", "|---|---|---|"]
    for row in timeline:
        title, purpose = STAGES[row["stage"]]
        lines.append(f"| {title} | {purpose} | {row['artifact']} |")
    proposal = archive.get("proposal")
    if proposal:
        lines += ["", "本轮行动：" + OPERATORS[proposal["operator"]]["meaning"] + "。"]
        if "purpose" in proposal:
            lines += ["", "实验目的：" + ("区分机制解释，不用于替换原版。" if proposal["purpose"] == "mechanism_probe" else "检验去偏改法能否保持质量。"),
                      "", "若主要解释成立，预期：" + proposal["prediction_if_hypothesis"],
                      "", "若竞争解释成立，预期：" + proposal["prediction_if_alternative"]]
    handoff = archive.get("plan_handoff")
    if handoff and handoff["planned_operator"]:
        lines += ["", "与上轮计划的衔接：" + ("沿用计划中的行动。" if handoff["action_followed"] else "本轮修订了行动。") + handoff["reason"]]
    comparison = archive.get("comparison")
    if comparison:
        lines += ["", "| 本轮候选相对原版 | 变化 |", "|---|---:|",
                  f"| 推荐质量 | {comparison['quality_change']:+.4f} |",
                  f"| 额外热门倾斜 | {comparison['bias_change']:+.4f} |"]
    execution = archive.get("execution")
    if execution:
        lines += ["", ("本轮复用第 " + execution["source_trial"] + " 轮相同实验的结果，没有重复训练或排名；这不算一次独立重复验证。")
                  if execution["reused"] else "本轮实际执行了候选实验。"]
    if archive.get("candidate_evidence"):
        lines += ["", "逐条排名差异和笔记变化见 candidate_evidence.json；其摘要已交给本轮总结及下一轮诊断。"]
    gates = archive.get("metric_gates")
    if gates:
        lines += ["", "| 开发区检查 | 结果 |", "|---|---|"]
        for name, value in gates.items():
            lines.append(f"| {name} | {'待补证据' if value is None else '通过' if value else '未通过'} |")
    decision = archive.get("decision", {})
    reason = {"guard evidence is incomplete": "学习与笔记内容的证据尚未齐备，保留原版。",
              "baseline cannot support relative comparison": "原版偏差过小或评价不足，不能可靠判断相对去偏。",
              "at least one independent gate failed": "质量或偏差至少一项未达到约定条件，保留原版。",
              "candidate output or independent guard failed": "候选输出或独立检查未通过，保留原版。"}.get(decision.get("reason"), decision.get("reason", "本轮尚未作效果判断。"))
    lines += ["", "决定原因：" + reason, "", "流程能执行与方法有效分别判断；本轮没有确认性统计晋升或 RSI 结论。"]
    update = archive.get("research_update")
    if update:
        lines += ["", "研究员对本轮的解释（模型判断）：" + update["lesson"],
                  "", "下一轮问题：" + update["next_question"], "", "下一轮拟议对照：" + update["next_test"]]
    (Path(directory) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research(directory, model, world, prompts, config, history=(), source_arm=None, experiments=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timeline = []
    archive = {"status": "running", "selected": "B0", "observation": None,
               "trial_id": directory.name,
               "execution_status": "running", "research_status": "not_evaluated",
               "strategy_version": config["strategy_version"], "proposal": None,
               "mode_note": "执行固定研究策略；模型提出解释，程序负责实施与评价。",
               "statistical_support": "not_tested"}

    def mark(stage, artifact):
        timeline.append({"stage": stage, "artifact": artifact, "time_utc": datetime.now(timezone.utc).isoformat()})
        atomic_json(directory / "timeline.json", timeline)
        atomic_json(directory / "archive.json", archive)

    atomic_json(directory / "world.json", world)
    atomic_json(directory / "strategy.json", config)
    metadata = model_metadata(model)
    atomic_json(directory / "protocol.json", {"purpose": "development_research_loop", "strategy_version": config["strategy_version"],
                "quality_floor_ratio": 0.99, "bias_ceiling_ratio": 0.90, **metadata,
                "world_protocol": world["protocol"], "candidate_limit": 1, "statistical_support": "not_tested"})
    atomic_json(directory / "baseline_prompts.json", prompts)
    atomic_json(directory / "history_input.json", list(history))
    atomic_json(directory / "archive.json", archive)
    atomic_json(directory / "timeline.json", [])

    def ask(role, instruction, payload, properties):
        model.scope = {"research_trial": directory.name, "research_role": role}
        result = model.ask(instruction + " 用中文回答，引用已有证据，明确未知部分。\n" + json.dumps(payload, ensure_ascii=False),
                           object_schema(properties), {"stage": role})
        atomic_json(directory / (role + ".json"), result)
        mark(role, role + ".json")
        return result

    try:
        snapshot = load_baseline(source_arm, world, prompts, metadata=metadata) if source_arm else None
        events = snapshot["events"] if snapshot else ordered_events(world["public"]["events"]["high"], world["private"]["popular"], "pair_0")
        event_ids = [e["event_id"] for e in events]
        event_list = {"type": "array", "items": {"type": "string", "enum": sorted(event_ids)}}
        if source_arm:
            source_arm = Path(source_arm)
            for name, source in (("initial", source_arm.parent / "initial"), ("B0", source_arm)):
                target = directory / name
                target.mkdir()
                for path in source.glob("*.json"):
                    (target / path.name).write_bytes(path.read_bytes())
            before = score_arm(world, snapshot["initial_pair"])
            baseline_pair = snapshot["baseline_pair"]
            contract = snapshot["contract"]
            archive["baseline_source"] = str(source_arm.resolve())
            archive["comparison_scope"] = "Cached baseline vs later candidate; development evidence, not a time-matched confirmatory comparison."
        else:
            model.scope = {"research_trial": directory.name, "research_arm": "initial"}
            before = score_arm(world, execute_arm(directory / "initial", model, prompts, world["public"], []))
            model.scope = {"research_trial": directory.name, "research_arm": "B0"}
            baseline_pair = execute_arm(directory / "B0", model, prompts, world["public"], events)
            contract = make_baseline_contract(world, prompts, events, metadata)
        atomic_json(directory / "baseline_contract.json", contract)
        baseline = score_arm(world, baseline_pair)
        baseline_state = read(directory / "B0/state.json")
        baseline_identity = hashlib.sha256(json.dumps({"contract": contract, "state": baseline_state,
                                "rankings": baseline_pair}, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        records = [read(directory / "B0" / (eid + ".json")) for eid in event_ids]
        evidence = visible_evidence(world, events, records, before, baseline, list(history))
        agenda = research_agenda(history, evidence["question"])
        evidence["question"] = agenda["research_question"]
        evidence["agenda"] = agenda
        archive["research_question"] = agenda["research_question"]
        archive["agenda"] = agenda
        archive["observation"] = evidence["observation"]
        atomic_json(directory / "agenda.json", agenda)
        atomic_json(directory / "evidence.json", evidence)
        atomic_json(directory / "B0/score.json", baseline)
        mark("observe", "evidence.json")
        diagnosis = ask("diagnose", "你是推荐系统研究员。根据收益、代价与事件轨迹提出可检验的机制解释。必须给竞争解释；优先解释 previous_research 中的验证失败或反例，并据此修订旧计划。不要把热门或一次分数变化直接判作因果偏差。",
                        evidence, {"hypothesis": TEXT_SCHEMA, "evidence_event_ids": event_list,
                                   "alternative_explanation": TEXT_SCHEMA, "missing_evidence": TEXT_SCHEMA})
        archive["diagnosis"] = diagnosis
        proposal = ask("design", "把机制假设变成一个最小对照实验。优先回答 agenda 中的上轮问题；沿用或修订其行动都要说明理由。区分查原因的 mechanism_probe 与检验去偏改法的 debias。写清主要解释和竞争解释各自预测什么不同信号。每次只改一项，参照保持原版 B0。dependency_replay 指定源事件；其他改法的 root_event_ids 留空。exploration 需要新曝光反馈。",
                       {"evidence": evidence, "diagnosis": diagnosis, "agenda": agenda, "rerank_strength": config["rerank_strength"]},
                       experiment_fields(event_ids) | {"parent": {"type": "string", "enum": ["B0"]},
                        "rationale": TEXT_SCHEMA, "expected_direction": TEXT_SCHEMA, "falsifier": TEXT_SCHEMA,
                        "plan_reason": TEXT_SCHEMA})
        if proposal["parent"] != "B0" or proposal["operator"] not in OPERATORS:
            raise InvalidOutput("unsupported research proposal")
        roots = proposal["root_event_ids"]
        validate_experiment_scope(proposal, event_ids)
        archive["proposal"] = proposal
        archive["candidate_delta"] = describe_delta(proposal, config["rerank_strength"])
        atomic_json(directory / "candidate_delta.json", archive["candidate_delta"])
        planned = agenda["suggested_experiment"]
        archive["plan_handoff"] = {"planned_operator": planned["operator"] if planned else None,
                    "actual_operator": proposal["operator"],
                    "action_followed": bool(planned) and all(proposal[k] == planned[k] for k in ("operator", "purpose", "root_event_ids")),
                    "reason": proposal["plan_reason"]}
        atomic_json(directory / "plan_handoff.json", archive["plan_handoff"])
        review = ask("review", "独立复核此实验能否区分假设和竞争解释，指出仍可能存在的混杂因素。你决定是否值得执行，不决定效果是否通过。",
                     {"evidence": evidence, "diagnosis": diagnosis, "proposal": proposal},
                     {"proceed": {"type": "boolean"}, "reason": TEXT_SCHEMA, "counterexample": TEXT_SCHEMA})
        archive["review"] = review
        if not review["proceed"]:
            archive.update(status="design_rejected", decision={"reason": "复核认为当前对照不足；保留实验设计和反对理由。"})
        elif proposal["operator"] == "exploration":
            archive.update(status="needs_new_exposure_data", decision={"reason": "已提出探索实验；执行它需要真实的新曝光与反馈，当前只保存设计。"})
        elif proposal["operator"] == "retain":
            archive.update(status="no_change", decision={"reason": "研究员选择保留当前规则，没有把无改动算作改进。"})
        else:
            model.scope = {"research_trial": directory.name, "research_arm": "candidate", "operator": proposal["operator"]}
            atomic_json(directory / "candidate_policy.json", {"parent": "B0", "operator": proposal["operator"],
                        "purpose": proposal["purpose"],
                        "root_event_ids": roots, "rerank_strength": config["rerank_strength"] if proposal["operator"] == "popularity_rerank" else None})
            mark("execute", "candidate/")
            identity = execution_key(proposal, baseline_identity, config["rerank_strength"], metadata)
            cached = (experiments or {}).get(identity)
            if cached:
                shutil.copytree(Path(cached["candidate_dir"]), directory / "candidate")
                ranking_dir = directory / "candidate/ranking" if proposal["operator"] == "dependency_replay" else directory / "candidate"
                saved = read(ranking_dir / "rankings.json")
                pair = saved["forward"], saved["reverse"]
            else:
                pair = execute_candidate(directory / "candidate", model, prompts,
                       world["public"] | {"exposure": visible_exposure(world)}, directory / "B0", proposal, config["rerank_strength"])
                if experiments is not None:
                    experiments[identity] = {"candidate_dir": str((directory / "candidate").resolve()), "trial_id": directory.name}
            archive["execution"] = {"experiment_id": identity, "reused": cached is not None,
                                    "source_trial": cached["trial_id"] if cached else directory.name,
                                    "new_execution": cached is None, "independent_repetition": False}
            atomic_json(directory / "execution.json", archive["execution"])
            candidate = score_arm(world, pair)
            atomic_json(directory / "candidate/score.json", candidate)
            comparison = {"reference": "B0", "purpose": proposal["purpose"],
                          "quality_change": candidate["utility"] - baseline["utility"],
                          "bias_change": candidate["bias"] - baseline["bias"],
                          "reference_invalid_rankings": baseline["invalid_rankings"],
                          "candidate_invalid_rankings": candidate["invalid_rankings"],
                          "statistical_support": "not_tested"}
            archive["comparison"] = comparison
            atomic_json(directory / "comparison.json", comparison)
            full_evidence = build_candidate_evidence(world, directory / "B0", directory / "candidate",
                                                    baseline_pair, pair, proposal["operator"])
            full_evidence["execution"]["current_trial_execution"] = deepcopy(archive["execution"])
            atomic_json(directory / "candidate_evidence.json", full_evidence)
            archive["candidate_evidence"] = compact_candidate_evidence(full_evidence)
            guards = {"position": candidate["position_top1_disagreement"] <= baseline["position_top1_disagreement"],
                      "budget": model.requests <= model.max_requests, "memory_coverage": None, "learning": None}
            archive["metric_gates"] = {"质量至少保留原版的99%": candidate["utility"] >= 0.99 * baseline["utility"],
                    "额外热门倾斜至少减少10%": candidate["bias"] <= 0.90 * baseline["bias"] if baseline["bias"] >= .001 else None,
                    "位置变化未加重": guards["position"], "笔记内容检查": None, "学习证据检查": None}
            decision = ({"status": "probe_completed", "reason": "机制对照已执行；质量和偏差变化用于区分解释，不作方法晋升或因果认定。",
                         "statistical_support": "not_tested"} if proposal["purpose"] == "mechanism_probe"
                        else point_decision(baseline, candidate, guards, minimum_bias=0.001))
            archive.update(status=decision["status"], decision=decision, metrics={"B0": baseline, "candidate": candidate})
            archive["research_status"] = decision["status"]
            mark("evaluate", "candidate/score.json")
        archive["research_update"] = ask("learn", "根据本轮实际执行和外部评价，修正研究解释并提出下一轮可执行的实验及两种解释的不同预测。dependency_replay 计划必须指定已有源事件；其他行动的 root_event_ids 留空。没有执行的设计不能写成实验发现；一次分数变化不能证明因果或跨环境成立。解释属于你的研究判断，不改变外部评价。",
                      {"observation": archive["observation"], "diagnosis": diagnosis, "proposal": proposal,
                       "review": review, "test_status": archive["status"], "metrics": archive.get("metrics"),
                       "comparison": archive.get("comparison"), "candidate_evidence": archive.get("candidate_evidence"),
                       "experiment_execution": archive.get("execution"), "previous_research": list(history),
                       "external_decision": archive["decision"], "operators": OPERATORS},
                      {"evidence_state": {"type": "string", "enum": ["supported_in_development", "weakened", "inconclusive", "not_tested"]},
                       "updated_explanation": TEXT_SCHEMA, "lesson": TEXT_SCHEMA,
                       "next_experiment": object_schema(experiment_fields(event_ids))})
        update = archive["research_update"]
        validate_experiment_scope(update["next_experiment"], event_ids)
        update["next_question"] = update["next_experiment"]["research_question"]
        update["next_test"] = OPERATORS[update["next_experiment"]["operator"]]["meaning"]
        archive["execution_status"] = "completed"
        archive["research_status"] = archive["status"]
    except (InvalidOutput, ModelFailure, BudgetExceeded) as exc:
        archive.update(status="failed", failure={"error_type": type(exc).__name__})
        archive["execution_status"] = "failed"
        archive.setdefault("decision", {"reason": "本轮执行未完成，保留已有证据及原规则。"})
    except Exception as exc:
        archive.update(status="failed", execution_status="failed", failure={"error_type": type(exc).__name__})
        archive.setdefault("decision", {"reason": "执行异常，已有研究记录保留。"})
        raise
    finally:
        mark("archive", "archive.json")
        atomic_json(directory / "archive.json", archive)
        render_report(directory, archive, timeline)
    return archive


def render_batch_report(directory, archive):
    trials = archive.get("trials", [])
    execution = archive.get("execution_status", "running")
    lines = ["# 推荐系统研究员：研究过程", "", "流程执行：" + EXECUTION_LABELS[execution] + "。",
             "", "各轮共用原始 B0 与同一预算；后续轮读取前一轮的研究所得和候选证据。",
             "", "| 轮次 | 实验目的 | 研究行动 | 结果 | 是否实际重跑 |", "|---|---|---|---|---|"]
    for index, result in enumerate(trials):
        operator = result.get("proposal")
        purpose = "查原因" if operator and operator.get("purpose") == "mechanism_probe" else "检验去偏改法"
        action = OPERATORS[operator['operator']]['meaning'] if operator else '未产生可执行设计'
        physical = result.get("execution")
        executed = ("复用已有结果" if physical["reused"] else "实际执行") if physical else "未执行候选"
        lines.append(f"| {index + 1} | {purpose} | {action} | {STATUS_LABELS.get(result['status'], result['status'])} | {executed} |")
    update = archive.get("research_update")
    if update:
        lines += ["", "最后一轮研究所得（模型判断）：" + update["lesson"], "", "下一步问题：" + update["next_question"]]
    lines += ["", "相同输入和改法只执行一次；复用结果不算独立重复验证。每轮的解释、逐条证据、评分与决定见 trials 对应目录。",
              "", "流程完成不等于去偏有效。"]
    (Path(directory) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconcile_execution(directory, terminal):
    """The parent process records a timeout even when the worker cannot finish writing."""
    directory = Path(directory)
    path = directory / "archive.json"
    archive = read(path) if path.exists() else {"status": "failed", "selected": "B0", "trials": [],
                                               "research_status": "not_evaluated"}
    archive["execution_status"] = terminal["status"]
    if terminal["status"] != "completed":
        archive["status"] = "failed"
        archive["terminal"] = terminal
        for trial_dir in sorted((directory / "trials").glob("*")):
            trial_path = trial_dir / "archive.json"
            if not trial_path.exists():
                continue
            record = read(trial_path)
            if record.get("execution_status") == "running":
                record.update(status="failed", execution_status=terminal["status"], terminal=terminal)
                atomic_json(trial_path, record)
                render_report(trial_dir, record, read(trial_dir / "timeline.json"))
            archive["trials"] = [t for t in archive["trials"] if t.get("trial_id") != record.get("trial_id")]
            archive["trials"].append(record)
        if archive["trials"]:
            archive["research_status"] = archive["trials"][-1].get("research_status", "not_evaluated")
    atomic_json(path, archive)
    render_batch_report(directory, archive)
    return archive


def run_cycles(directory, model, world, prompts, config, history=(), source_arm=None, journal=None):
    """Keep one fixed B0, one execution index, and one allowance as questions evolve."""
    candidate_limit = config.get("candidate_limit", 1)
    if type(candidate_limit) is not int or not 1 <= candidate_limit <= 8:
        raise ValueError("a research batch needs between one and eight candidate trials")
    directory = Path(directory)
    (directory / "trials").mkdir(parents=True)
    prior, experiments = (journal.research_history() if journal else []) + list(history), {}
    archive = {"status": "running", "execution_status": "running", "research_status": "not_evaluated",
               "selected": "B0", "trials": [], "strategy_version": config["strategy_version"],
               "request_count": model.requests, "statistical_support": "not_tested"}
    atomic_json(directory / "archive.json", archive)
    baseline_source = source_arm
    try:
        for index in range(candidate_limit):
            trial = directory / "trials" / f"{index + 1:04d}"
            try:
                result = run_research(trial, model, world, prompts, config, prior, baseline_source, experiments)
            except Exception:
                if (trial / "archive.json").exists():
                    failed = read(trial / "archive.json")
                    archive["trials"].append(failed)
                    if journal is not None:
                        journal.record_trial(trial, failed)
                raise
            if journal is not None:
                result["candidate_id"] = journal.record_trial(trial, result)
                atomic_json(trial / "archive.json", result)
            archive["trials"].append(result)
            prior.append(result)
            baseline_source = directory / "trials/0001/B0"
            archive.update(status=result["status"], research_status=result["research_status"],
                           research_update=result.get("research_update"))
            archive["request_count"] = model.requests
            atomic_json(directory / "archive.json", archive)
            atomic_json(directory / "experiment_index.json", experiments)
            if result["execution_status"] == "failed":
                archive["execution_status"] = "failed"
                break
            if result["status"] in ("needs_new_exposure_data", "no_change"):
                break
        else:
            archive["execution_status"] = "completed"
        if archive["execution_status"] == "running":
            archive["execution_status"] = "completed"
    except Exception:
        archive.update(status="failed", execution_status="failed")
        if archive["trials"]:
            archive["research_status"] = archive["trials"][-1].get("research_status", "not_evaluated")
        raise
    finally:
        archive["request_count"] = model.requests
        archive["unique_candidate_experiments"] = len(experiments)
        atomic_json(directory / "experiment_index.json", experiments)
        atomic_json(directory / "archive.json", archive)
        render_batch_report(directory, archive)
    return archive


def inspect_existing(source, output):
    """Turn existing real measurements into a research question; never call a model."""
    source, output = Path(source).resolve(), Path(output)
    if (source / "trials").is_dir():
        candidates = [p for p in sorted((source / "trials").iterdir()) if (p / "candidate/score.json").exists()]
        if not candidates:
            raise ValueError("this research batch has no scored candidate to inspect")
        source = candidates[-1]
    if (source / "completed_worlds.json").exists():
        paired = defaultdict(list)
        for row in read(source / "completed_worlds.json"):
            paired[row["seed"]].append(row)
        scores = {}
        for arm in ("initial", "original_update"):
            scores[arm] = {key: statistics.mean(statistics.mean(r["arms"][arm]["score"][key] for r in group)
                                              for group in paired.values()) for key in ("utility", "bias")}
        observation = change_observation(scores["initial"], scores["original_update"])
        evidence_scope = "已有学习前后测量；同一编号的层先合并。缺少曝光因素对照，不能归因。"
        required = ["protocol.json", "completed_worlds.json", "run_result.json"]
    else:
        observation = change_observation(read(source / "B0/score.json"), read(source / "candidate/score.json"))
        evidence_scope = "已有原版和单一候选的开发比较；没有新模型调用。"
        required = ["protocol.json", "B0/score.json", "candidate/score.json",
                    "archive.json" if (source / "archive.json").exists() else "decision.json"]
    output.mkdir(parents=True, exist_ok=False)
    archive = {"status": "research_question_prepared", "selected": "B0", "observation": observation,
               "proposal": None, "source": str(source), "source_protocol": read(source / "protocol.json"),
               "mode_note": "回看已有真实测量，新增模型调用为 0。" + evidence_scope,
               "decision": {"reason": "下一步解释指标冲突，设计能区分原因的对照；本次未重新实验。"},
               "research_question": "收益来自偏好学习还是额外热门倾斜？什么干预能保持质量并减少后者？",
               "next_evidence_needed": ["替代机制与反证条件", "同初态、同事件和同评价的干预对照", "独立质量与偏差判断"],
               "model_calls": 0, "statistical_support": "not_tested"}
    atomic_json(output / "source_manifest.json", {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in required})
    atomic_json(output / "archive.json", archive)
    timeline = [{"stage": "observe", "artifact": "archive.json"}]
    atomic_json(output / "timeline.json", timeline)
    render_report(output, archive, timeline)
    return archive
