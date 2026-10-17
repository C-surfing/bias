"""A researcher operates real CPU recommenders using observed feedback only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics

from rsi_researcher.evaluation.observed import evaluate_observed, paired_comparison
from rsi_researcher.recommenders.itemcf import ItemCF
from rsi_researcher.research.journal import ExperimentJournal
from rsi_researcher.runtime.client import atomic_json, object_schema, TEXT_SCHEMA


DEFAULT_STRATEGY = {"version": "research_strategy_v1", "strength_grid": [0, .05, .1, .2, .4],
                    "evidence_focus": "overall", "preferred_operator": "popularity_penalty"}
OPERATORS = ["retain", "popularity_penalty", "user_calibrated", "support_preserving"]
METRICS = ("ndcg", "panel_excess_head_share", "head_share", "tail_relevant_recall",
           "quality_users", "zero_ideal_users", "evaluated_users", "tail_supported_users", "invalid_rankings")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _metrics(result):
    return {key: result[key] for key in METRICS}


def _score(dataset, recommender, rows, policy):
    # Only IDs cross into the recommender. Labels stay with the external evaluator.
    candidates = [{"user": r["user"], "item": r["item"]} for r in rows]
    rankings = recommender.rank_panels(candidates, policy)
    score = evaluate_observed(rows, rankings, recommender.train_counts,
               catalog_items=dataset["item_ids"], k=5, positive_threshold=dataset["positive_threshold"],
               gain_floor=dataset["gain_floor"])
    return score, rankings


def _examples(baseline, candidate, focus, limit=6):
    before = {row["user"]: row for row in baseline["per_user"]}
    rows = []
    for row in candidate["per_user"]:
        old = before[row["user"]]
        if row["ndcg"] is None:
            continue
        rows.append({"user": row["user"], "quality_change": row["ndcg"] - old["ndcg"],
                     "bias_change": row["panel_excess_head_share"] - old["panel_excess_head_share"],
                     "ndcg": row["ndcg"], "panel_excess_head_share": row["panel_excess_head_share"],
                     "tail_relevant_recall": row["tail_relevant_recall"]})
    if not rows:
        return []
    # Always include adverse cases; the research strategy controls the remaining focus.
    picked = [min(rows, key=lambda r: r["quality_change"]), max(rows, key=lambda r: r["bias_change"])]
    if focus == "tradeoff_users":
        ordered = sorted(rows, key=lambda r: (r["quality_change"], -abs(r["bias_change"])))
    elif focus == "tail_relevance":
        ordered = sorted(rows, key=lambda r: (r["tail_relevant_recall"] is None, r["tail_relevant_recall"] or 0))
    else:
        ordered = sorted(rows, key=lambda r: -r["panel_excess_head_share"])
    result, seen = [], set()
    for row in picked + ordered:
        if row["user"] not in seen:
            result.append(row)
            seen.add(row["user"])
        if len(result) == limit:
            break
    return result


def _change_statistics(baseline, candidate):
    before = {r["user"]: r for r in baseline["per_user"]}
    changes = [r["ndcg"] - before[r["user"]]["ndcg"] for r in candidate["per_user"] if r["ndcg"] is not None]
    return {"quality_evaluable_users": len(changes), "quality_loss_users": sum(v < 0 for v in changes),
            "quality_gain_users": sum(v > 0 for v in changes), "quality_unchanged_users": sum(v == 0 for v in changes),
            "largest_quality_loss": min(changes) if changes else None,
            "case_selection_note": "Six selected examples are not a representative sample; population counts come from all evaluable users."}


def _profile(dataset, recommender):
    train, development = dataset["train"], dataset["development"]
    def mean(rows):
        return statistics.mean(r["rating"] for r in rows) if rows else None
    def positive_head(rows):
        positives = [r for r in rows if r["rating"] >= dataset["positive_threshold"]]
        return sum(r["item"] in recommender.head_items for r in positives) / len(positives) if positives else None
    support = recommender.diagnose_panels(development)
    support.pop("per_user")
    return {"dataset": dataset["dataset_id"], "training_rows": len(train), "development_rows": len(development),
            "catalog_items": len(dataset["item_ids"]), "training_mean_feedback": mean(train),
            "random_development_mean_feedback": mean(development),
            "training_positive_head_share": positive_head(train),
            "random_development_positive_head_share": positive_head(development),
            "score_support": support,
            "interpretation": "Training popularity is an observation-frequency proxy; distribution differences do not identify a cause.",
            "quality_scope": "NDCG@5 on observed random-feedback candidate panels; missing pairs remain unknown."}


def _ask(model, directory, stage, payload, properties):
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / (stage + "_input.json"), payload)
    model.scope = {"backend": "observed_itemcf", "research_trial": str(directory.name),
                   "study": directory.parent.parent.name, "dataset": payload.get("current_dataset")}
    instructions = {
        "observed_plan": "你是推荐系统研究员。依据current_dataset的真实开发证据提出机制假设和竞争解释，再选一个最小实验。历史source_dataset来自别的任务，只用于形成假设；其用户、指标与结论都不是当前观察，质量底线只用当前baseline计算。优先解释当前任务的上轮反证。只能对实际执行的operator下结论；retain表示未改动，不能写成测试了新算法。置信区间跨0表示不确定，不能据此否定未测试的机制。热门下降本身不等于去偏；不能宣称因果或RSI成立。strength必须来自给定网格，未观察反馈保持未知。用中文。",
        "observed_learn": "你是推荐系统研究员。根据current_dataset程序实际计算的收益、代价和受损用户修正判断，提出下一问题和具体行动。next_strength只能从strategy.strength_grid选择；网格外的强度可以在next_question提出供以后研究，但本轮不能执行。区分来源任务经验与当前观察；用户编号只在所属数据集中有意义。只解释实际执行的policy，未测试算法不能写成发现；retain是未改动。没有改善就承认失败，区间跨0表示不确定而非不存在去偏机制；点门槛不等于统计证实。用中文。",
        "strategy_revision": "你要改进研究策略本身。根据前一任务的开发和独立验证，修改下一任务的搜索强度网格、证据阅读重点和优先行动。说明可推翻的预期。不要修改评分规则。用中文。",
    }
    answer = model.ask(instructions[stage] + "\n" + json.dumps(payload, ensure_ascii=False),
                       object_schema(properties), {"stage": stage})
    atomic_json(directory / (stage + ".json"), answer)
    return answer


def _policy(plan):
    return {"operator": plan["operator"], "strength": float(plan["strength"]) if plan["operator"] != "retain" else 0.0}


def _next_action(history, strategy):
    for record in reversed(history):
        update = record.get("research_update") or {}
        if "next_operator" in update:
            return {"operator": update["next_operator"],
                    "strength": update["next_strength"] if update["next_operator"] != "retain" else 0,
                    "question": update["next_question"], "source": "previous_research_feedback"}
    positive = [v for v in strategy["strength_grid"] if v > 0]
    return {"operator": strategy["preferred_operator"],
            "strength": min(positive) if positive and strategy["preferred_operator"] != "retain" else 0,
            "question": "哪种改法能保留质量并降低面板额外热门倾斜？", "source": "strategy_starting_preference"}


def source_experience(source_result):
    """Transfer aggregate evidence, never equate users or cases across datasets."""
    def aggregate(metrics):
        return {arm: {k: v for k, v in score.items() if k in METRICS}
                for arm, score in metrics.items()}
    validation = source_result.get("validation")
    validation = ({"status": validation["status"], "metrics": aggregate(validation["metrics"]),
                   "comparison": validation["comparison"]} if validation else None)
    return {"source_dataset": source_result["dataset"],
            "source_fingerprint": source_result["dataset_fingerprint"],
            "policy_trace": [{"policy": t["policy"], "metrics": aggregate(t["metrics"]),
                              "comparison": t["comparison"]} for t in source_result["trials"]],
            "validation": validation,
            "transfer_scope": "Source-task aggregate experience only; not current users, current thresholds, or current findings."}


def _select(trials):
    passing = [t for t in trials if t["comparison"]["status"] == "point_gates_passed"]
    if not passing:
        return {"operator": "retain", "strength": 0.0}, "没有候选同时达到开发区质量与偏差点门槛，保留原版。"
    best = min(passing, key=lambda t: (t["metrics"]["candidate"]["panel_excess_head_share"],
                                      -t["metrics"]["candidate"]["ndcg"]))
    return best["policy"], "在通过质量和偏差点门槛的开发候选中，选择面板额外热门倾斜最小的一项。"


def _finalize(directory, dataset, recommender, selected_policy, baseline_dev, selected_dev, journal, candidate_id):
    atomic_json(directory / "frozen_candidate.json", {"policy": selected_policy,
                "selection_split": "development", "dataset_fingerprint": dataset["fingerprint"],
                "evaluation_protocol": baseline_dev["protocol"], "candidate_redesign": False})
    baseline_test, base_rankings = _score(dataset, recommender, dataset["test"], None)
    selected_test, selected_rankings = _score(dataset, recommender, dataset["test"], selected_policy)
    comparison = paired_comparison(baseline_test, selected_test)
    result = {"status": "completed", "policy": selected_policy, "scope": "held_out_observed_random_panels",
              "metrics": {"B0": _metrics(baseline_test), "candidate": _metrics(selected_test)},
              "comparison": comparison, "test_candidate_selection": False, "online_experiment": False,
              "statistical_promotion": False}
    atomic_json(directory / "validation.json", result)
    atomic_json(directory / "validation_scores.json", {"B0": baseline_test, "candidate": selected_test})
    atomic_json(directory / "validation_rankings.json", {"B0": base_rankings, "candidate": selected_rankings})
    if journal is not None:
        journal.record_validation(candidate_id, result)
    return result


def run_study(directory, model, dataset, *, rounds=3, history_enabled=True, strategy=None,
              prior_history=(), journal=None, cache=None, compute_test=True):
    """Spend a fixed candidate budget; freeze dev selection before held-out scoring."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    strategy = deepcopy(strategy or DEFAULT_STRATEGY)
    cache = {} if cache is None else cache
    recommender = ItemCF(dataset["train"], dataset["item_ids"], dataset["positive_threshold"])
    baseline, baseline_rankings = _score(dataset, recommender, dataset["development"], None)
    atomic_json(directory / "baseline.json", baseline)
    atomic_json(directory / "baseline_rankings.json", baseline_rankings)
    atomic_json(directory / "strategy.json", strategy)
    evidence = {"profile": _profile(dataset, recommender), "baseline": _metrics(baseline),
                "initial_examples": [{key: value for key, value in row.items()
                                      if key not in ("quality_change", "bias_change")}
                                     for row in _examples(baseline, baseline, strategy["evidence_focus"])],
                "initial_state": "Baseline descriptive evidence only. No current-task intervention has run yet; there is no previous retain trial.",
                "population_definitions": {"evaluated_users": "All observed-panel users.",
                    "quality_users": "Users with positive ideal gain, eligible for NDCG and its change.",
                    "zero_ideal_users": "Separate users with no positive ideal gain, excluded from quality_users; not quality-loss cases."},
                "operators": {"retain": "保留原版", "popularity_penalty": "按训练观察频次惩罚热门分数",
                              "user_calibrated": "根据训练正反馈热门比例，为不同用户缩放惩罚",
                              "support_preserving": "按原分数比例而非减去固定量惩罚热门；强度小于1时保留正分与零分的证据顺序，不保证去偏有效"},
                "evaluation": "固定随机评分面板；质量≥原版99%，面板额外热门倾斜≤原版90%；最终反馈不可用于选候选。"}
    for example in evidence["initial_examples"]:
        example["dataset"] = dataset["dataset_id"]
    result = {"execution_status": "running", "research_backend": "observed_itemcf", "dataset": dataset["dataset_id"],
              "dataset_fingerprint": dataset["fingerprint"], "history_enabled": history_enabled,
              "strategy": strategy, "trials": [], "research_status": "not_evaluated"}
    atomic_json(directory / "archive.json", result)
    history = list(prior_history) if history_enabled else []
    try:
        for index in range(rounds):
            target = directory / "trials" / f"{index + 1:04d}"
            payload = {"current_dataset": dataset["dataset_id"], "evidence": evidence,
                       "history": history if history_enabled else [], "strategy": strategy,
                       "next_action": _next_action(history if history_enabled else [], strategy),
                       "budget": {"round": index + 1, "candidate_limit": rounds}}
            plan = _ask(model, target, "observed_plan", payload,
                        {"hypothesis": TEXT_SCHEMA, "alternative_explanation": TEXT_SCHEMA,
                         "operator": {"type": "string", "enum": strategy.get("allowed_operators", OPERATORS[:3])},
                         "strength": {"type": "number", "enum": strategy["strength_grid"]},
                         "prediction": TEXT_SCHEMA, "falsifier": TEXT_SCHEMA, "reason": TEXT_SCHEMA})
            policy = _policy(plan)
            key = _hash({"dataset": dataset["fingerprint"], "model": recommender.metadata, "policy": policy})
            reused = key in cache
            if reused:
                candidate, rankings = cache[key]
            else:
                candidate, rankings = _score(dataset, recommender, dataset["development"], policy)
                cache[key] = (candidate, rankings)
            atomic_json(target / "candidate.json", candidate)
            atomic_json(target / "rankings.json", rankings)
            comparison = paired_comparison(baseline, candidate)
            record = {"status": comparison["status"], "execution_status": "completed", "trial_id": target.name,
                      "dataset": dataset["dataset_id"],
                      "research_backend": "observed_itemcf", "proposal": plan, "policy": policy,
                      "execution": {"experiment_id": key, "reused": reused, "independent_repetition": False},
                      "comparison": comparison, "metrics": {"B0": _metrics(baseline), "candidate": _metrics(candidate)},
                      "change_statistics": _change_statistics(baseline, candidate),
                      "examples": _examples(baseline, candidate, strategy["evidence_focus"])}
            for example in record["examples"]:
                example["dataset"] = dataset["dataset_id"]
            suggested = payload["next_action"]
            record["plan_handoff"] = {"suggested": suggested, "actual": policy,
                   "followed": policy["operator"] == suggested["operator"] and policy["strength"] == suggested["strength"],
                   "reason": plan["reason"]}
            atomic_json(target / "archive.json", record)
            update = _ask(model, target, "observed_learn", {"current_dataset": dataset["dataset_id"],
                          "trial": record, "history": history if history_enabled else [], "strategy": strategy},
                       {"lesson": TEXT_SCHEMA, "next_question": TEXT_SCHEMA,
                        "next_operator": {"type": "string", "enum": strategy.get("allowed_operators", OPERATORS[:3])},
                        "next_strength": {"type": "number", "enum": strategy["strength_grid"]},
                        "evidence_state": {"type": "string", "enum": ["supported", "weakened", "inconclusive"]}})
            record["research_update"] = update
            atomic_json(target / "archive.json", record)
            if journal is not None:
                record["candidate_id"] = journal.record_trial(target, record)
            result["trials"].append(record)
            if history_enabled:
                history = history + [record]
            atomic_json(directory / "archive.json", result)
        selected, reason = _select(result["trials"])
        selected_dev, _ = _score(dataset, recommender, dataset["development"], selected)
        result.update(selected_policy=selected, selection_reason=reason,
                      development={"B0": _metrics(baseline), "candidate": _metrics(selected_dev)},
                      research_status="candidate_frozen", execution_status="completed")
        if compute_test:
            matches = [t for t in result["trials"] if t["policy"] == selected]
            candidate_id = matches[-1].get("candidate_id") if matches else None
            if journal is not None and candidate_id is None:
                candidate_id = journal.record_trial(directory / "retained_baseline", {
                    "status": "no_change", "research_backend": "observed_itemcf", "proposal": selected,
                    "execution": {"experiment_id": _hash({"dataset": dataset["fingerprint"], "policy": selected})}})
            result["validation"] = _finalize(directory, dataset, recommender, selected, baseline, selected_dev, journal, candidate_id)
        return result
    except Exception as exc:
        result.update(execution_status="failed", failure={"error_type": type(exc).__name__})
        raise
    finally:
        atomic_json(directory / "archive.json", result)


def fixed_control(directory, dataset, strengths=(.05, .1, .2), operator="popularity_penalty"):
    """A predeclared CPU-only search, using the same number of candidate proposals."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    recommender = ItemCF(dataset["train"], dataset["item_ids"], dataset["positive_threshold"])
    baseline, _ = _score(dataset, recommender, dataset["development"], None)
    trials = []
    for strength in strengths:
        policy = {"operator": operator, "strength": strength}
        candidate, _ = _score(dataset, recommender, dataset["development"], policy)
        trials.append({"policy": policy, "metrics": {"candidate": _metrics(candidate)},
                       "comparison": paired_comparison(baseline, candidate)})
    selected, reason = _select(trials)
    result = {"execution_status": "completed", "control": "fixed_predeclared_operator_grid", "operator": operator,
              "candidate_proposals": len(trials), "model_requests": 0, "selected_policy": selected,
              "selection_reason": reason, "trials": trials}
    selected_dev, _ = _score(dataset, recommender, dataset["development"], selected)
    result["development"] = {"B0": _metrics(baseline), "candidate": _metrics(selected_dev)}
    result["validation"] = _finalize(directory, dataset, recommender, selected, baseline, selected_dev, None, None)
    atomic_json(directory / "archive.json", result)
    return result


def revise_strategy(directory, model, source_result):
    original = source_result["strategy"]
    answer = _ask(model, Path(directory), "strategy_revision", {
        "original_strategy": original, "source_research": source_experience(source_result),
        "editable_fields": ["strength_grid", "evidence_focus", "preferred_operator"],
        "next_task": "另一个数据集；相同候选轮数和模型接口对比v1/v2，不能读取其最终评价。"},
        {"strength_grid": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": .5},
                           "minItems": 2, "maxItems": 5, "uniqueItems": True},
         "evidence_focus": {"type": "string", "enum": ["overall", "tradeoff_users", "tail_relevance"]},
         "preferred_operator": {"type": "string", "enum": original.get("allowed_operators", OPERATORS[:3])},
         "reason": TEXT_SCHEMA, "prediction": TEXT_SCHEMA, "falsifier": TEXT_SCHEMA})
    strategy = {"version": "research_strategy_v2", "strength_grid": sorted(set([0] + answer["strength_grid"])),
                "evidence_focus": answer["evidence_focus"], "preferred_operator": answer["preferred_operator"]}
    if "allowed_operators" in original:
        strategy["allowed_operators"] = original["allowed_operators"]
    atomic_json(Path(directory) / "strategy_v2.json", strategy)
    return strategy, answer


def run_real_campaign(directory, model, source_dataset, transfer_dataset=None, config=None):
    """Compare research with controls, then consume a learned strategy on a new task."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = config or {}
    if config.get("rounds", 3) != 3:
        raise ValueError("this predeclared feasibility campaign requires three source candidate rounds")
    journal = ExperimentJournal(directory / "journal.json")
    result = {"status": "running", "research_scope": "real_data_feasibility_not_RSI_proof", "datasets": {}}
    atomic_json(directory / "campaign.json", result)
    for dataset in [source_dataset] + ([transfer_dataset] if transfer_dataset else []):
        result["datasets"][dataset["dataset_id"]] = {key: dataset[key] for key in ("fingerprint", "protocol", "statistics", "source")}
    cache = {}
    initial_strategy = config.get("research_strategy", DEFAULT_STRATEGY)
    try:
        result["with_history"] = run_study(directory / "with_history", model, source_dataset,
                      rounds=config.get("rounds", 3), strategy=initial_strategy, journal=journal, cache=cache)
        result["without_history"] = run_study(directory / "without_history", model, source_dataset,
                      rounds=config.get("rounds", 3), strategy=initial_strategy, history_enabled=False, journal=journal, cache=cache)
        control_strengths = config.get("control_strengths", (.05, .1, .2))
        if len(control_strengths) != config.get("rounds", 3):
            raise ValueError("fixed control must have the same candidate count")
        result["fixed_control"] = fixed_control(directory / "fixed_control", source_dataset, control_strengths,
                                               config.get("control_operator", "popularity_penalty"))
        with_score = json.loads((directory / "with_history/validation_scores.json").read_text(encoding="utf-8"))["candidate"]
        without_score = json.loads((directory / "without_history/validation_scores.json").read_text(encoding="utf-8"))["candidate"]
        fixed_score = json.loads((directory / "fixed_control/validation_scores.json").read_text(encoding="utf-8"))["candidate"]
        result["direct_controls"] = {"history_vs_no_history": paired_comparison(without_score, with_score),
                                      "history_vs_fixed": paired_comparison(fixed_score, with_score)}
        atomic_json(directory / "campaign.json", result)
        strategy, revision = revise_strategy(directory / "strategy_revision", model, result["with_history"])
        result["strategy_revision"] = {"strategy": strategy, "reasoning": revision,
            "actual_consumers": ["proposal_schema.strength.enum", "case_selection.evidence_focus",
                                 "next_action.strategy_starting_preference_not_forced_operator"]}
        if transfer_dataset:
            # Equal candidate and model-call allowance; identical source-task experience.
            prior = [source_experience(result["with_history"])]
            result["transfer_v1"] = run_study(directory / "transfer_v1", model, transfer_dataset,
                    rounds=config.get("transfer_rounds", 2), strategy=initial_strategy, prior_history=prior, journal=journal, cache=cache)
            result["transfer_v2"] = run_study(directory / "transfer_v2", model, transfer_dataset,
                    rounds=config.get("transfer_rounds", 2), strategy=strategy, prior_history=prior, journal=journal, cache=cache)
            original = json.loads((directory / "transfer_v1/validation_scores.json").read_text(encoding="utf-8"))["candidate"]
            revised = json.loads((directory / "transfer_v2/validation_scores.json").read_text(encoding="utf-8"))["candidate"]
            result["strategy_transfer_comparison"] = paired_comparison(original, revised)
            result["strategy_transfer_comparison"]["interpretation"] = "One exploratory strategy-transfer contrast, not replicated RSI evidence."
        result["status"] = "completed"
        return result
    except Exception as exc:
        result.update(status="failed", failure={"error_type": type(exc).__name__})
        raise
    finally:
        result["model_requests"] = model.requests
        atomic_json(directory / "campaign.json", result)
        render_campaign_report(directory, result)


def render_campaign_report(directory, result):
    lines = ["# 官方真实数据研究：可行性验证", "", "执行状态：" + result["status"], "",
             "推荐器在本机CPU执行；真实模型只提出研究实验、解释反馈及修改研究策略。", "",
             "| 对照 | 冻结改法 | 外部质量 | 面板额外热门倾斜 | 点门槛 |", "|---|---|---:|---:|---|"]
    for key, title in (("with_history", "带历史研究员"), ("without_history", "无历史研究员"),
                       ("fixed_control", "固定规则"), ("transfer_v1", "新任务策略v1"), ("transfer_v2", "新任务策略v2")):
        arm = result.get(key)
        if not arm or not arm.get("validation"):
            continue
        validation = arm["validation"]
        score = validation["metrics"]["candidate"]
        policy = arm["selected_policy"]
        quality = f"{score['ndcg']:.5f}" if score['ndcg'] is not None else "不可评价"
        bias = f"{score['panel_excess_head_share']:.5f}" if score['panel_excess_head_share'] is not None else "不可评价"
        lines.append(f"| {title} | {policy['operator']} / {policy['strength']:.4g} | {quality} | {bias} | {validation['comparison']['status']} |")
    lines += ["", "质量是随机反馈中已观察候选的NDCG@5，额外热门倾斜也只在同一面板中定义。未观察反馈保持未知。",
              "", "候选按开发反馈选择并冻结，外部反馈只评价冻结候选。Coat验证反馈用于改进下一任务研究策略，因此不是整个元研究流程的最终未见测试。",
              "", "固定规则与研究员使用相同候选提案数，模型费用不同；有/无历史研究员的角色调用数相同。新任务v1/v2候选预算相同。",
              "", "逐轮假设、竞争解释、实际指标及下一行动在各臂 trials 目录。研究策略变化及程序使用位置在 strategy_revision 目录。",
              "", "点门槛与配对用户不确定区间分别保存；小型试验只能验证自主研究和策略修改可执行，不能单独证明因果去偏或RSI稳定成立。",
              "", f"本批模型实际请求：{result.get('model_requests', 0)}（包括重试）。"]
    (Path(directory) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_transfer_check(directory, model, dataset, source_result, revised_strategy):
    """One new-cohort feasibility contrast after fixing source-context attribution."""
    directory = Path(directory)
    journal = ExperimentJournal(directory / "journal.json")
    prior = [source_experience(source_result)]
    result = {"status": "running", "scope": "context_attribution_followup_not_confirmatory_RSI",
              "dataset_fingerprint": dataset["fingerprint"], "dataset_protocol": dataset["protocol"],
              "dataset_statistics": dataset["statistics"], "candidate_rounds_per_arm": 1}
    cache = {}
    try:
        for name, strategy in (("transfer_v1", DEFAULT_STRATEGY), ("transfer_v2", revised_strategy)):
            result[name] = run_study(directory / name, model, dataset, rounds=1, strategy=strategy,
                                     prior_history=prior, journal=journal, cache=cache)
            atomic_json(directory / "campaign.json", result)
        before = json.loads((directory / "transfer_v1/validation_scores.json").read_text(encoding="utf-8"))["candidate"]
        after = json.loads((directory / "transfer_v2/validation_scores.json").read_text(encoding="utf-8"))["candidate"]
        result["strategy_transfer_comparison"] = paired_comparison(before, after)
        result["status"] = "completed"
        return result
    except Exception as exc:
        result.update(status="failed", failure={"error_type": type(exc).__name__})
        raise
    finally:
        result["model_requests"] = model.requests
        atomic_json(directory / "campaign.json", result)
        render_campaign_report(directory, result)
