"""Executable research actions; no diagnosis, scoring, or promotion decisions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from rsi_researcher.runtime.client import API_PROTOCOL, MODEL_PARAMETERS, RANKING_PROTOCOL, atomic_json
from rsi_researcher.recommenders.agentcf import AgentCF
from rsi_researcher.recommenders.execution import execute_arm, intervention
from rsi_researcher.recommenders.replay import repair


OPERATORS = {
    "retain": {"meaning": "保留当前规则，承认证据不足", "executable": False},
    "uncertain_feedback": {"meaning": "未消费的比较项按未知处理，再从相同初态学习", "executable": True},
    "dependency_replay": {"meaning": "修正指定源事件的反馈解释，并重建读过它的后续记忆", "executable": True},
    "popularity_rerank": {"meaning": "用前置热门计数作固定强度的输出重排，记忆不变", "executable": True},
    "no_sharing": {"meaning": "每人使用自己的物品笔记，切断跨用户物品共享", "executable": True},
    "swap_training_candidates": {"meaning": "只交换训练候选的前后位置，事件与反馈不变", "executable": True},
    "exploration": {"meaning": "改变实际曝光并收集新反馈；固定旧日志不能提供这些答案", "executable": False},
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe_delta(proposal, rerank_strength):
    """Name the one implemented change, separate from a researcher's explanation."""
    operator = proposal["operator"]
    changes = {
        "uncertain_feedback": ("feedback_semantics", {"feedback_templates": "unobserved_is_unknown"}),
        "no_sharing": ("memory_sharing", {"item_notebook_scope": "per_user"}),
        "swap_training_candidates": ("training_input_order", {"candidate_order": "reverse_each_pair"}),
        "popularity_rerank": ("output_ranking", {"preperiod_popularity_penalty": rerank_strength}),
        "dependency_replay": ("memory_repair", {"root_event_ids": sorted(set(proposal["root_event_ids"])),
                                                "root_feedback": "unobserved_is_unknown"}),
        "retain": ("none", {}), "exploration": ("exposure_data_collection", {"execution": "requires_new_feedback"}),
    }
    target, change = changes[operator]
    return {"reference": "B0", "target": target, "change": change,
            "implementation": "preimplemented_intervention_not_generated_code"}


def execution_key(proposal, baseline_fingerprint, rerank_strength, metadata=None):
    """Question wording and experiment purpose do not change a physical execution."""
    value = {"baseline": baseline_fingerprint, "operator": proposal["operator"],
             "roots": sorted(set(proposal["root_event_ids"])),
             "model_execution": metadata or {"api_protocol": API_PROTOCOL,
                             "ranking_protocol": RANKING_PROTOCOL, "model_parameters": MODEL_PARAMETERS},
             "rerank_strength": rerank_strength if proposal["operator"] == "popularity_rerank" else None}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def rerank(pair, public, strength):
    """Fixed output control using observed preperiod counts, not preference truth."""
    counts = public["exposure"]["preperiod_counts"]
    scale = max(counts.values()) or 1
    result = []
    for ranking in pair:
        changed = {}
        for qid, order in ranking.items():
            if order is None:
                changed[qid] = None
                continue
            positions = {item: rank for rank, item in enumerate(order)}
            changed[qid] = sorted(order, key=lambda item:
                                 positions[item] / max(1, len(order) - 1) + strength * counts[item] / scale)
        result.append(changed)
    return tuple(result)


def execute_candidate(directory, model, prompts, public, baseline_dir, proposal, rerank_strength):
    """Run one action from the fixed B0 records and return forward/reverse rankings."""
    directory, baseline_dir = Path(directory), Path(baseline_dir)
    state = _read(baseline_dir / "state.json")
    records = [_read(baseline_dir / (eid + ".json")) for eid in state["completed"]]
    events = [record["event"] for record in records]
    operator = proposal["operator"]
    if operator == "uncertain_feedback":
        return execute_arm(directory, model, intervention(prompts, "uncertain_feedback"), public, events)
    if operator in ("no_sharing", "swap_training_candidates"):
        probe_events = deepcopy(events)
        if operator == "swap_training_candidates":
            for event in probe_events:
                event["candidates"].reverse()
        return execute_arm(directory, model, prompts, public, probe_events,
                           memory_policy="no_sharing" if operator == "no_sharing" else "original_update")
    if operator == "dependency_replay":
        repaired = repair(records, proposal["root_event_ids"], AgentCF(model, prompts),
                          AgentCF(model, intervention(prompts, "uncertain_feedback")), directory)
        return execute_arm(directory / "ranking", model, prompts, public | {"initial": repaired}, [])
    if operator == "popularity_rerank":
        saved = _read(baseline_dir / "rankings.json")
        pair = rerank((saved["forward"], saved["reverse"]), public, rerank_strength)
        directory.mkdir()
        atomic_json(directory / "state.json", state)
        atomic_json(directory / "rankings.json", {"forward": pair[0], "reverse": pair[1]})
        return pair
    raise ValueError("this action needs no execution or requires new exposure data")
