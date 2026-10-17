"""Fixed replies for an offline workflow demonstration, never research evidence.

This adapter performs no network requests and reads no credentials. Its request
counter counts demonstration steps, not paid API usage or tokens.
"""

from copy import deepcopy
import json
import re

from jsonschema import Draft202012Validator

from rsi_researcher.runtime.client import BudgetExceeded, InvalidOutput, MEMORY_CONTROL_PROTOCOL, RANKING_PROTOCOL


class OfflineResearchModel:
    """Exercise the same controller interface using deliberately scripted answers."""

    def __init__(self, operator="uncertain_feedback", proceed=True, purpose="debias", follow_agenda=True,
                 budget=300, limits=None):
        if type(budget) is not int or budget < 0:
            raise ValueError("offline request allowance must be a nonnegative integer")
        self.requests = 0
        self.max_requests = min(budget, limits.max_requests) if limits is not None else budget
        self.research_operator, self.operator, self.proceed = operator, operator, proceed
        self.purpose, self.follow_agenda = purpose, follow_agenda
        self.limits, self.scope, self.calls = limits, {}, []
        self.metadata = {"api_protocol": "offline_test_double_v1", "ranking_protocol": RANKING_PROTOCOL,
                         "memory_control_protocol": MEMORY_CONTROL_PROTOCOL,
                         "model_parameters": {"model": "offline_fixed_responses", "temperature": 0}}

    def ask(self, prompt, schema, context):
        if self.limits is not None:
            self.limits.remaining_seconds()
        if self.requests >= self.max_requests:
            raise BudgetExceeded("offline demonstration request allowance exhausted")
        self.requests += 1
        self.calls.append((prompt, deepcopy(context)))
        stage = context["stage"]
        if stage in ("diagnose", "design", "review", "learn"):
            payload = json.loads(prompt.split("\n", 1)[1])
            result = self._research_reply(stage, payload)
        elif stage == "diagnose_and_propose":
            result = {"parent": "B0", "operator": self.research_operator,
                      "mechanism": "离线固定假设", "evidence": "离线流程输入",
                      "expected_direction": "检验质量与偏差的变化", "falsifier": "实际变化不符合预期"}
        elif stage == "select":
            ids = re.findall(r"ID=([^;\s]+);", prompt)
            if not ids:
                raise InvalidOutput("offline selection prompt contains no candidate IDs")
            result = {"selected_id": ids[0], "reason": "离线固定规则：选择第一个候选"}
        elif stage == "user_update":
            result = {"memory": "Offline demonstration user memory; this text is a fixed reply."}
        elif stage == "item_update":
            result = {"first_memory": "Offline demonstration first item memory.",
                      "second_memory": "Offline demonstration second item memory."}
        elif stage == "rank":
            result = {"ranking": re.findall(r"ID=([^;\s]+);", prompt)}
        elif stage == "rank_missing_insertions":
            omitted = json.loads(prompt.rsplit("\nOmitted IDs: ", 1)[1])
            result = {"insertions": [{"item_id": item, "before_id": None} for item in omitted]}
        else:
            raise InvalidOutput("unsupported offline demonstration stage: " + stage)
        Draft202012Validator(schema).validate(result)
        if self.limits is not None:
            self.limits.remaining_seconds()
        return result

    def _research_reply(self, stage, payload):
        if stage == "diagnose":
            event_ids = [event["event_id"] for event in payload.get("events", [])]
            return {"hypothesis": "未观察项被当作不喜欢，可能改变后续共享笔记。",
                    "evidence_event_ids": event_ids[:1],
                    "alternative_explanation": "也可能只是候选位置变化。",
                    "missing_evidence": "需要区分记忆和输出重排的作用。"}
        if stage == "design":
            agenda = payload["agenda"]
            plan = agenda["suggested_experiment"] if self.follow_agenda else None
            event_ids = [event["event_id"] for event in payload["evidence"].get("events", [])]
            result = {"parent": "B0", "operator": self.research_operator,
                      "root_event_ids": event_ids[:1] if self.research_operator == "dependency_replay" else [],
                      "rationale": "只改变一个机制，与原版比较。", "expected_direction": "质量保持且偏差降低。",
                      "falsifier": "质量损失或没有减少额外热门倾斜。", "purpose": self.purpose,
                      "research_question": agenda["research_question"],
                      "prediction_if_hypothesis": "改变反馈解释后，额外热门倾斜减弱。",
                      "prediction_if_alternative": "若原因只在候选位置，改变反馈解释仍不能消除位置差异。",
                      "plan_reason": "沿用上轮的待查问题。" if plan else "根据当前可见证据设计本轮行动；必要时修订旧计划。"}
            if plan:
                result.update(deepcopy(plan))
            return result
        if stage == "learn":
            shared_probe = (payload["proposal"]["operator"] == "no_sharing"
                            and payload["test_status"] == "probe_completed")
            return {"evidence_state": "inconclusive", "updated_explanation": "当前对照仍不足以归因。",
                    "lesson": "共享切断对照已执行，下一步检查训练输入位置。" if shared_probe else "需要区分输出重排和共享记忆传播。",
                    "next_experiment": {
                        "research_question": "只交换训练候选位置会不会改变结果？" if shared_probe else "切断物品共享后还会出现同样现象吗？",
                        "purpose": "mechanism_probe", "operator": "swap_training_candidates" if shared_probe else "no_sharing",
                        "root_event_ids": [],
                        "prediction_if_hypothesis": "若输入位置是原因，交换位置可能改变响应。" if shared_probe else "若共享传播是原因，切断共享后偏差变化会减弱。",
                        "prediction_if_alternative": "若只由共享机制驱动，保持共享而交换位置不应造成系统变化。" if shared_probe else "若只是输入位置，切断共享仍会保留相应差异。"}}
        return {"proceed": self.proceed, "reason": "对照可执行，但需要保留替代解释。",
                "counterexample": "热门可能正好符合真实兴趣。"}
