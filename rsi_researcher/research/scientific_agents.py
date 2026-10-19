"""Real model-backed scientific roles; scoped inputs, prompts and JSON contracts.

The four agents are distinct Qwen calls with separate responsibilities.
They have no shell, write, evaluator, or hidden-label tools. The orchestrator
and registered experiment executor own all external actions.
"""
from __future__ import annotations

import json
from jsonschema import Draft202012Validator


def strict(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


TEXT = {"type": "string", "minLength": 12}
DIAGNOSIS = strict({
    "hypothesis": TEXT, "competing_explanation": TEXT,
    "missing_evidence": TEXT, "why_this_matters": TEXT,
})
CRITIQUE = strict({
    "decision": {"type": "string", "enum": ["approve", "reject"]},
    "reason": TEXT, "confounding_risk": TEXT,
    "better_discriminating_test": TEXT,
})
REFLECTION = strict({
    "evidence_state": {"type": "string", "enum": ["supported", "weakened", "inconclusive"]},
    "numeric_interpretation": TEXT, "remaining_alternative": TEXT,
    "next_question": TEXT, "candidate_research_skill": TEXT,
    "suggested_next_task": {"type": "string", "enum": ["stop"]},
    "next_task_reason": TEXT,
})


def design_schema(actions, metrics):
    return strict({
        "action": {"type": "string", "enum": sorted([*actions, "abstain"])},
        "hypothesis": TEXT,
        "competing_explanation": TEXT,
        "falsifier": TEXT,
        "metric": {"type": "string", "enum": sorted(metrics)},
        "direction": {"type": "string", "enum": ["increase", "decrease", "approximately_zero"]},
        "minimum_abs_effect": {"type": "number", "minimum": 0, "maximum": 1},
        "budget_justification": TEXT,
    })


class ResearchAgent:
    role = ""
    instruction = ""
    schema = {}

    def __init__(self, model):
        self.model = model

    def act(self, observation, episode_id, schema=None):
        contract = schema or self.schema
        Draft202012Validator.check_schema(contract)
        self.model.scope = {"scientific_role": self.role, "episode": episode_id}
        result = self.model.ask(
            self.instruction
            + "\nOnly the provided DEVELOPMENT evidence is visible. "
              "Never claim independent confirmation, run a shell command, "
              "invent a tool action, or alter measured numeric evidence.\n"
            + json.dumps(observation, ensure_ascii=False, sort_keys=True),
            contract, {"stage": self.role},
        )
        Draft202012Validator(contract).validate(result)
        return result


class DiagnosisAgent(ResearchAgent):
    role = "diagnosis"
    instruction = (
        "You are the diagnosis scientist. Infer a falsifiable mechanism from "
        "the given evidence and a serious alternative explanation. Identify "
        "what evidence is missing and why the experiment is scientifically useful."
    )
    schema = DIAGNOSIS


class DesignerAgent(ResearchAgent):
    role = "design"
    instruction = (
        "You are the experiment-design scientist. Select ONE legal task action "
        "with a machine-checkable metric, effect direction, minimum magnitude, "
        "strong falsifier and cost justification. Use only the supplied allowed "
        "actions and metrics. Your predicted direction must match the falsifier; "
        "do not claim the predicted result already happened. The minimum "
        "effect must meet the evaluator's fixed threshold for your metric. "
        "Compare the diagnostic value of legal actions. Abstain if available "
        "evidence cannot support a discriminating intervention. Explain why "
        "the chosen action distinguishes the competing mechanism, or why "
        "neither does, in the budget_justification and falsifier."
    )

    def propose(self, observation, episode_id, actions, metrics):
        return self.act(observation, episode_id, design_schema(actions, metrics))


class CriticAgent(ResearchAgent):
    role = "critic"
    instruction = (
        "You are a skeptical peer reviewer. Assess whether the registered "
        "single-variable action distinguishes the primary hypothesis from "
        "the alternative under the PREDECLARED development conditions. "
        "Do not demand independent confirmation before allowing a diagnostic "
        "development experiment; require it only for final causal claims. "
        "A paired frozen-state prompt intervention with unchanged candidate "
        "panels and no-change repeat can test a conditional prompt mechanism, "
        "NOT population causal popularity bias. Reject unsupported global "
        "causal claims, unpaired controls and non-discriminating actions. "
        "Your approval NEVER grants execution permission."
    )
    schema = CRITIQUE


class ReflectionAgent(ResearchAgent):
    role = "reflection"
    instruction = (
        "You are the research reflection scientist. Interpret the independent "
        "measured evidence, the frozen forecast verdict and the experiment cost. "
        "Propose the next discriminating scientific question and ONE TENTATIVE "
        "research skill. Do not mark a skill verified or reverse a failed gate."
    )
    schema = REFLECTION

    def reflect(self, observation, episode_id, allowed_followups):
        schema = json.loads(json.dumps(self.schema))
        schema["properties"]["suggested_next_task"]["enum"] = ["stop", *allowed_followups]
        return self.act({**observation, "allowed_next_tasks": allowed_followups},
                        episode_id, schema)
