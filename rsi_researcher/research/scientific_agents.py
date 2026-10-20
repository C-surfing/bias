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
DEBIAS_CRITIQUE = strict({
    **CRITIQUE["properties"],
    "counterexample_if_null_true": TEXT,
    "unexcluded_competing_mechanism": TEXT,
    "registered_falsifier_measurable": {"type":"boolean"},
    "relevance_to_frozen_null": {"type":"string","enum":[
        "logically_discriminating","relevant_but_not_identifying",
        "not_relevant"]},
})
REFLECTION = strict({
    "evidence_state": {"type": "string", "enum": ["supported", "weakened", "inconclusive"]},
    "numeric_interpretation": TEXT, "remaining_alternative": TEXT,
    "next_question": TEXT, "candidate_research_skill": TEXT,
    "suggested_next_task": {"type": "string", "enum": ["stop"]},
    "next_task_reason": TEXT,
})


def authoritative_evidence(audit=None):
    """Only the Python numerical audit, never the LLM, grants evidence levels.

    Both historic science/tree episodes and five-bias Debias Director share
    this single, intentionally small, evidence authority boundary.
    """
    audit = audit or {}
    confirmation = audit.get("independent_confirmation") is True
    # Older experimental protocols always record zero confirmed findings;
    # a fresh confirmation needs a separately frozen evaluator.
    development = bool(
        audit.get("development_effect_passes")
        or audit.get("debias_candidate_eligible")
        or audit.get("forecast_passes_both_panels"))
    return {
        "authority": "python_numeric_audit_only",
        "statistical_significance": "NOT_ESTABLISHED_BY_THIS_WORKFLOW",
        "sequential_error_control": "NOT_AVAILABLE",
        "independent_confirmation": confirmation,
        "development_check_passed": development,
        "causal_identification": (
            "SUPPORTED_WITHIN_REGISTERED_SIMULATOR_ONLY"
            if audit.get("evidence_class","").startswith("SYNTHETIC_")
            else "NOT_ESTABLISHED_BY_THIS_WORKFLOW"),
        "model_reflection_is_hypothesis_not_evidence": True,
        "confirmed_scientific_discoveries": 0,
    }


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
        "Apply the task-specific registered protocol rather than imposing "
        "an AgentCF ranking intervention protocol on Coat. For Coat selection "
        "tasks, independently scored randomized-development user panels and "
        "fixed propensity/gamma/placebo controls are the relevant tests; "
        "AgentCF paired memory/ranking tests require frozen trained states, "
        "unchanged ranking repeats and linked-writer support. Reject a "
        "proposal only if its specific frozen test is unable to distinguish "
        "the named competing mechanism, is unpaired, or claims unsupported "
        "causal effects; cite the relevant protocol failure precisely. "
        "Development admissibility never implies independent confirmation. "
        "For each experiment, attempt a CONCRETE counterexample: can the "
        "primary null be true while the proposed measurable falsifier passes? "
        "Inspect the task-specific data requirements and support before "
        "asking for propensity, placebo, or Gamma controls from another bias. "
        "Never treat randomized reference prediction as identified policy OPE. "
        "Your approval NEVER grants execution permission."
    )
    schema = CRITIQUE

    def review_debias(self, observation, episode_id):
        # Same Critic Agent with a stronger task-specific POPPER-style
        # logical link contract; not an added role or a new LLM service.
        return self.act(observation, episode_id, schema=DEBIAS_CRITIQUE)


class ReflectionAgent(ResearchAgent):
    role = "reflection"
    instruction = (
        "You are the research reflection scientist. Interpret the independent "
        "measured evidence, the frozen forecast verdict and the experiment cost. "
        "Propose the next discriminating scientific question and ONE TENTATIVE "
        "research skill. The Python numerical audit is the ONLY evidence "
        "authority. No significance test or independent confirmation was "
        "performed here: never call an improvement statistically significant, "
        "causally confirmed, or say further confirmation is unnecessary. "
        "Do not mark a skill verified or reverse a failed gate."
    )
    schema = REFLECTION

    def reflect(self, observation, episode_id, allowed_followups):
        schema = json.loads(json.dumps(self.schema))
        schema["properties"]["suggested_next_task"]["enum"] = ["stop", *allowed_followups]
        return self.act({**observation, "allowed_next_tasks": allowed_followups},
                        episode_id, schema)
