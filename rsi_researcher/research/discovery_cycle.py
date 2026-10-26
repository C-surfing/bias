"""Autonomous discovery research cycle over the existing model and Journal.

No custom model transport, tree search, shell-capable agent, or grading oracle.
The LLM only sees the observation and controlled experimental results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from rsi_researcher.research.discovery_lab import FIELDS, CONDITIONS

def lab_query(secret, task, action, split="development", factor=None,
              condition="all", tag=1, n=384, seed=None):
    """Calls the black-box environment in a SEPARATE process; never imports its truth."""
    command = [sys.executable, "-B", "-m", "rsi_researcher.research.discovery_lab",
               "--mode", "query", "--private", str(secret), "--task", task,
               "--action", action, "--split", split, "--condition", condition,
               "--tag", str(tag), "--n", str(n)]
    if factor:
        command += ["--factor", factor]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)

def private_grade(secret, submitted_path):
    command = [sys.executable, "-B", "-m", "rsi_researcher.research.discovery_lab",
               "--mode", "grade", "--private", str(secret),
               "--submissions", str(submitted_path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)

from rsi_researcher.runtime.client import atomic_json

def contract(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}

T = {"type": "string", "minLength": 8}
FACTOR = {"type": "string", "enum": list(FIELDS)}
CONDITION = {"type": "string", "enum": list(CONDITIONS)}
TAG = {"type": "integer", "enum": [0, 1]}
DIRECTION = {"type": "string", "enum": ["increase", "decrease", "no_effect"]}
HYPOTHESIS = contract({
    "hypothesis": T, "mechanism_explanation": T, "competing_explanation": T,
    "falsifier": T, "scope_explanation": T, "factor": FACTOR,
    "condition": CONDITION, "tag": TAG, "direction": DIRECTION,
    "confidence": {"type": "string", "enum": ["low", "medium", "high"]}})
EXPLORER = contract({"question": T, "feature_to_inspect": FACTOR, "reason": T})
CONCEPT_EXPLORER = contract({
    "factor": FACTOR,
    "observational_signal": T,
    "why_informative": T,
    "competing_explanation": T,
    "falsification_plan": T})
DESIGNER = contract({
    "decision": {"type": "string", "enum": ["test", "reject", "stop"]},
    "factor": FACTOR, "condition": CONDITION, "tag": TAG,
    "contrast_reason": T, "predicted_direction": DIRECTION})
REVIEW = contract({"assessment": {"type": "string", "enum": ["supported", "falsified", "inconclusive"]},
                   "critic": T, "remaining_alternative": T, "next_question": T})

def ask(model, role, instruction, payload, schema, episode):
    from rsi_researcher.research.scientific_agents import ResearchAgent
    agent = ResearchAgent(model)
    agent.role = role
    agent.instruction = instruction
    return agent.act(payload, episode, schema)

def is_effect(result, direction):
    low, high = result["ci95"]
    return (direction == "increase" and low > 0.0 or
            direction == "decrease" and high < 0.0)

def record_claim(hyp, arm, evidence, state):
    return {"kind": "ResearchClaim", "arm": arm,
            "observation": "user-level grouped rankings, not ground truth",
            "hypothesis": hyp["hypothesis"],
            "mechanism_explanation": hyp["mechanism_explanation"],
            "scope": hyp["condition"], "scope_explanation": hyp["scope_explanation"],
            "factor": hyp["factor"], "tag": hyp["tag"],
            "direction": hyp["direction"], "confidence": hyp["confidence"],
            "competing_explanation": hyp["competing_explanation"],
            "proposed_falsification": hyp["falsifier"],
            "experiment_refs": [e["id"] for e in evidence],
            "status": state}

def static_hyp(factor, condition, tag, direction, arm):
    return {"hypothesis": f"{factor} changes tag {tag} exposure conditional on {condition}",
            "mechanism_explanation": "Candidate scores change under a matched setting change.",
            "competing_explanation": "Observational difference or random ranking variation.",
            "falsifier": "A paired intervention at fixed candidates should show no effect.",
            "scope_explanation": f"Conditional effect in {condition} users",
            "factor": factor, "condition": condition, "tag": tag,
            "direction": direction, "confidence": "low"}

def fixed_scan(secret, task, seed):
    # Complete predeclared grid, rather than a handicapped three-rule baseline.
    # It is intentionally expensive and records every attempted comparison.
    evidence, by_key = [], {}
    for factor in FIELDS:
        for tag in (0, 1):
            for condition in CONDITIONS:
                result = lab_query(secret, task, "contrast", factor=factor,
                                   condition=condition, tag=tag)
                record = {"kind": "ExperimentRecord",
                          "id": f"e{len(evidence):02d}",
                          "phase": "development", "action": result["intervention"],
                          "result": result}
                evidence.append(record)
                by_key[(factor, tag, condition)] = result
    candidates = []
    for factor in FIELDS:
        for tag in (0, 1):
            all_effect = by_key[(factor, tag, "all")]["effect"]
            effect_0 = by_key[(factor, tag, "segment_0")]["effect"]
            effect_1 = by_key[(factor, tag, "segment_1")]["effect"]
            if abs(effect_1) > 0.07 and abs(effect_0) < 0.025:
                condition = "segment_1"
            elif abs(effect_0) > 0.07 and abs(effect_1) < 0.025:
                condition = "segment_0"
            else:
                condition = "all"
            measure = by_key[(factor, tag, condition)]
            candidates.append((abs(measure["effect"]), factor, tag, condition, measure))
    _, factor, tag, condition, result = max(candidates)
    direction = "increase" if result["effect"] > 0 else "decrease"
    significant = abs(result["effect"]) >= 0.07 and is_effect(result, direction)
    hyp = static_hyp(factor, condition, tag, direction, "fixed")
    return record_claim(hyp, "fixed", evidence,
                        "supported" if significant else "inconclusive"), evidence, []

def one_shot(model, observation, task):
    hyp = ask(model, "discovery-hypothesis",
              "Propose one falsifiable mechanism for a previously unspecified bias, or express no effect. "
              "Use measurable variables and avoid classic bias labels without a mechanism. "
              "You only see observed ranking behavior. Hypotheses are not validated findings.",
              {"observation": observation}, HYPOTHESIS, task+"-oneshot")
    return record_claim(hyp, "one_shot", [], "proposed"), [], [{"role": "hypothesis", "output": hyp}]

def concept_test(model, secret, task, observation, seed):
    """Open concept proposals + registered paired falsification, without a search tree.

    The LLM chooses up to three different factors, with no mechanism labels.
    Both user segments are tested with identical candidate sets and paired
    randomness, so an effect of the opposite sign or only in a subgroup
    can generate a REVISED claim, explicitly distinct from the initial guess.
    Only a new cohort can subsequently confirm it.
    """
    evidence, trajectory, previous = [], [], []
    tested_factors = set()
    last_hyp = None
    for index in range(3):
        remaining = [field for field in FIELDS if field not in tested_factors]
        # These are public associations for every available factor, not a
        # hidden-answer ranking or an encoded Bias type list.
        cards = [card for card in observation["contrast_cards"]
                 if card["field"] in remaining]
        # Present all public observational contrasts, most discriminating first,
        # without revealing latent mechanisms or choosing experiments for the LLM.
        # Signal/interval precision is a *heuristic for exploration*, not a
        # mechanism label and never a causal or confirmatory test.
        cards.sort(key=lambda card: abs(card["observed_difference_tag1"]) /
                   max(1e-5, card["descriptive_interval95"][1] -
                       card["descriptive_interval95"][0]), reverse=True)
        explorer_schema = {**CONCEPT_EXPLORER, "properties": {
            **CONCEPT_EXPLORER["properties"],
            "factor": {"type": "string", "enum": remaining}}}
        exploration = ask(model, "discovery-concept-explorer",
                          "Act as a data-driven scientific Explorer. Select ONE untested "
                          "variable for the next falsification. Use the numeric per-segment "
                          "observational contrasts, not semantic stereotypes of known Biases. "
                          "Identify an unexpectedly large or conditional association, give "
                          "a competing explanation, and propose a paired intervention that "
                          "could disprove it. Small nonzero observational differences "
                          "can be sampling noise. This is not causal proof. "
                          "Budget allows only three distinct factors; scrutinize the "
                          "effect size and interval of ALL public variables before choosing. "
                          "The cards are sorted by observational signal/uncertainty, NOT hidden "
                          "mechanism likelihood. Test the most discriminating evidence rather "
                          "than habitual semantic choices; prefer a rival-sensitive test.",
                          {"field_cards": cards,
                           "previous_measured_experiments": previous,
                           "available_fields": remaining,
                           "remaining_test_budget": 3-index},
                          explorer_schema, f"{task}-explorer-{index}")
        trajectory.append({"role": "data_driven_explorer", "index": index,
                           "public_observation_cards": cards, "output": exploration})
        chosen = exploration["factor"]
        schema = {
            **HYPOTHESIS,
            "properties": {**HYPOTHESIS["properties"],
                           "factor": {"type": "string", "enum": [chosen]}}}
        hyp = ask(model, "discovery-concept",
                  "Develop a falsifiable scientific hypothesis about the Explorer-selected "
                  "variable using the observed recommendation data, without relying on "
                  "named classical Bias categories. Explicitly distinguish observational "
                  "association from a causal claim. Exposure@3 means fraction of top-three "
                  "items of a particular tag, not exposure at the third position. "
                  "History measures simulated earlier exposure events, not user engagement. "
                  "Specify an alternative explanation, scope and falsification. "
                  "Do not assume the initial observed direction is causal.",
                  {"observations": observation, "exploration": exploration,
                   "previous_measured_experiments": previous,
                   "selected_factor": chosen},
                  schema, f"{task}-concept-{index}")
        tested_factors.add(hyp["factor"])
        last_hyp = hyp
        trajectory.append({"role": "proposed_concept", "index": index, "output": hyp})
        tests = {}
        for segment in ("segment_0", "segment_1"):
            result = lab_query(secret, task, "contrast", seed=seed,
                               factor=hyp["factor"], condition=segment,
                               tag=hyp["tag"])
            record = {"kind": "ExperimentRecord", "id": f"e{len(evidence):02d}",
                      "phase": "development", "action": result["intervention"],
                      "result": result}
            evidence.append(record)
            tests[segment] = result
            trajectory.append({"role": "registered_paired_experiment",
                               "proposed_concept_index": index, "record": record})
        changes = {}
        for segment, result in tests.items():
            direction = "increase" if result["effect"] > 0 else "decrease"
            if abs(result["effect"]) >= 0.05 and is_effect(result, direction):
                changes[segment] = direction
        previous.append({"tested_factor": hyp["factor"], "tag": hyp["tag"],
                         "segments": {k: {"effect": v["effect"], "ci95": v["ci95"]}
                                      for k,v in tests.items()},
                         "conclusion": "measured_effect" if changes else "no_detected_effect"})
        if not changes:
            continue
        # Contradictory subgroup directions do not justify one scoped claim.
        if len(changes)==2 and len(set(changes.values()))>1:
            continue
        scope = "all" if len(changes)==2 else next(iter(changes))
        direction = next(iter(changes.values()))
        revised_schema = {**HYPOTHESIS, "properties": {
            **HYPOTHESIS["properties"],
            "factor": {"type": "string", "enum": [hyp["factor"]]},
            "condition": {"type": "string", "enum": [scope]},
            "tag": {"type": "integer", "enum": [hyp["tag"]]},
            "direction": {"type": "string", "enum": [direction]}}}
        revision = ask(model, "discovery-claim-revision",
                       "Update the mechanism explanation from the ACTUALLY EXECUTED registered "
                       "paired results, correcting any mistaken initial direction or scope. "
                       "The measured factor, scope, tag and direction are fixed by the numeric "
                       "audit. Give a clear causal testable hypothesis and genuine alternative. "
                       "Exposure@3 is the fraction of top-three candidate items with a "
                       "given tag, not the single third rank position. History counts prior "
                       "simulated exposure events, not engagement. Separate what the paired "
                       "experiment measured from any unverified explanatory speculation. "
                       "Never imply this exploratory evidence is independent confirmation.",
                       {"initial_candidate": hyp, "paired_tests": tests,
                        "reason_for_revision": "measured effect changes and subgroup boundaries"},
                       revised_schema, f"{task}-concept-revision-{index}")
        trajectory.append({"role": "evidence_driven_revision", "index": index,
                           "original": hyp, "revised": revision,
                           "measured_direction": direction, "measured_scope": scope})
        claim = record_claim(revision, "concept_test", evidence, "supported")
        claim["evidence_based_effect_statement"] = (
            f"At fixed user identity, candidates and RNG conditions, changing "
            f"{revision['factor']} from {tests[next(iter(tests))]['intervention']['levels'][0]} "
            f"to {tests[next(iter(tests))]['intervention']['levels'][1]} "
            f"changes tag-{revision['tag']} fraction among top-three items "
            f"for {scope} in the measured {direction} direction. "
            f"This is a simulator-level paired outcome, not a fully identified "
            f"real-world mechanism.")
        claim["interpretation_needs_scientific_review"] = True
        claim["initial_hypothesis"] = hyp
        claim["candidate_concepts_tested"] = len(previous)
        claim["changed_after_experiment"] = (
            hyp["direction"]!=revision["direction"]
            or hyp["condition"]!=revision["condition"])
        return claim, evidence, trajectory
    if last_hyp is None:
        raise ValueError("no generated hypotheses")
    claim = record_claim(last_hyp, "concept_test", evidence, "inconclusive")
    claim["candidate_concepts_tested"] = len(previous)
    return claim, evidence, trajectory

def active(model, secret, task, observation, seed, rounds=3):
    experiments, trajectory, revisions, supported_candidates = [], [], [], []
    context = {"initial_observation": observation, "previous_trials": []}
    last_hyp = None
    for turn in range(rounds):
        explorer = ask(model, "discovery-explorer",
                       "Select which public variable and group comparison is most informative. "
                       "Do not assume classic biases; prefer tests distinguishing explanations.",
                       context, EXPLORER, f"{task}-explore-{turn}")
        focused = {"field": explorer["feature_to_inspect"],
                   "cells": [row for row in observation["cells"]
                             if row["field"] == explorer["feature_to_inspect"]]}
        trajectory.append({"role": "explorer", "turn": turn, "output": explorer,
                           "executed_public_filter": focused})
        hyp = ask(model, "discovery-researcher",
                  "Propose or revise one falsifiable, potentially nonstandard ranking mechanism. "
                  "Specify measurable predictions, an alternative mechanism, and falsification. "
                  "Previous negative findings must change your hypothesis if appropriate. "
                  "No ground truth is visible. The no-effect conclusion is allowed.",
                  {"context": context, "exploration": explorer, "focused_observation": focused}, HYPOTHESIS,
                  f"{task}-hyp-{turn}")
        last_hyp = hyp
        trajectory.append({"role": "researcher", "turn": turn, "output": hyp})
        design = ask(model, "discovery-designer",
                     "Choose a registered paired intervention, or reject/stop. "
                     "The test must distinguish current hypothesis from its competing explanation. "
                     "All user/candidate identities and random scores will be matched by the executor.",
                     {"context": context, "hypothesis": hyp, "options": {
                         "factors": list(FIELDS), "conditions": list(CONDITIONS),
                         "item_tags": [0, 1], "budget_left": rounds-turn}},
                     DESIGNER, f"{task}-design-{turn}")
        trajectory.append({"role": "designer", "turn": turn, "output": design})
        if design["decision"] != "test":
            revisions.append({"turn": turn, "decision": design["decision"]})
            break
        result = lab_query(secret, task, "contrast", seed=seed,
                           factor=design["factor"], condition=design["condition"],
                           tag=design["tag"])
        evidence = {"kind": "ExperimentRecord", "id": f"e{turn:02d}",
                    "phase": "development", "action": result["intervention"],
                    "result": result}
        experiments.append(evidence)
        numerical = ("supported" if is_effect(result, hyp["direction"])
                     and design["factor"] == hyp["factor"]
                     and design["condition"] == hyp["condition"]
                     and design["tag"] == hyp["tag"] else "inconclusive")
        review = ask(model, "discovery-critic-reflection",
                     "Critique the ACTUALLY EXECUTED paired evidence. Explain plausible confounders, "
                     "competing explanations and what should be tested next. Never declare independent "
                     "confirmation or override numeric CI. Revisit prior conclusions when contradicted. "
                     "Read manipulation_check: unchanged top-1 is an OUTCOME, not proof the registered "
                     "setting failed to change. Exact zero paired effects are possible under a true "
                     "deterministic null with matched random variates.",
                     {"hypothesis": hyp, "design": design, "executed_experiment": evidence,
                      "numeric_development_state": numerical,
                      "previous_experiments": experiments[:-1]}, REVIEW,
                     f"{task}-review-{turn}")
        trajectory.append({"role": "experiment", "turn": turn, "record": evidence})
        trajectory.append({"role": "critic_reflection", "turn": turn, "output": review})
        revisions.append({"turn": turn, "review": review, "numerical_state": numerical})
        if numerical == "supported":
            supported_candidates.append({"hypothesis": hyp,
                                         "experiment_id": evidence["id"],
                                         "turn": turn})
        # Cross-round memory records experimental facts and unresolved questions,
        # not the prior model's full essays or every raw observation row.
        context["previous_trials"].append({
            "hypothesis": hyp["hypothesis"][:240],
            "prediction": {k: hyp[k] for k in ("factor", "condition", "tag", "direction")},
            "test": {k: design[k] for k in ("factor", "condition", "tag")},
            "effect": result["effect"], "ci95": result["ci95"],
            "numeric_state": numerical,
            "critic_state": review["assessment"],
            "next_question": review["next_question"][:240]})
    if not last_hyp:
        raise ValueError("agent did not propose any hypothesis")
    # Preserve prospectively tested successful hypotheses instead of discarding
    # them whenever the final exploratory turn was inconclusive. Hypotheses
    # cannot claim support from an older test they did not actually predict.
    selected = supported_candidates[-1] if supported_candidates else None
    final_hyp = selected["hypothesis"] if selected else last_hyp
    state = "supported" if selected else "inconclusive"
    claim = record_claim(final_hyp, "active", experiments, state)
    claim["selected_supporting_experiment"] = selected["experiment_id"] if selected else None
    claim["revisions"] = revisions
    return claim, experiments, trajectory

def confirm(secret, task, claim, n=768):
    """Confirmation is conducted once AFTER research stops, without agent access."""
    if claim["status"] != "supported":
        return {"kind": "EvidenceAssessment", "status": "not_eligible",
                "reason": "no validated development claim"}
    result = lab_query(secret, task, "contrast", split="confirmation",
                       factor=claim["factor"], condition=claim["scope"],
                       tag=claim["tag"], n=n)
    passes = is_effect(result, claim["direction"])
    # Subgroup scope requires negative control in the complementary segment.
    opposite = None
    subgroup_checks = {}
    if claim["scope"] != "all":
        other = "segment_0" if claim["scope"] == "segment_1" else "segment_1"
        opposite = lab_query(secret, task, "contrast", split="confirmation",
                             factor=claim["factor"], condition=other,
                             tag=claim["tag"], n=n)
        passes = passes and opposite["ci95"][0] <= 0 <= opposite["ci95"][1]
    else:
        # An all-users claim cannot be inferred from a significant pooled
        # average alone: an unrecognized subgroup interaction is a competing
        # explanation and means the stated scope was not recovered.
        subgroup_checks = {
            subgroup: lab_query(secret, task, "contrast", split="confirmation",
                                factor=claim["factor"], condition=subgroup,
                                tag=claim["tag"], n=n)
            for subgroup in ("segment_0", "segment_1")
        }
        passes = passes and all(is_effect(value, claim["direction"])
                               for value in subgroup_checks.values())
    return {"kind": "EvidenceAssessment", "status": "confirmed" if passes else "inconclusive",
            "independent_seed_and_users": True,
            "test": result, "scope_negative_control": opposite,
            "scope_subgroup_checks": subgroup_checks,
            "inference_limit": "single post-selection confirmation; no global FDR control"}

def run(secret, output, config=None, rounds=3, arms=("fixed", "one_shot", "concept_test", "active")):
    from rsi_researcher.research.journal import ExperimentJournal
    # Public case IDs are not derived from hidden mechanism names.
    # The agent runner never loads the private truth file into its process.
    manifest = Path(output).parent / ("manifest-" + Path(output).name + ".json")
    if not manifest.exists():
        raise ValueError("missing pre-generated public manifest adjacent to output")
    public = json.loads(manifest.read_text(encoding="utf8"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Public manifest contains no seed, secret path, mechanism names, or parameters.
    atomic_json(output/"protocol.json", {
        "version": "discovery-cycle-v2", "tasks": list(public["tasks"]),
        "arms": arms, "max_rounds": rounds, "development_n": 384,
        "confirmation_n": 768, "claim_authority": "independent_numeric_evaluator",
        "model_provider": (config or {}).get("llm", {}).get("provider", "none"),
        "scope": "simulator rediscovery only",
        "source_sha256": {
            filename: hashlib.sha256((Path(__file__).with_name(filename)).read_bytes()).hexdigest()
            for filename in ("discovery_lab.py", "discovery_cycle.py")
        }})
    journal = ExperimentJournal(output/"journal.json")
    model = None
    if any(a != "fixed" for a in arms):
        if config is None:
            raise ValueError("model-backed arms need explicit model configuration")
        from rsi_researcher.runtime.limits import RunLimits
        from rsi_researcher.runtime.models import create_model
        model_dir = output/"model_calls"
        model_dir.mkdir()
        limits = RunLimits(model_dir, max_requests=100, max_tokens=1200000,
                           max_seconds=3600)
        model = create_model(model_dir, limits.max_requests, limits, config)
        if config["llm"]["provider"] == "vllm":
            model.health()
    claims_by_arm = {a: {} for a in arms}
    model_calls_by_arm = {a: 0 for a in arms}
    for task in public["tasks"]:
        # No secret or mechanism config in the agent context or research journal.
        observation = lab_query(secret, task, "observe")
        for arm in arms:
            before_calls = model.requests if model else 0
            arm_dir = output / arm / task
            arm_dir.mkdir(parents=True)
            if arm == "fixed":
                claim, experiments, trajectory = fixed_scan(secret, task, 0)
            elif arm == "one_shot":
                claim, experiments, trajectory = one_shot(model, observation, task)
            elif arm == "concept_test":
                claim, experiments, trajectory = concept_test(model, secret, task, observation, 0)
            elif arm == "active":
                claim, experiments, trajectory = active(model, secret, task, observation, 0, rounds)
            else:
                raise ValueError("unknown arm")
            independent = confirm(secret, task, claim)
            if independent["status"] == "confirmed":
                claim["status"] = "confirmed"
            atomic_json(arm_dir/"observation.json", observation)
            atomic_json(arm_dir/"claim.json", claim)
            atomic_json(arm_dir/"experiments.json", experiments)
            atomic_json(arm_dir/"trajectory.json", trajectory)
            atomic_json(arm_dir/"independent_assessment.json", independent)
            journal.record_trial(arm_dir, {"arm": arm, "task": task,
                                           "proposal": claim, "experiments": experiments,
                                           "execution": {"experiment_id": arm+"-"+task},
                                           "evidence_assessment": independent})
            claims_by_arm[arm][task] = claim
            model_calls_by_arm[arm] += (model.requests - before_calls) if model else 0
    summary = {"groups": {}, "note": "Full truth labels are only revealed by the private evaluator."}
    for arm, claims in claims_by_arm.items():
        submitted = {task: {**claim, "condition": claim["scope"],
                            "status": "supported" if claim["status"] == "confirmed" else "inconclusive"}
                     for task, claim in claims.items()}
        submissions = output / (arm + "-submissions.json")
        atomic_json(submissions, submitted)
        report = private_grade(secret, submissions)
        report["experimental_tests"] = sum(len(json.loads((output/arm/task/"experiments.json").read_text()))
                                           for task in claims)
        report["model_calls"] = model_calls_by_arm[arm]
        report["valid_discoveries_per_experiment"] = (
            report["valid_discoveries"] / report["experimental_tests"]
            if report["experimental_tests"] else None)
        summary["groups"][arm] = report
    atomic_json(output/"summary.json", summary)
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--private")
    p.add_argument("--seed", type=int)
    p.add_argument("--output", required=True)
    p.add_argument("--config")
    p.add_argument("--arms", default="fixed,one_shot,concept_test,active")
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()
    if (args.private is None) == (args.seed is None):
        p.error("specify exactly one of --private (existing suite) or --seed (generate)")
    secret_path = args.private
    if args.seed is not None:
        # Distinct owner-only private suite; only the manifest enters runs/.
        # Historical top-level private suites remain untouched. New suites
        # have one owner-only location instead of cluttering /data/cjh/rsi/.
        private_root = Path("/data/cjh/rsi/discovery_private_20261024/suites")
        private_root.mkdir(mode=0o700, exist_ok=True)
        target = private_root / Path(args.output).name
        target.mkdir(mode=0o700, exist_ok=False)
        secret_path = str(target/"truth.json")
        command = [sys.executable, "-B", "-m", "rsi_researcher.research.discovery_lab",
                   "--mode", "make", "--seed", str(args.seed), "--private", secret_path]
        manifest = subprocess.run(command, capture_output=True, text=True, check=True)
        manifest_path = Path(args.output).parent / ("manifest-" + Path(args.output).name + ".json")
        manifest_path.write_text(manifest.stdout, encoding="utf8")
    cfg = json.loads(Path(args.config).read_text()) if args.config else None
    result = run(secret_path, args.output, config=cfg, rounds=args.rounds,
                 arms=tuple(args.arms.split(",")))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
