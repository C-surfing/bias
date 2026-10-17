"""Author-path researcher closed loop.

Python does all arithmetic, gate decisions and case extraction; Qwen only
reads pre-computed evidence, registers a falsifiable prediction, and revises a
conditional hypothesis after reading the real (externally scored) results.
The model never scores itself.
"""
import json
from pathlib import Path

from rsi_researcher.runtime.client import TEXT_SCHEMA, object_schema, InvalidOutput
from rsi_researcher.evaluation.author_probes import run_author_probes
from rsi_researcher.research.author_experiment import AUTHOR_OPERATORS


EFFECT_LEVELS = ["decrease", "increase", "no_change", "maintain_or_improve", "unknown"]

PREDICTION_SCHEMA = object_schema({
    "primary_hypothesis": TEXT_SCHEMA,
    "competing_hypothesis": TEXT_SCHEMA,
    "selected_operator": {"type": "string", "enum": sorted(AUTHOR_OPERATORS)},
    "supplementary_check": TEXT_SCHEMA,
    "expected_effects": object_schema({
        "false_dislike_rate": {"type": "string", "enum": EFFECT_LEVELS},
        "quality_ndcg": {"type": "string", "enum": EFFECT_LEVELS},
        "popularity_bias": {"type": "string", "enum": EFFECT_LEVELS},
    }),
    "falsification_condition": TEXT_SCHEMA,
})

REVISION_SCHEMA = object_schema({
    "applicability_conditions": TEXT_SCHEMA,
    "confirmed_mechanisms": TEXT_SCHEMA,
    "falsified_expectations": TEXT_SCHEMA,
    "unresolved_competing_explanations": TEXT_SCHEMA,
    "next_priority_action": {"type": "string", "enum": sorted(AUTHOR_OPERATORS)},
    "honest_assessment": TEXT_SCHEMA,
})


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_researcher_evidence(run_dir):
    """Compact, bounded evidence for the researcher from a committed run."""
    run_dir = Path(run_dir)
    probes = run_author_probes(run_dir)
    summary = _read(run_dir / "summary.json")
    arms = summary["arms"]
    def metrics(arm):
        top5 = arms[arm]["mean_top5_head_share"]
        cand = arms[arm]["mean_candidate_head_share"]
        return {"ndcg_at_5": arms[arm]["mean_ndcg_at_5"],
                "hit_at_5": arms[arm]["mean_hit_at_5"],
                "top5_head_share": top5,
                "candidate_head_share": cand,
                "excess_popularity_exposure": top5 - cand}
    return {
        "dataset": probes["dataset"], "committed_events": probes["committed_events"],
        "ranking_queries": probes["ranking_queries"],
        "frozen": metrics("frozen"), "original_update": metrics("original_update"),
        "exposure_bias": {"false_dislike_memory_rate": probes["exposure_bias"]["false_dislike_memory_rate"],
                          "false_dislike_events": probes["exposure_bias"]["false_dislike_events"],
                          "representative_cases": probes["exposure_bias"]["representative_cases"][:2]},
        "conformity_bias": {"select_reason_popularity_rate": probes["conformity_bias"]["select_reason_popularity_rate"],
                            "user_memory_popularity_rate": probes["conformity_bias"]["user_memory_popularity_rate"]},
        "selection_paac": {group: {"mean_update_count": v["mean_update_count"],
                                   "mean_memory_chars": v["mean_memory_chars"],
                                   "mean_growth_ratio": v["mean_growth_ratio"]}
                           for group, v in probes["selection_paac"]["groups"].items()},
        "position_bias": {"input_order_following_tau": probes["position_bias"]["input_order_following_kendall_tau"]},
    }


def register_prediction(model, evidence):
    prompt = ("You are the recommendation-system researcher. Read the pre-computed development evidence "
              "below and register ONE falsifiable prediction before running any intervention. "
              "Choose exactly one allowed operator; state a primary and a competing hypothesis; predict the "
              "direction of each effect; and name the condition that would falsify your claim. "
              "Do not claim causality from these small-sample observations. Do not invent numbers.\n\n"
              + json.dumps(evidence, ensure_ascii=False))
    return model.ask(prompt, PREDICTION_SCHEMA, {"stage": "researcher_predict"})


def validate_prediction(prediction):
    """Contract checks only — never a judgment about the prediction's truth."""
    if prediction.get("selected_operator") not in AUTHOR_OPERATORS:
        raise InvalidOutput("selected_operator is not permitted")
    effects = prediction.get("expected_effects", {})
    if prediction.get("selected_operator") == "retain":
        # A no-op cannot be expected to change a metric.
        for key, level in effects.items():
            if level in ("decrease", "increase"):
                raise InvalidOutput("retain cannot predict a change in " + key)


def compare_arms(baseline_dir, candidate_dir, quality_floor=0.99, bias_ceiling=0.9,
                 minimum_reference_bias=0.001):
    """Python-precomputed paired comparison and gate decisions."""
    baseline_dir, candidate_dir = Path(baseline_dir), Path(candidate_dir)
    b = build_researcher_evidence(baseline_dir)
    c = build_researcher_evidence(candidate_dir)

    def ratio(new, old, floor):
        return new / old if old else (1.0 if new == 0 else 0.0)

    b_qual = b["original_update"]["ndcg_at_5"]
    c_qual = c["original_update"]["ndcg_at_5"]
    b_excess = b["original_update"]["excess_popularity_exposure"]
    c_excess = c["original_update"]["excess_popularity_exposure"]
    b_dislike = b["exposure_bias"]["false_dislike_memory_rate"]
    c_dislike = c["exposure_bias"]["false_dislike_memory_rate"]

    quality_gate = ratio(c_qual, b_qual, quality_floor) >= quality_floor
    bias_gate = (c_excess <= bias_ceiling * b_excess) if abs(b_excess) >= minimum_reference_bias else c_excess <= b_excess
    dislike_gate = c_dislike < b_dislike

    return {
        "baseline_dir": str(baseline_dir), "candidate_dir": str(candidate_dir),
        "metrics": {
            "false_dislike_memory_rate": {"baseline": b_dislike, "candidate": c_dislike,
                "absolute_change": c_dislike - b_dislike,
                "relative_change": (c_dislike - b_dislike) / b_dislike if b_dislike else None,
                "gate": "PASS" if dislike_gate else "FAIL"},
            "ndcg_at_5": {"baseline": b_qual, "candidate": c_qual,
                "absolute_change": c_qual - b_qual,
                "relative_change": (c_qual - b_qual) / b_qual if b_qual else None,
                "gate": "PASS" if quality_gate else "FAIL", "floor_ratio": quality_floor},
            "excess_popularity_exposure": {"baseline": b_excess, "candidate": c_excess,
                "absolute_change": c_excess - b_excess,
                "relative_change": (c_excess - b_excess) / b_excess if abs(b_excess) >= minimum_reference_bias else None,
                "gate": "PASS" if bias_gate else "FAIL", "ceiling_ratio": bias_ceiling},
            "conformity_select_reason": {"baseline": b["conformity_bias"]["select_reason_popularity_rate"],
                "candidate": c["conformity_bias"]["select_reason_popularity_rate"]},
            "position_following_tau": {"baseline": b["position_bias"]["input_order_following_tau"],
                "candidate": c["position_bias"]["input_order_following_tau"]},
        },
        "representative_cases": {"baseline": b["exposure_bias"]["representative_cases"],
                                 "candidate": c["exposure_bias"]["representative_cases"]},
        "gates_pass": {"false_dislike": dislike_gate, "quality": quality_gate, "bias": bias_gate},
        "interpretation": "Gates are Python-computed thresholds, not model judgments; a failed gate is a "
                          "normal scientific falsification, never a format error.",
    }


def revise_hypothesis(model, prediction, comparison):
    payload = {"registered_prediction": prediction, "actual_results": {
        k: v for k, v in comparison.items() if k in ("metrics", "gates_pass")}}
    prompt = ("Compare your registered prediction against the Python-computed actual results below. "
              "Be honest: state which expectations were confirmed and which were falsified by the data. "
              "Produce a conditional hypothesis card. Never spin a failed gate as success.\n\n"
              + json.dumps(payload, ensure_ascii=False))
    return model.ask(prompt, REVISION_SCHEMA, {"stage": "researcher_revise"})


def run_researcher_loop(model, baseline_dir, candidate_dir, output_dir):
    """Predict -> validate -> compare -> revise, all driven by pre-computed numbers."""
    from rsi_researcher.runtime.client import atomic_json
    output_dir = Path(output_dir)
    evidence = build_researcher_evidence(baseline_dir)
    prediction = register_prediction(model, evidence)
    validate_prediction(prediction)
    comparison = compare_arms(baseline_dir, candidate_dir)
    revision = revise_hypothesis(model, prediction, comparison)
    result = {"evidence": evidence, "prediction": prediction, "comparison": comparison,
              "revision": revision, "scope": "researcher audit; gates computed by Python, never by the model"}
    atomic_json(output_dir / "researcher.json", result)
    return result


def run_cli(args):
    from rsi_researcher.runtime.models import create_model
    from rsi_researcher.runtime.limits import RunLimits
    from rsi_researcher.runtime.client import atomic_json, usage_summary

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    limits = RunLimits(directory, args.max_requests, args.max_tokens, args.max_seconds)
    model = create_model(directory, limits.max_requests, limits, config)
    result = run_researcher_loop(model, args.baseline, args.candidate, directory)
    if (directory / "calls.jsonl").exists():
        atomic_json(directory / "usage.json", usage_summary(directory))
    print(json.dumps({"prediction": result["prediction"], "comparison": result["comparison"],
                      "revision": result["revision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Author-data researcher closed loop.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="register prediction, compare, revise (needs vLLM)")
    run.add_argument("--baseline", required=True, help="baseline run directory")
    run.add_argument("--candidate", required=True, help="intervention run directory")
    run.add_argument("--output", required=True)
    run.add_argument("--config", required=True, help="config with the vllm llm block")
    run.add_argument("--max-requests", type=int, default=10)
    run.add_argument("--max-tokens", type=int, default=60000)
    run.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.command == "run":
        run_cli(args)
