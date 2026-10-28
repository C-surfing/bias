"""Data-agnostic reference and estimand contracts.

Scientific declarations are hypotheses. Only executed evidence may change status.
This module validates provenance and alignment; it does not select Bias classes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class ReferenceSpec:
    kind: str
    target_quantity: str
    target_reference: str
    affected_population: str
    identification_assumptions: str
    source_scope: str
    status: str = "proposed_unidentified"

    @classmethod
    def from_choice(cls, choice: dict, capabilities: dict) -> "ReferenceSpec":
        kind = choice.get("reference_kind", "none")
        capability = {
            "randomized_exposure": "randomized_exposure_probe",
            "dense_relevance": "dense_relevance",
            "controlled_model": "replayable_recommender",
        }.get(kind)
        if kind not in ("none", "randomized_exposure", "dense_relevance", "controlled_model"):
            raise ValueError("Unsupported reference type")
        if capability and not capabilities.get(capability, False):
            raise ValueError(f"Dataset does not support {kind} reference; "
                             "measure another reference or document the gap")
        return cls(
            kind=kind,
            target_quantity=choice.get("proposed_distortion", ""),
            target_reference=choice.get("target_reference", ""),
            affected_population=choice.get("contrast_filter", ""),
            identification_assumptions=choice.get("reference_identification", ""),
            source_scope={
                "randomized_exposure": "restricted_random_insertion_not_normative",
                "dense_relevance": "observed_relevance_not_platform_allocation",
                "controlled_model": "constructed_recommender_only",
                "none": "unidentified",
            }[kind])

    def as_record(self):
        return asdict(self)


@dataclass(frozen=True)
class EstimandSpec:
    source_table: str
    unit_expression: str
    population_filter: str
    group_expression: str
    control_value: str
    treatment_value: str
    outcome_expression: str
    analysis: str

    @classmethod
    def from_choice(cls, choice: dict) -> "EstimandSpec":
        return cls(
            source_table=choice["contrast_source"],
            unit_expression="uid",
            population_filter=choice.get("contrast_filter", ""),
            group_expression=choice["contrast_group"],
            control_value=choice["control_value"],
            treatment_value=choice["treatment_value"],
            outcome_expression=choice["contrast_outcome"],
            analysis=choice["analysis"])

    def compare_executed(self, executed: dict) -> dict:
        """Distinguish exact target alignment from a conditional sensitivity test."""
        declared = asdict(self)
        differences = {k: [v, executed.get(k)] for k, v in declared.items()
                       if str(v).strip() != str(executed.get(k, "")).strip()}
        return {
            "same_estimand": not differences,
            "same_core_contrast": all(k == "population_filter" for k in differences),
            "population_shifted": "population_filter" in differences,
            "differences": differences,
        }

    def as_record(self):
        return asdict(self)


def observed_direction(statistics: dict, meaningful_margin: float) -> str:
    """Numerical evidence only; an insignificant effect is NOT 'no_change'."""
    ci = statistics.get("ci95_bootstrap") or ()
    if len(ci) != 2 or meaningful_margin < 0:
        return "inconclusive"
    low, high = map(float, ci)
    if not all(isfinite(v) for v in (low, high)) or low > high:
        return "inconclusive"
    if low > meaningful_margin:
        return "increase"
    if high < -meaningful_margin:
        return "decrease"
    if meaningful_margin > 0 and -meaningful_margin < low <= high < meaningful_margin:
        return "no_change"
    return "inconclusive"


def rival_prediction_audit(statistics: dict, h1: str, h2: str,
                           margin: float) -> dict:
    """Evaluate directions only for a valid executed same-target experiment."""
    direction = observed_direction(statistics, margin)
    return {
        "observed_direction": direction,
        "prediction_h1": h1,
        "prediction_h2": h2,
        "distinguishes_predictions": bool(
            direction != "inconclusive" and h1 != h2
            and direction in (h1, h2)),
        "supports": ("H1" if direction == h1 and h1 != h2 else
                     "H2" if direction == h2 and h1 != h2 else "inconclusive"),
        "note": "Observational prediction agreement does not identify a causal Bias.",
    }
