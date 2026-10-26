"""Blinded, deterministic mechanism-discovery benchmark.

The discovery agent receives only public observations and experiment responses.
Hidden labels are used exclusively by an independent evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from random import Random
from statistics import mean
import json


MECHANISMS = ("null", "position_group_interaction", "history_amplification", "position_only")


@dataclass(frozen=True)
class DiscoveryCase:
    case_id: str
    seed: int
    mechanism: str

    def __post_init__(self):
        if self.mechanism not in MECHANISMS:
            raise ValueError("unknown mechanism")

    def sample(self, *, n: int = 256, intervention: str = "observational", seed: int = 0):
        """Generate paired recommendation outcomes with an explicit intervention.

        'randomize_position' removes the association between group and position.
        'reset_history' blocks feedback accumulation. Noise remains paired across
        interventions for the same seed, preventing a false difference from RNG.
        """
        if intervention not in ("observational", "randomize_position", "reset_history"):
            raise ValueError("unknown intervention")
        if n < 16:
            raise ValueError("n must be >=16")
        rng = Random(self.seed * 100003 + seed)
        result = []
        for i in range(n):
            group = i % 2
            history = (i // 2) % 4
            position = (group if intervention != "randomize_position" else (i // 4) % 2)
            noise = rng.uniform(-0.04, 0.04)
            score = 0.50 + noise
            if self.mechanism == "position_only":
                score += 0.16 * (1 - position)
            elif self.mechanism == "position_group_interaction":
                score += 0.20 * (1 - position) * group
            elif self.mechanism == "history_amplification":
                score += 0.055 * history * group * (intervention != "reset_history")
            result.append({"group": group, "position": position, "history": history,
                           "score": round(score, 6)})
        return result


class BlindDiscoveryBenchmark:
    """The evaluator holds mechanism identities; the public view never does."""

    def __init__(self, seed: int = 20261023):
        rng = Random(seed)
        kinds = list(MECHANISMS)
        rng.shuffle(kinds)
        self._cases = {
            f"case-{i:02d}": DiscoveryCase(f"case-{i:02d}", seed + i * 17, kind)
            for i, kind in enumerate(kinds)
        }
        self._queries = {key: 0 for key in self._cases}

    def cases(self):
        return tuple(self._cases)

    def observe(self, case_id: str, *, split: str = "development", n: int = 256):
        if split != "development":
            raise ValueError("confirmation observations are evaluator-only")
        return self.experiment(case_id, "observational", n=n)

    def experiment(self, case_id: str, intervention: str, *, n: int = 256):
        case = self._cases[case_id]
        if intervention not in ("observational", "randomize_position", "reset_history"):
            raise ValueError("unregistered intervention")
        count = self._queries[case_id]
        self._queries[case_id] += 1
        return case.sample(n=n, intervention=intervention, seed=count)

    def assess(self, submissions: dict):
        """Evaluator-only: one explicit mechanism or abstention per case.

        Scores exact mechanism recovery and null rejection separately. It does
        not mistake an agent's p-value or verbal confidence for confirmation.
        """
        grades = {}
        for case_id, case in self._cases.items():
            response = submissions.get(case_id, "abstain")
            if response not in (*MECHANISMS, "abstain"):
                raise ValueError(f"invalid response for {case_id}")
            grades[case_id] = {
                "correct": response == case.mechanism,
                "false_discovery": response not in ("abstain", "null")
                                   and response != case.mechanism,
                "null_case": case.mechanism == "null",
                "queries": self._queries[case_id],
            }
        return {
            "recovery": mean(v["correct"] for v in grades.values()),
            "false_discoveries": sum(v["false_discovery"] for v in grades.values()),
            "queries": sum(v["queries"] for v in grades.values()),
            "cases": grades,
        }


def grouped_signal(rows):
    """Descriptive anomaly only, never a validated causal finding."""
    groups = {g: [row["score"] for row in rows if row["group"] == g] for g in (0, 1)}
    return round(mean(groups[1]) - mean(groups[0]), 6)


def demo(seed: int = 20261023):
    benchmark = BlindDiscoveryBenchmark(seed)
    return {"protocol": "hidden-mechanism-development-v1",
            "blind_case_ids": benchmark.cases(),
            "observation_summaries": {
                case: grouped_signal(benchmark.observe(case)) for case in benchmark.cases()
            },
            "note": "No mechanism labels or confirmation samples are exposed."}


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
