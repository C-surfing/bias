from __future__ import annotations

"""Scientific quality separation: raw model formatting is not debias gain."""
from math import ceil

from .dataset import summarize, cpb


def _raw_complete(row):
    """Every vote returned a full un-repaired permutation; no policy fault."""
    return (row.get("ranking") is not None
            and row.get("observed_votes", 0) > 0
            and row.get("repaired_votes", 0) == 0
            and row.get("policy_execution_failures", 0) == 0)


def paired_clean_evidence(baseline, changed, popularity, catalog,
                          k=5, min_fraction=0.60):
    """Measure paired *unrepaired* comparisons on the SAME users in both arms.

    Baseline/changed are independent raw rankings. The intersection must be
    computed separately per comparison; it never selects on correctness or
    which items are targets. Both random and position-probe orders must
    have clean completions for a user to be eligible.
    """
    names=("random","first")
    panels={}
    size=None
    for arm in names:
        left=baseline[arm]
        right=changed[arm]
        if len(left)!=len(right):
            raise ValueError("unpaired ranking panels")
        if size is None:
            size=len(left)
        if len(left)!=size or any(a["user"]!=b["user"] for a,b in zip(left,right)):
            raise ValueError("mismatched user or position arms")
        if len({r["user"] for r in left})!=len(left):
            raise ValueError("duplicate panel user")
        panels[arm]=(left,right)
    if [r["user"] for r in panels["first"][0]]!=[r["user"] for r in panels["random"][0]]:
        raise ValueError("position arm users are not aligned")
    indices=[i for i in range(size) if all(
        _raw_complete(panels[arm][0][i]) and _raw_complete(panels[arm][1][i])
        for arm in names
    )]
    minimum=max(5,ceil(size*min_fraction))
    support=len(indices)>=minimum
    details={
        "protocol":"matched_user_both_arms_unrepaired_v1",
        "total_users":size,"paired_clean_users":len(indices),
        "excluded_users":size-len(indices),
        "minimum_users":minimum,"sufficient_support":support,
        "format_inclusion_not_based_on_target_hit":True,
        "baseline_repaired_users":sum(any(
            not _raw_complete(panels[arm][0][i]) for arm in names)
            for i in range(size)),
        "candidate_repaired_users":sum(any(
            not _raw_complete(panels[arm][1][i]) for arm in names)
            for i in range(size))
    }
    if not indices:
        return {**details,"baseline":None,"candidate":None}
    def metrics(side):
        main=summarize([panels["random"][side][i] for i in indices],
                       popularity,catalog,k)
        probe=summarize([panels["first"][side][i] for i in indices],
                        popularity,catalog,k)
        for suffix in ("HR","NDCG"):
            key=f"{suffix}@{k}"
            main[f"CandidatePositionBias_{key}"]=cpb(probe[key],main[key])
        return {str(k):main}
    return {**details,"baseline":metrics(0),"candidate":metrics(1)}
