"""Controls for the executable, independently referenced KuaiRec recommender."""
import pytest

from rsi_researcher.research.discovery.real_intervention import (
    WINDOW, bucket, reference_panel, profile, rank, ndcg, ci_paired,
)


def test_split_and_state_replay_are_deterministic():
    assert bucket(42) == bucket(42)
    assert reference_panel(42, 11) == reference_panel(42, 11)
    categories = {0: (2,), 1: (3,), 2: (2,), 3: (3,)}
    priors = {i: .5 for i in categories}
    prefix = [(float(i), i % 4, i % 2) for i in range(90)]
    recent = [(float(90 + i), 0, 1) for i in range(12)]
    candidates = [(0, 1), (1, 0), (2, 1), (3, 0)]
    stale, _ = rank(candidates, prefix, categories, priors)
    restored, _ = rank(candidates, prefix + recent, categories, priors)
    sham, _ = rank(candidates, prefix[12:] + recent, categories, priors)
    assert restored == sham  # deletion outside the finite memory has no effect
    assert profile((prefix+recent), categories) != profile(prefix, categories)
    assert stale != restored  # a real setting intervention can change ranking
    assert ndcg(restored, 2) is not None
    assert ci_paired([.2] * 50, 42) == pytest.approx([.2, .2])


def test_no_reference_label_in_profile_and_zero_positive_is_undefined():
    assert ndcg([(0., 5, 0)], 0) is None
    assert profile([(0., 5, 1)], {5: (3,)}) == {3: 1 / 7}
