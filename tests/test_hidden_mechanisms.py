from rsi_researcher.research.hidden_mechanisms import (
    BlindDiscoveryBenchmark, MECHANISMS, grouped_signal,
)


def test_blind_protocol_deterministic_and_private():
    first = BlindDiscoveryBenchmark(42)
    second = BlindDiscoveryBenchmark(42)
    assert first.cases() == second.cases()
    assert not any(name in repr(first.cases()) for name in MECHANISMS)
    assert first.observe(first.cases()[0]) == second.observe(second.cases()[0])
    assert first._queries[first.cases()[0]] == 1
    try:
        first.observe(first.cases()[0], split="confirmation")
    except ValueError:
        pass
    else:
        assert False, "holdout leaked"


def test_injected_mechanisms_and_negative_control():
    benchmark = BlindDiscoveryBenchmark(11)
    results = {}
    for case_id, case in benchmark._cases.items():
        observational = case.sample(intervention="observational")
        randomized = case.sample(intervention="randomize_position")
        reset = case.sample(intervention="reset_history")
        if case.mechanism == "null":
            assert all(abs(row["score"] - 0.5) <= 0.041 for row in observational)
        elif case.mechanism == "position_group_interaction":
            assert grouped_signal(randomized) > grouped_signal(observational) + 0.06
        elif case.mechanism == "history_amplification":
            assert grouped_signal(observational) > grouped_signal(reset) + 0.06
        elif case.mechanism == "position_only":
            assert abs(grouped_signal(randomized)) < 0.02
        results[case_id] = case.mechanism
    grade = benchmark.assess(results)
    assert grade["recovery"] == 1
    assert grade["false_discoveries"] == 0


if __name__ == "__main__":
    test_blind_protocol_deterministic_and_private()
    test_injected_mechanisms_and_negative_control()
    print("2 tests passed")
