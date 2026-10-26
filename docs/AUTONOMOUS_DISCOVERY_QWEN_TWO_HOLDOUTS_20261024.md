# Autonomous Bias Discovery - Qwen3-8B hidden evaluation

Qwen3-8B via local vLLM. Open concept generation, paired ranking intervention, numerical critique, independent test.
Two new evaluator-private mechanism seeds after protocol/source freeze; four arms, four tasks per suite.

## Pooled summary, 6 planted mechanisms + 2 no-mechanism controls

| Arm | Confirmed / 6 | False discoveries | Development interventions | LLM requests | LLM tokens |
|---|---:|---:|---:|---:|---:|
| fixed | 6/6 | 0 | 192 | 0 | 0 |
| one_shot | 0/6 | 0 | 0 | 8 | 13,538 |
| concept_test | 4/6 | 0 | 36 | 22 | 36,446 |
| active | 0/6 | 0 | 16 | 64 | 103,010 |

## discovery-qwen-v2-holdout-a-20261024

| Case | Private mechanism, unsealed only after experiment | Fixed | One-shot raw | Concept+Test | Active |
|---|---|---|---|---|---|
| task-00 | null | correct null | raw miss | correct null | correct null |
| task-01 | single | confirmed | raw miss | miss | miss |
| task-02 | dynamic | confirmed | raw miss | confirmed | miss |
| task-03 | interaction | confirmed | raw miss | confirmed | miss |

| Arm | Requests | Prompt Tokens | Completion Tokens | Experiments |
|---|---:|---:|---:|---:|
| fixed | 0 | 0 | 0 | 96 |
| one_shot | 4 | 5803 | 960 | 0 |
| concept_test | 11 | 16006 | 2250 | 18 |
| active | 32 | 47024 | 4540 | 8 |

## discovery-qwen-v2-holdout-b-20261024

| Case | Private mechanism, unsealed only after experiment | Fixed | One-shot raw | Concept+Test | Active |
|---|---|---|---|---|---|
| task-00 | single | confirmed | raw miss | miss | miss |
| task-01 | interaction | confirmed | raw miss | confirmed | miss |
| task-02 | dynamic | confirmed | raw miss | confirmed | miss |
| task-03 | null | correct null | raw miss | correct null | correct null |

| Arm | Requests | Prompt Tokens | Completion Tokens | Experiments |
|---|---:|---:|---:|---:|
| fixed | 0 | 0 | 0 | 96 |
| one_shot | 4 | 5807 | 968 | 0 |
| concept_test | 11 | 16002 | 2188 | 18 |
| active | 32 | 46957 | 4489 | 8 |

## Interpretation

- The agent is not given the hidden task mechanism, direction, segment or injected coefficient.
- Agent hypotheses are unrestricted prose with a registered public set of measurable factors; no Bias-type answer list.
- A positive claim requires controlled paired user-level evidence and an independently seeded 768-user confirmation.
- Scope is audited separately: a subgroup claim requires a null contrast on the opposite segment, an all-user claim requires effects in both segments.
- Null environments and negative experiments remain first-class valid outcomes.
- Counts measure hidden planted mechanism rediscovery; this is NOT proof of a previously unknown real-world bias.
- User-level normal 95% intervals are exploratory; sequential familywise or FDR control has not been demonstrated.
- Two different seeds measure preliminary repeatability, not population-wide robustness.
- Exact source/config SHA256 recorded in runs/discovery-qwen-v2-frozen-sha256-20261024.txt and verified for both runs.
- Per-run evidence: protocol.json, summary.json, journal.json, each arm/task claim.json, trajectory.json, experiments.json, independent_assessment.json, and model_calls/calls.jsonl.

## Decision

Preserve the smallest productive discovery cycle; refine epistemic fidelity, not the scientific tree. Future real-data AgentCF/LLMRank study needs an explicit reference standard and counterfactual interventions; new bias existence cannot be inferred from a plausible name alone.


## Failure analysis for the unspecified single-factor mechanism

The independent evaluator found that the single-factor mechanism was
related to the public slot setting in BOTH sealed task suites.
Concept Discovery failed to test slot on either suite:

- Hidden suite A tested device, history, context (both user segments), all null.
- Hidden suite B tested history, device, context (both user segments), all null.

Thus the consistent 0/2 misses came from limited concept coverage /
experiment prioritization, not from the paired estimator failing after
a true intervention. Simply increasing candidate coverage to all four factors
would make this an exhaustive registered-feature scan, a scientifically
weaker claim to autonomous discovery.

Next highest-value change: provide descriptive observational contrast
statistics for the public variables as Explorer evidence, without supplying
mechanism labels or a hand-selected Bias registry. Then let the Agent
justify its own novel, testable mechanism and choose differentiating
interventions. Freeze changes BEFORE completely new blind tasks and compare
against the same exhaustive scan. Do not tune using the two already
unsealed truths.
