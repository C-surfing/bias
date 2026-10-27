# Open Discovery: domain-science audit and Qwen3-8B end-to-end results

Date: 2026-10-27. Main project: /data/cjh/rsi/project.

## Scientific verdict

The earlier domain research contract (9e59139) was directionally correct but incomplete in the real execution route. The final changes enable a full Qwen-selected contrast -> typed Workbench -> Critic countertest -> distinct numeric Workbench -> Reflection path. They **do not establish a new recommender Bias**. Preference-associated click effects are not a credible recommender-induced exposure/utility deviation.

Data: isolated KuaiRand-27K development-only DuckDB, 21,824,051 events, 21,682,292 feature rows, 2,608 development users, event days 2022-04-08 through 2022-04-30. All arms use the same development database and local Qwen3-8B configuration: temperature 0.45, seed 20261024, max_tokens 1700, at most 12 research rounds. No independent holdout was accessed.

## Actual equal-configuration pilot runs

| Arm | Outcome | Qwen requests initiated | Mechanism candidates | Candidates with Workbench | Distinct numeric countertests | Valid H1/H2 discriminations | Credible reference deviations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: pre-scientific-contract commit 200512b | model_failed | 21 | 4 | 0 | 0 | 0 | 0 |
| B: scientific contract before route fixes | development_inconclusive | 18 | 2 | 0 | 0 | 0 | 0 |
| C: first compiler fix | model_failed | 25 | 4 | 1 | 0 | 0 | 0 |
| E: compact context | development_inconclusive | 18 | 2 | 1 | 0 | 0 | 0 |
| F: countertest contract fix | development_inconclusive | 22 | 2 | 1 | 1 | 0 | 0 |
| G: final code with action pruning | development_inconclusive | 22 | 2 | 1 | 1 | 0 | 0 |

Evidence paths under runs/: kuairand27k-pre-contract-A-20261027 and kuairand27k-bias-contract-[B,C,E,F,G]-20261027. Each directory contains journal.json, research_state.json, research_opportunities.json and model_calls/calls.jsonl. C and A terminated early, and numbers of actual requests differ; this is a fixed-ceiling development pilot, **not** statistical proof of an improvement in scientific discovery rates.

## Distinct user-level countertest in G

H001 compared prior-click group 0 versus 1, outcome current click, on shared exposure-mode dates. Initial paired-user difference: +0.149939, 95% bootstrap CI [+0.146290, +0.153696], 2,600 paired users. Critic proposed a different conditional experiment restricting previous_hate = 1. Its paired difference was +0.023549, 95% CI [-0.046669, +0.092228], 147 paired users. The second interval crosses zero, and the subgroup is not randomized, so this is not a valid causal effect-difference estimate. The Critic recorded rival_discriminated=false and unmeasured reference; Reflection attempted a cosmetic revision that was vetoed. Real machine-readable remaining gaps: numeric_reference_deviation_unmeasured, competing_mechanism_not_discriminated, recommendation_consequence_not_identified.

## Code audit and implemented fixes

1. The default compact Designer path replaced the Bias science instructions. The real compact path now retains reference, mechanism and H1/H2 constraints.
2. Qwen selected existing dataset columns, but previous typed compilation required registered features, leading to invalid aggregated SQL. A selected measured native group can now compile to a legal per-event user-level experiment; the native-data regression test exercises Workbench.
3. Reject bare exposure-mode CTR as sufficient evidence of recommendation-induced bias. Opportunity selection now exposes measured, supported native contexts without enumerating known Bias types.
4. Critic countertest previously discarded its supported selection contract, generating empty JSON Schema enums and HTTP 400. It now uses the same verified source/group/outcome contract and can select an informative subgroup.
5. Overlong model input (over 16,384-token vLLM context) duplicated opportunity history and observation cards. Model-facing information is compacted while complete evidence stays in the saved journal.
6. Dynamic action options exclude already blocked concepts, ungrounded experiments, and repeated no-information SQL loops. Critic-to-Reflection evidence gaps are refreshed after executed tests.
7. Reflection had an incorrectly unindented recheck invocation. It now rechecks only a cosmetic failed revision.

## What remains scientifically unresolved

The Critic's structured API still lacks separate machine-verifiable reference-validity, actual consequence and novelty judgments. Scientific-gap audits and the freeze gate partly mitigate this but cannot manufacture identification. KuaiRand random insertions are a restricted exposure-mode diagnostic, not a randomized user-history experiment or normative recommendation target. No independent new Bias claim can be frozen from these episodes. The next substantive research improvement must be an agent-created **item-side allocation vs defensible response/relevance reference**, with a distinct same-estimand H1/H2 prediction and a real independent confirmation, rather than further generic prompt expansion.

Relevant methods: DiscoveryBench, ICLR 2025 (https://github.com/allenai/discoverybench); POPPER, ICML 2025 (https://github.com/snap-stanford/POPPER); KuaiRand, CIKM 2022 (https://doi.org/10.1145/3511808.3557624). We use their scientific principles, not POPPER's formal sequential Type-I guarantee.
