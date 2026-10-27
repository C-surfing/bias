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


## Evidence-driven Qwen pilot after removing fixed budgets

Run: runs/kuairand27k-evidence-driven-qwen-20261027 on the same
isolated 2,608-user development database and local Qwen3-8B.
The CLI omitted --max-rounds: protocol shows max_rounds=null and
research_stop_policy=evidence_or_stall. No model_calls/limits.json exists.

Outcome: 17 Scientist actions (rounds 0-16), 20 Journal entries,
27 initiated model requests, four distinct development SQL results,
two hypothesis candidates, one valid Workbench user-level estimate,
zero independently executed rival-discriminating countertests and
zero credible normative reference deviations. It stopped with
scientific_stalled because six consecutive actions yielded no
new observed evidence, test or substantive revision. It was NOT
terminated by a request, time, token or total-round quota.

Initial exploratory contrast: the user-level difference for previous_click
groups on current click was +0.160357; bootstrap 95% CI
[+0.157020, +0.163415] across 2,608 paired users. This is not
recommendation-induced distortion or a new Bias. Numerous repeated
SQL/feature errors remained after the initial measurement.

This validates an unlimited-round **execution and stopping mechanism**,
not better scientific Bias discovery. The prior bounded experiments
remain an exploratory fixed-cap comparator; there is no claim of
significant discovery-rate improvement or of independent confirmation.


## 2026-10-27 post-BADF experiment-driven science audit

Same frozen development-only KuaiRand-27K DB: runs/kuairand27k-open-agent-enriched-development-20261027.duckdb (2,608 dev users). Local Qwen3-8B, GPU7. Independent holdout was NOT consulted.

| Run directory under runs/ | Setting | Verdict |
| --- | --- | --- |
| kuairand27k-open-phenomena-exp-seed20261027 | Original open phenomena, fixed rounds, seed 20261027 | budget_exhausted, numerical Workbench countertest but invalid mechanism/estimand alignment |
| kuairand27k-open-phenomena-thinking-seed20261027 | Thinking on, max_tokens 3800 | model_failed with HTTP 400 |
| kuairand27k-open-phenomena-relational-seed20261028 | Relational-evidence-only concept gate | scientific_stalled, marginals did not become false scientific candidates |
| kuairand27k-open-phenomena-episodes-seed20261028 | Same seed, deterministic episode sampling on | scientific_stalled, five experiment actions, but reversed conditional interpretation |
| kuairand27k-open-phenomena-thinking2200-seed20261029 | Lower-token Thinking, directional evidence audit | scientific_stalled on repeated SQL and illegal aggregate-as-feature |
| kuairand27k-open-phenomena-direction-replay-seed20261028 | Same seed and episode sampling with directional evidence audit | scientific_stalled; misleading phenomenon excluded; repeated actions persisted |

Initial development-only user-level previous_long_view-to-click estimate was +0.1517559, bootstrap CI95 [0.1482565, 0.1557442] across 2,607 paired users; a critic-subgroup previous_hate=1 estimate was +0.0774358, CI95 [0.0223653, 0.1343656] on 242 paired users. These DID NOT measure the claimed activity-dependent exposure-allocation mechanism, and lack an identified reference, so neither is a new Bias discovery.

Harness changes: relational conditional evidence prerequisite before concept formation; measured-direction label (mean outcome BY group) and provenance guard; optional reproducible per-episode seeds only in research roles. Same-seed comparison yielded more exploratory experiments but ZERO additional validated bias mechanisms. Targeted tests: 27 pass.

Scientific status: 0 discovered and independently confirmed novel recommender biases; 0 verified interventions. Actual bottleneck is mechanism-to-estimand alignment and repetitive no-information SQL/derived feature actions. Next high-value experiment: Qwen identifies an item-level allocation against defensible response/relevance reference, makes distinct same-estimand H1/H2 predictions, and uses independently executed falsification. Randomized insertions are a limited reference and cannot identify whole-policy counterfactuals. Preserve frozen confirmation boundary. Source journals and model calls are retained in the six run directories.
