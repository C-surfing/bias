# Autonomous Discovery — Generic Experimental Workbench (2026-10-24)

## Decision and implementation

Keep `rsi_researcher.research.discovery explore` as the single active Discovery entry. No extra Director or duplicate science tree.

Implemented **P0 (executable experiment contract)** and **P1 (evidence-driven counterexperiment/reflection loop)**:

- `rsi_researcher/research/discovery/workbench.py`: `EXPERIMENT_SCHEMA` and `ExperimentalWorkbench`.
- `rsi_researcher/research/discovery/open_research.py`: new `experiment` scientific action. Research Designer writes the specification; Workbench runs it on physically isolated development DuckDB; independent Critic reviews actual numeric output. Critic's `test` result prompts Designer to generate a DIFFERENT plan, run by the SAME Workbench. Reflection can revise mechanism/falsifier/measurement, abandon, or request literature; journal keeps prior versions.
- The historic `sql`/`test`/`revise` actions remain available; the model is not forced into a particular Bias family.
- Previously implemented Literature Researcher remains available via the `literature` action. No claims of new literature novelty follow just from retrieval.

## Scientific contract

The Scientist/Designer, not hardcoded Bias tasks, chooses:
- Research question; full per-event (non-aggregated) `source_query` if needed; development table.
- Population condition; `unit_expression` (normally user ID); treatment/control groups; outcome; strata.
- `paired_user` or `cluster_weighted`; an optional separate placebo source and grouping; falsifier and context-specific meaningful margin.

Workbench returns:
- Group support, independent-unit/paired-unit counts, event-weighted group means;
- User-equal-weight paired effects or user-cluster-weighted effects;
- Fixed-seed 200-draw user-cluster Bootstrap 95% percentile interval;
- Within-stratum descriptive contrasts;
- Optional negative-control result and difference relative to the primary result;
- Actual reproducible compiled SQL, original user-defined specification, and explicit `identifies_causal_bias: false` status.

The model can derive new row-level variables via SQL without editing Python code. Workbench rejects a source query that pre-aggregates away independent user units, one-arm-only comparisons, inadequate paired users, fabricated columns and an identical treatment repeated as a placebo. Normalizes leading `WHERE` and SQL-literal-wrapped labels as non-scientific formatting.

**Important distinction**: Stratum effects are subgroup descriptions, NOT an automatically propensity-adjusted causal estimator. The Bootstrap is over independent users and presumes the user was the appropriate unit. No automatic multiple-comparison correction or sequential peeking adjustment. An agent-chosen margin is not an independently justified clinical/scientific threshold. Neither a low p-value nor a tight interval establishes an algorithm's harmful Bias.

## Verification

Run:

```bash
cd /data/cjh/rsi/project
PYTHONDONTWRITEBYTECODE=1 /data/cjh/rsi/discovery_private_20261024/env/bin/python \
 -m pytest -q tests/test_experimental_workbench.py tests/test_open_research.py \
 tests/test_discovery.py tests/test_kuairand_temporal_discovery.py
```

**30 tests passed**, including real DuckDB data, per-user bootstrap, placebo, derived source query, multiple group strata, schema-invalid experiments, contradictory frozen candidates, Designer→Critic counterexperiment→Reflection and no holdout access.

## Actual KuaiRand-27K development experiments

Database: `runs/kuairand27k-gpt-qwen-parity-v1-20261024/shared_explore.duckdb` (303 previously-used development users). **No confirmation data opened** for this round.

1. Unconstrained, model-directed Qwen `explore`:
`runs/kuairand27k-scientific-workbench-pilot-20261024/qwen-open-experiment/`.
Qwen selected the new `experiment` action, but its generated specifications had invalid population conditions (`WHERE` keyword and false date bounds), so no scientific result was computed; run marked `development_inconclusive`. The harmless `WHERE` syntax problem is now normalized and event-day range is provided in future Designer prompts.

2. Standalone, actual-Qwen designer stress tests:
`qwen-real-designer/` and `qwen-real-designer-v2/`.
First trial pre-aggregated `source_query` and removed `uid`, then repeated an invalid placebo. Second trial invented `clock_state` as though a prior SELECT alias were a persistent table column and supplied SQL-quoted labels. The last error was normalized, but the research-variable provenance problem remains. **Neither test produced a valid Qwen-designed Workbench result.** Full Qwen output and errors retained.

3. Genuine Workbench calculation with a *GPT-6 interactive-authored*, not Qwen-authored, Experiment Specification:
`runs/kuairand27k-scientific-workbench-pilot-20261024/gpt-authored-workbench-check.json`.

Research question: whether equal-millisecond records behave differently from records with an advanced timestamp. User-paired conditional exposure-group outcome is click, with `previous_click` stratification. Independent placebo groups are assigned by a deterministic video hash in a derived SELECT.

Results on **development** users:
- Main user-equal-weight paired difference **−0.02396** (−2.40 percentage points), user bootstrap 95% CI **[−0.02857, −0.01937]**.
- Hash placebo paired difference **−0.00156** (−0.16 pp), bootstrap CI **[−0.00411, +0.00021]**.
- Main descriptive stratum gaps when `previous_click=0`: **−0.05893**; when `previous_click=1`: **+0.01461**. The change of direction warns against a simplistic batch-state causal interpretation.
- This validates the generic Workbench's real-data execution, **not** Qwen scientific autonomy or independent confirmation, and not a new Bias discovery.

## Remaining work / scientific acceptance

P0 software capability: complete. P1 generic counterexperiment/reflection capability: complete with tested model-action contract, but **autonomous Qwen end-to-end scientific acceptance has NOT passed**. Distinguish software readiness from scientific research success.

Next decision should target the **model/tool interface**, not add fixed Bias cards:
- Let Designer explicitly register/compile derived columns before using them in an Experiment Specification; distinguish row-level source query from aggregated exploratory summaries.
- Test a stronger model with the *same* Workbench to evaluate hypothesis quality and data-schema fidelity.
- For independently promising, preregistered mechanisms, separately build P3 holdout executor and mechanism-level interventions; leave existing sealed confirmation/official scientific Holdout untouched until then.

No historical C00 result or unrelated worktree change was deleted. No Git commit in this round.
