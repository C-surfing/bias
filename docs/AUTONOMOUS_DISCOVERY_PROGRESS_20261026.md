# Autonomous Bias Discovery — 2026-10-26 implementation and evidence

Main checkout: `/data/cjh/rsi/project`. The existing Discovery baseline was preserved
on `main` as `65aa4a4` before the following implementation changes. This note
distinguishes *executable research infrastructure* from confirmed scientific discovery.

## A. Qwen → Derived feature → Experiment Specification → Workbench

Changes are located in `research/discovery/open_research.py` and `workbench.py`.

1. DuckDB's `define_feature` executes and registers a new derived variable on
   physically isolated development-only data. Statistical Workbench now resolves
   the **same registered view** as interactive SQL (including custom SELECT sources).
2. Experiment Designer sees actual data values, exact columns, event-date range,
   and registered features. Invalid model-generated designs receive actionable
   execution errors and one model-directed correction opportunity. The model is
   specifically asked to **reference** its registered variable, not reproduce
   the SQL expression under a different name.
3. `outcome_expression` is ONE per-event numeric scalar, not AVG/COUNT or an
   output alias. Derived *predictors* cannot leak current click/hate/long_view
   into a hypothesis about those outcomes. Opaque numeric video-tag codes are
   never interpreted as semantic content. Invalid copied placebos are marked
   *not performed*, not credited as falsifications.
4. Critic can request a genuinely different Workbench user-level test. If its
   Designer repeats the same specification, Critic's separate SQL experiment
   is executed and marked **descriptive only**, not a paired CI. Reflection sees
   actual numeric evidence. A substantive revision requires a changed
   falsification/measurement and a *new executable* query, not just new prose.
5. Full journal preserves complete experiment outputs; only a bounded subset
   of extensive stratified results enters the model context.

Prior real runs `qwen-workbench-v1` through `v5` show failures as well as
successful individual experiments. Their `journal.json`, `research_state.json`
and `model_calls/calls.jsonl` are kept in
`runs/discovery-implementation-20261026/`. Failed hypotheses have **not**
been promoted to independent scientific discoveries.

Acceptance for Step 1 is STRICT: one real Qwen run must visibly register and
compute a feature, *reference it* in an executable Workbench specification,
obtain a Critic numerical countertest, and undergo a verifiably changed
measurement/falsification plan. Simulated-model integration tests do not
substitute for this proof.

## B. Concept+Test on wholly new hidden tasks

Fresh evaluator-private seeds: `2026102701`, `2026102702`.
Results under `runs/discovery-implementation-20261026/blind-a` and `blind-b`.

| Arm | Injected mechanisms recovered | false discoveries | Development interventions | LLM requests |
|---|---:|---:|---:|---:|
| Fixed exhaustive scan | 6/6 | 0 | 192 | 0 |
| Concept+Test with observational-contrast prioritization | 6/6 | 0 | 26 | 32 |

These are 2 independent newly seeded *simulator* suites, each with 3 injected
mechanisms and 1 null control. Their original private truths are stored outside
the repository in the evaluator-private directory. Observational signals were
sorted by absolute standardized contrast, and the LLM selected the tests;
this did not introduce task-specific bias labels. Earlier Concept+Test was 4/6
on *different* private suites, so the numbers are **not** a matched causal
ablation or a significance claim. Do not re-tune on these newly unsealed tasks.

The search-tree Active RSI was intentionally **not** expanded or rerun.

## C. Controlled recommender mechanism on real KuaiRec feedback

Source: `data/external/kuairec/KuaiRec.zip` (official near-full KuaiRec
`small_matrix.csv`, category metadata, and actual watch feedback).
Implementation: `research/discovery/real_intervention.py`.

- Train on disjoint users to build an item prior.
- For each development/confirmation user, isolate item-level observed feedback
  for held-out relevance reference (watch_ratio >= 1); this relevance is not
  used to build that user's profile.
- Execute the same *implemented* category-memory recommender twice with
  identical candidates and user/item identities, once omitting recent
  feedback writes and once restoring the writes.
- Negative control removes old writes outside the memory window, producing no
  effect. Compare paired held-out NDCG@10 and user bootstrap intervals.

| Split | independent users | NDCG with recent writes omitted | NDCG with writes restored | Paired delta (restored - omitted) | 95% CI |
|---|---:|---:|---:|---:|---|
| Development | 283 | 0.834602 | 0.831239 | -0.003363 | [-0.013449, +0.008165] |
| Confirmation | 120 | 0.842640 | 0.842170 | -0.000470 | [-0.014430, +0.013583] |

The intervention changes rank order in 98.23% / 99.17% of users respectively.
This establishes an **identifiable state-write effect in our reproducible
recommender**. The interval for *utility harm* contains zero. It neither
identifies the unobserved algorithm inside KuaiRec's production service nor
proves a harmful, previously unknown Bias. No debias success claim is made.

## Review commands

```bash
cd /data/cjh/rsi/project
PYTHONDONTWRITEBYTECODE=1 /data/cjh/rsi/discovery_private_20261024/env/bin/python -m pytest -q \
  tests/test_discovery.py tests/test_hidden_mechanisms.py tests/test_open_research.py \
  tests/test_experimental_workbench.py tests/test_real_intervention.py \
  tests/test_kuairand_temporal_discovery.py
```

Only the bounded scientific runtime and newly changed code should be committed.
Do not accidentally stage unrelated uncommitted LLMRank/Four-Bias changes,
proprietary source datasets, private mechanism truth, or large `runs/` artifacts.

## October 26 continuation: review of real Qwen outcomes

The dedicated reviewer is executable without an LLM:

```bash
/data/cjh/rsi/discovery_private_20261024/env/bin/python -m \
  rsi_researcher.research.discovery.review \
  runs/discovery-implementation-20261026/qwen-workbench-v9
```

The workflow acceptance contract distinguishes feature lineage, successful
Workbench calculation, materially different Critic calculation, and a
substantive model-selected revision **or explicit abandonment**. An SQL COUNT
probe alone does not qualify as a distinct user-level counterexperiment.
Confirmation on independently withheld cases and recommendation utility harm
remain separate scientific objectives.

Observed trajectories:
- v8: Qwen defined/reused features, executed Workbench analyses and explicitly
  abandoned a contradicted concept after descriptive Critic SQL. No distinct
  inferential Critic counterexperiment.
- v9: First user-level contrast found an association of +0.14135085
  [0.12890597, 0.15241406], but changing only `strata_expression` left the
  aggregate estimate unchanged; model's cosmetic revision was rejected.
- v10: Four successful feature-bound primary Workbench analyses, but no
  valid materially distinct Critic counterexperiment. Model often selected
  unsupported or non-discriminating population filters.
- v11: Additional Qwen execution of the support-guided design iteration
  recorded under its own run directory; assess the completed journal rather
  than inferring success from process exit alone.

The current implementation provides actual feature evidence cards, development
support checks for model-proposed subgroups, first-experiment feature binding,
a controlled model-selected countertest mutation, and a separate machine-readable
quality audit. It does not automatically turn an observational signal into an
identified, previously unknown causal recommender bias.

Useful related-paper/code review: `docs/DISCOVERY_LITERATURE_CODE_REVIEW_20261026.md`.

### v11 final result (supersedes the pending status above)

The real local `qwen3-8b` run completed. The evidence-only reviewer reports
`full_qwen_workflow_accepted=true` and `development_inconclusive`.

- Qwen defined `tag_previous_click_interaction` as
  `CASE WHEN previous_click = 1 AND tags LIKE '%39%' THEN 1 ELSE 0 END`,
  computed it on actual development events and reused the registered field.
- Primary paired-user click contrast:
  **+0.0557897**, bootstrap 95% CI **[+0.0461094, +0.0645504]**,
  **303 paired users**. This is observational and not causal identification.
- Critic demanded a separate numeric countertest. Rather than accepting a
  copied experiment, Designer selected a *different* supported subgroup:
  `(previous_click = 1) AND previous_hate = 1`.
- Workbench actually computed its separate paired-user contrast:
  **+0.0573461**, CI **[-0.152922, +0.293838]**, only **14 paired users**.
  Wide uncertainty prevents resolving the competing explanation.
- Qwen Reflection **abandoned H001** because the mechanism and proposed
  alternative were not discriminated. It did not relabel that association as a
  validated new Bias.

This completes the **end-to-end research workflow acceptance** (definition
→ computed feature → referenced specification → Workbench → material Critic
numeric countertest → evidence-responsive abandonment). It does **not**
establish a previously unknown, harmful platform-level bias, nor prove that
the original hypothesis is false. The small countertest subgroup and
unidentified recommender policy mechanism remain important limitations.
The rejected H001, model call logs, exact SQL and run journal remain preserved.
