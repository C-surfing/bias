# Mechanism-discriminating experiment selection — 2026-10-26

The single Discovery loop remains Explorer / Opportunity Pool /
Designer / Workbench / Critic / Reflection. No new Director, search tree,
preset list of Bias classes, or separate model-side memory was added.

## Scientific contract
- Concept-formation proposes candidate phenomena and rival explanations from
  actual development observations, without inventing tag semantics.
- The researcher chooses the next question, registered feature, two observed
  arms, one numeric event outcome, population, independent statistical unit
  and predicted directions H1/H2 on the **same** contrast. Different prose
  alone is not a discriminating prediction.
- Workbench checks native DuckDB outcome type, observed finite outcomes,
  arm-specific independent users and users observed in both arms before
  inferential statistics. It returns numerical support, not a model score.
  The output and the supporting source columns originate from the current
  registered research view, not preset named-bias actions.
- Invalid contrasts are rejected and surfaced to Qwen with the measured
  error. If simply removing the filter would recover valid comparison support,
  that is reported as an observed alternative and available to the model as a
  constrained repair option. Invalid twice -> a journaled rejected choice;
  research may continue instead of falsely announcing a valid experiment.
- The Designer receives this selected contract and must implement the same
  outcome, grouping, arms and analysis. Workbench's feature binding checks
  remain mandatory. Critic can test a different estimand; Reflection revises
  or abandons in light of real numerical results.
- `paired_user` requires at least 5 users actually in both arms.
  `cluster_weighted` requires at least 5 independent users per arm,
  permitting properly analyzed mutually exclusive user cohorts.
  Fewer than 20 independent users trigger a low-information warning
  (not fabricated confirmation).
- The machine reviewer separately verifies an actually supported selection
  with distinct prediction directions, and the executed Experiment
  Specification matching the chosen contrast. An end-to-end run with
  no valid selected contrast is not counted as a scientific workflow success.
- A bounded run writes its evidence upon model failure/budget exhaustion.

## Direct code inspiration
- DiscoveryBench / AllenAI: separate context, variables, relations and
  executed workflows, but audit SQL lineage and numeric evidence rather
  than relying on free-form text evaluation.
  https://github.com/allenai/discoverybench
- ResearchBench / ACL Findings 2026: preserve multiple rival concepts;
  use evidence-based selection, and later evaluate reversed candidate order
  and hypothesis mutation within the *same* opportunity interface, rather
  than adding new Agent roles.
  https://github.com/ankitala/ResearchBench
- Biases in the Blind Spot / ICML 2026: follow concept generation with
  contrasting tests, progressive sample checks and multiple-testing control.
  Its input perturbations are not automatically causal interventions on an
  opaque logged platform.
  https://github.com/FlyingPumba/biases-in-the-blind-spot
- MechHypoBench / October 2026: score unseen-condition consequences only
  after freezing the mechanism and protocol; currently NOT a confirmed
  novel Bias on KuaiRand.
  https://arxiv.org/abs/2610.05197

## Evidence
See `runs/discovery-opportunity-20261026/selection-improvement/`.
Do not interpret failed choice or numerical association as a confirmed
mechanism, recommendation harm, or independent validation. Concept+Test
remains the primary cost-controlled discovery baseline.

## Next scientific test
Frozen new mechanisms, independent withheld conditions, competing
mechanism prediction accuracy, and real recommender-state interventions.
Holdout inspection must not inform new variable or hypothesis selection.

## Real-Qwen validation after implementation
Frozen runs: `runs/discovery-opportunity-20261026/selection-improvement/`.
Use the `discovery.review` module to independently audit any journal.

- v1: Qwen selected `next_click` from the wrong table, and its first
  correction repeated the invalid source. The tool rejected the selection.
- v2/v3: a proposed filter left 303 treated users but **zero** control
  users; pre-selection numerical support rejected it and recorded the
  invalid population. The v3 budget cap also demonstrated why bounded
  model runs need an explicit, persisted `budget_exhausted` status.
- v4: removing the unsupported filter recovered **303 paired users**, but
  Designer used a high-cardinality multi-column stratum and Workbench
  rejected excessive grouping cardinality.
- **v5**: Qwen chose H002 with registered `tag_click_interaction`, numeric
  `click`, and 303 paired users. The original user-level estimate was
  +0.141351 with Bootstrap 95% CI [0.128906,0.152414]. Qwen requested
  valid distinct Workbench countertests and abandoned H001/H002. The
  deterministic full-workflow reviewer accepted the run. Some later
  primary tests repeated an already measured estimand and one Critic
  countertest yielded exactly identical numeric statistics, leading to
  the subsequent anti-duplication and numeric-distinctness changes.
- **v6** (after the new duplicate/retirement rules): Qwen selected H001,
  correctly previewed 303 paired users and ran one primary estimate
  +0.055790 (95% CI [0.046109,0.064550]) and a genuinely different
  Critic contrast +0.036514. However, Qwen still failed to produce a
  scientifically substantive revised measurement or abandon the
  hypothesis, and repeated primary estimates were rejected rather
  than executed. The run completed with `development_inconclusive`
  and `full_qwen_workflow_accepted=false`. This is the correct audit,
  not a regression in numerical experiment execution.

The achieved advancement is a *real-data, typed, supported and traced*
opportunity-to-experiment contract that does not silently hallucinate
numeric columns or grant credit to identical countertests. It has not
established a novel, causal or harmful recommender bias. The actual
mechanistic rivalry of the generated tag-click hypotheses remains weak.
`ResearchBench`'s reverse-order ranking, blind-spot progressive
multiplicity controls, and `MechHypoBench` hidden-condition mechanism
prediction scoring have **not** yet been implemented; apply them next
only against a frozen set of discriminating predictions and a sound
independent evaluation environment.
