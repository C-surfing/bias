# Discovery: research papers → specific engineering choices (2026-10-26)

The work is Autonomous Bias Discovery **for recommender-system Debias RSI**.
Do not equate a statistically significant association in logs with a new
causal bias mechanism, and do not expand the RSI search tree before concept
coverage and experimental correctness are established.

## 1. DiscoveryBench (ICLR 2025)
- Paper/code: https://github.com/allenai/discoverybench
- Code inspection starting points: `discovery_agent.py`, `agents/`,
  `discovery_eval.py`, `eval/`; real and synthetic tasks.
- Key distinction: evaluating **experimental workflow fidelity** separately
  from the **meaning and quality of the resulting hypothesis**.
- Adopt: journaled feature definitions, executable experiment specifications,
  independent numerical reviewer, open-form mechanism evaluation.
- Do not blindly adopt: lexical matching of free-form mechanisms as scientific
  truth; our instrumented recommender and holdout must supply evidence.

## 2. ResearchBench (ACL Findings 2026)
- Paper: https://aclanthology.org/2026.findings-acl.644/
- Code: https://github.com/ankitala/ResearchBench
- Inspiration retrieval → hypothesis composition (mutate/refine/recombine)
  → pairwise ranking; compare both presentation orders.
- Adopt *after* Qwen has a working experimental loop: produce a small,
  diverse collection of hypotheses, rank by explanatory scope, rival
  discriminability and expected information per cost; reverse hypothesis
  presentation order to measure chooser instability.
- Current immediate borrowing: single-field *experiment* mutation when
  Qwen3-8B duplicates all fields of a Workbench specification. One
  Qwen-selected modification is executed by our existing Workbench.
- Do not copy its gold-hypothesis ranking as a truth signal for unknown
  real-world Bias discoveries.

## 3. Biases in the Blind Spot (ICML 2026)
- Paper: https://proceedings.mlr.press/v306/arcuschin26a.html
- Code: https://github.com/FlyingPumba/biases-in-the-blind-spot
- Inspect: `biases_in_the_blind_spot/`, data preparation and the concept
  pipeline, `experiments/` and result JSON. Automated candidate concepts
  are paired with positive/negative variants, tested on progressively
  larger samples, and evaluated with multiple-testing controls.
- Adopt: positive/negative *interventions* where the recommendation model
  can actually be run with fixed candidates, paired user units, an explicit
  manipulation check, negative controls and fixed budget.
- Cannot be transposed directly: synthetic counterfactual text variants are
  not observed logged recommendation interventions. KuaiRand random insertion
  does not reveal the platform's internal preference/model state.

## 4. MechHypoBench (2026-10-04)
- Paper: https://arxiv.org/abs/2610.05197
- Official code repository remains unverified.
- Distinguishes **Driver, Context, History, Phase** and evaluates mechanisms
  by predictions on *withheld, one-factor-at-a-time condition pairs*.
- Adopt for the **scientific-quality tier**: after a candidate mechanism is
  frozen, request directional predictions for hidden paired conditions;
  run these through an independent simulator or real controlled recommender
  to compute correct direction / zero-effect identification. Verbal
  similarity to a gold explanation is not sufficient.
- Our current implementation **does not yet perform this mechanism
  consequence evaluation**. The report must mark it "not_evaluated".

## Current implementation map
- `rsi_researcher/research/discovery/open_research.py`: Qwen roles,
  action protocol, feature definition, observed value summaries, evidence
  continuity, model-chosen countertest deltas, reflection and rejection.
- `rsi_researcher/research/discovery/workbench.py`: read-only,
  per-user/cluster statistics, bootstrap intervals, placebo, provenance.
- `rsi_researcher/research/discovery/review.py`: machine-checked steps
  and *explicit unknowns*, distinct from mechanism truth.
- `rsi_researcher/research/discovery_cycle.py`: Concept+Test versus
  Fixed in injected hidden mechanisms, previously 4/6, new blinded suites 6/6.
- `rsi_researcher/research/discovery/real_intervention.py`: paired
  intervention on *our* category-memory recommender using KuaiRec ground
  feedback. Confirms ranking-state sensitivity, NOT harmful Bias.

## Quality rubric (review scientific claim **after** runtime correctness)
1. **Grounded observable**: an actually executed feature with nontrivial
   sample support and correct time order; no future/current-outcome leakage.
2. **Mechanistic explanation**: specify a driver and the mediator/state,
   context/history/phase, and a specific falsifying prediction. "Tags cause
   more clicks" without an algorithmic mechanism is not enough.
3. **Real rival**: user composition, source logging, item relevance,
   position/candidate construction, seasonality, or random exposure—not
   simply "the null hypothesis."
4. **Discriminating experiment**: valid paired users, feasible group support,
   different primary estimand, uncertainty and material effect comparison.
   Mere rewording and re-stratification do not count.
5. **Unseen consequences**: frozen condition-level predictions measured on
   independent held-out users/seeds or simulator interventions. Missing is
   not failure, but no claim of verified mechanism.
6. **Scientific novelty and harm**: identified difference relative to cited
   prior work; a trustworthy recommendation-relevance reference and utility
   harm measured under the actual debias intervention.

This order follows the user's priority: first secure feature → experiment
continuity, then evaluate true hypothesis quality. Only successful candidates
advance to debias; failed experiments remain valuable negative evidence.

## Interpreting Qwen's real KuaiRand runs
- v8: actual registered feature and 2 statistical experiments. Critic's
  secondary SQL was descriptive only. Qwen abandoned H001; valid evidence-
  responsive **rejection**, but full numeric countertest acceptance was not met.
- v9: reused the derived feature, produced first contrast:
  +0.14135085 with 95% bootstrap [+0.12890597, +0.15241406]. Qwen
  modified only the `strata_expression` to form a countertest; the pooled
  effect and interval remained identical. It proposed a cosmetic revision,
  which the audit rejected.
- None of these constitute identification of a novel recommender bias:
  the 'tag 39/42' effects are descriptive associations with potential
  confounding, and platform policy internals are unobservable.

## Implemented proof of a complete research workflow

`qwen-workbench-v11` is the first fully auditable real Qwen sequence:
a model-proposed feature and bound experiment, actual user-level statistics,
an independently requested *materially different* subgroup countertest, and
an agent-chosen abandonment after inspecting the noisy countertest. The
`review.py` report grades this workflow as accepted while keeping
mechanism identification, novelty, harmfulness and hidden consequence
agreement explicitly unverified. Scientific quality evaluation must come
**after** this execution contract, never be conflated with it.
