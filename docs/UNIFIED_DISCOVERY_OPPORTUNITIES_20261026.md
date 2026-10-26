# Autonomous Discovery: one loop, one research state

## The architectural decision
Observation, opportunity discovery/selection, variable construction,
experimental specification, Workbench, Critic, Reflection and subsequent
observations are **one continuous research loop** in
`rsi_researcher/research/discovery/open_research.py`. A research opportunity
pool is structured research *memory*, not a new scheduler or Agent pipeline.

The model may first inspect raw event distributions, user trajectories, temporal
changes and conditional relationships. It selects its own variables and SQL.
No popularity, exposure, filter-bubble or fairness categories are enumerated
as candidate research paths. Numeric tag codes have no semantic category names.

Once actual development observations exist, Qwen proposes competing mechanisms.
Each opportunity has evidence trace, source concept, mechanism and rival,
falsifier, measurable predictions, a next observation, prospective cost and
development experimental history. Qwen selects its own opportunity based on
where the rival predictions disagree and whether a feasible numeric test
could separate them. The decision and model-written query remain in the same
Journal and `research_opportunities.json`; Workbench still validates
registered-variable names and performs user-level inference on isolated
development data. Critic asks for an actual different experiment. Reflection
revises or abandons based on the data; pool state reflects that decision.

## Correspondence to official research code
- DiscoveryBench: dataset-supported Context / Variables / Relationship, plus
  workflow fidelity rather than accepting plausible natural-language outputs.
  Code: https://github.com/allenai/discoverybench (`eval/new_eval.py`).
- ResearchBench: separate inspiration observation, hypothesis composition and
  rival-oriented comparison; generate more than one mechanism and let the
  model explicitly select a research opportunity. Double-order hypothesis
  comparison and mutation are future *optional* improvements after repeated
  reliable runs. Code: https://github.com/ankitala/ResearchBench
  (`src/researchbench/generation.py`, `ranking.py`).
- Biases in the Blind Spot: candidate concepts independent of predefined
  categories, matched negative controls and sequential testing. Do not
  confuse a description of a recommendation log with a system intervention.
  Code: https://github.com/FlyingPumba/biases-in-the-blind-spot
  (`biases_in_the_blind_spot/concept_pipeline/`).
- MechHypoBench: prefer predictions of **held-out observable consequences**
  to verbal similarity between hypotheses. Paper:
  https://arxiv.org/abs/2610.05197.

## Evidence and acceptance
- The previously confirmed Qwen v11 chain remains intact: computed variable,
  bound experiment, independent Critic-requested *different* user-level
  contrast and Qwen abandonment. This was a workflow success, not discovery
  of a causal, harmful recommender bias.
- The initial unified-pool v1 run defined a raw-data feature and produced
  opportunities, but Designer repeatedly supplied a text-valued result
  (`next_tags`) where a numeric outcome was required. Workbench correctly
  refused to run it. This is an experimental-design quality failure.
- New work records both choice and the next model-directed observation.
  Machine audit separately reports whether a Qwen opportunity choice,
  Workbench execution, distinct countertest and evidence-responsive revision
  or abandonment occurred.
- The latest real-Qwen run details and its exact paths should be taken from
  each run's `journal.json`, `research_opportunities.json`, and review
  command. Failed model requests and unfinished runs must not be reported
  as end-to-end success.

## Remaining scientific research work
1. Make Qwen consistently choose **numeric outcomes with observed support**,
   rather than retry an unchanged invalid Experiment Specification.
2. Verify that mechanism-vs-rival predictions differ before running
   a discriminating test. Do not rank solely by significance.
3. After controlled tests, measure hidden-condition predictive accuracy,
   independent holdout confirmation, novelty and recommendation-utility harm.
   Only then attempt mechanism-specific Debias RSI.
4. Keep Concept+Test and fixed exhaustive search as separate cost-aware
   discovery baselines. Workflow self-improvement is a supporting
   optimization, not the main research contribution.

## Shared link limitation
The supplied chatgpt.com/share/6adf266d-1d10-83ea-b2a2-ad2bbd1e769f
was explicitly attempted through the web tool, the Linux host and local
Windows curl. None yielded readable conversation text; no claim is made to
have analyzed its inaccessible contents. The above changes reflect the
user-supplied design details and verified paper repositories.

## Actual local Qwen acceptance results
The following results are stored in `runs/discovery-opportunity-20261026/`
and the deterministic reviewer can be rerun independently of the model.

- **qwen-integrated-v1**: two model-generated opportunities were recorded.
  Qwen repeatedly used a categorical `next_tags` as a DOUBLE-valued outcome.
  Workbench correctly rejected that design, so no successful inferential
  experiment is claimed.
- **qwen-integrated-v2**: the Agent explicitly chose opportunity H001 over H002
  with its own explanation, executed a development observation, used its
  registered feature in a user-level test: +0.141351 (303 paired users; 95% CI
  [0.128906,0.152414]); Critic then caused a distinct numerical test,
  estimate +0.031793. Qwen abandoned H001; the journal's workflow review
  passes. The process subsequently stopped on an unrelated model request
  HTTP 400. This is a successful recorded research sequence, NOT a clean
  process-level completion.
- **qwen-integrated-v3**: Qwen explicitly wrote two different rival predictions
  for H001, selected a research opportunity and executed two primary/counter
  comparisons. It completed with `development_inconclusive`; the Critic's
  restricted subgroup only contained 15 paired users. Qwen requested a
  revision but did not change the falsifier or executable measurement enough
  to pass the numeric audit. The full workflow reviewer correctly reports
  `false` for this run.

Neither run proves a new causal or harmful recommendation bias. The next
scientific improvement should target **choice of discriminating variables,
outcomes and falsifiers**, not extra tree size or more Agent roles.
