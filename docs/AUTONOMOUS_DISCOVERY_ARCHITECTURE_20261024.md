# Autonomous Bias Discovery — architecture and decision authority

**Project:** `/data/cjh/rsi/project`

**Research objective:** let Qwen discover a previously unspecified candidate from raw, sequential recommendation behavior; independent experiments must distinguish observed regularities from a bias with utility harm and an effective targeted intervention. Do not count new names or simulator rediscovery as real-world novelty.

## Canonical command

```bash
cd /data/cjh/rsi/project
PYTHONPATH=. /data/cjh/rsi/discovery_private_20261024/env/bin/python -B \
 -m rsi_researcher.research.discovery status \
 --dataset data/external/kuairand-27k

# Only AFTER validating the official full archive:
PYTHONPATH=. /data/cjh/rsi/discovery_private_20261024/env/bin/python -B \
 -m rsi_researcher.research.discovery prepare \
 --dataset data/external/kuairand-27k/KuaiRand-27K \
 --output runs/kuairand27k-discovery-science-20261024-v1 \
 --modulo 8 --seed 20261024

# Only with owned local Qwen running; dev evidence is frozen before holdout read.
PYTHONPATH=. /data/cjh/rsi/discovery_private_20261024/env/bin/python -B \
 -m rsi_researcher.research.discovery review \
 --output runs/kuairand27k-discovery-science-20261024-v1 \
 --config configs/discovery_qwen_blind_20261024.json
```

## Ownership and boundaries

| File/directory | Responsibility |
|---|---|
| `rsi_researcher/research/discovery/__main__.py` | A single **real-data command entry**; delegates to existing modules, no duplicate science stack |
| `rsi_researcher/research/kuairand_temporal_discovery.py` | Reads complete 27K history, builds public developmental observation cards and fixed, disjoint user/time evaluation material; actual SQL joins and user units |
| `rsi_researcher/research/kuairand_discovery.py` | Historical Pure development evidence, **not complete sequential truth** |
| `rsi_researcher/research/discovery_lab.py` | Black-box synthetic mechanism simulator, never a real-data novelty claim |
| `rsi_researcher/research/discovery_cycle.py` | Synthetic baselines and evidence-controlled research loop, preserved for comparison |
| `rsi_researcher/research/scientific_agents.py`, `runtime/models.py`, `research/journal.py` | Shared Qwen science/LLM/Journaling components, **not reimplemented** |
| `data/external/kuairand-27k` | Immutable official raw downloaded data and checksum provenance |
| `runs/kuairand27k-*` | Derived DuckDB database, public exploratory cards, evaluator-private holdout file, frozen Qwen claims and reviewer evidence |
| `/data/cjh/rsi/discovery_private_20261024/suites/<suite>/` | Canonical **new synthetic** mechanism truths, owner-only permissions, distinct from real-data holdout files |

Six historical `/data/cjh/rsi/discovery_private_discovery-*` directories (small synthetic evaluator truths) were **moved, not deleted**, on 2026-10-26 to `/data/cjh/rsi/archive/discovery-private-legacy-20261024/` to declutter the root. Original paths, archived paths, and SHA-256 hashes are recorded in its private `relocations.json`; replay older runs by passing the archived truth path with `--private`. The active `discovery_private_20261024` directory (including its isolated Python environment and new private suites) remains in place. Do not mix that environment with experiment results.

The Qwen model is **not granted shell/file tools** by this workflow. The evaluator has access to confirmation data and synthetic secrets but transmits only development cards to the model. This is an LLM interface boundary, **not security-grade Unix-user process isolation** (the execution user owns the files). `holdout_assessment.json` is not part of Qwen's prompt.

## Three research stages, three independent verdicts

1. **Anomaly / concept:** exact source rows, timestamps, sample/support counts, conditional differences and competitive hypotheses. Observations do not demonstrate causal influence.
2. **Mechanism / harm:** preregistered discriminating predictions, independent user-and-time evidence, content-and-user-matched falsification, placebo tests and randomized pool reference when support exists. A random insertion in the feed does not randomize historic user state or permit arbitrary policy OPE.
3. **Debias / value:** explicit intervention in a *real executable recommender*, fixed candidate/utility contract, remove–restore where possible, control and placebo, paired independent utility, worst affected users and costs.

A Qwen hypothesis passes stage 1 but does not automatically pass stages 2 or 3. If stage 2 cannot be causally identified, report `replicated_observational_concept` or `not_identified`; never infer online policy gains from simple static exposure reallocations.

## Statistical controls already wired

- Event history uses **all standard and randomly inserted exposure events**, with random events marked and separated from standard recommendation outcomes.
- User sequences use millisecond timestamps. Recent windows exclude the current feedback. `next_click` outcomes are only counted for next events in the same day within 2 hours; no cross-day label leak.
- 'Other-user' prior-day signals subtract the target user's own events and always use earlier calendar dates.
- Factorial interaction residuals explicitly distinguish `a&b` from the sum of `a` and `b` single-factor contrasts. All such comparisons are descriptive.
- Development and confirmation use disjoint user folds and separate dates. Only development evidence and a few source-keyed event examples are passed to Qwen.
- After proposals are frozen, the evaluator computes independent comparisons within users and content tags and checks restricted randomized-insertion response where possible.
- No post-selection/online FDR guarantee, no inferred policy propensities, no counterfactual whole-catalog recommendations.

## Constraints

Only local Qwen3-8B via existing serving stack. Do not use DeepSeek. No unrelated tree/MCTS modules, no destructive code cleanup of concurrently modified research files, and no claim of causal novel Bias unless the numerical evidence and targeted debias truly support it.

## 2026-10-26 decision: discovery-first research and review map

**Single research loop:** Explore -> formulate competing hypotheses -> run a registered falsification experiment -> revise/abandon -> *independent* confirmation. Only independently supported mechanisms proceed to real recommender intervention and debias. These are responsibilities, **not new agents, Directors, trees, or model clients**.

**Current evidence boundary:** `runs/kuairand27k-open-agent-development-20261024.duckdb` has 21,824,051 `events` and 21,682,292 `features` rows covering 2,608 development users (2022-04-08 to 2022-04-30). Qwen sees schemas and bounded query results, never a full CSV in its prompt. `DevelopmentTools.define_feature` already builds ephemeral registered DuckDB views for subsequent exploration/experiments. Do not make another feature-registry layer.

**Canonical owners, in the order reviewers should follow:**

| Read | Question to verify |
|---|---|
| `rsi_researcher/research/kuairand_temporal_discovery.py` | How are source events, timestamps, sequence outcomes and disjoint users/time constructed? |
| `rsi_researcher/research/discovery/open_research.py`: `DevelopmentTools` | Does Qwen see real schema; can it define and then reuse a row-level feature? |
| `open_research.py`: `ACTION_SCHEMA`, `CONCEPT_PAIR_SCHEMA` | Does the model propose a mechanism, credible rival, falsifier and executable measurement? |
| `rsi_researcher/research/discovery/workbench.py`: `EXPERIMENT_SCHEMA`, `ExperimentalWorkbench` | Does an experiment retain user units, compare supported groups and return executable SQL, estimate, cluster CI and placebo? |
| `open_research.py`: experiment -> Critic -> countertest -> Reflection | Did an independent numerical counterexperiment actually run, and did the revised hypothesis materially change? |
| `research/discovery_cycle.py` and `discovery_lab.py` | Hold fixed and Concept+Test baselines against freshly sealed hidden synthetic tasks; do not tune on revealed tasks. |
| `runs/*/journal.json`, `model_calls/` | Distinguish development-only evidence, model-proposed tests, critic vetoes and independently confirmed results. |

**Immediate implementation review findings:**

1. In `workbench.py::_summary`, a mandatory minimum paired-user check is applied even when `analysis='cluster_weighted'`. This incorrectly blocks legitimate between-user, mutually exclusive arms. Restrict paired support checks to `paired_user`; assess independent clusters per arm for `cluster_weighted`, and add one regression case.
2. The full Qwen autonomous acceptance has not yet passed: one real Qwen-defined derived column -> real Experiment Specification -> successful Workbench statistics -> distinct numerical Critic countertest -> substantive hypothesis revision/abandonment, with full provenance.
3. The `4/6` Concept+Test vs `0/6` Active RSI result is on planted **synthetic** mechanisms. Preserve Concept+Test, improve observation and experiment selection, then compare on entirely new concealed tasks.
4. Random insertion in KuaiRand supplies partial randomized exposure evidence, **not** control over the historical recommender's internal state. `confirmed causal recommendation bias` requires a controlled real recommender intervention, not merely held-out log correlation.
5. Same LLM in Scientist and Critic roles is a reasoning separation, **not** independent evidence. The latter requires an executor, disjoint samples or interventions and frozen plans.

**Priority literature and code reading sequence (skim paper + follow actual functions):**

- KuaiRand (CIKM 2022): https://kuairand.com/ ; https://github.com/chongminggao/KuaiRand
- DiscoveryBench (ICLR 2025): https://github.com/allenai/discoverybench ; `agents/coder_agent.py`, `agents/react_agent.py`, `eval/eval.py`; learn task representation and facet-based evaluation, *not* unrestricted novelty claims.
- ResearchBench (ACL 2026): https://github.com/ankitala/ResearchBench ; learn inspiration retrieval, composition and order-invariant hypothesis ranking.
- Biases in the Blind Spot (ICML 2026): https://github.com/FlyingPumba/biases-in-the-blind-spot ; learn free concept proposal, counterfactual variations, multiplicity-aware screening (adapt interventions to an executable recommender).
- POPPER (ICML 2025): https://github.com/snap-stanford/POPPER ; `popper/popper.py`, `popper/agent.py`, `popper/prompt_utils.py`; learn falsification protocol; **do not claim sequential Type-I control without validating its assumptions**.
- FIRE-Bench (ICML 2026): https://github.com/maitrix-org/FIRE-Bench ; learn verifiable atomic claims, research-cost accounting and sealed rediscovery.
- MechHypoBench (October 2026): https://arxiv.org/abs/2610.05197 ; distinguish data regularities from mechanisms by checking consequences under withheld conditions.

**Next vertical slice:** review the above data -> model-action -> executor path; fix the between-user statistical contract; run one Qwen-designed feature + valid Workbench experiment + numerical countertest + meaningful revision. Do not touch independent holdout or start a Debias claim until a candidate survives development audit.
