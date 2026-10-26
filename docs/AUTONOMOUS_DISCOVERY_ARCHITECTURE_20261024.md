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

Historical `/data/cjh/rsi/discovery_private_discovery-*` directories are referenced by older experimental logs and retained unchanged. They contain tiny historical synthetic evaluator truths. There is **no migration or deletion**, and new suites do not add top-level clutter. The `discovery_private_20261024` directory also contains an isolated Python environment; do not mix that environment with experiment results.

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
