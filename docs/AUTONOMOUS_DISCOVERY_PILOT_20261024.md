# Autonomous Bias Discovery: v2 pilot — 2026-10-24

## Current decision

The existing Debias RSI scientific tree is not modified. This pilot adds one black-box
recommendation laboratory, one agent research-cycle entry point, and an independent
structured evaluator. It reuses `ResearchAgent`, `create_model`, `RunLimits`, and
`ExperimentJournal`. No second LLM client, MCTS, evaluation framework, or shared
source modifications were introduced. Previously dirty files on `main` remain untouched.

## What the three papers contribute

- **Biases in the Blind Spot** ([paper](https://arxiv.org/html/2602.10117),
  [code](https://github.com/FlyingPumba/biases-in-the-blind-spot)):
  propose candidate concepts from clustered observations, and keep concept
  generation distinct from paired statistical testing. The official
  `concept_pipeline/statistics.py` implements McNemar for binary paired decisions,
  Bonferroni and staged stopping. Its `variations_quality_checker.py` examines
  whether concept interventions isolate the intended factor.
  **We port:** free hypothesis fields, paired interventions on fixed users/candidates,
  and explicit scope/competing explanations. **We do not port:** McNemar for
  continuous exposure@3 differences, automatic early stopping, or classifier
  verbalization filters. Our normal CIs are exploratory; no FWER guarantee is made.
- **POPPER** ([paper](https://arxiv.org/abs/2502.09858),
  [code](https://github.com/snap-stanford/POPPER)):
  operationalize free hypotheses as measurable consequences, select falsifications,
  execute on registered data, and accumulate evidence. Official `popper/popper.py`
  delegates to `SequentialFalsificationTest`, with configurable alpha, max tests,
  and E-value aggregation. **We port:** separate prediction, registered experiment
  execution, critique/revision, negative results. **We do not port:** POPPER's
  sequential Type-I guarantees; our agent chooses tests adaptively and our
  analysis does not construct valid sequential e-values.
- **FIRE-Bench** ([paper](https://arxiv.org/html/2602.02905),
  [code](https://github.com/maitrix-org/FIRE-Bench)):
  measure rediscovery against existing independently verifiable findings and
  atomized claims, distinguishing execution from scientific conclusion. The
  official `benchmark/tree_parser.py` decomposes research into concrete
  experiments; `eval/RAGChecker` supports claim evaluation. **We port:**
  sealed task truths, separate grading of feature, conditional scope,
  effect direction, null controls and research cost. **We do not port:**
  free-form LLM judge matching; first-stage truth is explicitly structured.

## Lab and evidence boundaries

- The private generator is `rsi_researcher/research/discovery_lab.py`.
  The hidden truth JSON lives **outside** the project under
  `/data/cjh/rsi/discovery_private_20261024/`, chmod 600 in a 700 directory.
  Public manifests contain only anonymous task IDs, variables and allowed tools.
- The researcher process **does not parse private truth**. Registered query calls
  execute the simulator in a separate subprocess; only JSON observations and
  aggregate paired effects return. Model prompts never contain private paths,
  latent mechanism parameters, evaluator scores or answer options.
- The orchestrator can invoke a separate private grader **after** the agent
  finishes. Agent roles have no shell/filesystem tools.
  This is prompt/tool isolation, **not OS-user or VM isolation**; for an
  adversarial code-executing researcher, separate Unix users or containers
  are needed.
- Each simulated recommendation re-ranks the **same six candidate items**
  for each fixed user under matched random states. Interventions change only
  the registered field (device, context, slot or prior exposure events).
  Observational group summaries are clearly distinguished from interventions.
- Dynamic behavior has real previous recommendation events and a per-user
  exposure-memory update, then a downstream ranking call; the test exposed and
  repaired an initial absorbing-state failure.
- Paired user-level differences in tagged-item exposure@3 use normal 95% CIs.
  Development cohort: 384 simulated users, independent confirmation cohort:
  768 newly generated users under a different RNG seed. Conditional mechanism
  confirmation also demands a null effect outside its proposed scope.
- In binary-tag ranking, boosting tag 0 is observationally equivalent to
  suppressing tag 1. The corrected scorer grades the **identifiable relative
  contrast direction**, not an unidentifiable absolute score shift.

## Claim semantics

- `ResearchClaim`: observation, free mechanism hypothesis, scoped factor/tag
  and direction, alternative, falsifier, experiment IDs, status.
- `ExperimentRecord`: intervention levels, target subgroup, paired estimate,
  CI, randomization unit, actual top-rank changes and identification scope.
- `EvidenceAssessment`: independent replication and out-of-scope negative control.
  Only an actual independent repeat can transition supported to confirmed.
- States: proposed, supported, falsified, inconclusive, confirmed.
  The journal keeps trials and outcomes using existing `ExperimentJournal`.

## Runs executed — *fixed-rule scanner only*

Two completely separate mechanism suites (different latent task draws) were run.
The initial wrong-direction grade exposed the binary-tag identifiability issue.
Original `summary.json` files were **preserved**, then a revised independent
evaluation was saved as `summary-regraded.json`.

| Suite | Source seed | Mechanisms recovered | False claims | Exploratory tests | Model requests |
|---|---:|---:|---:|---:|---:|
| development | 20261024 | 3/3 | 0 | 96 | 0 |
| hidden new tasks | 20261025 | 3/3 | 0 | 96 | 0 |

Both null controls were correctly rejected by fixed exhaustive scanning.
Each suite also executed confirmation interventions for eligible claims.
The fixed scanner tests 4 factors × 2 tag representations × 3 user scopes per
task (24 per task) and chooses the strongest relevant effect; this is an
expensive, strong deterministic control, not an agent.

## Real model-driven development results (post-run independent audit)

A real model was subsequently recovered: using the existing
`StrictDeepSeek` JSON-schema client, `deepseek-flash` completed a
schema smoke test and 49 successful scientific calls. Authentication
was supplied from an already authorized local WSL environment over
SSH stdin; the API key was never written to source, prompts or logs.
The 50th call returned HTTP **402 Payment Required**. The research process
stopped immediately without fabricating the missing response. The campaign
is **interrupted**, not a full completed four-arm benchmark.

All available artifacts are in
`runs/discovery-model-dev-20261024/`: each completed task/arm has
`observation.json`, `claim.json`, `experiments.json`,
`trajectory.json`, `independent_assessment.json`; the shared
`journal.json` and raw `model_calls/calls.jsonl` retain real
model responses and reported usage. Evaluator-only
`partial_summary.json` preserves the per-task structural scores.

Fair comparison on the **three common completed cases** (2 positive
mechanisms + 1 null negative control):

| Method | Valid confirmed mechanisms | False discoveries | Real development interventions |
|---|---:|---:|---:|
| fixed full scan | 2/2 | 0 | 72 |
| one-shot LLM (unverified) | 0/2 | 0 | 0 |
| LLM concept + registered test | **2/2** | 0 | **3** |
| active LLM exploration / falsification | 0/2 | 1 | 8 |

On all four cases, one-shot recovered 2/3 raw mechanism structures but
zero **validated** discoveries because it never executed a test. The
concept+test method got **3/3** valid independent-sample confirmations,
zero false discoveries, and exactly four development interventions.
Its validation used a new user cohort (seed offset 32771) but these
are still the *development benchmark mechanism instances*: this is
**not confirmation of LLM generalization to a newly generated secret
mechanism suite**. Only the fixed baseline was tested on a completely
separate hidden mechanism generation seed.

Real model usage by arm:

| Arm | Started calls | Prompt tokens (successful) | Completion tokens | Total tokens |
|---|---:|---:|---:|---:|
| one-shot | 4 | 6,405 | 2,295 | 8,700 |
| concept+test | 4 | 6,345 | 2,007 | 8,352 |
| active | 42 (41 completed, 1 HTTP 402) | 116,808 | 21,738 | 138,546 |

The active method was neither accurate nor economical in this run.
A concrete failure: on the null task the Critic misclassified a
zero ranking change after a **real setting intervention** as proof
the intervention was a no-op. This confuses a manipulation check with
a downstream outcome. After the completed campaign was frozen,
the environment was improved to report the **observed setting change**
as a separate manipulation check. The agent's reflection prompt now
makes that distinction explicit, and cross-round evidence was reduced
from large self-generated essays to numeric summaries. The active
agent also now preserves a prospectively supported hypothesis rather
than always choosing its most recent one. The independent evaluator
now verifies subgroup effect direction for any claim of **all-user
scope**; a significant pooled effect alone cannot establish that an
interaction is absent. This additional requirement is verified by
a dedicated test for rejecting an overbroad interaction claim. **These changes are
engineering fixes for the next version; the original run has NOT been
relabelled or retrospectively treated as improved.** Model-backed
performance of the revised code has not yet been measured.

The strongest provisional finding is that free concept generation
plus *one registered falsification test* may provide a much better
information-per-token trade-off than unstructured multi-round
Agent deliberation, at least on a small known-mechanism simulator.
This does not establish genuinely novel bias discovery in live
recommendation logs.

## Reproducible commands

For an authorized existing model key in the environment, from
`/data/cjh/rsi/project`, using the task-local environment that has
`jsonschema` and project dependencies:

```bash
/data/cjh/rsi/discovery_private_20261024/env/bin/python -B -m   rsi_researcher.research.discovery_cycle   --seed 20261101   --output runs/discovery-new-suite-20261101   --arms fixed,one_shot,concept_test,active   --rounds 3   --config configs/discovery_deepseek_20261024.json
```

This command creates one owner-only hidden mechanism file outside the
project, one public manifest, outputs and an isolated Journal. The
agent cannot call arbitrary shell/code tools. `--private <private-file>`
can instead reuse an existing truth suite if a *matching public
manifest* is placed adjacent to the output with the expected name.
No run directory can be overwritten silently.

To test the environment without requesting LLM:

```bash
PYTHONPATH=. /data/cjh/rsi/discovery_private_20261024/env/bin/python -B tests/test_discovery.py
```

All **five** current targeted tests passed on the server: private/public
boundary, deterministic paired recommendation interventions, zero/null
manipulation verification, structural truth scoring, and independent
subgroup-scope confirmation.

The legacy fixed-only results are in
`runs/discovery-benchmark-v2-dev-20261024`,
`runs/discovery-benchmark-v2-holdout-20261024`
and `runs/discovery-cli-validation-20261024`.
`summary-regraded.json` is authoritative after correcting
the binary-tag gauge equivalence; old `summary.json` remains
unaltered for audit.

## Limitations and next step

The environment is deterministic, small, binary-tagged, and mechanism families
are investigator-specified. The agent is not told their identities, but this
does not yet establish a wholly novel bias in live recommender data. Synthetic
causal effects do not imply actual unfairness outside the declared invariance
benchmark. The evaluator currently measures structural claim recovery, not
semantic free-text mechanism adequacy. No global multiple-testing error
guarantee or calibrated novel-bias prevalence estimate has been established.

Next priority: secure valid DeepSeek credit or reuse an existing authorized
model endpoint, run the revised active logic on development tasks, freeze
the prompts and environment, then evaluate all model-backed arms on an
entirely **new blind mechanism suite**. Compare independent-task mechanism
recovery, false discovery, scope, direction, model tokens, and actual
environment intervention cost; do not use the already inspected seed
20261025 for the model-group final claim.

After independent validation, map a confirmed `ResearchClaim` with a causal
intervention record to existing `BiasTaskRegistry` as a scoped new task:
estimator, comparison, measured target and utility safeguard, falsifier and
source experiment refs. Debias RSI consumes this contract later; its present
ranking and tree code remain untouched.
