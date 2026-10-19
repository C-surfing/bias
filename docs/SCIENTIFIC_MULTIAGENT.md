# RSI Scientific Multi-Agent: implementation and scientific boundaries

> Development architecture as implemented on `feat/rsi-multiagent-20261017`.
> Mainline is not overwritten. This document describes VERIFIED code paths,
> not any claim that RSI or debiasing has been scientifically demonstrated.

## One project / one CLI / one Qwen backend for new research

**Research object:** author-data AgentCF user/item memory, feedback and rankings.
**Scientific targets:** classical Bias Task Registry (Selection/Exposure/
Popularity/Filter Bubble/Fairness); currently executable development
falsifiers: Coat Selection and AgentCF Memory/Popularity.
Do not promote Filter Bubble or Fairness until identifiable data and
protocols exist. KuaiRand Exposure remains a support/identifiability audit.

```text
python -m rsi_researcher science          # scientific campaign
python -m rsi_researcher science-compare  # matched research-policy arms
python -m rsi_researcher bias-tasks       # frozen classic-five registry
```

**Qwen scientific roles:** `DiagnosisAgent` forms falsifiable hypotheses and
strong alternative explanations; `DesignerAgent` proposes a task-legal,
prospective numeric experiment; `CriticAgent` rejects confounded or
unidentifiable design; `ReflectionAgent` interprets the independent numeric
audit and proposes a *candidate* skill and next task. They are four distinct,
schema-validated model invocations sharing one Qwen3-8B vLLM service. None
is authorized to run a shell, edit data, modify an evaluator, inspect
confirmation labels, or approve its own discovery.

**Python governance:** `scientific_loop.py` owns budget limits, episode
states, evidence references, one-plan freeze, Critic rejection/revision,
predeclared fixed-rule control, next-task ceiling and immutable run directory.
`scientific_experiments.py` alone trains AgentCF states, executes registered
interventions, calculates numeric effects and applies quality safeguards.
`scientific_skills.py` separately manages candidate and provenance-checked
promoted skills. `scientific_comparison.py` runs arms on matched development
cohorts with identical ceilings and isolated logs. This is NOT five separate
Agent stacks.

## Actions and exact numeric contracts

| Task | Registered action | Numeric claim | Important limit |
|---|---|---|---|
| Coat propensity | `gamma_stress` | `gamma_reference_margin`: best gamma=1 Brier vs best 0.75/1.25 | Gamma sensitivity, not causal proof |
| Coat propensity | `propensity_shuffle` | `propensity_placebo_margin`: best shuffled Brier minus matched Brier | Negative propensity control |
| Coat utility | `split_utility` | `brier_gain`, `ndcg_delta`, `tail_delta` | 99% quality guard |
| Coat utility | `gamma_stress` | `gamma_reference_margin` | Separate sensitivity question |
| AgentCF Memory | broad Item Memory, targeted cross-user Item Memory, popularity prompt | General and linked-reader rank-change excess, NDCG, head excess | Broad memory effects vs reader-specific propagation |
| AgentCF Popularity | `popularity_prompt`, `item_memory_ablation` | Same measured paired effects | Head exposure != identified popularity bias |

Python refuses any task/action/metric mismatch EVEN IF Qwen Critic approves it.
Diagnosis sees real pre-decision *trained AgentCF* observations: training
event support, bounded User/Item Memory before/after examples, candidate
coverage, cross-user Item Memory write/read links, and actual baseline rankings.
The observations hide validation relevance and experiment outcomes. They are
diagnostic evidence, NOT a causal claim that memory or popularity is harmful.
The trained state, diagnostic evidence and baseline ranks are frozen before
the hypothesis and its numeric intervention forecast.

Each metric has a shared minimum development effect fixed by Python before
Qwen proposes a number (Brier/gamma/placebo 0.0001, NDCG/tail/head 0.02,
excess rank changes, including linked readers, 0.20). A model cannot pass a prediction by setting
a smaller threshold. Near-zero point estimates in tiny panels are NOT
evidence of equivalence. These thresholds are effect screens, not p-values.
Numbers are computed from exactly two disjoint *development* panels.
Unchanged repeat rankings estimate stochastic noise. Changes without sufficient
training-memory candidate overlap terminate as inconclusive. A task about
shared Item Memory also abstains when no other-user memory writer appears in
a ranking user's candidate panel, or when one of the two development panels
has no cross-user memory path. Such abstentions are NOT completed experiments. The randomized
Coat *holdout* and AgentCF hidden/unlabeled relevance never enter this loop.

### Scientific outcome semantics and experiment choice

The independent audit now separates mechanism direction/effect evidence,
known-target NDCG@5 utility costs, provisional debias proxy changes,
falsification limits and independent confirmation. A measurable mechanism
may reduce utility; the older combined quality-plus-forecast field is
retained only for compatible consumers. Near-zero point estimates do not
establish equivalence, and development observations do not exclude all
competing causal explanations.

The Designer can explicitly abstain BEFORE the Critic and execution when
there is no discriminating legal experiment. Abstention is NOT counted as
an executed or successful experiment. Otherwise, the Designer chooses one
task-legal action with a prospective numeric forecast and explains its
diagnostic value and cost.

A new targeted cross-user Item Memory ablation reverts only a candidate
memory written by OTHER training users, not by the current reader. User
memories, other item memories, query candidates and prompt remain unchanged.
Its registered primary metric measures linked-reader ranking changes
beyond same-condition repeat noise; unlinked readers are separately
reported as a negative-control diagnostic. Both development panels must
contain eligible writer-reader links or the mechanism task abstains.
This is narrower than broad Item Memory ablation; it does not identify
the specific origin of a false negative-feedback interpretation or prove
that changed recommendations are harmful.

New science actions are defined outside code-hashed historical Cross-Bias
task definitions. Strategy comparisons now also report whether actual
audited history caused different frozen actions or numeric predictions.
Neither changed wording nor observed history consumption proves RSI gains.

## Three ways to run

**Offline regression:** `/data/cjh/rsi/envs/research/bin/python -B -m unittest
discover -s tests -t . -q` in an isolated Git worktree. Tests use a
deterministic `FakeModel` and *never* support scientific claims.

**Real bounded research:** configure `configs/qwen_scientific_loop.json` to
an already-authorized Qwen vLLM instance. It does NOT automatically start,
restart or terminate any shared GPU process.

```bash
cd /data/cjh/rsi/experiments/rsi-multiagent-20261017-chatgpt
/data/cjh/rsi/envs/research/bin/python -B -m rsi_researcher science \
  --config configs/qwen_scientific_loop.json \
  --mode fresh --tasks agentcf_memory agentcf_popularity --auto \
  --arm qwen_no_history --cohort-size 6 \
  --output runs/UNIQUE_RUN_NAME
```

Use `--mode archive` ONLY to test interpretation of old numeric artifacts;
`ARCHIVED_DEVELOPMENT_REPLAY` has zero new observations. Use the explicit
`--fallback-fixed-rule` only when a rejected Qwen plan should be followed
by a separately labeled predeclared control, not to claim Qwen success.

**Equal-budget comparison:**

```bash
/data/cjh/rsi/envs/research/bin/python -B -m rsi_researcher science-compare \
  --config configs/qwen_scientific_loop.json \
  --tasks agentcf_popularity --cohort-size 6 \
  --prior-campaign runs/PREVIOUS_IMMUTABLE_DEV_CAMPAIGN \
  --output runs/NEW_MATCHED_COMPARISON
```

Each arm gets the SAME model-request/token/time ceiling, cohort seed, dataset
and numerical audit. For AgentCF, paired status requires matching frozen
initial state, training events, queries and prompts (one SHA-256 fingerprint);
matching user IDs alone is insufficient. For new AgentCF comparisons, the
trainer runs ONCE before any science arm; all arms receive one frozen trained
state and identical pre-decision ranking observations. The common training and
diagnostic Qwen calls are logged under shared_training/, separately from
each research arm's Qwen budget and usage. policy_arms_share_trained_state
means exactly that; it does NOT prove isolated causal research-policy gains
because stochastic ranking and limited task replication remain. If an arm
rejects the design or abstains for insufficient evidence, the comparison is
INCOMPLETE. Actual arm usage and shared preparation costs are reported
separately; equal ceilings are not equal actual costs. An arm with rejected Qwen
proposals is recorded as not executed. Without a frozen prior campaign,
`qwen_audited_history` and `qwen_no_history` may be identical:
`memory_comparison_interpretable=false`, **no RSI memory claim**. Even with a prior campaign path, the history arm
must actually receive audited numeric records before memory comparison
becomes interpretable.
`qwen_promoted_skill` requires a provenance-verified skill library.

## Memory, research Skill evolution and independent confirmation

- Numeric Research Memory: only SHA-256-grounded previous development audits.
  Research reflection narrative never becomes a verified fact.
- Domain priors: uncertainty around exposure/negative feedback, preservation
  of user/item state, repeated rankings, unknown popularity causality.
- Candidate Skill: `candidate_skill.json` includes task, version, source
  audit fingerprint, limitations and `validated_skill=false`.
- Evaluation gate: 8 distinct previously unseen, independently audited matched
  tasks, all four arms (evolved/strong rule/no history/shuffled history),
  explicit identical cohort fingerprint and common cost caps; algorithm requires
  at least 7/8 paired wins, higher confirmed discoveries than EVERY comparator,
  and total tokens no more than 110% of the cheapest comparator.
  Passing this mechanical screen requires **separate confirmation** and is not
  itself a p-value, final independent discovery, or automatic promotion.
- In the current author CDs and reused Coat development studies, confirmed
  scientific discoveries remain **zero** until a distinct pre-registered
  independent confirmation experiment exists.

## Development cohort feasibility (CPU only, no new Qwen requests)

Exploratory support check on seed integers 20270001–20270012, exactly two
observed training events per user, using the existing author CDS train/valid
candidate IDs and excluding known development users. Count a possible
cross-user memory path when one candidate item was positively observed in
another participating user's training history. Require >=1 such possible
path in **each** half of the sorted query panel.

| Users | Seeds with possible path in both panels / 12 |
| --- | ---: |
| 4 | 2 |
| 6 | 7 |
| 8 | 10 |
| 12 | 12 |

This checks only *candidate support* before Qwen training; even a possible
path does not guarantee the generated memory was actually changed, that it
affects ranking, or that it causes harmful bias. These 12 seeds are now
development diagnostics, not fresh confirmatory trials. Prefer larger
predeclared development cohorts for shared-memory questions, subject to
actual inference cost and unchanged independent confirmation boundaries.
Do not cherry-pick a successful seed or call these support proportions
scientific discovery rates.

## Stop / failure semantics

No fixed probability of scientific success is assumed. A campaign stops on:
budget exhaustion, no legal action, missing identifiability, Critic rejection,
lack of matching Item Memory evidence, Qwen network/schema failure, exhausted
preapproved task list, or finite episode cap. All run outputs are unique and
immutable; partial failures must be diagnosed before launching a new run,
rather than being replayed as successful episodes.

## Operations and repository cleanup policy

Read Karpathy Guidelines from
`https://github.com/multica-ai/andrej-karpathy-skills/blob/main/skills/karpathy-guidelines/SKILL.md`:
understand first, keep the simplest implementation, change only what is
necessary, and verify every result. The new science entry always requires vLLM
and never falls back to DeepSeek. Legacy ItemCF/DeepSeek code is outside the
new workflow but is still called by older commands/tests; do not delete it
without a dependency-aware frozen-release migration. Historical scientific
records are immutable. Only the isolated worktree is modified.
