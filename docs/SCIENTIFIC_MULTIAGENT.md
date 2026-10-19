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
| AgentCF Memory | `item_memory_ablation`, `popularity_prompt` | `rank_change_excess`, `ndcg_delta`, `head_excess_delta` | Existing candidate support required |
| AgentCF Popularity | `popularity_prompt`, `item_memory_ablation` | Same measured paired effects | Head exposure != identified popularity bias |

Python refuses any task/action/metric mismatch EVEN IF Qwen Critic approves it.
Numbers are computed from exactly two disjoint *development* panels.
Unchanged repeat rankings estimate stochastic noise. Changes without sufficient
training-memory candidate overlap terminate as inconclusive. The randomized
Coat *holdout* and AgentCF hidden/unlabeled relevance never enter this loop.

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
matching user IDs alone is insufficient. If an arm does not actually execute,
the comparison is INCOMPLETE, not a successful paired study. Actual model
calls and token usage are reported, not treated as equal because the caps are
equal. An arm with rejected Qwen
proposals is recorded as not executed. Without a frozen prior campaign,
`qwen_audited_history` and `qwen_no_history` may be identical:
`memory_comparison_interpretable=false`, **no RSI memory claim**.
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
