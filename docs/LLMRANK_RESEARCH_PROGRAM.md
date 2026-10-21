# RSI / LLMRank self-improving research program

This is the human-maintained research contract for the **existing**
`rsi_researcher science-tree` harness. The LLMRank adapter must not establish
a separate competing RSI scientific authority. The four science-agent roles,
vLLM runtime, run quotas, scientific skill provenance, and the original
tree search remain part of the single mainline research platform.

## Research question

Can a model-driven research policy produce **generalizable, measurable
recommendation-debias interventions** on a pure zero-shot Qwen ranker
while preserving user-relevant ranking utility and not exceeding the
fixed-search compute budget?

A lower ARP or higher APLT is evidence of a **proxy change**, not by
itself identified causal popularity-bias correction. Position bias is
tested with paired candidate-permutation experiments. A valid negative
result is valuable; do not optimize for the phrase "success".

## Autoresearch contract (Karpathy-inspired)

- **Fixed, outside the editable surface:** input dataset split, validation
  and holdout labels, candidate source, negative sampling, evaluator,
  metrics, seeds, request/time limits, all raw model outputs and audit trails.
- **Editable experiment surface:** human-readable ranking instructions;
  history selection/order; candidate presentation; Borda vote count;
  train-popularity rerank coefficient; composable safe score expression
  on `base`, `pop`, and `match`.
- **Human maintains this file.** Scientists generate explicit, executable
  proposals and falsifiers rather than editing the evaluation code.
- Establish a real frozen baseline first. Make a prospective modification.
  Execute paired development probes. Record numeric effects and costs.
  The independent Python auditor determines whether to keep the strategy;
  reject or archive otherwise. Every rejection should inform the next node.
- Store artifacts in a unique run directory, never rewrite prior runs.
  A node can be proposed, rejected by protocol, executed and rejected
  by measurements, development-qualified, or independently confirmed;
  these are different states. The last category requires a separately
  planned untouched study, not the same cohort.
- Never treat repeated samples from the same cohort or the same model
  response as independent evidence. Track request counts, prompt and
  completion tokens, invalid outputs and cache hits.

## Workflow

```
Dataset adapters (ML-1M / Games / Amazon 2014 5-core)
   -> freeze train/dev/heldout users + candidate sets
   -> Qwen LLMRank baseline with user/item histories
   -> independent HR/NDCG/ARP/APLT/PopREO/position/validity audit
   -> read-only evidence tools and DiagnosisAgent
   -> LLMRankDesignerAgent proposes a compositional strategy + falsifier
   -> executable-contract validator + task-specific CriticAgent
   -> execute on the same frozen development cases
   -> independent numerical audit, keep/discard, ReflectionAgent
   -> scientific_skills.proposal (unverified, provenance-linked memory)
   -> scientific_tree node expansion / backtrack within budget
   -> freeze one strategy BEFORE evaluating held-out test
   -> paired bootstrap CI, fixed-strategy controls, request/token costs
```

## Current scientific acceptance screen (development only)

A proposal is eligible for deployment to heldout only if:
1. all model outputs are valid and no worse in invalidity than baseline;
2. NDCG@5 has no more than 5% relative degradation and HR@5 no more than
   one sampled hit loss; and
3. ARP falls at least 2% relative, APLT rises more than 0.02, or absolute
   paired candidate-position bias falls by more than 0.02.

This is a provisional *screen*, NOT scientific discovery or significance.
When a candidate improves bias but hurts relevance, store the Pareto
trade-off and expand a weaker variant instead of discarding the science.
Independent assessment after strategy freeze is mandatory.

## Critic's responsibility

The Critic can veto data leakage, nonpaired comparisons, unexecutable
formulae, invalid outcome claims or unfalsifiable tests. It cannot
reject a runnable preliminary experiment merely because it might
fail, is not causal proof, or might have a relevance/bias trade-off.
A rejected plan receives one revision. Both attempts remain in history.

## Relation to upstream

- Official candidate-file `has_gt=True` retrieval for ML-1M and Games:
  `ml-1m.random` / `Games.random`; `sasrec` is an alternative.
- ML-1M item split here is custom chronological leave-two-out.
  Games uses authors' preprocessed valid/test targets.
- Therefore this is **LLMRank-author-candidate Qwen adaptation**,
  NOT bit-for-bit full LLMRank paper reproduction.
- Amazon Beauty, Sports, CDs use McAuley 2014 5-core reviews/meta
  processed into the common format. These have no official LLMRank
  candidate-file baselines; label their experiments frozen-random.

## Reproduction commands

```bash
cd /data/cjh/rsi/integrated-llmrank-20261021
PY=/data/cjh/rsi/envs/research/bin/python
$PY -m rsi_researcher science-tree \
  --task llmrank_bias --config configs/llmrank_qwen.json \
  --data /data/cjh/rsi/llmrank-rsi-20261021/external \
  --dataset ml-1m --retrieval random --cohort-size 24 \
  --seeds 20261021,20261022,20261023 \
  --max-nodes 2 --max-requests 600 --output runs/UNIQUE
```

Requires an explicitly owned local Qwen vLLM service and available GPU.
Never launch Qwen implicitly, stop unrelated GPU processes, or claim
runs which have not completed. A 24-user smoke test does not establish
statistical significance. 200-1000 users with multiple seeds and
independent external cohorts are the later validation standard.

## References and design principles

- https://github.com/karpathy/autoresearch: immutable evaluator,
  single editable experiment surface, fixed budget, result log,
  keep/discard based on an objective metric.
- https://lilianweng.github.io/posts/2026-07-04-harness/:
  files as persistent memory, tool-grounded experiment evidence,
  explicit subagent roles, separation of harness from evaluator,
  auditable evolution, testable failure records, and diversity
  preservation. Do not copy the autoresearch training loop as a
  recommender or require weight training for LLMRank.
