# Scientific discovery vertical slice: scoped mechanism and sealed confirmation (2026-10-28)

The research goal remains autonomous discovery of **previously unspecified recommender Bias**. This change provides a necessary falsifiable *constructed-model* mechanism path, **not a claim of real-world platform Bias**.

## What changed

- Agent discovers a candidate and selects a controlled-model reference if that dataset capability is supported.
- An Agent-designed intervention runs on the owned KuaiRec recommender, holding users, candidates, labels and model implementation fixed.
- The independent numeric H1/H2 direction audit, nonzero development interval, executed negative control, current hypothesis version, Critic scoped verdict and literature check all must agree before `frozen_scoped_mechanism_unconfirmed`.
- The old normative/platform reference gate remains strict. It is NOT bypassed by a constructed-model effect.
- A different CLI invocation, without any model access, evaluates the **frozen same specification** on disjoint confirmation users. Both statistical and scientific scope are recorded.

## Reproducible path (run on the project server)

```bash
cd /data/cjh/rsi/project
PYTHONPATH=. python -m rsi_researcher.research.discovery explore \
  --source-db data/external/kuairec/KuaiRec.zip \
  --dataset kuairec \
  --config configs/discovery_qwen_open_phenomena.json \
  --output runs/kuairec-open-scoped-20261028 \
  --max-rounds 24
```

Inspect `frozen_candidate.json`, `journal.json`, `research_opportunities.json`, and `model_calls/`. Do **not** call confirmation on `development_inconclusive` or `scientific_stalled`.

Only if `frozen_candidate.json.status == "frozen_scoped_mechanism_unconfirmed"`:

```bash
PYTHONPATH=. python -m rsi_researcher.research.discovery confirm-controlled \
  --source-zip data/external/kuairec/KuaiRec.zip \
  --frozen runs/kuairec-open-scoped-20261028/frozen_candidate.json \
  --output runs/kuairec-open-scoped-20261028/confirmation.json
```

The confirmation evaluator does not ask Qwen for new variables, groups or interventions. It evaluates the frozen intervention once on disjoint users. The output can report `confirmed_scoped_model_mechanism`; the output always records `platform_causal_bias_established=false` and `real_world_novelty_established=false`.

## What remains to discover a real new Bias

The KuaiRand insertion probe alone does not identify the unknown whole-platform ranking policy, and KuaiRec dense feedback is not an observed production allocation mechanism. To claim a real new recommendation Bias, the next research step must identify an **item allocation vs defensible relevance/opportunity reference** under a valid design, test a mechanism with different H1/H2 predictions, and independently replicate on sealed data. Do not promote the scoped-model gate to a real-platform causal conclusion.

Avoid changing the frozen hypothesis in response to confirmation outcomes; a new hypothesis requires a fresh development/confirmation split.
