# Discovery dataset and scientific reference contract

This change is a **development-only execution slice**, not a claim that a new
recommender Bias has been identified. The same Scientist → Designer → Workbench →
Critic → Reflection loop now uses a selected dataset adapter.

## Source boundaries

| Adapter | --source-db | Research events | Reference supported |
| --- | --- | --- | --- |
| `kuairand27k` (default) | frozen KuaiRand trajectory DuckDB | development users `user_fold != 0`, bounded days | random insertion *diagnostic only*, same observed item pool |
| `kuairec` | official KuaiRec ZIP with `small_matrix.csv` and `item_categories.csv` | development-bucket users, excluding their item evaluation panel | near-full observed user-item relevance, **not** production exposure or policy allocation |
| `development_duckdb` | an already isolated DuckDB | caller-certified `events` and `features` tables | none by default; no automatic source reference |

The prepared tables use canonical `uid` and `vid` roles; adapters expose
capability metadata and outcome/item fields. The generic entry does **not**
silently copy a source DB that might include confirmation users: use
`--development-db` pointing to the exact same pre-isolated database.

```bash
python -m rsi_researcher.research.discovery explore \
  --dataset kuairand27k --source-db /path/to/frozen_source.duckdb \
  --config configs/discovery_qwen_open_phenomena.json \
  --output runs/kuairand-new-dev

python -m rsi_researcher.research.discovery explore \
  --dataset kuairec --source-db /path/to/KuaiRec.zip \
  --config configs/discovery_qwen_open_phenomena.json \
  --output runs/kuairec-new-dev

python -m rsi_researcher.research.discovery explore \
  --dataset development_duckdb --source-db /path/to/isolated_dev.duckdb \
  --development-db /path/to/isolated_dev.duckdb \
  --config configs/discovery_qwen_open_phenomena.json \
  --output runs/other-dataset-new-dev
```

No environment secrets or raw data are committed.

## Scientific contracts

`ReferenceSpec` checks whether a requested source is supported by the
declared dataset capabilities. The supported reference kinds describe evidence
**sources**, not predefined Bias categories. A random-insertion contrast is
recorded as `measured_unidentified`; it does not assert a legitimate target
policy or causal recommendation harm.

`EstimandSpec` records the same study population, unit, group, two arms,
outcome and analysis that a selected H1/H2 prediction intends to measure.
The executed Workbench experiment is compared against that contract.
Conditioning on a new subgroup is recorded as a population change, not silently
reported as the identical estimand. A post-countertest Critic sees the actual
numerical direction audit, but noncausal observational evidence cannot by
itself establish a recommender-caused Bias.

A currently absent capability is returned as unsupported. **Do not** change
`normative_target_validated` to `True` to make a freeze pass. A future
controlled recommender reference must first produce executed, scoped,
independently auditable evidence.

## Validation and next work

Automated tests exercise real DuckDB execution and a small synthetic
KuaiRec-format ZIP using the same development-only split functions as the
constructed-recommender evaluation. They are interface tests, not live Qwen
discovery or external dataset replication.

Next: link controlled recommender experiments to a scoped normative reference,
let Agent actively select new evidence while retaining multiple hypotheses,
and run equal-cost local Qwen and stronger-model experiments with independent
confirmation users protected.

## Executable reference: Agent-controlled model intervention

For `kuairec`, `replayable_recommender=true` means **our owned,
transparent category-memory re-ranker**, not a reconstructed production
KuaiRec algorithm. The Agent first proposes a phenomenon and competing H1/H2,
selects a `controlled_model` reference, then may request
`controlled_experiment`. A separate Designer chooses one mechanism parameter:
history window, recent feedback write omission, item-prior weight, user-profile
weight, positive-feedback weight, or negative-feedback weight. These are model
operations, **not predefined Bias categories**. The Designer also declares
one metric (NDCG@10 or Recall@10), two intervention values, distinct directional
predictions and a meaningful effect margin.

The Workbench executes both settings on the same development users, the
same candidate lists, and the same held-out user-item evaluation labels. The
constructed recommender receives only its history and train-user item priors,
**not** the panel labels. It reports a paired-user treatment-minus-control
estimand, 95% user-bootstrap CI, user count, ranking changes and a negative
control that deletes writes outside the active history window. The negative
control checks the *implementation*, not the rival scientific mechanism.

The Reference Ledger records `identified_model_effect_only` after checking
that the actually executed estimand matches the design. This status establishes
that an effect is **measurable/identifiable within the constructed recommender**;
it does NOT prove a nonzero effect, its normative desirability, a production
platform Bias, or novel real-world discovery. The general Bias freeze gate
remains closed because `normative_target_validated=false`. If Reflection
changes the falsifier or measured quantity, the old experiment remains
versioned as evidence but the new version requires a fresh reference test.

Machine evidence is written to `research_opportunities.json`, `journal.json`,
and the usual research-state outputs. The LLM sees aggregates, not per-user
masked evaluation labels; the full reproducible record stays with the evaluator.

**Present implementation limit:** the intervention grammar covers six safe
dimensions of this specific executable model. It does not claim universal
intervention ability across arbitrary external models. Extending to another
recommender requires an Adapter exposing a comparable paired intervention and
an explicit, scoped reference; it must not copy KuaiRec-specific mechanics
into the shared Workbench.
