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
