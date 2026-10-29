# P0 Discovery scientific identification: verified execution contract (2026-10-28)

## Research question

The system must discover a *previously unnamed* candidate recommendation distortion, and show its observable, defensible reference and a falsifiable comparison against a user-preference rival. The Agent chooses concepts, source variables, control/treatment arms, and mechanisms. The Workbench executes pre-registered comparisons and audits their provenance.

## Executed contract

1. **Observed candidate:** Scientist and Concept Formation propose mechanisms using development-only data. A model-selected opportunity registers the **exact** source table, uid unit, cohort filter, item/user group, two actual arm labels, numeric outcome and analysis in `EstimandSpec`. Unsupported arms or hallucinated features are rejected.
2. **Experiment:** Primary Designer can no longer change the selected cohort filter or unit. A changed cohort belongs to a *separately labeled* Critic sensitivity test, never the original H1/H2 estimand. Workbench publishes bootstrap intervals and numerical H1/H2 directional comparison. This direction audit is explicitly observational, not a causal proof.
3. **Restricted randomized-reference mode (KuaiRand):** Agent proposes the item-side feature and arm labels. Workbench returns observed standard-vs-random group allocation gap, response contrasts and **the standard-minus-random response-gap interaction**, plus a matched-four-cell within-user sensitivity calculation and support. The registered prediction audit compares the *interaction*, not an unrelated marginal CTR, and rejects mismatched fields.
4. **Owned-model reference (KuaiRec):** Before a controlled experiment, Workbench executes the Agent-selected observational estimand on the development-only observed feedback. Designer sees the result and must choose one executable model parameter and explain how the observed signal motivates this test. The evidence record binds observed source, exact estimand, outcome, model parameter, predicted directions, scope and version.
5. **Scientific gate:** A constructed-model hypothesis may freeze only if the observed source signal supports predeclared H1, the controlled intervention supports its own preregistered H1 over H2 with nonzero CI and executed negative control, Critic accepts scoped validity and literature was reviewed. The frozen scheme can then be replayed on the sealed confirmation users via `confirm-controlled`.

## What the two datasets do and do not identify

| Dataset | What can be identified from these tools | What remains unidentified |
|---|---|---|
| KuaiRand random insertions | Selected within-item-pool observational allocation/response differences and their uncertainty; user-matched sensitivity | Normative merit for production ranking, historical user-state randomization, causality of opaque learned policy |
| KuaiRec dense feedback + *owned* recommender | A proposed response regularity and paired causal effect of an explicit state/score intervention inside **our** model | The mechanism actually implemented by KuaiRec's production platform, out-of-distribution policy utility, real-world new Bias |
| Independent confirmation | Whether a *frozen scoped-model mechanism* reproduces on disjoint confirmation users | Whether the underlying platform uses that model mechanism |

An Agent-written assertion that a reference is credible is not identification evidence; reference status is assigned by executed workbench scope. Even a positive numerical H1/H2 direction match cannot distinguish a causal real-world Bias where assignment, confounding and the production ranking policy are unknown.

## Reproduce / inspect

```bash
cd /data/cjh/rsi/project
git pull --ff-only origin main
python -m pytest -q tests/test_discovery_dataset_reference.py tests/test_discovery_controlled_reference.py tests/test_discovery_compact_agent_route.py
PYTHONPATH=. python -m rsi_researcher.research.discovery explore \
  --source-db data/external/kuairec/KuaiRec.zip --dataset kuairec \
  --config configs/discovery_qwen_open_phenomena.json \
  --output runs/kuairec-p0-evidence-20261028 --max-rounds 24
```

Inspect `journal.json`: `selection_contract_audit`, `observational_prediction_audit`, `randomized_reference_prediction_audit` where applicable, `observational_anchor`, `controlled_experiment`, `numeric_prediction_audit`, `freeze_gate`. Inspection of `research_opportunities.json` should reveal the actual evidence status and provenance.

Actual local Qwen experiments and the KuaiRand production-logging study have **not** been run in GitHub CI. Do not interpret a passing fixture test as a discovered novel Bias.
