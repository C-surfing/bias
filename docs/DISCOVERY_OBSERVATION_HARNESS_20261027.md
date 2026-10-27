# Observation-driven KuaiRand Bias Discovery: implementation and review

Canonical modules: rsi_researcher/research/discovery/open_research.py, observations.py, experiment_contract.py, workbench.py and opportunities.py.

## Evidence and scientific ownership

- The isolated development database owns all SQL and statistics. The Qwen model never directly receives held-out event rows.
- Observation cards O001-O007 summarize supported comparison populations on common dates, item tag and author groups, previous-feedback, calendar changes, user activity and interface tab contexts. Every card records its source SQL, rows, and limitations. No card contains a hard-coded Bias diagnosis.
- Real Qwen Concept Formation must provide the target reference and its identification assumptions, measurable distortion, and distinct mechanism/rival predictions. An unmeasured target is not a confirmed reference.
- Compact Experiment Intent lets Qwen choose a registered item or state group, observed arm values, a numeric event outcome, population restriction, user-level analysis, and prediction/falsifier. Python constructs the existing Workbench experiment contract and verifies observed support.
- The Critic is a separate model role but NOT independent evidence. A Workbench contrast, an actual countertest and a credible reference must determine scientific quality.
- Random insertion is NOT randomization of a complete ranking policy. The numeric random reference is a diagnostic, not a validated optimal allocation target. Frozen/confirmed causal Bias cannot be inferred from descriptive response differences.

## Development database

Upstream: runs/kuairand27k-science-frozen-v1-20261024/seq.duckdb
Enhanced isolated dev DB: runs/kuairand27k-open-agent-enriched-development-20261027.duckdb
Counts: events 21,824,051; features 21,682,292. The enhanced events table additionally includes author, tab, activity, sequence_index, previous_long_view, recent_click and older_click when available upstream.

## Equal-budget A/B/C

All three configurations use exactly the same Qwen model parameters, seed, development database, and 14 rounds.
A: configs/discovery_qwen_ablation_A_legacy.json (cards off, typed contract off).
B: configs/discovery_qwen_ablation_B_observations.json (cards on, typed contract off).
C: configs/discovery_qwen_ablation_C_observations_compact.json (cards on, typed contract on).

Run from /data/cjh/rsi/project, after local vLLM is available at the base_url in the configs:

    for arm in A_legacy B_observations C_observations_compact; do
      PYTHONPATH=. /data/cjh/rsi/discovery_private_20261024/env/bin/python -B -m rsi_researcher.research.discovery explore \
        --source-db runs/kuairand27k-science-frozen-v1-20261024/seq.duckdb \
        --development-db runs/kuairand27k-open-agent-enriched-development-20261027.duckdb \
        --config configs/discovery_qwen_ablation_${arm}.json \
        --output runs/discovery-ablation-20261027/${arm} --max-rounds 14 --min-queries 2
    done

    PYTHONPATH=. python scripts/discovery_ablation_summary.py runs/discovery-ablation-20261027/*

Each run writes observation_cards.json, journal.json, research_opportunities.json, frozen_candidate.json, discovery_protocol.json and model_calls/calls.jsonl. Compare valid Workbench experiments, real countertests, cost per evidence-backed candidate, not LLM-generated Bias names.

## Tested scope and limits

Development-only item-side contrasts and the compiler/Workbench mainline have been exercised against actual KuaiRand events; these do not establish new Bias.
Local vLLM at 127.0.0.1:18523 was offline during the implementation review. Autonomous Qwen A/B/C runs have not been executed here.
Independent holdout confirmation and intervention/Debias intentionally remain separate and untouched.
