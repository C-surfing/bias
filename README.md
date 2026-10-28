# RSI -- Autonomous Recommender Bias Discovery

Active goal: let Qwen independently propose previously unspecified recommender bias mechanisms from real logs, quantify systemic distortion relative to a defensible reference, and run competing H1/H2 falsification experiments.

This main branch prioritizes discovery and scientific confirmation, not predefined four-bias evaluation or LLMRank debias search. Earlier results and code remain in Git branch archive/pre-discovery-only-20261027 at commit 2113c0a.

## Primary code
- rsi_researcher/research/discovery/__main__.py: single CLI entry for KuaiRand preparation, exploration and review.
- rsi_researcher/research/discovery/open_research.py: Scientist, concept formation, opportunity selection, Designer, Critic, Reflection and state.
- rsi_researcher/research/discovery/observations.py: schema-derived field distributions and generic contrasts.
- rsi_researcher/research/discovery/workbench.py: user-level statistics and randomized-exposure diagnostics.
- rsi_researcher/research/discovery/phenomenon_miner.py: evidence-grounded extraction and consolidation.\n- rsi_researcher/research/discovery/opportunities.py: rival hypotheses and evidence records.
- rsi_researcher/research/discovery/review.py: scientific-quality assessment.
- rsi_researcher/research/scientific_agents.py and runtime/: structured Qwen calls and optional fixed-budget evaluation.

Detailed current protocol and data setup: docs/AUTONOMOUS_DISCOVERY_ARCHITECTURE_20261024.md.

Cross-dataset, development-only adapter and reference / estimand contract:
docs/DISCOVERY_DATASET_REFERENCE_CONTRACT.md. Use `explore --dataset` to select
KuaiRand, KuaiRec, or an explicitly isolated DuckDB event dataset. Reference
capabilities are evidence limits; they never independently establish a new
algorithmic Bias.

## Command
Run the discovery CLI in the isolated project environment with a local authorized Qwen service:
PYTHONPATH=. /data/cjh/rsi/discovery_private_20261024/env/bin/python -B -m rsi_researcher.research.discovery explore --source-db /path/to/frozen_source.duckdb --config configs/discovery_qwen_open_phenomena.json --output runs/unique-discovery-development

Omitting --max-rounds uses evidence-driven research until a scientific stop, a defensible freeze, repeated no-information actions, or model failure. Use --max-rounds 14 only for fixed-budget benchmark comparisons. This does not mean accepting unverified discoveries.

Development users never include independent confirmation users. Statistical correlation is a lead, not a discovered recommender bias. Random insertion in KuaiRand supplies a restricted exposure reference, not general off-policy identification.


Open-phenomena discovery uses a schema-derived profile, uniform field scans, general conditional statistics, and Qwen evidence-ID-linked phenomena before mechanism formation. None of these descriptive patterns certifies recommender-caused distortion; Workbench and independent reference checks still decide the claim. Explicit --max-rounds remains optional for future A/B/C equal-cost ablations.
