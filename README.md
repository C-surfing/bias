# RSI -- Autonomous Recommender Bias Discovery

Active goal: let Qwen independently propose previously unspecified recommender bias mechanisms from real logs, quantify systemic distortion relative to a defensible reference, and run competing H1/H2 falsification experiments.

This main branch prioritizes discovery and scientific confirmation, not predefined four-bias evaluation or LLMRank debias search. Earlier results and code remain in Git branch archive/pre-discovery-only-20261027 at commit 2113c0a.

## Primary code
- rsi_researcher/research/discovery/__main__.py: single CLI entry for KuaiRand preparation, exploration and review.
- rsi_researcher/research/discovery/open_research.py: Scientist, concept formation, opportunity selection, Designer, Critic, Reflection and state.
- rsi_researcher/research/discovery/observations.py: grounded development evidence cards.
- rsi_researcher/research/discovery/workbench.py: user-level statistics and randomized-exposure diagnostics.
- rsi_researcher/research/discovery/opportunities.py: rival hypotheses and evidence records.
- rsi_researcher/research/discovery/review.py: scientific-quality assessment.
- rsi_researcher/research/scientific_agents.py and runtime/: structured Qwen calls and budgets.

Detailed current protocol and data setup: docs/AUTONOMOUS_DISCOVERY_ARCHITECTURE_20261024.md.

## Command
Run the discovery CLI in the isolated project environment with a local authorized Qwen service:
PYTHONPATH=. /data/cjh/rsi/discovery_private_20261024/env/bin/python -B -m rsi_researcher.research.discovery explore --source-db /path/to/frozen_source.duckdb --config configs/discovery_qwen_ablation_C_observations_compact.json --output runs/unique-discovery-development --max-rounds 14

Development users never include independent confirmation users. Statistical correlation is a lead, not a discovered recommender bias. Random insertion in KuaiRand supplies a restricted exposure reference, not general off-policy identification.
