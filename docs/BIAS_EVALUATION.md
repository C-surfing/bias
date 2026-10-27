# Target-bias evaluation for the LLMRank RSI mainline

Reference: Jiang et al., *Beyond Utility: Evaluating LLM as Recommender*
(WWW 2025), https://arxiv.org/pdf/2411.00331
Sections 3.4.1, 3.4.4 and Appendix A.1 (Equations 14-23).
This protocol adapts those metrics; it does not reproduce the paper.

## One research target per campaign

The canonical objectives and thresholds are in
rsi_researcher/llmrank/bias_objectives.py. Each new campaign freezes
bias_target and bias_objective_contract in its manifest; inheritance rejects
different targets, ranking parsers or metric versions. Older experiments are
not reinterpreted or rewritten.

* popularity (default): primary PopREO@5. This is five popularity groups'
  relevant-target recall dispersion (lower better), following the paper.
  Require >=5 observed positive targets in each group, a defined result in
  both arms, and >0.02 absolute reduction to pass the *development* screen.
  ARP and APLT remain exposure/novelty proxies: their changes cannot substitute
  for lower PopREO. Small samples with missing strata are INCONCLUSIVE.
* candidate_position: primary absolute CandidatePositionBias_NDCG@5,
  computed from target-first and random candidate-presentation arms (lower
  absolute value is better). Require at least ten paired users, defined
  metrics, and >0.02 absolute reduction. Position-biased prompt ranking is
  not the same as propensity bias in logged clicks. The HR version is
  secondary.

These effect thresholds are local, explicit development screening heuristics,
not statistical significance claims or thresholds prescribed by the paper.
No metric alone proves causal bias correction.

## Three independent decision dimensions

1. TARGET BIAS: supported primary outcome improves in the registered
   direction. A different proxy cannot rescue an unsuccessful target.
2. RECOMMENDATION UTILITY: NDCG@5 >= 95% of the same-cohort baseline and
   HR@5 loss at most one sample hit, plus valid complete ranking and
   executable intervention. Any trade-off stays visible.
3. EVIDENCE: format-clean matched evidence must agree with full-panel
   results; insufficient format support or group support is inconclusive.
   Candidate development success is not independent confirmation. Stop
   search and freeze policy before the sole untouched held-out assessment.

Held-out output separately records primary target values/support, paired-user
bootstrap CI of target bias reduction (if support permits), NDCG and HR, other
proxies, format sensitivity, and model requests. No post-hoc selection.
No sequential testing correction or identified causal effects are claimed.

Selection, exposure, fairness and filter bubble are distinct registered
scientific tasks; they retain their own data and identification protocols in
configs/bias_tasks.json. Do not claim they can be evaluated by replacing
their target with the two LLMRank objectives here.

## Example (not yet executed)

    python -B -m rsi_researcher llmrank-cycle \
      --config configs/llmrank_qwen.json \
      --data datasets/amazon/processed/Beauty --dataset Beauty \
      --retrieval bm25 --bias-target popularity \
      --users 200 --seeds 20261021,20261022,20261023 \
      --max-rounds 2 --max-nodes 2 \
      --requests-per-round 1000 --max-total-requests 5000 \
      --output runs/UNIQUE_NEW_CAMPAIGN

Requires an authorized, already running Qwen vLLM service. Adjust sample
size/budget against GPU resources before executing. The current task only
changes the evaluation code and checks; it does not consume GPU resources.
