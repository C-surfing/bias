# Recommender Bias Discovery — research target and evidence stages

## A scientific distinction
A user behavior association is not a recommender-system Bias.
An evidence-backed correctable Bias candidate requires:
1. An explicit target distribution, relevance reference or controlled-policy comparison.
2. A measured systematic deviation of observed feedback, trained scores or rankings from that target.
3. A falsifiable mechanism entering the data-collection, model-learning or
   recommender-decision process, with an identifiable intervention.
Independent confirmation and recommendation-utility harm are additional
scientific gates, not properties of a well-written hypothesis.

This is a **project operational definition**, not a quotation from any paper.
It deliberately does not supply Qwen a list of predefined bias categories.

## What is implemented
- `ResearchOpportunityPool` now keeps a separate `bias_relevance` research
  proposal, including `reference_kind`, `target_reference`,
  `proposed_distortion`, `mechanism_entry_point`, and
  `corrective_intervention`.
- Initial `evidence_status=behavioral_pattern`. Qwen may propose the
  missing reference, but cannot upgrade this status merely by producing
  a causal-sounding description or statistical significance.
- The existing mechanism-vs-rival experiment selector must still provide
  a numeric outcome, actually supported treatment/control users and
  different conditional predictions. No new Agent, runner or policy search
  architecture has been introduced.
- The review explicitly reports reference discrepancy, mechanism
  identification and independent confirmation as **false** until those
  independent scientific requirements are genuinely implemented and
  evaluated. Do not promote them based on a model answer.

## Research environments
- KuaiRand-27K random insertion events can support selected logging and
  time-sequence contrasts, but they do not randomize the entire recommender
  state or identify each user's long-term ideal preferences.
- KuaiRec dense feedback gives a useful *feedback-defined* evaluation panel.
  A controllable local recommender can be intervened upon, but this is not
  evidence about the opaque production platform recommender.
- Mechanisms found on disjoint item/user populations must be expressed in
  transferable conditions rather than the literal KuaiRand tag IDs.

## Next result-bearing experiments
1. Compare Qwen's original generic observation task with the revised
   system-distortion task using identical development observations,
   experiment budgets and reproducible seeds.
2. Freeze H1/H2 predictions about a target reference and competing
   mechanisms before showing the hidden-condition result.
3. Evaluate the frozen predictions on disjoint users or controlled
   recommendation-state interventions.
4. If confirmation finds systematic target mismatch, test a causal
   correction while retaining recommendation quality. Only then route
   to the Debias RSI scientific tree.
5. Compare models under equal tool/request budgets: Qwen3-8B, DeepSeek
   Flash (and optionally stronger DeepSeek), plus fixed/Concept+Test
   baselines. Record valid discoveries per cost, not just JSON compliance.

## Secret handling for the DeepSeek backend
The Windows user-level DEEPSEEK_API_KEY is set, but non-forwarded SSH shells
currently do not inherit it. The source code reads DEEPSEEK_API_KEY from the
environment of the executing Linux process. Do not hardcode a key in the
repository or run journal. Load it interactively within the target shell
using `read -r -s -p 'DeepSeek API key: ' DEEPSEEK_API_KEY; echo;
export DEEPSEEK_API_KEY` and run the experiment in that same shell.
