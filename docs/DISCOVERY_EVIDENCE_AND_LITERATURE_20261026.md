# Autonomous Bias Discovery: Research Contract and Literature Review (2026-10-26)

## Decision
Scientific validity is gated in order: variable computed -> variable bound by name -> executable experiment -> distinct numerical countertest -> substantive revision or abandonment -> independent mechanism and utility confirmation.
Do not claim a novel harmful recommender Bias from a development association, Critic prose, or a copied control.

## Current evidence
- A local main checkpoint was made at 65aa4a4. Follow-up Discovery work is scoped separately.
- Real Qwen registered tag_previous_click_interaction and executed Workbench statistics using the exact variable name and read-only derived view. A controlled first experiment estimated a within-user click association of +0.05579 (95% user bootstrap CI [+0.04611,+0.06455]); this association is not identified as causal feedback.
- A genuine countertest on the is_random=1 subgroup measured +0.03651 instead. Critic triggered Qwen Reflection. Qwen requested 'revise', but did not materially change the mechanism, falsifier or measurement. The numeric auditor rejected that pseudo-revision.
- An independent-context evaluation for model-level KuaiRec feedback writes changed >98% of rankings, but confirmed no clear utility harm. This is NOT a verified online bias.
- Separate novel hidden mechanism suites recovered 6/6 injected mechanisms by Concept+Test (26 development interventions and 32 Qwen requests), but these are different tasks than an earlier 4/6 result and cannot be treated as a paired before/after gain.

## Adopted from code and papers
1. DiscoveryBench (ICLR 2025): https://github.com/allenai/discoverybench ; https://arxiv.org/abs/2407.01725 ; code eval/new_eval.py and agents/. Hypotheses are evaluated by Context, Variables, Relationship and full workflow. Here, use deterministic SQL lineage as the primary correctness evidence; a semantic judge is advisory only.
2. ResearchBench (ACL Findings 2026): https://github.com/ankitala/ResearchBench ; https://aclanthology.org/2026.findings-acl.644/ ; code src/researchbench/retrieve.py, generation.py, ranking.py. Inspiration retrieval, hypothesis composition, mutation, pairwise ranking with both candidate orders belong in hypothesis selection AFTER the feature/tool contract is stable.
3. Biases in the Blind Spot (ICML 2026): https://github.com/FlyingPumba/biases-in-the-blind-spot ; https://proceedings.mlr.press/v306/arcuschin26a.html ; concept pipeline generates candidate concepts and validates them via paired positive/negative perturbations and progressive testing with multiplicity control. Transfer only where we can intervene in a recommender, not to observational log artifacts.
4. MechHypoBench (2026-10): https://arxiv.org/abs/2610.05197 ; tests mechanistic hypotheses by observable consequences under withheld conditions, not by resemblance of textual explanations. No verified official code at review time.
5. LLM-AutoSciLab (NeurIPS 2026): https://scientific-discovery.github.io/llm-autoscilab-project/ ; select experiments where competing mechanisms disagree. Later extension, not reason to enlarge Active RSI yet.
6. Robin (Nature 2026): https://www.nature.com/articles/s41586-026-10652-y ; measured experimental feedback drove revisions; the authors simplified an agent workflow that ran near-deterministically. Prefer our simple end-to-end workflow until stronger evidence merits new agent layers.

## Implementation anchors
- research/discovery/open_research.py: Qwen roles, reusable derived-variable cards, experiment and Critic/Reflection journals.
- research/discovery/workbench.py: generic inference and feature binding contract, subgroup support checks, explicit handling of invalid placebo.
- research/discovery/review.py: process audit versus scientific claim audit; descriptive Critic SQL is not an independent Workbench test.
- research/discovery_cycle.py and discovery_lab.py: Concept+Test and hidden-mechanism independent evaluator baseline.
- research/discovery/real_intervention.py: controllable offline KuaiRec state-write model and held-out utility reference.

## Outstanding acceptance gap
Qwen must use a truly different counterexperiment and then produce a measured falsifiable revision (or explicitly abandon the mechanism). The Critic's alternative and an invalid rewording are not enough. Only after this is stable should we score proposed concept novelty, mechanism predictions on withheld conditions, recommendation utility and discoveries per model request.
