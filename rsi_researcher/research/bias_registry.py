"""Frozen five-classic-bias task definitions. No new Agent roles or execution code."""
from dataclasses import dataclass,asdict
from hashlib import sha256
from pathlib import Path
import json

VERSION="classic-five-bias-tasks-v1.0"
@dataclass(frozen=True)
class BiasTask:
    id:str
    definition:str
    primary_dataset:str
    auxiliary_datasets:tuple[str,...]
    observational_proxy:tuple[str,...]
    evaluation_metrics:tuple[str,...]
    required_evidence:tuple[str,...]
    allowable_interventions:tuple[str,...]
    not_identified_if:tuple[str,...]
    current_status:str
    caveat:str

TASKS=(
 BiasTask("popularity",
  "Observed recommendation exposure or scores disproportionately favor frequently observed items relative to a specified utility/opportunity benchmark.",
  "agentcf_cds",("coat","kuairand_pure"),
  ("training_item_frequency","extra_head_share_over_panel_ideal"),
  ("paired_NDCG_at_5","Hit_at_5","head_exposure_share","tail_relevant_recall"),
  ("preintervention_item_popularity","matched_candidate_panels","independent_relevance_reference"),
  ("popularity_penalty","user_calibrated_rerank","support_preserving_rerank"),
  ("no_relevance_reference_to_distinguish_popularity_from_legitimate_demand",),
  "DEVELOPMENT_PROXY_ONLY","Prior AgentCF popularity prompts primarily measured behavior/proxies; not causal popularity correction."),
 BiasTask("exposure",
  "Feedback is biased by which items a logging policy exposes to a user, making unexposed preferences unavailable.",
  "kuairand_pure",("controlled_synthetic","agentcf_cds"),
  ("observed_exposure_frequency","policy_item_support_overlap"),
  ("randomized_log_click_utility","policy_support_coverage","effective_sample_size","SNIPS_if_propensities_identified"),
  ("impression_log","logging_policy_or_known_randomization","support_positivity","observed_click_or_rating","time_window_alignment"),
  ("exposure_constraint_rerank","controlled_exploration","identified_propensity_weighting"),
  ("no_exposure_records","no_identifiable_logging_propensities_or_randomized_reference","positivity_violated"),
  "DATA_GROUNDING_EXPLORATORY","AgentCF unsupported-negative-memory text is a semantic mechanism, not exposure propensity."),
 BiasTask("selection",
  "Recorded ratings/interactions are a nonrandom selection of user-item preferences (MNAR), changing the observed training distribution.",
  "coat",("kuairand_pure","controlled_synthetic"),
  ("self_selected_vs_random_rating_positive_rate_gap","observed_item_frequency"),
  ("random_panel_Brier","random_panel_NDCG_at_5","random_panel_Hit_at_5","paired_utility_uncertainty"),
  ("nonrandom_training_ratings","random_reference_panel","observed_label_mask","train_only_propensity_estimator"),
  ("unweighted_item_mean","estimated_propensity_weighted_item_mean","clipped_weight_sensitivity"),
  ("no_random_reference_and_no_identification_assumptions","unknown_or_zero_support_propensities"),
  "BENCHMARK_READY_FOR_DEVELOPMENT","Coat propensities.ascii contains learned estimates, not true assignment probabilities; zero ratings are missing."),
 BiasTask("filter_bubble",
  "Repeated exposure-feedback-updates shrink topic/interest diversity beyond a predeclared relevance-preserving counterfactual.",
  "agentcf_cds",("kuairand_pure","controlled_synthetic"),
  ("topic_concentration_over_time","personal_history_similarity"),
  ("longitudinal_topic_coverage","diversity_at_k","utility_over_time","counterfactual_trajectory_gap"),
  ("timestamped_repeated_exposures","item_topic_features","user_history","matched_initial_states","intervention_trajectory"),
  ("diversity_aware_rerank","bounded_interest_exploration"),
  ("single_snapshot_only","item_topic_labels_unavailable","feedback_trajectories_not_comparable"),
  "NOT_YET_IDENTIFIED","One-shot diversity metric cannot establish a filter bubble."),
 BiasTask("fairness",
  "A predeclared subject (user/item/provider group) experiences unequal utility or opportunity under an explicit fairness criterion.",
  "kuairand_pure",("coat","controlled_synthetic"),
  ("groupwise_exposure_counts","groupwise_utility_proxy"),
  ("groupwise_NDCG_gap","group_opportunity_gap","worst_group_utility","overall_utility_cost"),
  ("group_definition_and_scope","group_membership","minimum_group_support","group_utility_or_exposure_reference"),
  ("group_constrained_rerank","equal_opportunity_with_utility_guard"),
  ("groups_not_defined","insufficient_group_support","fairness_norm_not_declared"),
  "NOT_YET_IDENTIFIED","Head/tail is not automatically a social-fairness group; normative choice must be documented.")
)

def verify():
    assert len(TASKS)==5 and {t.id for t in TASKS}=={"popularity","exposure","selection","filter_bubble","fairness"}
    for t in TASKS:
        assert all([t.id,t.definition,t.primary_dataset,t.observational_proxy,t.evaluation_metrics,
                    t.required_evidence,t.allowable_interventions,t.not_identified_if,t.caveat])
        assert t.current_status in ("DEVELOPMENT_PROXY_ONLY","DATA_GROUNDING_EXPLORATORY",
                                    "BENCHMARK_READY_FOR_DEVELOPMENT","NOT_YET_IDENTIFIED")
    return True

def freeze(path):
    verify()
    dst=Path(path)
    if dst.exists():raise FileExistsError("Frozen registry already exists; make a new version")
    obj={"version":VERSION,"meaning":"Five classic recommendation bias research TASKS, not five Agent roles",
         "legacy_agentcf_semantic_probes":"auxiliary mechanism diagnosis only",
         "acceptance_rule":"Never promote proxy-only improvements as confirmed real data debiasing",
         "tasks":[asdict(t) for t in TASKS]}
    data=json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n"
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(data,encoding="utf8")
    digest=sha256(dst.read_bytes()).hexdigest()
    dst.with_suffix(".sha256").write_text(digest+"\n",encoding="ascii")
    print(json.dumps({"version":VERSION,"tasks":len(TASKS),"sha256":digest},ensure_ascii=False))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--out",required=True)
    freeze(p.parse_args().out)
