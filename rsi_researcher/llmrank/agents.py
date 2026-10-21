"""LLMRank-specific contracts using the EXISTING scientific agent harness."""
from rsi_researcher.research.scientific_agents import (
    ResearchAgent, DiagnosisAgent, CriticAgent, ReflectionAgent, CRITIQUE, strict, TEXT
)
from rsi_researcher.llmrank.strategy import BASELINE

STRATEGY = strict({
    "history_n":{"type":"integer","enum":[3,5,10,20,50]},
    "history_order":{"type":"string","enum":["chronological","reverse"]},
    "candidate_order":{"type":"string","enum":["frozen","reversed","rotated"]},
    "votes":{"type":"integer","enum":[1,3]},
    "popularity_lambda":{"type":"number","minimum":0,"maximum":2},
    "rerank_policy":{"type":"string","enum":[
        "global","personalized_gate","protect_top3"]},
    "instruction":{"type":"string","maxLength":350},
    "score_expression":{"type":"string","minLength":1,"maxLength":320},
    "policy_program":{"type":"string","maxLength":1600},
})
PROPOSAL=strict({
    "hypothesis":TEXT,
    "alternative":TEXT,
    "falsifier":TEXT,
    "predicted_metric":{"type":"string","enum":[
        "ARP@5","APLT@5","CandidatePositionBias_NDCG@5","NDCG@5"]},
    "strategy": STRATEGY
})

class LLMRankDesignerAgent(ResearchAgent):
    role="design"
    instruction=(
        "You are a scientific agent inventing new mechanisms for zero-shot "
        "recommendation debiasing, NOT merely choosing from a menu. "
        "Design a distinct, testable, label-blind intervention. "
        "You control history framing, candidate presentation, any finite "
        "continuous popularity_lambda from 0 to 2, a conditional scoring "
        "expression, and optional multistage JSON policy_program. "
        "The expression can use base (normalized Qwen rank), score (current "
        "stage score), pop (train popularity), match (historical title overlap), "
        "recent_match (last 3 items), user_pop (user historical median popularity), "
        "pop_excess (pop above user profile), novelty, head (top 20 percent "
        "of candidate popularity), rank_norm, history_len, pop_percentile. "
        "Use +,-,*,/, min,max,abs,clamp,sqrt,log1p and Python-like ternary "
        "expressions, e.g. 'base - (0.3*pop_excess if recent_match < 0.15 "
        "else 0.05*pop)'. None may use target or test labels. "
        "policy_program is a JSON ARRAY SERIALIZED AS A STRING, or empty string. "
        "The program has 1..5 stages. Legal stage fields: "
        "score has op and expr; mmr has op and relevance_weight (continuous 0..1); "
        "protect has op and top (integer 0..5); head_budget has op, top and "
        "max_head (integers <=10). Compose stages only if the scientific "
        "mechanism demands them; do not include every stage reflexively. "
        "Pick your OWN stage sequence, conditional features and real coefficients "
        "rather than copying a canned formula. On each retry, contrast your "
        "new mechanism with the EXACT prior strategy and explain what changed. "
        "These stages combine a flexible score, candidate-level redundancy "
        "correction, relevance prefix lock and contextual popularity budget. "
        "Prefer a NONEMPTY policy_program with at least one reasoned stage "
        "so that this experiment tests an actual new algorithm, not only a "
        "cosmetic prompt edit. Leave it empty only if every stage would "
        "contradict your explicit falsifiable hypothesis. "
        "You MUST state competing explanation and a DEVELOPMENT falsifier. "
        "Previous numeric failures should guide proposals: harsh popularity "
        "penalties often reduce ARP by sacrificing NDCG, so consider adaptive "
        "gating and keeping relevance. Admit small exploratory experiments "
        "even if uncertain; only evaluator decides success. "
        "Never change datasets, split, evaluator, user target, metric code, "
        "or model weights. Return all schema fields."
    )
    schema=PROPOSAL
    def propose(self,observation,episode_id):
        return self.act(observation,episode_id)

class LLMRankCriticAgent(CriticAgent):
    instruction=(
        "You are the Critic of a controlled LLMRank development experiment. "
        "Your job is protocol ADMISSIBILITY, not predicting success or demanding "
        "causal proof from one exploratory experiment. APPROVE when it is "
        "executable on frozen candidates, preserves label isolation, includes "
        "a measurable falsifier and can be compared to baseline on identical "
        "users. REJECT only actual leakage, missing controls, unexecutable "
        "strategy, or a logically untestable measured claim. Do not reject "
        "merely because an intervention might fail, is not novel enough, "
        "or could cause a trade-off; those are for independent evaluation. "
        "If rejecting, identify the exact contract failure and a repair."
    )
    schema=CRITIQUE

class RankGateway:
    """Adapter from the new dataset runner into existing quota/schema vLLM model."""
    def __init__(self,scientific_model,cache_path=None,preload_paths=()):
        self.model=scientific_model
        self.cache={}
        self.cache_hits=0
        self.cache_path=cache_path
        self.preloaded_rankings=0
        # Immutable ranking evidence is carried across development rounds.
        # The key includes exact model metadata, prompt, schema and system
        # instruction; changed strategies still issue genuine Qwen requests.
        # Holdout responses are NEVER imported into this development cache.
        for source in preload_paths:
            from pathlib import Path
            path=Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"missing prior rank cache: {path}")
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row=__import__("json").loads(line)
                    key=row["prompt_sha256"]
                    answer=row["ranking"]
                    if not isinstance(answer,dict) or not isinstance(answer.get("ranking"),list):
                        raise ValueError("corrupted previous model ranking")
                    if key in self.cache and self.cache[key]!=answer:
                        raise ValueError("conflicting frozen rank cache for the same request")
                    self.cache[key]=answer
                    self.preloaded_rankings+=1
    @property
    def calls(self):
        return self.model.requests
    def ask(self,system,prompt,tag):
        import hashlib
        import json
        import re
        aliases=sorted(set(re.findall(r"C\d{2}",prompt)))
        schema=strict({"ranking":{"type":"array","items":{
            "type":"string","enum":aliases},"minItems":len(aliases),
            "maxItems":len(aliases)}})
        key=hashlib.sha256(json.dumps({
            "system":system,"prompt":prompt,"schema":schema,
            "model":self.model.metadata},ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            self.cache_hits+=1
            return self.cache[key]
        self.model.scope={"scientific_role":"ranking","episode":tag}
        response=self.model.ask(system+"\n"+prompt,schema,{"stage":"rank","case":tag})
        self.cache[key]=response
        if self.cache_path is not None:
            with self.cache_path.open("a",encoding="utf-8") as output:
                output.write(json.dumps({
                    "prompt_sha256":key,"tag":tag,"ranking":response
                },ensure_ascii=False)+"\n")
        return response
