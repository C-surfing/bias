"""LLMRank-specific contracts using the EXISTING scientific agent harness."""
from rsi_researcher.research.scientific_agents import (
    ResearchAgent, DiagnosisAgent, CriticAgent, ReflectionAgent, CRITIQUE, strict, TEXT
)
from rsi_researcher.llmrank.strategy import BASELINE

STRATEGY = strict({
    "history_n":{"type":"integer","enum":[5,10,20,50]},
    "history_order":{"type":"string","enum":["chronological","reverse"]},
    "candidate_order":{"type":"string","enum":["frozen","reversed","rotated"]},
    "votes":{"type":"integer","enum":[1,3]},
        "popularity_lambda":{"type":"number","enum":[0,0.05,0.1,0.2,0.4]},
    "rerank_policy":{"type":"string","enum":[
        "global","personalized_gate","protect_top3"]},
    "instruction":{"type":"string","maxLength":350},
    "score_expression":{"type":"string","minLength":1,"maxLength":160},
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
        "You are the RSI experimental designer for LLM zero-shot ranking. "
        "Propose one COMPOSABLE concrete intervention; not a label from a small menu. "
        "You can change history framing, candidate presentation, sampling, "
        "an original generalizable prompt instruction and a small safe score formula. "
         "Use rerank_policy global for unconditional penalty, personalized_gate "
        "to subtract only popularity exceeding the user's history median, "
        "or protect_top3 to freeze the original first 3 predictions. "
        "score_expression uses ONLY numbers and base, pop, match; e.g. "
        "'base - 0.1 * pop + 0.1 * match'. Do not use code execution, labels, "
        "test answers, or change evaluation. Make the proposal genuinely distinct "
        "from previous experiments. State a measurable DEVELOPMENT falsifier: "
        "the intervention FAILS if bias does not improve or utility drops "
        "beyond registered tolerance. A development experiment is exploratory; "
        "it need not prove causal identification to be allowed. "
        "Use exact strategy object fields and valid enum values."
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
