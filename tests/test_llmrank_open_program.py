"""One meaningful open-program test, not a new defensive test framework."""
import json
from unittest import TestCase
from jsonschema import Draft202012Validator

from rsi_researcher.llmrank.formula import validate,score
from rsi_researcher.llmrank.strategy import (
    BASELINE,canonical,rank,stable_aliases,functional_fingerprint
)
from rsi_researcher.llmrank.program import compile_program,execute_program
from rsi_researcher.llmrank.agents import PROPOSAL

class DummyQwen:
    def __init__(self,aliases):
        self.aliases=aliases
        self.calls=0
    def ask(self,*args):
        self.calls+=1
        return {"ranking":self.aliases}

class OpenProgramTests(TestCase):
    def setUp(self):
        self.items=[f"I{i:02d}" for i in range(20)]
        self.case={"user":"dev-user","history":self.items[:5],
                   "target":self.items[-1],"candidates":self.items}
        self.titles={
            item:("lavender shampoo hair" if n%4<2 else
                  "football gloves grip" if n%4==2 else
                  "acoustic guitar strings")
            for n,item in enumerate(self.items)
        }
        self.pop={item:(200 if n<6 else 2) for n,item in enumerate(self.items)}
        self.aliases=stable_aliases(self.case)
        self.reverse={v:k for k,v in self.aliases.items()}
        self.qwen_output=[self.aliases[i] for i in self.items]
    def test_non_linear_user_condition_is_real_executable_code(self):
        tree=validate("base - (0.9 * pop_excess if recent_match < 0.25 else 0.02*pop)")
        self.assertAlmostEqual(score(tree,{"base":.8,"recent_match":0.1,
                                           "pop_excess":.5,"pop":.7}),.35)
        self.assertAlmostEqual(score(tree,{"base":.8,"recent_match":0.9,
                                           "pop_excess":.5,"pop":.7}),.786)
        with self.assertRaises(ValueError):
            validate("__import__('os').system('rm -rf /')")
        with self.assertRaises(ValueError):
            validate("item.__class__")
    def test_agent_can_propose_composed_program_with_continuous_params(self):
        steps=[
            {"op":"score","expr":"score + (0.3*recent_match if user_pop < 0.5 else 0.05*match)"},
            {"op":"mmr","relevance_weight":0.75},
            {"op":"protect","top":1},
            {"op":"head_budget","top":5,"max_head":2}
        ]
        proposal={**BASELINE,"popularity_lambda":0.137,"policy_program":json.dumps(steps)}
        strategy=canonical(proposal)
        Draft202012Validator(PROPOSAL).validate({
            "hypothesis":"User-conditioned diversity corrects redundant popular items.",
            "alternative":"Changes may come only from the candidate mix.",
            "falsifier":"Reject if NDCG falls >5 percent or ARP does not fall.",
            "predicted_metric":"ARP@5","strategy":strategy})
        q=DummyQwen(self.qwen_output)
        ranked,meta=rank(q,self.case,self.items,self.titles,strategy,
                         self.pop,3,"real-program",return_details=True)
        self.assertEqual(q.calls,1)
        self.assertEqual(set(ranked),set(self.items))
        self.assertEqual(len(ranked),20)
        self.assertEqual(meta["observed_votes"],1)
        self.assertLessEqual(sum(1 for x in ranked[:5] if x in self.items[2:6]),2)
    def test_qwen_single_quoted_stages_are_safe_data_not_code(self):
        qwen="""[{'op': 'score', 'expr': 'score + 0.3 * recent_match'},
                  {'op':'mmr','relevance_weight':0.8},
                  {'op':'protect','top':2}]"""
        candidate=canonical({**BASELINE,"policy_program":qwen})
        self.assertEqual(len(compile_program(candidate["policy_program"])),3)
        self.assertIn('"op":"score"',candidate["policy_program"])
        self.assertEqual(candidate["policy_program"],
                         canonical({**BASELINE,
                                    "policy_program":candidate["policy_program"]})["policy_program"])
        with self.assertRaises(ValueError):
            compile_program("[__import__('os').system('echo should-not-run')]")
    def test_nested_qwen_program_compiles_to_same_semantic_steps(self):
        nested="""[{'score': {'op': 'add', 'expr': 'base - 0.2 * pop_excess'}},
                   {'protect': {'op':'keep_top','top':3},
                    'mmr': {'op':'add','relevance_weight':0.7}}]"""
        direct="""[{'op':'score','expr':'base - 0.2 * pop_excess'},
                   {'op':'protect','top':3},
                   {'op':'mmr','relevance_weight':0.7}]"""
        a=canonical({**BASELINE,"policy_program":nested})
        b=canonical({**BASELINE,"policy_program":direct})
        self.assertEqual(a["policy_program"],b["policy_program"])
        self.assertEqual(len(compile_program(nested)),3)
        with self.assertRaises(ValueError):
            compile_program("[{'score': {'op': 'execute', 'expr': 'base'}}]")
    def test_program_compilation_rejects_unbounded_code_and_preserves_fingerprint(self):
        a=canonical({**BASELINE,"policy_program":
            '[{"op":"score","expr":"score + 0.2 * novelty"}]'})
        b=canonical({**BASELINE,"policy_program":
            '[ { "expr": "score + 0.2 * novelty", "op": "score" } ]'})
        self.assertEqual(functional_fingerprint(a,[self.case]),
                         functional_fingerprint(b,[self.case]))
        for value in ('[{"op":"python","source":"open(\"file\")"}]',
                      '[{"op":"score","expr":"__import__(\"os\")"}]',
                      '[{"op":"mmr","relevance_weight":999}]',
                      '[{"op":"protect","top":200}]'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    compile_program(value)
