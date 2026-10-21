"""Role-specific sampling: exploratory scientists, frozen rankers."""
import json
from pathlib import Path
from unittest import TestCase
from rsi_researcher.runtime.models import VLLMModel,model_description
from rsi_researcher.research.scientific_agents import strict

class ResearchSamplerTests(TestCase):
    def test_design_diversity_cannot_modify_ranking_sampling(self):
        path=Path(__file__).resolve().parent.parent/"configs"/"llmrank_open_policy_qwen.json"
        config=json.loads(path.read_text())
        model=VLLMModel.__new__(VLLMModel)
        model.metadata=model_description(config)
        model.model_parameters=model.metadata["model_parameters"]
        schema=strict({"decision":{"type":"string"}})
        model.scope={"scientific_role":"ranking","episode":"same-case"}
        baseline=model.build_payload("rank",schema)
        self.assertEqual(baseline["temperature"],0.0)
        self.assertEqual(baseline["seed"],20261021)
        model.scope={"scientific_role":"design","episode":"research-node1"}
        first=model.build_payload("design",schema)
        same=model.build_payload("design",schema)
        self.assertEqual(first["temperature"],0.75)
        self.assertEqual(first["top_p"],0.95)
        self.assertEqual(first["seed"],same["seed"])
        model.scope={"scientific_role":"design","episode":"research-node2"}
        next_turn=model.build_payload("design",schema)
        self.assertNotEqual(first["seed"],next_turn["seed"])
        model.scope={"scientific_role":"critic","episode":"research-node2"}
        critic=model.build_payload("critique",schema)
        self.assertEqual(critic["temperature"],0.)
        self.assertEqual(critic["seed"],20261021)
