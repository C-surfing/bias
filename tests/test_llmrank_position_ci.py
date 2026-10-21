"""Paired position-bias uncertainty must use the same users/arm order."""
from unittest import TestCase
from rsi_researcher.llmrank.strategy import bootstrap_paired

class PositionCIContract(TestCase):
    def test_position_bias_uses_both_frozen_presentation_arms(self):
        items=[f"I{i:02d}" for i in range(20)]
        before={"random":[],"first":[]}
        after={"random":[],"first":[]}
        for i in range(24):
            uid=f"u{i}"
            target=items[i%20]
            others=[x for x in items if x!=target]
            def rank_at(position):
                x=others.copy()
                x.insert(position,target)
                return x
            before["random"].append({"user":uid,"target":target,"ranking":rank_at(2 if i%2 else 8)})
            before["first"].append({"user":uid,"target":target,"ranking":rank_at(1 if i%3 else 7)})
            after["random"].append({"user":uid,"target":target,"ranking":rank_at(2 if i%4 else 7)})
            after["first"].append({"user":uid,"target":target,"ranking":rank_at(1 if i%5 else 8)})
        scores=bootstrap_paired(before,after,
            {item:i+1 for i,item in enumerate(items)},
            {i:i for i in items},k=5,seed=99,draws=300)
        self.assertIn("CandidatePositionBias_NDCG@5",scores)
        self.assertEqual(scores["NDCG@5"]["paired_n"],24)
        self.assertGreater(scores["CandidatePositionBias_NDCG@5"]["valid_resamples"],0)
        self.assertIsNotNone(scores["CandidatePositionBias_NDCG@5"]["ci95"])
        with self.assertRaises(ValueError):
            bootstrap_paired(before,{"random":after["random"][:-1],"first":after["first"]},
                             {item:j+1 for j,item in enumerate(items)},{i:i for i in items},5,99,10)
