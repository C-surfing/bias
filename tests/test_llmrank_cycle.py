"""Science-loop regression tests: finite rounds and unobserved holdout."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from rsi_researcher.llmrank.cycle import conduct,ensure_frozen

class RecursiveResearchTests(TestCase):
    def test_frozen_cohort_mismatch_fails(self):
        cohort={k:"x" for k in (
            "seed","dataset","dev_users","holdout_users",
            "data_archive_sha256","candidate_source","retrieval")}
        ensure_frozen(cohort,dict(cohort))
        mismatch={**cohort,"dev_users":"different"}
        with self.assertRaisesRegex(ValueError,"dev_users"):
            ensure_frozen(cohort,mismatch)

    def test_recursive_research_does_not_peek_holdout(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)/"research"
            previous=[]
            def fake_search(destination,**kwargs):
                previous.append(kwargs["prior_campaign"])
                destination.mkdir()
                seed=kwargs["seeds"][0]
                (destination/f"seed-{seed}").mkdir()
                assert kwargs["development_only"] is True
                out={"holdout":None,"development":{
                    "selected_rsi":{"merit":0.0}}}
                (destination/f"seed-{seed}"/"result.json").write_text(json.dumps(out))
                (destination/"summary.json").write_text(json.dumps({"seeds_done":[{}]}))
                return {"requests":20}
            seen=[]
            def fake_confirm(directory,**kwargs):
                seen.append(kwargs["frozen_campaign"])
                return ([{"seed":1,"baseline":{"5":{"NDCG@5":0.2,"ARP@5":50}},
                    "rsi":{"5":{"NDCG@5":0.2,"ARP@5":50}}}],10)
            with patch("rsi_researcher.llmrank.cycle.search",fake_search),\
                 patch("rsi_researcher.llmrank.cycle.independent_confirmation",fake_confirm):
                result=conduct(root,config={},data="unused",
                    dataset="ml-1m",retrieval="random",users=12,
                    seeds=[1],max_rounds=4,max_nodes=1,
                    requests_per_round=80,max_total_requests=500,patience=1)
            self.assertEqual(result["rounds"],2)
            self.assertIsNone(previous[0])
            self.assertEqual(previous[1],root/"round-01")
            self.assertEqual(seen,[root/"round-02"])
            self.assertEqual(result["combined_requests"],50)
            self.assertEqual(result["independent_confirmed_scientific_discoveries"],0)
            self.assertEqual(json.loads((root/"cycle_index.json").read_text())
                             ["independent_confirmation"],False)
