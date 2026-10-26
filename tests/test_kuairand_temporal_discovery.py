"""Focused contract tests for complete KuaiRand-27K temporal discovery."""
import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from rsi_researcher.research.kuairand_temporal_discovery import (
    BASIC,RANDOM_LOG,STANDARD_PARTS,USERS,discover_files,
    prepare_trajectories,mine_observational_cards,
    random_reference_audit, matched_independent_review
)

class TemporalDiscoveryTests(unittest.TestCase):
    def test_reject_pure_or_incomplete_sequence(self):
        with tempfile.TemporaryDirectory() as work:
            data=Path(work)/"KuaiRand-Pure"/"data"
            data.mkdir(parents=True)
            (data/"log_random_4_22_to_5_08_pure.csv").write_text("user_id,video_id\n")
            with self.assertRaises(FileNotFoundError):
                discover_files(work)

    def test_prior_temporal_features_and_future_user_holdout(self):
        import duckdb
        from rsi_researcher.research.kuairand_random_assignment import create_random_assignment_cards
        with tempfile.TemporaryDirectory() as work:
            data=Path(work)/"KuaiRand-27K"/"data"
            data.mkdir(parents=True)
            columns=("user_id","video_id","date","time_ms","is_click",
                     "is_hate","is_like","long_view","tab","is_rand")
            files=[]
            for filename in STANDARD_PARTS:
                f=(data/filename).open("w",newline="")
                writer=csv.writer(f)
                writer.writerow(columns)
                files.append((f,writer))
            try:
                for user in range(25):
                    for idx in range(75):
                        day=20220411 if idx<45 else (20220426 if idx<58 else 20220503)
                        time_ms=int(datetime.strptime(str(day),"%Y%m%d").replace(
                            tzinfo=timezone.utc).timestamp()*1000)+idx*25000+user
                        part=0 if idx<45 else 2
                        files[part][1].writerow((user,(user+idx)%12,day,time_ms,
                           int((user+idx)%4 !=0),int(idx%11==0),0,
                           int((user+idx)%5==0),1,0))
            finally:
                for f,w in files:f.close()
            with (data/RANDOM_LOG).open("w",newline="") as f:
                w=csv.writer(f);w.writerow(columns)
                t=int(datetime.strptime("20220426","%Y%m%d").replace(
                    tzinfo=timezone.utc).timestamp()*1000)+50*25000+12000
                w.writerow((0,2,20220426,t,1,0,0,0,1,1))
            with (data/BASIC).open("w",newline="") as f:
                w=csv.writer(f);w.writerow(("video_id","tag","author_id","video_duration"))
                for item in range(12):w.writerow((item,str(item%3),str(item%4),8000))
            with (data/USERS).open("w",newline="") as f:
                w=csv.writer(f);w.writerow(("user_id","user_active_degree"))
                for user in range(25):w.writerow((user,"high_active"))
            db=Path(work)/"test.duckdb"
            outcome=prepare_trajectories(work,db,user_modulo=1,threads=2)
            self.assertEqual(outcome["tables"]["events"],1876)
            self.assertEqual(outcome["tables"]["random_probe_features"],1)
            self.assertEqual(outcome["tables"]["anomaly_features"],875)
            with duckdb.connect(str(db),read_only=True) as con:
                first=con.execute(
                    "SELECT event_day,previous_click,older_click "
                    "FROM observed_sequences WHERE uid=0 ORDER BY moment_ms LIMIT 1"
                ).fetchone()
                self.assertEqual(first[0],20220411)
                self.assertIsNone(first[1])
                self.assertIsNone(first[2])
                first_future=con.execute(
                    "SELECT COUNT(*) FROM anomaly_features WHERE event_day>=20220501 "
                    "AND user_fold=0").fetchone()[0]
                self.assertGreater(first_future,0)
                probe=con.execute(
                    "SELECT source_file FROM random_probe_features").fetchone()[0]
                self.assertEqual(probe,RANDOM_LOG)
                before_next_day=con.execute(
                    "SELECT next_click FROM anomaly_features "
                    "WHERE uid=0 AND event_day=20220426 "
                    "ORDER BY moment_ms DESC LIMIT 1").fetchone()[0]
                self.assertIsNone(before_next_day)
            shock=create_random_assignment_cards(db,seed=20261023,min_events=1)
            self.assertIsInstance(shock,list)
            with duckdb.connect(str(db),read_only=True) as con:
                recorded=con.execute("SELECT COUNT(*) FROM random_assignment_episodes").fetchone()[0]
                self.assertEqual(recorded,1)
            candidates=mine_observational_cards(db,seed=20261023,
                                                min_events=5,min_users=2)
            self.assertGreater(len(candidates),0)
            for row in candidates:
                self.assertFalse(row["is_causal_bias"])
                self.assertEqual(row["development"]["split"],"development")
                if row["confirmation"]:
                    self.assertEqual(row["confirmation"]["split"],"confirmation")
                if "&" in row["development"]["concept_id"]:
                    self.assertIn("factorial_interaction_residual",row["development"])
                    self.assertEqual(len(row["development"]["four_condition_cells"]),4)
            selected=candidates[0]["development"]
            matched=matched_independent_review(db,selected)
            self.assertFalse(matched["causal_identification"])
            support=random_reference_audit(db,selected)
            self.assertFalse(support.get("causal_debias_utility_identified",False))

if __name__=="__main__":
    unittest.main(verbosity=2)
