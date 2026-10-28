"""Dataset-specific preparation and reference-support views for open discovery.

Adapters describe evidence and capabilities; they do not propose Bias mechanisms.
The common research loop only sees a development-only DuckDB with events/features.
"""
from __future__ import annotations

import ast
import csv
import io
from datetime import datetime, timezone
from pathlib import Path
import zipfile


class DatasetAdapter:
    dataset_id = "development_duckdb"
    outcome_columns = ("click", "hate", "long_view", "next_click",
                       "next_tags", "watch_ratio", "relevance")
    item_features = ("vid", "tags", "author")
    capabilities = {
        "observed_responses": True, "randomized_exposure_probe": False,
        "dense_relevance": False, "logging_propensities": False,
        "replayable_recommender": False,
    }
    guidance = (
        "Investigate supported fields and actual study units. Observed response "
        "differences are not proof of recommender-caused distortion. "
        "When a reference is absent, record an identification gap."
    )

    def prepare(self, source: str, target: str) -> dict:
        """Accept only an existing physically isolated development DB.

        The caller, not the researcher, owns the user split for this mode.
        """
        import duckdb
        if not Path(source).exists():
            raise FileNotFoundError(source)
        with duckdb.connect(source, read_only=True) as con:
            tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
            if not {"events", "features"} <= tables:
                raise ValueError("Development DuckDB needs events and features tables")
        if Path(source).resolve() != Path(target).resolve():
            raise ValueError("development_duckdb expects --development-db to equal "
                             "--source-db; never copy unknown holdout rows")
        return {"path": source, "reused": True,
                "split": "caller_certified_development_only"}

    def install_reference_views(self, connection, enabled: bool) -> list[str]:
        return []

    def evaluate_controlled_reference(self, source: str, specification: dict) -> dict:
        raise ValueError(f"{self.dataset_id} cannot replay an owned recommender")

    def describe(self) -> dict:
        return {"dataset_id": self.dataset_id,
                "capabilities": dict(self.capabilities),
                "outcome_columns": list(self.outcome_columns),
                "item_features": list(self.item_features),
                "guidance": self.guidance}


class KuaiRandAdapter(DatasetAdapter):
    dataset_id = "kuairand27k"
    capabilities = {
        **DatasetAdapter.capabilities,
        "randomized_exposure_probe": True,
    }
    guidance = (
        "KuaiRand random insertions provide restricted exposure-mode response "
        "diagnostics, not randomized full ranking policies or normative merit. "
        "shared_item_pool_events (when available) is restricted to videos observed "
        "in random insertions and overlapping UI/period. Tags are opaque IDs; "
        "hate is an observed outcome, not a semantic tag. "
        "Use reference_gap only for a supported categorical item field."
    )

    def prepare(self, source_db: str, target_db: str) -> dict:
        """Materialize KuaiRand development folds; never export user_fold=0."""
        import duckdb
        dst = Path(target_db)
        if dst.exists():
            return {"path": str(dst), "reused": True}
        dst.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(dst))
        try:
            source = str(Path(source_db).resolve()).replace("'", "''")
            con.execute("ATTACH '" + source + "' AS source_db (READ_ONLY)")
            con.execute("""CREATE TABLE features AS SELECT
                uid,vid,event_day,moment_ms,source_file,activity,tags,
                click,long_view,next_click,after_long_gap,after_nonclick,
                after_negative_feedback,immediate_repeat,topic_changed,
                recent_response_drop,previous_interval_ms,recent_click,older_click
                FROM source_db.anomaly_features
                WHERE user_fold<>0 AND event_day<=20220430""")
            fields = {x[0] for x in con.execute(
                "DESCRIBE SELECT * FROM source_db.observed_sequences").fetchall()}
            extra = "".join(",s." + x for x in (
                "author", "tab", "activity", "sequence_index",
                "previous_long_view", "recent_click", "older_click") if x in fields)
            con.execute("""CREATE TABLE events AS SELECT
                uid,vid,event_day,moment_ms,is_random,click,hate,
                long_view,tags,previous_video,previous_click,
                previous_hate,previous_moment,previous_tags,next_tags""" + extra + """
                FROM source_db.observed_sequences s
                WHERE s.uid IN (SELECT DISTINCT uid FROM features)
                  AND s.event_day<=20220430""")
            con.execute("DETACH source_db")
            counts = {name: con.execute("SELECT COUNT(*) FROM " + name).fetchone()[0]
                      for name in ("features", "events")}
        except Exception:
            con.close()
            dst.unlink(missing_ok=True)
            raise
        con.close()
        return {"path": str(dst), "reused": False, "rows": counts,
                "split": "user_fold_nonzero_and_event_day_le_20220430"}

    def install_reference_views(self, connection, enabled: bool) -> list[str]:
        if not enabled:
            return []
        existing = {r[1] for r in connection.execute(
            "PRAGMA table_info('events')").fetchall()}
        if not {"uid", "vid", "is_random"} <= existing:
            return []
        fields = [f for f in (
            "uid", "vid", "event_day", "is_random", "click", "long_view",
            "hate", "tags", "author", "activity", "tab", "previous_click",
            "previous_long_view", "previous_hate", "previous_tags")
            if f in existing]
        period = ("AND e.event_day >= (SELECT MIN(event_day) "
                  "FROM main.events WHERE is_random=1)"
                  if "event_day" in existing else "")
        if "tab" in existing:
            dominant = connection.execute(
                "SELECT tab FROM main.events WHERE is_random=1 "
                "GROUP BY tab ORDER BY COUNT(*) DESC LIMIT 1").fetchone()
            if dominant and dominant[0] is not None:
                tab = str(dominant[0]).replace("'", "''")
                period += " AND e.tab='" + tab + "'"
        connection.execute(
            "CREATE TEMP VIEW shared_item_pool_events AS SELECT " +
            ", ".join("e." + f for f in fields) +
            " FROM main.events e WHERE e.vid IN "
            "(SELECT DISTINCT vid FROM main.events WHERE is_random=1) " + period)
        return ["shared_item_pool_events"]


class KuaiRecAdapter(DatasetAdapter):
    dataset_id = "kuairec"
    capabilities = {
        **DatasetAdapter.capabilities,
        "dense_relevance": True,
        "replayable_recommender": True,
    }

    def evaluate_controlled_reference(self, source: str, specification: dict) -> dict:
        from rsi_researcher.research.discovery.controlled_reference import (
            evaluate_controlled,
        )
        return evaluate_controlled(source, specification)
    guidance = (
        "KuaiRec small_matrix gives relatively dense observed user-item feedback. "
        "watch_ratio and relevance are observed responses, NOT known platform "
        "exposures; never interpret them as randomized recommendations. "
        "This development view excludes confirmation users and separately held "
        "item-level evaluation panels. No production-ranking mechanism is known."
    )

    def prepare(self, source: str, target: str) -> dict:
        """Stream official KuaiRec archive into a development-only response view."""
        import duckdb
        from rsi_researcher.research.discovery.real_intervention import (
            bucket, reference_panel,
        )
        dest = Path(target)
        if dest.exists():
            return {"path": str(dest), "reused": True}
        dest.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(dest))
        try:
            con.execute("CREATE TABLE events (uid BIGINT,vid BIGINT,"
                        "event_day BIGINT,moment_ms BIGINT,"
                        "watch_ratio DOUBLE,relevance INTEGER,tags VARCHAR)")
            with zipfile.ZipFile(source) as archive:
                catpath = next(x for x in archive.namelist()
                               if x.endswith("/data/item_categories.csv"))
                matrixpath = next(x for x in archive.namelist()
                                  if x.endswith("/data/small_matrix.csv"))
                categories = {}
                with archive.open(catpath) as raw:
                    for row in csv.DictReader(io.TextIOWrapper(raw)):
                        categories[int(row["video_id"])] = ",".join(
                            str(x) for x in ast.literal_eval(row["feat"]))
                batch = []
                with archive.open(matrixpath) as raw:
                    for row in csv.DictReader(io.TextIOWrapper(raw)):
                        uid, vid = int(row["user_id"]), int(row["video_id"])
                        if (bucket(uid) != "development" or
                                reference_panel(uid, vid) or vid not in categories):
                            continue
                        stamp = float(row["timestamp"]) if row["timestamp"].strip() else 0.
                        if stamp > 0:
                            day = int(datetime.fromtimestamp(
                                stamp, timezone.utc).strftime("%Y%m%d"))
                        else:
                            day = 0
                        ratio = float(row["watch_ratio"])
                        batch.append((uid, vid, day, int(stamp * 1000),
                                      ratio, int(ratio >= 1), categories[vid]))
                        if len(batch) >= 2000:
                            con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?)", batch)
                            batch.clear()
                if batch:
                    con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?)", batch)
            con.execute("CREATE TABLE features AS SELECT * FROM events")
            n = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        except Exception:
            con.close()
            dest.unlink(missing_ok=True)
            raise
        con.close()
        return {"path": str(dest), "reused": False,
                "rows": {"events": n, "features": n},
                "split": "development_users_minus_item_evaluation_panel"}


_ADAPTERS = {
    "kuairand27k": KuaiRandAdapter,
    "kuairec": KuaiRecAdapter,
    "development_duckdb": DatasetAdapter,
}


def get_dataset_adapter(name: str) -> DatasetAdapter:
    try:
        return _ADAPTERS[name]()
    except KeyError:
        raise ValueError(f"Unknown dataset adapter {name!r}; choose {sorted(_ADAPTERS)}") from None
