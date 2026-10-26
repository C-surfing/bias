"""Unified command entry for KuaiRand Discovery.

The historic review command remains unchanged. The open-ended explore command
uses a separate development-only database and does not invoke holdout review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Bias discovery research workflow")
    commands = p.add_subparsers(dest="step", required=True)
    status = commands.add_parser("status")
    status.add_argument("--dataset", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--modulo", type=int, default=8)
    prep.add_argument("--seed", type=int, default=20261024)
    prep.add_argument("--min-events", type=int, default=180)
    review = commands.add_parser("review")
    review.add_argument("--output", required=True)
    review.add_argument("--config", required=True)
    explore = commands.add_parser("explore", help="agent-led open development investigation")
    explore.add_argument("--source-db", required=True,
                         help="frozen raw trajectory DuckDB (only opened during dev export)")
    explore.add_argument("--output", required=True)
    explore.add_argument("--config", required=True)
    explore.add_argument("--max-rounds", type=int, default=14)
    explore.add_argument("--min-queries", type=int, default=2)
    explore.add_argument("--development-db", help="optional existing isolated dev DB")
    args = p.parse_args()
    if args.step == "status":
        from rsi_researcher.research.kuairand_temporal_discovery import discover_files
        try:
            folder = discover_files(args.dataset)
            print(json.dumps({"complete_files_present": True, "data": str(folder),
                              "note": "file presence only; validate archive MD5 before formal experiment"}))
        except FileNotFoundError as e:
            print(json.dumps({"complete_files_present": False, "reason": str(e)}))
    elif args.step == "prepare":
        from rsi_researcher.research.kuairand_temporal_discovery import run
        run(args.dataset, args.output, args.modulo, args.seed, args.min_events)
    elif args.step == "review":
        from rsi_researcher.research.kuairand_temporal_discovery import qwen_open_discovery
        config = json.loads(Path(args.config).read_text())
        result = qwen_open_discovery(args.output, config)
        print(json.dumps({"hypotheses": len(result["hypotheses"]),
                          "model_requests": result["total_model_requests"]}))
    elif args.step == "explore":
        if args.max_rounds < 3 or args.min_queries < 1 or args.max_rounds <= args.min_queries:
            p.error("max-rounds must exceed min-queries and be >= 3")
        from rsi_researcher.research.discovery.open_research import run_open_discovery
        result = run_open_discovery(args.source_db, args.output,
                                    json.loads(Path(args.config).read_text()),
                                    args.max_rounds, args.min_queries, args.development_db)
        print(json.dumps({"status": result["status"],
                          "development_queries": len(result["development_queries"]),
                          "holdout_consulted": False}))


if __name__ == "__main__":
    main()
