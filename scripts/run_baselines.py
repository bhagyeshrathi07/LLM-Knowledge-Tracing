#!/usr/bin/env python
"""Score the trivial baselines on next-checkpoint prediction.

Writes data/processed/baseline_results.parquet (one row per predictor x model x target
checkpoint x skill) and prints a per-model summary table.

Usage:
    python scripts/run_baselines.py [--horizon 1] [--min-history 2] [--models m1 m2 ...]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from llm_kt.baselines import DEFAULT_BASELINES
from llm_kt.evaluate import evaluate, summarize
from llm_kt.ingest import MODELS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "data" / "processed" / "fluid_responses.parquet"
DEFAULT_OUT = None  # resolved per horizon below


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-history", type=int, default=2)
    args = parser.parse_args()
    if args.out is None:
        args.out = REPO_ROOT / "data" / "processed" / f"baseline_results_h{args.horizon}.parquet"

    responses = pd.read_parquet(args.inp)
    responses = responses[responses["model"].isin(args.models)]

    t0 = time.time()
    results = evaluate(
        responses, DEFAULT_BASELINES, min_history=args.min_history, horizon=args.horizon
    )
    results["horizon"] = args.horizon
    results.to_parquet(args.out, index=False)
    print(f"Wrote {len(results):,} rows to {args.out} in {time.time() - t0:.0f}s\n")

    pd.set_option("display.width", 160)
    print(f"Next-checkpoint prediction, horizon={args.horizon}, per model (lower log_loss/brier better)")
    print(
        summarize(results, by=["model", "predictor"])
        .set_index(["model", "predictor"])
        .round(4)
        .to_string()
    )
    print()
    print("Pooled over models")
    print(summarize(results, by=["predictor"]).set_index("predictor").round(4).to_string())


if __name__ == "__main__":
    main()
