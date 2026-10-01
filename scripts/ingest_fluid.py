#!/usr/bin/env python
"""Build data/processed/fluid_responses.parquet from the Fluid Benchmarking release.

Usage:
    python scripts/ingest_fluid.py [--out PATH] [--models m1 m2 ...] [--keep-truthfulqa]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_kt.ingest import DEFAULT_DROP_BENCHMARKS, MODELS, ingest_all

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "fluid_responses.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument(
        "--keep-truthfulqa",
        action="store_true",
        help="keep TruthfulQA, thresholding its continuous mc2 score at 0.5",
    )
    args = parser.parse_args()

    drop = () if args.keep_truthfulqa else DEFAULT_DROP_BENCHMARKS

    print(f"Ingesting {len(args.models)} models: {', '.join(args.models)}")
    df = ingest_all(models=tuple(args.models), drop_benchmarks=drop)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"Wrote {len(df):,} rows to {args.out}")
    print()
    print(
        df.groupby("model")
        .agg(checkpoints=("checkpoint_idx", "nunique"), items=("item_id", "nunique"))
        .to_string()
    )
    print()
    print(df.groupby("benchmark")["skill"].nunique().rename("skills").to_string())


if __name__ == "__main__":
    main()
