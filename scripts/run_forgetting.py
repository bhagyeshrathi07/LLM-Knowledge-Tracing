#!/usr/bin/env python
"""Peak-to-final paired forgetting test per (model, skill).

Writes data/processed/forgetting_peak_to_final.parquet and prints the skills whose
selection-adjusted p-value is below alpha, plus the seven candidate subjects named in
the project notes regardless of significance.

Usage:
    python scripts/run_forgetting.py [--models m1 m2 ...] [--alpha 0.05] [--min-items 50]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llm_kt.forgetting import peak_to_final_table
from llm_kt.ingest import MODELS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "data" / "processed" / "fluid_responses.parquet"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "forgetting_peak_to_final.parquet"

CANDIDATES = [  # OLMo-2 MMLU subjects with peak-minus-final accuracy > 0.08 (project notes, 2026-09-30)
    "machine_learning", "abstract_algebra", "global_facts", "public_relations",
    "college_mathematics", "econometrics", "business_ethics",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-items", type=int, default=50)
    parser.add_argument("--min-peak-idx", type=int, default=5, help="ignore peaks in the first N checkpoints")
    args = parser.parse_args()

    responses = pd.read_parquet(args.inp)
    responses = responses[responses["model"].isin(args.models)]
    table = peak_to_final_table(responses, min_items=args.min_items, min_peak_idx=args.min_peak_idx)
    table.to_parquet(args.out, index=False)
    print(f"Wrote {len(table):,} rows to {args.out}\n")

    pd.set_option("display.width", 200)
    cols = ["model", "skill", "peak_idx", "final_idx", "n", "acc_peak", "acc_final", "drop", "se",
            "b_1to0", "c_0to1", "p_mcnemar", "p_holm", "p_selection"]

    sig = table[table["p_selection"] < args.alpha].sort_values("p_selection")
    print(f"Skills with selection-adjusted p < {args.alpha}: {len(sig)}")
    if not sig.empty:
        print(sig[cols].round(4).to_string(index=False))

    print("\nOLMo-2 candidate subjects (from the curve inspection):")
    cand = table[(table["model"] == "olmo2-7b") & (table["skill"].isin(CANDIDATES))]
    print(cand[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
