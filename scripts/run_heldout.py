#!/usr/bin/env python
"""Run same-checkpoint held-out-item diagnosis and cross-benchmark transfer.

Usage:
    python scripts/run_heldout.py --in data/processed/fluid_responses.parquet

The within-MMLU test fits T-SKIRT on a deterministic half of each subject's items and
scores the other half at the same checkpoints. Drift is chosen on the chronological
validation block. Cross-benchmark transfer calibrates the mean MMLU skill trajectory to
each target benchmark on the train block and reports held-out later checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llm_kt.evaluate import summarize
from llm_kt.heldout import (
    evaluate_benchmark_transfer,
    evaluate_heldout_items,
    select_nu,
)
from llm_kt.ingest import MODELS
from llm_kt.static_irt import load_item_params

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "data" / "processed" / "fluid_responses.parquet"
DEFAULT_OUT = REPO_ROOT / "docs" / "results" / "heldout"
DEFAULT_TRANSFER = ["arc_challenge", "hellaswag", "winogrande"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--model", default="olmo2-7b", choices=MODELS)
    parser.add_argument("--holdout-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nu", nargs="+", type=float, default=[0.0, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--time-col", default="checkpoint_idx",
                        choices=["checkpoint_idx", "step", "tokens_seen_b"])
    parser.add_argument("--transfer-benchmarks", nargs="+", default=DEFAULT_TRANSFER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    responses = pd.read_parquet(args.inp)
    responses = responses[responses["model"] == args.model]
    item_params = load_item_params()
    item_scores, fit_ids, heldout_ids = evaluate_heldout_items(
        responses,
        item_params,
        model=args.model,
        nu_values=args.nu,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        time_col=args.time_col,
    )
    nu = select_nu(item_scores[item_scores["block"] == "val"])
    transfer_scores = evaluate_benchmark_transfer(
        responses,
        item_params,
        model=args.model,
        fit_item_ids=fit_ids,
        nu=nu,
        benchmarks=args.transfer_benchmarks,
        time_col=args.time_col,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    item_path = args.out_dir / "within_item_scores.csv"
    transfer_path = args.out_dir / "cross_benchmark_scores.csv"
    item_scores.to_csv(item_path, index=False)
    transfer_scores.to_csv(transfer_path, index=False)

    pd.set_option("display.width", 180)
    print(f"Fit {len(fit_ids):,} MMLU items; held out {len(heldout_ids):,}.")
    print(f"Selected T-SKIRT nu={nu:g} on validation-block item-weighted log-loss.")
    print("\nWithin-MMLU held-out item scores, by block (lower log-loss is better):")
    print(summarize(
        item_scores[item_scores["predictor"].isin(
            ["skill_mean", "global_mean", f"tskirt_nu{nu:g}_{args.time_col}"]
        )],
        by=["block", "predictor"],
    ).set_index(["block", "predictor"])[["n", "log_loss", "brier", "accuracy", "auc"]]
        .round(4).to_string())
    print("\nCross-benchmark transfer scores, by block:")
    print(summarize(
        transfer_scores,
        by=["benchmark", "block", "predictor"],
    ).set_index(["benchmark", "block", "predictor"])[
        ["n", "log_loss", "brier", "accuracy", "auc"]
    ].round(4).to_string())
    print(f"\nWrote {item_path} and {transfer_path}")


if __name__ == "__main__":
    main()
