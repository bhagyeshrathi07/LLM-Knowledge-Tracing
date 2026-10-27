#!/usr/bin/env python
"""Pick a DriftIRT setting on the validation block and compare it with item_mean.

Usage:
    python scripts/analyze_drift_irt.py [--grid models_drift_grid_h1.parquet] \
        [--horizon-tag drift_best]

Reads the h=1 grid written by run_models.py (DriftIRT settings plus item_mean on identical
rows), ranks DriftIRT settings by validation-block log-loss, and for the best one reports
pooled, validation and test log-loss and AUC with cluster-bootstrap SEs over skills
(n_boot=200), plus paired differences (DriftIRT minus item_mean) with 95% intervals. When
models_<horizon-tag>_h{5,20}.parquet exist, it adds test-block paired differences there.
Only the validation block is used for selection; test numbers are reported, never ranked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llm_kt.evaluate import paired_difference, summarize

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
ITEM_MEAN = "item_mean"
N_BOOT = 200
SCOPES = ("pooled", "val", "test")


def _scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    return df if scope == "pooled" else df[df["block"] == scope]


def compare(df: pd.DataFrame, best: str, scopes=SCOPES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-scope summary (with SEs) and paired differences for `best` against item_mean."""
    df = df[df["predictor"].isin([best, ITEM_MEAN])]
    tables, diffs = [], []
    for scope in scopes:
        g = _scope(df, scope)
        s = summarize(g, by=["predictor"], n_boot=N_BOOT, cluster="skill")
        s.insert(0, "scope", scope)
        tables.append(s)
        for metric in ("log_loss", "auc"):
            d = paired_difference(g, best, ITEM_MEAN, by=[], metric=metric, n_boot=N_BOOT)
            d.insert(0, "metric", metric)
            d.insert(0, "scope", scope)
            diffs.append(d)
    return pd.concat(tables, ignore_index=True), pd.concat(diffs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="models_drift_grid_h1.parquet")
    parser.add_argument("--horizon-tag", default="drift_best")
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    df = pd.read_parquet(PROC / args.grid)
    val = summarize(df[df["block"] == "val"], by=["predictor"]).sort_values("log_loss")
    print("Validation block, every predictor (lower log_loss better)")
    print(val.set_index("predictor")[["n", "log_loss", "brier", "accuracy", "auc"]].round(4).to_string())

    drift = val[val["predictor"].str.startswith("drift_irt")]
    best = drift.iloc[0]["predictor"]
    print(f"\nBest DriftIRT on validation log-loss: {best}")

    cols = ["n", "log_loss", "log_loss_se", "auc", "auc_se"]
    table, diffs = compare(df, best)
    print(f"\nh=1 scores, cluster-bootstrap SEs over skills (n_boot={N_BOOT})")
    print(table.set_index(["scope", "predictor"])[cols].round(4).to_string())
    print(f"\nh=1 paired difference, {best} minus item_mean")
    print(diffs.set_index(["scope", "metric"])[["diff", "se", "ci_low", "ci_high", "n_slices"]]
          .round(4).to_string())

    for h in (5, 20):
        path = PROC / f"models_{args.horizon_tag}_h{h}.parquet"
        if not path.exists():
            print(f"\n(no {path.name}; skipping h={h})")
            continue
        hdf = pd.read_parquet(path)
        table, diffs = compare(hdf, best, scopes=("pooled", "test"))
        print(f"\nh={h} scores")
        print(table.set_index(["scope", "predictor"])[cols].round(4).to_string())
        print(f"\nh={h} paired difference, {best} minus item_mean")
        print(diffs.set_index(["scope", "metric"])[["diff", "se", "ci_low", "ci_high", "n_slices"]]
              .round(4).to_string())


if __name__ == "__main__":
    main()
