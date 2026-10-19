#!/usr/bin/env python
"""Compare item_mean and recalibrated T-SKIRT as the forecast horizon grows (OLMo-2 MMLU).

Usage:
    python scripts/analyze_horizons.py

Reads models_mmlu_recal_a_h1.parquet (h=1), models_horizon_h{5,10,20}.parquet and,
when present, models_horizon_nu2_h{10,20}.parquet. Blocks come from assign_blocks(94)
on target_idx, so a row's block depends only on its target checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_kt.evaluate import _block_of, assign_blocks, paired_difference, summarize

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
ITEM_MEAN = "item_mean"
TSKIRT = "tskirt_nu1_checkpoint_idx_recal_tau2_taua0.5"
TSKIRT_NU2 = "tskirt_nu2_checkpoint_idx_recal_tau2_taua0.5"
HORIZONS = (1, 5, 10, 20)
N_CHECKPOINTS = 94


def load() -> pd.DataFrame:
    frames = []
    h1 = pd.read_parquet(PROC / "models_mmlu_recal_a_h1.parquet")
    frames.append(h1[h1["predictor"].isin([ITEM_MEAN, TSKIRT])])
    for h in HORIZONS[1:]:
        frames.append(pd.read_parquet(PROC / f"models_horizon_h{h}.parquet"))
        nu2 = PROC / f"models_horizon_nu2_h{h}.parquet"
        if nu2.exists():
            frames.append(pd.read_parquet(nu2))
    df = pd.concat(frames, ignore_index=True)
    ranges = assign_blocks(N_CHECKPOINTS)
    df["block"] = df["target_idx"].map(lambda i: _block_of(int(i), ranges))
    return df


def main() -> None:
    df = load()
    pd.set_option("display.width", 200)
    cols = ["log_loss", "log_loss_se", "auc", "auc_se", "n"]

    rows = []
    for (h, scope), g in [((h, "pooled"), df[df["horizon"] == h]) for h in HORIZONS] + [
        ((h, "test"), df[(df["horizon"] == h) & (df["block"] == "test")]) for h in HORIZONS
    ]:
        s = summarize(g, by=["predictor"], n_boot=300)
        s.insert(0, "scope", scope)
        s.insert(0, "horizon", h)
        rows.append(s)
    table = pd.concat(rows).sort_values(["horizon", "scope", "predictor"])
    print("Pooled and test-block scores (cluster-bootstrap SEs over skills, n_boot=300)")
    print(table.set_index(["horizon", "scope", "predictor"])[cols].round(4).to_string())

    print("\nPaired difference on test block, T-SKIRT minus item_mean (n_boot=1000)")
    diffs = []
    for h in HORIZONS:
        test = df[(df["horizon"] == h) & (df["block"] == "test")]
        for pred in (TSKIRT, TSKIRT_NU2):
            if pred not in set(test["predictor"]):
                continue
            for metric in ("log_loss", "auc"):
                d = paired_difference(test, pred, ITEM_MEAN, by=[], metric=metric, n_boot=1000)
                d.insert(0, "metric", metric)
                d.insert(0, "predictor", pred)
                d.insert(0, "horizon", h)
                diffs.append(d)
    print(pd.concat(diffs).set_index(["horizon", "predictor", "metric"])[
        ["diff", "se", "ci_low", "ci_high", "n_slices"]].round(4).to_string())

    print("\nDecay: test-block log-loss at h minus h=1 (nu2 compared with nu1's h=1)")
    test_ll = table[table["scope"] == "test"].pivot(index="predictor", columns="horizon", values="log_loss")
    base = test_ll[1].copy()
    base[TSKIRT_NU2] = base.get(TSKIRT)
    print(test_ll.sub(base, axis=0).drop(columns=1).round(4).to_string())

    print("\nTest-block origins/targets per horizon")
    t = df[df["block"] == "test"].groupby(["horizon", "predictor"]).agg(
        targets=("target_idx", "nunique"), first_origin=("origin_idx", "min"))
    print(t.to_string())


if __name__ == "__main__":
    main()
