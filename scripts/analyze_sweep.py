#!/usr/bin/env python
"""Select a recalibrated T-SKIRT (tau, tau_a) on the validation block, then report test.

Loads models_mmlu_recal_a_h1.parquet (item_mean + tau=2/tau_a=0.5) and the
models_sweep_taua*_h1.parquet grids. Selection uses val log-loss only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from llm_kt.evaluate import assign_blocks, paired_difference, summarize

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
PAT = re.compile(r"^tskirt_nu1_checkpoint_idx_recal_tau([\d.]+)_taua([\d.]+)$")


def load() -> pd.DataFrame:
    frames = [pd.read_parquet(PROC / "models_mmlu_recal_a_h1.parquet")]
    frames += [pd.read_parquet(p) for p in sorted(PROC.glob("models_sweep_taua*_h1.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["model"] == "olmo2-7b") & (df["horizon"] == 1)]
    df = df.drop_duplicates(["predictor", "model", "origin_idx", "target_idx", "skill"])
    ranges = assign_blocks(94)
    df["block"] = "train"
    for name, (lo, hi) in ranges.items():
        df.loc[df["target_idx"].between(lo, hi), "block"] = name
    return df


def main() -> None:
    df = load()
    print("blocks:", assign_blocks(94))
    grid = {}
    for p in df["predictor"].unique():
        m = PAT.match(p)
        if m:
            grid[p] = (float(m.group(1)), float(m.group(2)))

    # ---- selection: validation block only ----
    val = df[df["block"] == "val"]
    sv = summarize(val, by=["predictor"], n_boot=300).set_index("predictor")
    print("\nItem_mean (val):", sv.loc["item_mean", ["log_loss", "log_loss_se", "auc", "auc_se"]].round(4).to_dict())
    rows = [{"tau": t, "tau_a": a, **sv.loc[p, ["log_loss", "log_loss_se", "auc", "auc_se"]].to_dict()}
            for p, (t, a) in grid.items()]
    g = pd.DataFrame(rows).sort_values(["tau", "tau_a"])
    print("\n(a) Val-block grid")
    print(g.round(4).to_string(index=False))
    print("\nVal log-loss (rows tau, cols tau_a)")
    print(g.pivot(index="tau", columns="tau_a", values="log_loss").round(4).to_string())
    print("\nVal AUC (rows tau, cols tau_a)")
    print(g.pivot(index="tau", columns="tau_a", values="auc").round(4).to_string())
    best = min(grid, key=lambda p: sv.loc[p, "log_loss"])
    print(f"\n(b) Val-selected: {best}  tau={grid[best][0]:g}, tau_a={grid[best][1]:g}")

    # ---- only now look at test ----
    test = df[df["block"] == "test"]
    st = summarize(test[test["predictor"].isin([best, "item_mean"])], by=["predictor"], n_boot=300)
    print("\n(c) Test block")
    print(st.set_index("predictor")[["n", "log_loss", "log_loss_se", "auc", "auc_se"]].round(4).to_string())
    print("\n(d) Paired difference on test (selected minus item_mean), n_boot=1000")
    for metric in ("log_loss", "auc"):
        d = paired_difference(test, best, "item_mean", by=[], metric=metric, n_boot=1000)
        print(metric, d.round(4).to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()
