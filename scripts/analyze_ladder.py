#!/usr/bin/env python
"""The baseline ladder: global mean -> skill mean -> item mean, from every angle.

Reads baseline_results_h{1,5,10,20}.parquet and fluid_responses.parquet and writes:

  data/processed/ladder_table.csv            pooled ladder per (model, benchmark, horizon)
  data/processed/ladder_over_training.csv    MMLU, h=1, per (model, target checkpoint)
  data/processed/variance_decomposition.csv  MMLU, per (model, checkpoint)
  docs/figures/ladder_over_training.png
  docs/figures/variance_decomposition.png

and prints a summary. Definitions are in src/llm_kt/ladder.py.

Usage:
    python scripts/analyze_ladder.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from llm_kt.ladder import (
    ALL,
    LADDER,
    gap_trend,
    ladder_over_training,
    ladder_table,
    skill_to_benchmark,
    variance_decomposition,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
FIG_DIR = REPO_ROOT / "docs" / "figures"
HORIZONS = (1, 5, 10, 20)
FOCUS_MODEL = "olmo2-7b"

# Style constants and model order follow scripts/plot_skill_curves.py.
MODEL_ORDER = ["pythia-3b", "pythia-7b", "amber-7b", "olmo1-7b", "olmo2-7b", "k2-65b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Fixed categorical slots, in ladder order (validated: CVD and normal-vision separation pass;
# the third slot is low-contrast on the surface, so every figure carries a legend).
PRED_COLOR = {"global_mean": "#2a78d6", "skill_mean": "#eb6834", "item_mean": "#1baf7a"}
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
X_LABEL = "normalized checkpoint index (index / last index; not training progress)"


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)


def _models(df: pd.DataFrame) -> list[str]:
    present = set(df["model"])
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def _panel_grid(n: int) -> tuple[plt.Figure, list[plt.Axes]]:
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows), sharex=True, sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    # one shared x-label: per-panel labels collide at this width
    fig.supxlabel(X_LABEL, fontsize=8, color=INK_SECONDARY)
    return fig, flat[:n]


def _header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, y=0.985, ha="left")
    fig.text(0.01, 0.93, subtitle, fontsize=8, color=INK_SECONDARY)


def plot_ladder_over_training(ot: pd.DataFrame, out: Path) -> None:
    models = _models(ot)
    fig, axes = _panel_grid(len(models))
    for i, (ax, model) in enumerate(zip(axes, models)):
        g = ot[ot["model"] == model].sort_values("target_idx")
        style_axes(ax)
        for pred in LADDER:
            ax.plot(
                g["target_norm"], g[f"log_loss_{pred}"], color=PRED_COLOR[pred], linewidth=2,
                label=pred,
            )
        ax.set_title(f"{model}  ({len(g)} targets)", fontsize=9, color=INK, loc="left")
        if i % 3 == 0:
            ax.set_ylabel("log-loss (lower is better)", fontsize=8, color=INK_SECONDARY)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper right")
    _header(
        fig,
        "Baseline ladder over training: MMLU, next-checkpoint (h=1) log-loss",
        "Overall accuracy, subject accuracy and the item's own history as predictors of the "
        "next checkpoint; the gap between the upper two and item_mean is the item gap",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_variance_decomposition(vd: pd.DataFrame, out: Path) -> None:
    models = _models(vd)
    fig, axes = _panel_grid(len(models))
    for i, (ax, model) in enumerate(zip(axes, models)):
        g = vd[vd["model"] == model].sort_values("checkpoint_idx")
        style_axes(ax)
        ax.set_ylim(0, 1)
        ax.plot(
            g["checkpoint_norm"], g["item_mean_r2"], color=SERIES_1, linewidth=2,
            label="item running-mean R² (past checkpoints only)",
        )
        ax.plot(
            g["checkpoint_norm"], g["between_frac"], color=SERIES_2, linewidth=2,
            label="between-skill fraction (in-sample)",
        )
        ax.set_title(f"{model}  ({len(g)} checkpoints)", fontsize=9, color=INK, loc="left")
        if i % 3 == 0:
            ax.set_ylabel("share of correctness variance", fontsize=8, color=INK_SECONDARY)
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK_SECONDARY, loc="center right")
    _header(
        fig,
        "Where the variance in MMLU correctness lives, per checkpoint",
        "Between-skill fraction = Var(E[y | subject]) / Var(y); item R² = squared correlation "
        "of y with the item's mean over earlier checkpoints",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def load_results(horizons: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for h in horizons:
        path = PROC / f"baseline_results_h{h}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/run_baselines.py --horizon {h}")
        df = pd.read_parquet(path, columns=["predictor", "model", "origin_idx", "target_idx",
                                            "skill", "n", "log_loss", "brier", "accuracy",
                                            "auc", "horizon"])
        frames.append(df[df["predictor"].isin(LADDER)])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    args = parser.parse_args()

    responses = pd.read_parquet(
        PROC / "fluid_responses.parquet",
        columns=["model", "checkpoint_idx", "benchmark", "skill", "item_id", "correct"],
    )
    skill_bench = skill_to_benchmark(responses)
    results = load_results(tuple(args.horizons))

    table = ladder_table(results, skill_bench)
    table.to_csv(PROC / "ladder_table.csv", index=False)

    mmlu_skills = set(skill_bench[skill_bench == "mmlu"].index)
    h1_mmlu = results[(results["horizon"] == 1) & results["skill"].isin(mmlu_skills)]
    ot = ladder_over_training(h1_mmlu)
    ot.to_csv(PROC / "ladder_over_training.csv", index=False)

    vd = variance_decomposition(responses[responses["benchmark"] == "mmlu"])
    vd.to_csv(PROC / "variance_decomposition.csv", index=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_ladder_over_training(ot, FIG_DIR / "ladder_over_training.png")
    plot_variance_decomposition(vd, FIG_DIR / "variance_decomposition.png")
    for name in ("ladder_table.csv", "ladder_over_training.csv", "variance_decomposition.csv"):
        print(f"Wrote {PROC / name}")
    print(f"Wrote {FIG_DIR / 'ladder_over_training.png'}")
    print(f"Wrote {FIG_DIR / 'variance_decomposition.png'}\n")

    pd.set_option("display.width", 200)
    ll_cols = ["n", *[f"log_loss_{p}" for p in LADDER], "skill_gap", "item_gap", "auc_item_mean"]
    idx = ["model", "benchmark", "horizon"]

    print("Ladder, all models and benchmarks pooled, per horizon")
    t = table[(table["model"] == ALL) & (table["benchmark"] == ALL)]
    print(t.set_index(idx)[ll_cols].round(4).to_string(), "\n")

    print(f"Ladder, {FOCUS_MODEL} MMLU, per horizon")
    t = table[(table["model"] == FOCUS_MODEL) & (table["benchmark"] == "mmlu")]
    print(t.set_index(idx)[ll_cols].round(4).to_string(), "\n")

    print(f"Ladder, {FOCUS_MODEL} per benchmark, h=1")
    t = table[(table["model"] == FOCUS_MODEL) & (table["horizon"] == 1)]
    print(t.set_index(idx)[ll_cols].round(4).to_string(), "\n")

    print("Ladder, MMLU, per model, h=1")
    t = table[(table["benchmark"] == "mmlu") & (table["horizon"] == 1)]
    print(t.set_index(idx)[ll_cols].round(4).to_string(), "\n")

    print("Item gap over training (MMLU, h=1): slope per full run, Spearman, thirds")
    print(gap_trend(ot, "item_gap").set_index("model").round(4).to_string(), "\n")
    print("Skill gap over training (MMLU, h=1)")
    print(gap_trend(ot, "skill_gap").set_index("model").round(4).to_string(), "\n")

    print("Variance decomposition (MMLU), per model; item R^2 excludes checkpoint 0")
    vs = vd.groupby("model").agg(
        between_frac_min=("between_frac", "min"),
        between_frac_median=("between_frac", "median"),
        between_frac_max=("between_frac", "max"),
        between_frac_last=("between_frac", "last"),
        item_r2_min=("item_mean_r2", "min"),
        item_r2_median=("item_mean_r2", "median"),
        item_r2_max=("item_mean_r2", "max"),
        item_r2_last=("item_mean_r2", "last"),
    )
    print(vs.round(4).to_string())


if __name__ == "__main__":
    main()
