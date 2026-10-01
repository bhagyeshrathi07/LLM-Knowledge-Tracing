#!/usr/bin/env python
"""Plot per-skill accuracy curves from the canonical response table.

Produces two figures under docs/figures/:

  skill_curves_<model>.png   small multiples: one panel per selected MMLU subject,
                             accuracy vs checkpoint with a 95% Wilson band
  skill_<skill>_models.png   one skill, one line per model, direct-labelled

Usage:
    python scripts/plot_skill_curves.py [--model olmo2-7b] [--skill high_school_biology]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from llm_kt.curves import skill_accuracy

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "data" / "processed" / "fluid_responses.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "figures"

# Eight MMLU subjects spanning easy-to-hard and STEM/humanities, for the small multiples.
DEFAULT_SUBJECTS = [
    "elementary_mathematics",
    "high_school_mathematics",
    "college_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_physics",
    "world_religions",
    "professional_law",
]

# Fixed model order and fixed categorical slots (dataviz reference palette, light mode).
# Order never changes when a subset is plotted, so a model keeps its colour across figures.
MODEL_ORDER = ["pythia-3b", "pythia-7b", "amber-7b", "olmo1-7b", "olmo2-7b", "k2-65b"]
MODEL_COLOR = {
    "pythia-3b": "#2a78d6",
    "pythia-7b": "#eb6834",
    "amber-7b": "#1baf7a",
    "olmo1-7b": "#eda100",
    "olmo2-7b": "#e87ba4",
    "k2-65b": "#008300",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"


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
    ax.set_ylim(0, 1)


def plot_small_multiples(curves: pd.DataFrame, model: str, subjects: list[str], out: Path) -> None:
    sub = curves[(curves.model == model) & (curves.benchmark == "mmlu")]
    ncols = 4
    nrows = (len(subjects) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, skill in zip(axes.flat, subjects):
        s = sub[sub.skill == skill].sort_values("checkpoint_idx")
        style_axes(ax)
        ax.fill_between(s.checkpoint_idx, s.ci_lo, s.ci_hi, color=SERIES_1, alpha=0.15, linewidth=0)
        ax.plot(s.checkpoint_idx, s.acc, color=SERIES_1, linewidth=2)
        n = int(s.n.iloc[0])
        ax.set_title(f"{skill.replace('_', ' ')}  (n={n})", fontsize=9, color=INK, loc="left")
        ax.axhline(0.25, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))

    for ax in axes.flat[len(subjects) :]:
        ax.set_visible(False)
    for ax in axes[-1] if nrows > 1 else axes:
        ax.set_xlabel("checkpoint index", fontsize=8, color=INK_SECONDARY)
    for ax in axes[:, 0] if nrows > 1 else [axes[0]]:
        ax.set_ylabel("accuracy", fontsize=8, color=INK_SECONDARY)

    fig.suptitle(
        f"{model}: MMLU accuracy per subject across pretraining checkpoints",
        fontsize=11,
        color=INK,
        x=0.01,
        y=0.985,
        ha="left",
    )
    fig.text(
        0.01,
        0.935,
        "Line = raw accuracy, band = 95% Wilson interval, dotted = 4-option chance (0.25)",
        fontsize=8,
        color=INK_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_models_for_skill(curves: pd.DataFrame, skill: str, out: Path) -> None:
    sub = curves[curves.skill == skill]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    for model in MODEL_ORDER:
        s = sub[sub.model == model].sort_values("checkpoint_idx")
        if s.empty:
            continue
        # x = normalized checkpoint index, so runs with different checkpoint counts share an axis
        x = s.checkpoint_idx / s.checkpoint_idx.max()
        # six series: legend carries identity; direct labels would collide at the right edge
        ax.plot(x, s.acc, color=MODEL_COLOR[model], linewidth=2, label=model)

    ax.axhline(0.25, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
    ax.set_xlim(0, 1)
    ax.set_xlabel("normalized checkpoint index (index / last index; not training progress)", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("accuracy", fontsize=8, color=INK_SECONDARY)
    n = int(sub.n.iloc[0])
    ax.set_title(
        f"MMLU {skill.replace('_', ' ')} (n={n}): accuracy across pretraining, six models",
        fontsize=11,
        color=INK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default="olmo2-7b", choices=MODEL_ORDER)
    parser.add_argument("--skill", default="high_school_biology")
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    args = parser.parse_args()

    responses = pd.read_parquet(args.inp)
    curves = skill_accuracy(responses)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curves_path = args.out_dir.parent.parent / "data" / "processed" / "skill_curves.parquet"
    curves.to_parquet(curves_path, index=False)
    print(f"Wrote {len(curves):,} curve rows to {curves_path}")

    p1 = args.out_dir / f"skill_curves_{args.model}.png"
    plot_small_multiples(curves, args.model, args.subjects, p1)
    print(f"Wrote {p1}")

    p2 = args.out_dir / f"skill_{args.skill}_models.png"
    plot_models_for_skill(curves, args.skill, p2)
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
