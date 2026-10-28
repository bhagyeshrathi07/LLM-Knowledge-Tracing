#!/usr/bin/env python
"""Pick BKT settings on the validation block, compare them with item_mean, plot mastery.

Usage:
    python scripts/analyze_bkt.py [--grid models_bkt_grid_h1.parquet] [--horizon-tag bkt_best]
                                  [--mastery] [--kappa K --rho R]

Reads the h=1 grid written by run_models.py (BKT settings plus item_mean and skill_mean on
identical rows). Three predictors are singled out: the best skill-anchored setting and the best
Fluid-anchored setting by validation-block log-loss (kappa below the pin limit), and the
per-skill limit (kappa = 1e6, skill anchor), which is classic BKT with one guess/slip pair per
skill. For each, pooled / validation / test log-loss and AUC with skill-clustered bootstrap SEs
(n_boot=200) and paired differences against item_mean with 95% intervals; the per-skill limit
is also paired against skill_mean. When models_<horizon-tag>_h{5,20}.parquet exist, the same
comparison is repeated on their test block. Only the validation block is used for selection.

--mastery fits the selected skill-anchored setting once on the full OLMo-2 MMLU history and
writes data/processed/bkt_mastery_olmo2-7b.csv, bkt_skill_params_olmo2-7b.csv and the figure
docs/figures/bkt_mastery_olmo2-7b.png (per-subject P(learned) over training; learn vs forget).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from llm_kt.bkt import BKT, KAPPA_LIMIT
from llm_kt.evaluate import paired_difference, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
FIG_DIR = REPO_ROOT / "docs" / "figures"
FOCUS_MODEL = "olmo2-7b"
ITEM_MEAN = "item_mean"
SKILL_MEAN = "skill_mean"
LIMIT = f"bkt_k{1e6:g}_r1_skill"
N_BOOT = 200
SCOPES = ("pooled", "val", "test")
NAME_RE = re.compile(r"^bkt_k([^_]+)_r([^_]+)_(skill|fluid)")

# Style constants follow scripts/analyze_ladder.py (validated three-slot categorical palette).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
HIGHLIGHT = ["#2a78d6", "#eb6834", "#1baf7a"]
HIGHLIGHT_SKILLS = ["high_school_biology", "moral_scenarios", "professional_law"]


def parse_name(name: str) -> tuple[float, float, str]:
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"not a BKT predictor name: {name!r}")
    return float(m.group(1)), float(m.group(2)), m.group(3)


def load_grid(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; produce it with\n"
            "  python scripts/run_models.py --models olmo2-7b --benchmarks mmlu "
            "--predictors bkt item_mean skill_mean --kappa 1 5 20 100 1e6 --rho 1.0 0.9 0.7 "
            "--anchor skill fluid --horizon 1 --tag bkt_grid"
        )
    return pd.read_parquet(path)


def select(val: pd.DataFrame) -> tuple[str, str | None]:
    """(best skill-anchored, best Fluid-anchored or None) by validation log-loss, pins excluded."""
    bkt = val[val["predictor"].str.startswith("bkt_")].copy()
    parsed = bkt["predictor"].map(parse_name)
    bkt["kappa"] = [p[0] for p in parsed]
    bkt["anchor"] = [p[2] for p in parsed]
    free = bkt[bkt["kappa"] < KAPPA_LIMIT].sort_values("log_loss")
    skill = free[free["anchor"] == "skill"]
    fluid = free[free["anchor"] == "fluid"]
    if skill.empty:
        raise ValueError("no skill-anchored BKT setting with kappa below the pin limit in the grid")
    return skill.iloc[0]["predictor"], (None if fluid.empty else fluid.iloc[0]["predictor"])


def _scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    return df if scope == "pooled" else df[df["block"] == scope]


def compare(df: pd.DataFrame, best: str, against: str = ITEM_MEAN, scopes=SCOPES):
    """Per-scope summary (with SEs) for `best` and `against`, and paired differences best - against."""
    df = df[df["predictor"].isin([best, against])]
    tables, diffs = [], []
    for scope in scopes:
        g = _scope(df, scope)
        s = summarize(g, by=["predictor"], n_boot=N_BOOT, cluster="skill")
        s.insert(0, "scope", scope)
        tables.append(s)
        for metric in ("log_loss", "auc"):
            d = paired_difference(g, best, against, by=[], metric=metric, n_boot=N_BOOT)
            d.insert(0, "metric", metric)
            d.insert(0, "scope", scope)
            diffs.append(d)
    return pd.concat(tables, ignore_index=True), pd.concat(diffs, ignore_index=True)


def report(df: pd.DataFrame, best: str, against: str, label: str, scopes=SCOPES) -> None:
    cols = ["n", "log_loss", "log_loss_se", "auc", "auc_se"]
    table, diffs = compare(df, best, against, scopes)
    print(f"\n{label}: scores, cluster-bootstrap SEs over skills (n_boot={N_BOOT})")
    print(table.set_index(["scope", "predictor"])[cols].round(4).to_string())
    print(f"\n{label}: paired difference, {best} minus {against}")
    print(diffs.set_index(["scope", "metric"])[["diff", "se", "ci_low", "ci_high", "n_slices"]]
          .round(4).to_string())


# -- mastery figure ------------------------------------------------------------------------


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


def plot_mastery(mastery: pd.DataFrame, params: pd.DataFrame, out: Path, title_setting: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.55})
    fig.patch.set_facecolor(SURFACE)
    last = mastery["checkpoint_idx"].max()
    for skill, g in mastery.groupby("skill"):
        if skill in HIGHLIGHT_SKILLS:
            continue
        ax1.plot(g["checkpoint_idx"] / last, g["p_learned"], color=MUTED, alpha=0.35, lw=1.0)
    for color, skill in zip(HIGHLIGHT, HIGHLIGHT_SKILLS):
        g = mastery[mastery["skill"] == skill]
        if g.empty:
            continue
        ax1.plot(g["checkpoint_idx"] / last, g["p_learned"], color=color, lw=2.0, label=skill)
    style_axes(ax1)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.04, 1.04)
    ax1.set_xlabel("normalized checkpoint index (index / last index)", fontsize=8, color=INK_SECONDARY)
    ax1.set_ylabel("P(subject learned), filtered", fontsize=8, color=INK_SECONDARY)
    ax1.set_title("Mastery per MMLU subject (57 curves; three named)", fontsize=9, color=INK, loc="left")
    # the curves fill the panel, so the legend sits outside it (identity is never color-alone)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0),
               borderaxespad=0.0)

    ax2.scatter(params["learn"], params["forget"], s=18, color=MUTED, alpha=0.6, linewidths=0)
    for color, skill in zip(HIGHLIGHT, HIGHLIGHT_SKILLS):
        row = params[params["skill"] == skill]
        if row.empty:
            continue
        ax2.scatter(row["learn"], row["forget"], s=42, color=color, edgecolor=SURFACE, linewidths=1.5)
        ax2.annotate(skill, (row["learn"].iloc[0], row["forget"].iloc[0]), xytext=(5, 3),
                     textcoords="offset points", fontsize=7.5, color=INK_SECONDARY)
    style_axes(ax2)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("learn rate per checkpoint", fontsize=8, color=INK_SECONDARY)
    ax2.set_ylabel("forget rate per checkpoint", fontsize=8, color=INK_SECONDARY)
    ax2.set_title("Learn vs forget, one point per subject", fontsize=9, color=INK, loc="left")

    fig.suptitle(f"BKT on OLMo-2 MMLU ({title_setting})", fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.91, "Fitted once on all 94 checkpoints; per-item guess and slip shrunk toward "
             "the subject's values.", fontsize=8, color=INK_SECONDARY)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.14, wspace=0.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fit_mastery(kappa: float, rho: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    responses = pd.read_parquet(PROC / "fluid_responses.parquet")
    responses = responses[(responses["model"] == FOCUS_MODEL) & (responses["benchmark"] == "mmlu")]
    m = BKT(kappa=kappa, rho=rho)
    m.fit(responses)
    return m.mastery(), m.skill_params()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="models_bkt_grid_h1.parquet")
    parser.add_argument("--horizon-tag", default="bkt_best")
    parser.add_argument("--mastery", action="store_true", help="fit the selected setting on the full history and plot")
    parser.add_argument("--kappa", type=float, default=None, help="override kappa for the mastery fit")
    parser.add_argument("--rho", type=float, default=None, help="override rho for the mastery fit")
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    df = load_grid(PROC / args.grid)
    val = summarize(df[df["block"] == "val"], by=["predictor"]).sort_values("log_loss")
    print("Validation block, every predictor (lower log_loss better)")
    print(val.set_index("predictor")[["n", "log_loss", "brier", "accuracy", "auc"]].round(4).to_string())

    best_skill, best_fluid = select(val)
    print(f"\nSelected on validation log-loss: skill anchor {best_skill}; fluid anchor {best_fluid}")
    chosen = [(best_skill, "best skill-anchored"), (best_fluid, "best Fluid-anchored"), (LIMIT, "per-skill limit")]
    for name, label in chosen:
        if name is not None and name in set(df["predictor"]):
            report(df, name, ITEM_MEAN, f"h=1, {label} ({name}) vs item_mean")
    if LIMIT in set(df["predictor"]) and SKILL_MEAN in set(df["predictor"]):
        report(df, LIMIT, SKILL_MEAN, f"h=1, per-skill limit ({LIMIT}) vs skill_mean")

    for h in (5, 20):
        path = PROC / f"models_{args.horizon_tag}_h{h}.parquet"
        if not path.exists():
            print(f"\n(no {path.name}; skipping h={h})")
            continue
        hdf = pd.read_parquet(path)
        for name, label in chosen:
            if name is not None and name in set(hdf["predictor"]):
                report(hdf, name, ITEM_MEAN, f"h={h}, {label} ({name}) vs item_mean", scopes=("pooled", "test"))

    if args.mastery:
        kappa, rho, _ = parse_name(best_skill)
        kappa = args.kappa if args.kappa is not None else kappa
        rho = args.rho if args.rho is not None else rho
        mastery, params = fit_mastery(kappa, rho)
        mastery.to_csv(PROC / f"bkt_mastery_{FOCUS_MODEL}.csv", index=False)
        params.to_csv(PROC / f"bkt_skill_params_{FOCUS_MODEL}.csv", index=False)
        out = FIG_DIR / f"bkt_mastery_{FOCUS_MODEL}.png"
        plot_mastery(mastery, params, out, f"kappa={kappa:g}, rho={rho:g}")
        print(f"\nWrote {out} and the mastery / skill-parameter CSVs under {PROC}")
        print(params.sort_values("learn", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
