#!/usr/bin/env python
"""Pick BKT settings on the validation block, compare them with item_mean, plot mastery.

Usage:
    python scripts/analyze_bkt.py [--grid models_bkt_grid_h1.parquet] [--horizon-tag bkt_best]
                                  [--ewma-tag bkt_ewma] [--mastery] [--kappa K --rho R]

Reads the h=1 grid written by run_models.py (BKT settings plus item_mean and skill_mean on
identical rows). Three predictors are singled out: the best skill-anchored setting and the best
Fluid-anchored setting by validation-block log-loss (kappa below the pin limit), and the
per-skill limit (kappa = 1e6, skill anchor), which is classic BKT with one guess/slip pair per
skill. For each, pooled / validation / test log-loss and AUC with skill-clustered bootstrap SEs
(n_boot=200) and paired differences against item_mean with 95% intervals; the per-skill limit
is also paired against skill_mean. When models_<horizon-tag>_h{5,20}.parquet exist, the same
comparison is repeated on their test block. Only the validation block is used for selection.
When models_<ewma-tag>_h{1,5,20}.parquet exist (run_models.py --predictors item_mean_ewma
--rho R), the best skill-anchored setting is also paired against the recency-weighted item
mean with the same rho: the control that says whether BKT's gain over item_mean is more than
recency weighting alone. Three further comparisons go to extra_scores.csv and
extra_paired_differences.csv: the Fluid anchor vs the control, the Fluid anchor vs the skill
anchor, and the best skill-anchored setting with rho = 1 (recency off) vs item_mean.

Figures, all under docs/figures/: bkt_validation_grid.png (validation log-loss against kappa per
rho and anchor, with the references) and, when the horizon files exist, bkt_horizons.png
(test-block log-loss and AUC against horizon). --mastery fits the selected skill-anchored
setting and the per-skill limit (kappa 1e6, rho 1) once each on the full OLMo-2 MMLU history and
writes the mastery and skill-parameter CSVs and bkt_mastery_olmo2-7b.png /
bkt_mastery_olmo2-7b_k1e+06_r1.png (per-subject P(learned) over training; learn vs forget). With
--kappa / --rho only that setting is fitted and its files carry a _k<kappa>_r<rho> suffix.
--replot-mastery redraws the two mastery figures from the stored CSVs without refitting (seconds).

Everything printed is also stored under --out-dir (default docs/results/bkt, committed, unlike
data/processed): validation_table.csv, h{1,5,20}_scores.csv, h{1,5,20}_paired_differences.csv,
bkt_mastery_olmo2-7b.csv, bkt_skill_params_olmo2-7b.csv and analysis.txt (the console report).
"""

from __future__ import annotations

import argparse
import re
import sys
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
RESULTS_DIR = REPO_ROOT / "docs" / "results" / "bkt"
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


def ewma_name(bkt_name: str) -> str:
    """The recency-weighted item mean with the same rho as a BKT setting (its control)."""
    _, rho, _ = parse_name(bkt_name)
    return f"item_mean_ewma_r{rho:g}"


def select_no_recency(val: pd.DataFrame) -> str | None:
    """Best skill-anchored setting with rho = 1 below the pin limit: BKT with recency off."""
    names = [n for n in val.sort_values("log_loss")["predictor"] if NAME_RE.match(n)]
    for n in names:
        kappa, rho, anchor = parse_name(n)
        if anchor == "skill" and rho == 1.0 and kappa < KAPPA_LIMIT:
            return n
    return None


def extra_pairs(best_skill: str, best_fluid: str | None, no_recency: str | None, control: str,
                present: set[str]) -> list[tuple[str, str, str]]:
    """(a, b, label) comparisons beyond the item_mean ones, restricted to predictors present.

    Fluid anchor vs the recency control and vs the selected skill anchor (does a borrowed
    prior add anything?), and the best no-recency skill setting vs item_mean (how much of the
    gain is recency?).
    """
    pairs = [
        (best_fluid, control, "best Fluid-anchored vs recency control"),
        (best_fluid, best_skill, "best Fluid-anchored vs best skill-anchored"),
        (no_recency, ITEM_MEAN, "best skill-anchored with rho = 1 vs item_mean"),
    ]
    return [(a, b, lab) for a, b, lab in pairs if a is not None and a in present and b in present]


def _with_control(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Append the item_mean_ewma rows from `path` when it exists; otherwise return df."""
    if not path.exists():
        print(f"(no {path.name}; recency control not compared)")
        return df
    extra = pd.read_parquet(path)
    extra = extra[~extra["predictor"].isin(set(df["predictor"]))]
    return pd.concat([df, extra], ignore_index=True)


def load_grid(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; produce it with\n"
            "  python scripts/run_models.py --models olmo2-7b --benchmarks mmlu "
            "--predictors bkt item_mean skill_mean --kappa 0.03 0.1 0.3 1 5 20 100 1e6 "
            "--rho 1.0 0.9 0.7 --anchor skill fluid --theta-lo -1 -2 --theta-hi 1 2 "
            "--horizon 1 --tag bkt_grid (chunk it over several processes; see README)"
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


def report(df: pd.DataFrame, best: str, against: str, label: str, scopes=SCOPES):
    """Print the comparison and return (scores, paired differences) tagged with the label."""
    cols = ["n", "log_loss", "log_loss_se", "auc", "auc_se"]
    table, diffs = compare(df, best, against, scopes)
    print(f"\n{label}: scores, cluster-bootstrap SEs over skills (n_boot={N_BOOT})")
    print(table.set_index(["scope", "predictor"])[cols].round(4).to_string())
    print(f"\n{label}: paired difference, {best} minus {against}")
    print(diffs.set_index(["scope", "metric"])[["diff", "se", "ci_low", "ci_high", "n_slices"]]
          .round(4).to_string())
    table.insert(0, "comparison", label)
    diffs.insert(0, "comparison", label)
    diffs.insert(1, "a", best)
    diffs.insert(2, "b", against)
    return table, diffs


class _Tee:
    """Write to stdout and a file at once, so the console report is stored verbatim."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w")
        self._out = sys.stdout

    def write(self, s: str) -> None:
        self._out.write(s)
        self._f.write(s)

    def flush(self) -> None:
        self._out.flush()
        self._f.flush()


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
    style_axes(ax2)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    lo, hi = ax2.get_xlim()
    ax2.set_xlim(lo, hi * 6)  # room for the labels to the right of the rightmost point
    # highlighted subjects can share a point (e.g. both at the cold-start fixed point); a label
    # whose point sits on an already-labelled one is stacked upward instead of overprinting
    placed: list[tuple[float, float]] = []
    for color, skill in zip(HIGHLIGHT, HIGHLIGHT_SKILLS):
        row = params[params["skill"] == skill]
        if row.empty:
            continue
        x, y = float(row["learn"].iloc[0]), float(row["forget"].iloc[0])
        ax2.scatter([x], [y], s=42, color=color, edgecolor=SURFACE, linewidths=1.5, zorder=4)
        px, py = ax2.transData.transform((x, y))
        level = sum(1 for qx, qy in placed if abs(px - qx) < 10 and abs(py - qy) < 10)
        placed.append((px, py))
        ax2.annotate(skill, (x, y), xytext=(7, 4 + 12 * level), textcoords="offset points",
                     fontsize=7.5, color=color, annotation_clip=False)
    ax2.set_xlabel("learn rate per checkpoint", fontsize=8, color=INK_SECONDARY)
    ax2.set_ylabel("forget rate per checkpoint", fontsize=8, color=INK_SECONDARY)
    ax2.set_title("Learn vs forget, one point per subject", fontsize=9, color=INK, loc="left")

    fig.suptitle(f"BKT on OLMo-2 MMLU ({title_setting})", fontsize=11, color=INK, x=0.01, ha="left")
    pinned = "per-skill limit" in title_setting
    fig.text(0.01, 0.91, "Fitted once on all 94 checkpoints; "
             + ("one guess/slip pair per subject, every question pinned to it (classic BKT)."
                if pinned else "per-item guess and slip shrunk toward the subject's values."),
             fontsize=8, color=INK_SECONDARY)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.14, wspace=0.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


RHO_COLORS = {1.0: HIGHLIGHT[0], 0.9: HIGHLIGHT[1], 0.7: HIGHLIGHT[2]}


def _bkt_rows(val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, ll in zip(val["predictor"], val["log_loss"]):
        if NAME_RE.match(name):
            kappa, rho, anchor = parse_name(name)
            rows.append({"predictor": name, "kappa": kappa, "rho": rho, "anchor": anchor,
                         "thetas": name.split(anchor, 1)[1], "log_loss": ll})
    return pd.DataFrame(rows)


def plot_validation_grid(val: pd.DataFrame, best_skill: str, best_fluid: str | None, out: Path) -> None:
    """Validation log-loss against kappa, one line per rho; skill anchor solid, Fluid anchor dashed.

    Only the Fluid theta pair of `best_fluid` is drawn. Item mean, the recency control and the
    per-skill limit are horizontal references. The two selected settings are ringed.
    """
    rows = _bkt_rows(val)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    fig.patch.set_facecolor(SURFACE)
    finite = rows[rows["kappa"] < KAPPA_LIMIT]
    fluid_thetas = best_fluid.split("fluid", 1)[1] if best_fluid else None
    for anchor, ls, label in (("skill", "-", "skill anchor"), ("fluid", "--", "Fluid anchor")):
        sub = finite[finite["anchor"] == anchor]
        if anchor == "fluid":
            sub = sub[sub["thetas"] == fluid_thetas] if fluid_thetas else sub.iloc[0:0]
        for rho in sorted(sub["rho"].unique(), reverse=True):
            g = sub[sub["rho"] == rho].sort_values("kappa")
            ax.plot(g["kappa"], g["log_loss"], ls, color=RHO_COLORS.get(rho, MUTED), lw=2.0,
                    marker="o", ms=4.5, label=f"{label}, rho {rho:g}")
    refs = []
    for name, text in ((ITEM_MEAN, "item mean (the bar)"),
                       (next((n for n in val["predictor"] if n.startswith("item_mean_ewma")), None),
                        "recency-weighted item mean"),
                       (LIMIT, "per-skill limit (kappa = 1e6)")):
        if name is not None and name in set(val["predictor"]):
            y = float(val.loc[val["predictor"] == name, "log_loss"].iloc[0])
            refs.append((y, text))
    for (y, text), ls in zip(refs, (":", "-.", (0, (6, 2, 1, 2, 1, 2)))):
        ax.axhline(y, color=INK_SECONDARY, lw=1.0, ls=ls, label=text)
    for name in (best_skill, best_fluid):
        if name is None:
            continue
        kappa, _, _ = parse_name(name)
        y = float(val.loc[val["predictor"] == name, "log_loss"].iloc[0])
        ax.scatter([kappa], [y], s=110, facecolor="none", edgecolor=INK, linewidths=1.4, zorder=5,
                   label="selected on validation" if name == best_skill else None)
    style_axes(ax)
    ax.set_xscale("log")
    ax.set_xlabel("kappa (pseudo-count pulling each question toward its anchor; log scale)",
                  fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("validation log-loss (lower is better)", fontsize=8, color=INK_SECONDARY)
    ax.set_title("BKT grid on OLMo-2 MMLU, validation block (checkpoints 56 to 74), horizon 1",
                 fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    thetas_text = "none"
    if fluid_thetas:
        m = re.match(r"_lo(-?[\d.]+)_hi(-?[\d.]+)", fluid_thetas)
        thetas_text = f"theta_lo {m.group(1)}, theta_hi {m.group(2)}" if m else fluid_thetas
    fig.text(0.01, 0.93, f"Fluid anchor shown for its best theta pair ({thetas_text}); "
             "recency (rho below 1) helps at every kappa, and the curve is flat below kappa 1.",
             fontsize=8, color=INK_SECONDARY)
    fig.subplots_adjust(left=0.08, right=0.70, top=0.84, bottom=0.14)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_horizons(scores: pd.DataFrame, best_skill: str, best_fluid: str | None, control: str,
                  out: Path) -> None:
    """Test-block log-loss and AUC against horizon for item mean, the control and the two BKTs.

    `scores` has one row per (h, scope, predictor) with log_loss, log_loss_se, auc, auc_se.
    """
    series = [(ITEM_MEAN, "item mean (the bar)", MUTED),
              (control, "recency-weighted item mean", HIGHLIGHT[0]),
              (best_skill, "BKT, skill anchor", HIGHLIGHT[1])]
    if best_fluid:
        series.append((best_fluid, "BKT, Fluid anchor", HIGHLIGHT[2]))
    test = scores[scores["scope"] == "test"].drop_duplicates(["h", "predictor"])
    hs = sorted(test["h"].unique())
    xpos = {h: i for i, h in enumerate(hs)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    fig.patch.set_facecolor(SURFACE)
    for ax, metric, ylabel in zip(axes, ("log_loss", "auc"),
                                  ("test-block log-loss (lower is better)", "test-block AUC (higher is better)")):
        for k, (name, label, color) in enumerate(series):
            g = test[test["predictor"] == name].sort_values("h")
            if g.empty:
                continue
            x = [xpos[h] + (k - 1.5) * 0.04 for h in g["h"]]  # small dodge so error bars stay visible
            ax.errorbar(x, g[metric], yerr=g.get(f"{metric}_se"), color=color, lw=2.0, marker="o", ms=5,
                        capsize=2.5, label=label)
        style_axes(ax)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([str(h) for h in hs])
        ax.set_xlim(-0.4, len(hs) - 0.6)
        ax.set_xlabel("horizon (checkpoints ahead)", fontsize=8, color=INK_SECONDARY)
        ax.set_ylabel(ylabel, fontsize=8, color=INK_SECONDARY)
    axes[0].set_title("Log-loss", fontsize=9, color=INK, loc="left")
    axes[1].set_title("AUC", fontsize=9, color=INK, loc="left")
    axes[1].legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.suptitle("BKT against the bar further ahead, OLMo-2 MMLU test block (checkpoints 75 to 93)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.90, "Error bars: bootstrap standard error over the 57 subjects. Settings chosen at horizon 1.",
             fontsize=8, color=INK_SECONDARY)
    fig.subplots_adjust(left=0.07, right=0.80, top=0.80, bottom=0.14, wspace=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _selected_setting_text(params: pd.DataFrame) -> str:
    """Title text for the selected setting, read from the stored params when available."""
    if {"kappa", "rho"} <= set(params.columns):
        return f"selected: kappa={params['kappa'].iloc[0]:g}, rho={params['rho'].iloc[0]:g}"
    return "selected setting"


def replot_mastery(out_dir: Path, fig_dir: Path = FIG_DIR) -> list[Path]:
    """Redraw the mastery figures from the stored CSVs (no refit). Returns the files written."""
    written = []
    for suffix, setting in (("", None), (f"_k{KAPPA_LIMIT * 10:g}_r1", "per-skill limit: kappa=1e6, rho=1")):
        m_path = out_dir / f"bkt_mastery_{FOCUS_MODEL}{suffix}.csv"
        p_path = out_dir / f"bkt_skill_params_{FOCUS_MODEL}{suffix}.csv"
        if not (m_path.exists() and p_path.exists()):
            print(f"(no {m_path.name}; run with --mastery first)")
            continue
        mastery, params = pd.read_csv(m_path), pd.read_csv(p_path)
        if setting is None:
            setting = _selected_setting_text(params)
        out = fig_dir / f"bkt_mastery_{FOCUS_MODEL}{suffix}.png"
        plot_mastery(mastery, params, out, setting)
        written.append(out)
    return written


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
    parser.add_argument("--ewma-tag", default="bkt_ewma",
                        help="tag of the item_mean_ewma runs used as the recency control")
    parser.add_argument("--mastery", action="store_true",
                        help="fit the selected setting and the per-skill limit on the full history and plot")
    parser.add_argument("--replot-mastery", action="store_true",
                        help="redraw the mastery figures from the stored CSVs without refitting")
    parser.add_argument("--kappa", type=float, default=None, help="override kappa for the mastery fit")
    parser.add_argument("--rho", type=float, default=None, help="override rho for the mastery fit")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR,
                        help="where every table and the console report are stored")
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / "analysis.txt")
    scores: dict[int, list] = {1: [], 5: [], 20: []}
    diffs: dict[int, list] = {1: [], 5: [], 20: []}
    extra_scores: list = []
    extra_diffs: list = []

    df = load_grid(PROC / args.grid)
    df = _with_control(df, PROC / f"models_{args.ewma_tag}_h1.parquet")
    val = summarize(df[df["block"] == "val"], by=["predictor"]).sort_values("log_loss")
    print("Validation block, every predictor (lower log_loss better)")
    print(val.set_index("predictor")[["n", "log_loss", "brier", "accuracy", "auc"]].round(4).to_string())
    val.to_csv(out_dir / "validation_table.csv", index=False)

    best_skill, best_fluid = select(val)
    print(f"\nSelected on validation log-loss: skill anchor {best_skill}; fluid anchor {best_fluid}")
    chosen = [(best_skill, "best skill-anchored"), (best_fluid, "best Fluid-anchored"), (LIMIT, "per-skill limit")]
    for name, label in chosen:
        if name is not None and name in set(df["predictor"]):
            t, d = report(df, name, ITEM_MEAN, f"h=1, {label} ({name}) vs item_mean")
            scores[1].append(t)
            diffs[1].append(d)
    if LIMIT in set(df["predictor"]) and SKILL_MEAN in set(df["predictor"]):
        t, d = report(df, LIMIT, SKILL_MEAN, f"h=1, per-skill limit ({LIMIT}) vs skill_mean")
        scores[1].append(t)
        diffs[1].append(d)
    control = ewma_name(best_skill)
    if control in set(df["predictor"]):
        t, d = report(df, best_skill, control, f"h=1, best skill-anchored ({best_skill}) vs {control}")
        scores[1].append(t)
        diffs[1].append(d)
    no_recency = select_no_recency(val)
    for a, b, label in extra_pairs(best_skill, best_fluid, no_recency, control, set(df["predictor"])):
        t, d = report(df, a, b, f"h=1, {label} ({a} vs {b})")
        t.insert(1, "h", 1)
        d.insert(1, "h", 1)
        extra_scores.append(t)
        extra_diffs.append(d)

    for h in (5, 20):
        path = PROC / f"models_{args.horizon_tag}_h{h}.parquet"
        if not path.exists():
            print(f"\n(no {path.name}; skipping h={h})")
            continue
        hdf = _with_control(pd.read_parquet(path), PROC / f"models_{args.ewma_tag}_h{h}.parquet")
        for name, label in chosen:
            if name is not None and name in set(hdf["predictor"]):
                t, d = report(hdf, name, ITEM_MEAN, f"h={h}, {label} ({name}) vs item_mean",
                              scopes=("pooled", "test"))
                scores[h].append(t)
                diffs[h].append(d)
        if control in set(hdf["predictor"]):
            t, d = report(hdf, best_skill, control, f"h={h}, best skill-anchored ({best_skill}) vs {control}",
                          scopes=("pooled", "test"))
            scores[h].append(t)
            diffs[h].append(d)
        for a, b, label in extra_pairs(best_skill, best_fluid, no_recency, control, set(hdf["predictor"])):
            t, d = report(hdf, a, b, f"h={h}, {label} ({a} vs {b})", scopes=("pooled", "test"))
            t.insert(1, "h", h)
            d.insert(1, "h", h)
            extra_scores.append(t)
            extra_diffs.append(d)
    for h in (1, 5, 20):
        if scores[h]:
            pd.concat(scores[h], ignore_index=True).to_csv(out_dir / f"h{h}_scores.csv", index=False)
            pd.concat(diffs[h], ignore_index=True).to_csv(
                out_dir / f"h{h}_paired_differences.csv", index=False)

    if extra_scores:
        pd.concat(extra_scores, ignore_index=True).to_csv(out_dir / "extra_scores.csv", index=False)
        pd.concat(extra_diffs, ignore_index=True).to_csv(out_dir / "extra_paired_differences.csv", index=False)

    grid_fig = FIG_DIR / "bkt_validation_grid.png"
    plot_validation_grid(val, best_skill, best_fluid, grid_fig)
    print(f"\nWrote {grid_fig}")
    all_scores = pd.concat([t.assign(h=h) for h in (1, 5, 20) for t in scores[h]], ignore_index=True)
    if set(all_scores["h"]) > {1}:
        hz_fig = FIG_DIR / "bkt_horizons.png"
        plot_horizons(all_scores, best_skill, best_fluid, control, hz_fig)
        print(f"Wrote {hz_fig}")

    if args.replot_mastery and not args.mastery:
        for out in replot_mastery(out_dir):
            print(f"Redrew {out}")
    if args.mastery:
        kappa, rho, _ = parse_name(best_skill)
        if args.kappa is not None or args.rho is not None:
            # an overridden setting gets its own file names so the selected setting's files stay
            k = args.kappa if args.kappa is not None else kappa
            r = args.rho if args.rho is not None else rho
            fits = [(k, r, f"_k{k:g}_r{r:g}", "override")]
        else:
            fits = [(kappa, rho, "", "selected setting"),
                    (KAPPA_LIMIT * 10, 1.0, f"_k{KAPPA_LIMIT * 10:g}_r1", "per-skill limit")]
        for k, r, suffix, what in fits:
            mastery, params = fit_mastery(k, r)
            params.insert(1, "kappa", k)
            params.insert(2, "rho", r)
            mastery.to_csv(out_dir / f"bkt_mastery_{FOCUS_MODEL}{suffix}.csv", index=False)
            params.to_csv(out_dir / f"bkt_skill_params_{FOCUS_MODEL}{suffix}.csv", index=False)
            out = FIG_DIR / f"bkt_mastery_{FOCUS_MODEL}{suffix}.png"
            title = f"{what}: kappa={k:g}, rho={r:g}" if what != "per-skill limit" else "per-skill limit: kappa=1e6, rho=1"
            plot_mastery(mastery, params, out, title)
            print(f"\nWrote {out} ({what}) and the mastery / skill-parameter CSVs under {out_dir}")
            print(params.sort_values("learn", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
