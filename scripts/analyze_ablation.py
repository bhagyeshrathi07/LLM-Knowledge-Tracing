#!/usr/bin/env python
"""DriftIRT ablation: which ingredient carries DriftIRT's lead over the recency-weighted bar.

Usage:
    python scripts/analyze_ablation.py [--out-dir docs/results/ablation]

Implements docs/superpowers/specs/2026-11-03-driftirt-ablation.md (frozen before any arm was
run). OLMo-2 7B, MMLU, horizon 1, checkpoint clock; blocks by target checkpoint: train 0-55,
val 56-74, test 75-93. Arms, all DriftIRT with probit residual, plug-in forecast and residual
prior variance 1 except L0:

    L0   the bar, item_mean_ewma_r0.9                       models_ewma_grid_h1
    L1   no skill state, flat item prior (a = 1, b = 0)     models_abl_L1_h1     (_noskill_flat)
    L2   no skill state, Fluid 2PL prior                    models_abl_L2_h1     (_noskill)
    L3   static per-skill ability (q_theta 0)               models_abl_L3_h1
    L3g  static global ability                              models_abl_L3g_h1    (_gskill)
    L4   full DriftIRT                                      models_drift_grid_h1 (archived) and
                                                            models_abl_L4_h1 (re-tuning grid)
    L4g  drifting global ability                            models_abl_L4g_h1    (_gskill)
    L4s  full model, subjects shuffled, seed 0              models_abl_L4s_h1    (_shuf0)
         seeds 1, 2, 3 at the matched setting               models_abl_L4s123_h1 (_shuf1.._shuf3)

Analysis 1 (matched parameters, primary): q_delta 0.01 everywhere, q_theta 0.01 for drifting
states and 0 for static ones, no selection. Analysis 2 (re-tuned, exploratory): each arm's
setting chosen by validation log-loss on its own grid; L4s seeds 1 to 3 reuse seed 0's winner.

Statistics: cluster bootstrap over the 57 subjects, 2,000 replicates, seed 0. Every contrast,
including the difference-in-differences (L4 - L3) - (L4g - L3g), goes through
`contrast_bootstrap`, which forms the contrast within each slice and then resamples subjects,
so all rows of a table share one set of draws. The one confirmatory test is Analysis 1,
L4 - L3, test-block log-loss; every other row is exploratory. The script stops if
`align_slices` drops any row or if per-slice item counts differ between arms.

Writes to --out-dir: matched_ladder.csv, matched_contrasts.csv, retuned_validation.csv,
retuned_ladder.csv, retuned_contrasts.csv, shuffle_seeds.csv, analysis.txt (the console
report), and docs/figures/driftirt_ablation.png.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_bar import INK, INK_SECONDARY, MUTED, SURFACE, _Tee, style_axes, with_blocks

from llm_kt.evaluate import align_slices, contrast_bootstrap, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
FIG = REPO_ROOT / "docs" / "figures" / "driftirt_ablation.png"
RESULTS_DIR = REPO_ROOT / "docs" / "results" / "ablation"
FOCUS_MODEL = "olmo2-7b"
N_CHECKPOINTS = 94
N_SUBJECTS = 57
N_BOOT = 2000
SEED = 0
SCOPES = ("test", "val", "pooled")
METRICS = ("log_loss", "auc")

BAR = "item_mean_ewma_r0.9"
ARCHIVED_L4 = "drift_irt_qt0.01_qd0.01_pv1_fv0_zres"
ARMS = ("L0", "L1", "L2", "L3", "L3g", "L4", "L4g", "L4s")
SHUFFLE_SEEDS = (0, 1, 2, 3)
DESCRIPTION = {
    "L0": "the bar, recency-weighted item mean (rho 0.9)",
    "L1": "item-only probit random walk, flat item prior",
    "L2": "item-only probit random walk, Fluid prior",
    "L3": "static per-skill ability",
    "L3g": "static global ability",
    "L4": "full DriftIRT",
    "L4g": "drifting global ability",
    "L4s": "full model, subjects shuffled (seed 0)",
}
# arm -> (file tag, name suffix, drifting state?)
GRIDS = {
    "L1": ("abl_L1", "_noskill_flat", False),
    "L2": ("abl_L2", "_noskill", False),
    "L3": ("abl_L3", "", False),
    "L3g": ("abl_L3g", "_gskill", False),
    "L4": ("abl_L4", "", True),
    "L4g": ("abl_L4g", "_gskill", True),
    "L4s": ("abl_L4s", "_shuf0", True),
}
CONTRASTS = [  # (row name, weights, ingredient)
    ("L1 - L0", {"L1": 1, "L0": -1}, "link and filter plus prior treatment (two changes; descriptive)"),
    ("L2 - L1", {"L2": 1, "L1": -1}, "Fluid 2PL prior centre"),
    ("L3 - L2", {"L3": 1, "L2": -1}, "static per-subject level"),
    ("L4 - L3", {"L4": 1, "L3": -1}, "drift in the per-skill ability (primary)"),
    ("L3 - L3g", {"L3": 1, "L3g": -1}, "subject-specific versus one global static level"),
    ("(L4 - L3) - (L4g - L3g)", {"L4": 1, "L3": -1, "L4g": -1, "L3g": 1},
     "drift benefit, subject model versus global model"),
    ("L4 - L4s", {"L4": 1, "L4s": -1}, "real subject grouping versus a random grouping of equal sizes"),
]
PRIMARY = "L4 - L3"
DID = "(L4 - L3) - (L4g - L3g)"


def setting_name(qt: float, qd: float, suffix: str) -> str:
    return f"drift_irt_qt{qt:g}_qd{qd:g}_pv1_fv0_zres{suffix}"


def parse_setting(name: str) -> tuple[float, float]:
    m = re.match(r"drift_irt_qt([0-9.e-]+)_qd([0-9.e-]+)_", name)
    return float(m.group(1)), float(m.group(2))


def read(path: Path, names: list[str]) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"missing {path.relative_to(REPO_ROOT)}; see the README for the command that writes it")
    df = pd.read_parquet(path)
    df = df[(df["model"] == FOCUS_MODEL) & df["predictor"].isin(names)]
    missing = set(names) - set(df["predictor"])
    if missing:
        sys.exit(f"{path.name} lacks {sorted(missing)}")
    keep = ["predictor", "model", "origin_idx", "target_idx", "skill", "n",
            "log_loss", "brier", "accuracy", "auc"]
    return with_blocks(df[keep], N_CHECKPOINTS)


def load_arms(chosen: dict[str, tuple[Path, str]]) -> pd.DataFrame:
    """One frame with the predictor column renamed to the arm label, aligned and checked."""
    frames = [read(path, [name]).assign(predictor=arm) for arm, (path, name) in chosen.items()]
    df = pd.concat(frames, ignore_index=True)
    df, dropped = align_slices(df)
    if dropped["n_dropped"].sum() != 0:
        sys.exit(f"align_slices dropped rows; stopping:\n{dropped.to_string(index=False)}")
    n = df.pivot_table(index=["origin_idx", "target_idx", "skill"], columns="predictor", values="n")
    if n.isna().any().any() or not (n.nunique(axis=1) == 1).all():
        sys.exit("per-slice item counts n differ between arms; stopping")
    if df["skill"].nunique() != N_SUBJECTS:
        sys.exit(f"expected {N_SUBJECTS} subjects, found {df['skill'].nunique()}")
    print(f"Alignment: {len(chosen)} predictors, {len(df) // len(chosen):,} slices each, "
          "zero rows dropped by align_slices, identical per-slice n across arms.")
    return df


def _scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    return df if scope == "pooled" else df[df["block"] == scope]


def ladder(df: pd.DataFrame, order: list[str], settings: dict[str, str]) -> pd.DataFrame:
    rows = []
    for scope in SCOPES:
        s = summarize(_scope(df, scope), by=["predictor"], n_boot=N_BOOT, seed=SEED)
        s = s.set_index("predictor").loc[[a for a in order if a in set(s["predictor"])]].reset_index()
        s.insert(0, "scope", scope)
        rows.append(s)
    t = pd.concat(rows, ignore_index=True).rename(columns={"predictor": "arm"})
    t.insert(2, "setting", t["arm"].map(settings))
    t.insert(3, "description", t["arm"].map(DESCRIPTION))
    t["label"] = "exploratory"
    return t[["scope", "arm", "setting", "description", "n", "log_loss", "log_loss_se", "auc", "auc_se",
              "brier", "accuracy", "label"]]


def reading(row: str, lo: float, hi: float, metric: str) -> str:
    if lo <= 0 <= hi:
        return "no demonstrated difference"
    first_better = hi < 0 if metric == "log_loss" else lo > 0
    if row == DID:
        return ("drift benefit larger in the subject model" if first_better
                else "drift benefit smaller in the subject model")
    a, b = row.split(" - ")
    return f"{a} better" if first_better else f"{b} better"


def contrasts(df: pd.DataFrame, analysis: str, rows=CONTRASTS) -> pd.DataFrame:
    out = []
    for name, weights, ingredient in rows:
        for scope in SCOPES:
            for metric in METRICS:
                d = contrast_bootstrap(_scope(df, scope), weights, metric=metric, n_boot=N_BOOT, seed=SEED)
                r = d.iloc[0].to_dict()
                if r["n_clusters"] != N_SUBJECTS:
                    sys.exit(f"{name} {scope} {metric}: {r['n_clusters']} clusters, expected {N_SUBJECTS}")
                confirm = analysis == "matched" and name == PRIMARY and scope == "test" and metric == "log_loss"
                out.append({"analysis": analysis, "contrast": name, "ingredient": ingredient, "scope": scope,
                            "metric": metric, **r,
                            "reading": reading(name, r["ci_low"], r["ci_high"], metric),
                            "label": "confirmatory" if confirm else "exploratory"})
    return pd.DataFrame(out)


def shuffle_rows(df: pd.DataFrame, analysis: str) -> pd.DataFrame:
    """L4 - L4s for each seed (test block), then the mean and range of the four differences."""
    rows = []
    for metric in METRICS:
        diffs = []
        for seed in SHUFFLE_SEEDS:
            d = contrast_bootstrap(_scope(df, "test"), {"L4": 1, f"L4s{seed}": -1}, metric=metric,
                                   n_boot=N_BOOT, seed=SEED).iloc[0]
            diffs.append(d["diff"])
            rows.append({"analysis": analysis, "row": f"L4 - L4s seed {seed}", "scope": "test", "metric": metric,
                         "diff": d["diff"], "se": d["se"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                         "reading": reading("L4 - L4s", d["ci_low"], d["ci_high"], metric),
                         "label": "exploratory"})
        rows.append({"analysis": analysis, "row": "mean of the four seeds", "scope": "test", "metric": metric,
                     "diff": float(np.mean(diffs)), "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                     "range_low": float(min(diffs)), "range_high": float(max(diffs)),
                     "reading": "average of the four differences; range in range_low, range_high",
                     "label": "exploratory"})
    return pd.DataFrame(rows)


def print_ladder(t: pd.DataFrame) -> None:
    cols = ["setting", "n", "log_loss", "log_loss_se", "auc", "auc_se"]
    for scope in SCOPES:
        print(f"\n  {scope}" + (" block" if scope != "pooled" else " over the run"))
        print(t[t["scope"] == scope].set_index("arm")[cols].round(4).to_string())


def print_contrasts(c: pd.DataFrame, scopes=SCOPES) -> None:
    cols = ["diff", "se", "ci_low", "ci_high", "reading", "label"]
    for scope in scopes:
        print(f"\n  {scope}" + (" block" if scope != "pooled" else " over the run"))
        sub = c[c["scope"] == scope]
        print(sub.set_index(["contrast", "metric"])[cols].round(5).to_string())


# -- analysis 1 --------------------------------------------------------------------------------


def matched_files() -> dict[str, tuple[Path, str]]:
    f = {"L0": (PROC / "models_ewma_grid_h1.parquet", BAR),
         "L4": (PROC / "models_drift_grid_h1.parquet", ARCHIVED_L4)}
    for arm, (tag, suffix, drifting) in GRIDS.items():
        if arm != "L4":
            f[arm] = (PROC / f"models_{tag}_h1.parquet", setting_name(0.01 if drifting else 0, 0.01, suffix))
    return f


def analysis1(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n{'=' * 100}\nAnalysis 1: matched parameters (primary)\n{'=' * 100}")
    print("Every arm at the archived L4 setting wherever the ingredient exists: q_delta 0.01, q_theta 0.01 for")
    print("drifting states (L4, L4g, L4s), 0 for static states (L3, L3g), no ability for L1 and L2. No selection.")
    files = matched_files()
    shuf = {f"L4s{s}": (PROC / "models_abl_L4s123_h1.parquet", setting_name(0.01, 0.01, f"_shuf{s}"))
            for s in SHUFFLE_SEEDS[1:]}
    shuf["L4s0"] = files["L4s"]
    df = load_arms({**files, **shuf})
    df_main = df[df["predictor"].isin(ARMS)]
    settings = {arm: name for arm, (_, name) in files.items()}
    t = ladder(df_main, list(ARMS), settings)
    t.insert(0, "analysis", "matched")
    t.to_csv(out_dir / "matched_ladder.csv", index=False)
    print(f"\nLadder (SEs: cluster bootstrap over the {N_SUBJECTS} subjects, {N_BOOT} replicates, per arm)")
    print_ladder(t)
    c = contrasts(df_main, "matched")
    c.to_csv(out_dir / "matched_contrasts.csv", index=False)
    print(f"\nContrasts, first arm minus second (95% percentile interval, {N_BOOT} shared subject resamples,"
          " seed 0;\nnegative log-loss / positive AUC favour the first arm). Confirmatory: L4 - L3, test, log-loss only.")
    print_contrasts(c)
    s = shuffle_rows(df, "matched")
    print("\nShuffle-seed sensitivity (test block; seeds 1-3 at the matched setting like seed 0)")
    print(s.set_index(["metric", "row"])[["diff", "se", "ci_low", "ci_high", "range_low", "range_high"]]
          .round(5).to_string())

    # the archived L4 against the rerun of the same setting in the re-tuning grid
    rerun = read(PROC / "models_abl_L4_h1.parquet", [ARCHIVED_L4])
    arch = read(PROC / "models_drift_grid_h1.parquet", [ARCHIVED_L4])
    m = arch.merge(rerun, on=["origin_idx", "target_idx", "skill"], suffixes=("_a", "_r"))
    gap = float((m["log_loss_a"] - m["log_loss_r"]).abs().max())
    print(f"\nReproduction: the archived L4 and its rerun in models_abl_L4_h1 agree on {len(m):,} of "
          f"{len(arch):,} slices; largest per-slice log-loss difference {gap:.2e}.")
    return t, c, s


# -- analysis 2 --------------------------------------------------------------------------------


def analysis2(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n\n{'=' * 100}\nAnalysis 2: re-tuned on validation (secondary, exploratory)\n{'=' * 100}")
    print("q_delta in {0.003, 0.01, 0.03} for every arm; q_theta in {0.003, 0.01, 0.03} for L4, L4g, L4s only;")
    print("selection by validation log-loss (targets 56-74). L0 is fixed at rho 0.9. L4s seeds 1-3 reuse seed 0's winner.")
    val_rows, chosen = [], {"L0": (PROC / "models_ewma_grid_h1.parquet", BAR)}
    for arm, (tag, suffix, drifting) in GRIDS.items():
        path = PROC / f"models_{tag}_h1.parquet"
        qts = (0.003, 0.01, 0.03) if drifting else (0,)
        names = [setting_name(qt, qd, suffix) for qt in qts for qd in (0.003, 0.01, 0.03)]
        g = read(path, names)
        v = summarize(g[g["block"] == "val"], by=["predictor"], n_boot=N_BOOT, seed=SEED)
        v = v.sort_values("log_loss").reset_index(drop=True)
        v.insert(0, "arm", arm)
        v["q_theta"], v["q_delta"] = zip(*v["predictor"].map(parse_setting))
        v["chosen"] = v.index == 0
        v["minus_best"] = v["log_loss"] - v["log_loss"].iloc[0]
        val_rows.append(v)
        chosen[arm] = (path, v["predictor"].iloc[0])
    val = pd.concat(val_rows, ignore_index=True)
    val["label"] = "exploratory"
    cols = ["arm", "predictor", "q_theta", "q_delta", "n", "log_loss", "log_loss_se", "minus_best",
            "auc", "auc_se", "chosen", "label"]
    val[cols].to_csv(out_dir / "retuned_validation.csv", index=False)
    print("\nValidation block, every grid setting (sorted within arm; chosen = lowest validation log-loss)")
    print(val.set_index(["arm", "predictor"])[["log_loss", "log_loss_se", "minus_best", "auc", "chosen"]]
          .round(4).to_string())
    l4 = chosen["L4"][1]
    same = parse_setting(l4) == parse_setting(ARCHIVED_L4)
    print(f"\nL4 validation winner: {l4} "
          + ("(the archived setting)" if same else f"(differs from the archived setting {ARCHIVED_L4})"))

    qt0, qd0 = parse_setting(chosen["L4s"][1])
    shuf = {"L4s0": chosen["L4s"]}
    for seed in SHUFFLE_SEEDS[1:]:
        name = setting_name(qt0, qd0, f"_shuf{seed}")
        tag = "abl_L4s123" if (qt0, qd0) == (0.01, 0.01) else "abl_L4s123_retuned"
        shuf[f"L4s{seed}"] = (PROC / f"models_{tag}_h1.parquet", name)
    df = load_arms({**chosen, **shuf})
    df_main = df[df["predictor"].isin(ARMS)]
    settings = {arm: name for arm, (_, name) in chosen.items()}
    t = ladder(df_main, list(ARMS), settings)
    t.insert(0, "analysis", "retuned")
    t.to_csv(out_dir / "retuned_ladder.csv", index=False)
    print("\nLadder at each arm's chosen setting")
    print_ladder(t)
    c = contrasts(df_main, "retuned")
    c["settings"] = c["contrast"].map(lambda r: "; ".join(
        f"{a}: q_theta {parse_setting(settings[a])[0]:g}, q_delta {parse_setting(settings[a])[1]:g}"
        for a in dict.fromkeys(re.findall(r"L\d[gs]?", r)) if a != "L0"))
    c.to_csv(out_dir / "retuned_contrasts.csv", index=False)
    print("\nContrasts at the chosen settings (exploratory; not single-variable interventions where the chosen"
          " q values differ)")
    print_contrasts(c, scopes=("test",))
    s = shuffle_rows(df, "retuned")
    print(f"\nShuffle-seed sensitivity, re-tuned (all seeds at seed 0's winner q_theta {qt0:g}, q_delta {qd0:g})")
    print(s.set_index(["metric", "row"])[["diff", "se", "ci_low", "ci_high", "range_low", "range_high"]]
          .round(5).to_string())
    return val, c, s


# -- figure ------------------------------------------------------------------------------------

ARM_COLOR = {"L0": INK, "L1": "#898781", "L2": "#898781", "L3": "#2a78d6", "L3g": "#7fb0ea",
             "L4": "#eb6834", "L4g": "#f3a27f", "L4s": "#1baf7a"}


def plot(t: pd.DataFrame, c: pd.DataFrame, out: Path) -> None:
    """Left: Analysis 1 test log-loss per arm with one SE. Right: contrasts with 95% intervals."""
    test = t[t["scope"] == "test"].set_index("arm").loc[list(ARMS)]
    ct = c[(c["scope"] == "test") & (c["metric"] == "log_loss")].set_index("contrast")
    ct = ct.loc[[name for name, _, _ in CONTRASTS]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(ARMS))
    for i, arm in enumerate(ARMS):
        ax1.errorbar(i, test.loc[arm, "log_loss"], yerr=test.loc[arm, "log_loss_se"], color=ARM_COLOR[arm],
                     marker="o", ms=6, capsize=3, lw=0, elinewidth=2.0)
    ax1.axhline(test.loc["L0", "log_loss"], color=INK, lw=1.0, ls="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ARMS)
    ax1.set_ylabel("test-block log-loss (lower is better)", fontsize=8, color=INK_SECONDARY)
    ax1.set_title("Analysis 1 ladder, error bars one SE", fontsize=9, color=INK, loc="left")

    y = np.arange(len(ct))[::-1]
    for yi, (name, r) in zip(y, ct.iterrows()):
        primary = name == PRIMARY
        color = "#eb6834" if primary else INK_SECONDARY
        ax2.errorbar(r["diff"], yi, xerr=[[r["diff"] - r["ci_low"]], [r["ci_high"] - r["diff"]]],
                     color=color, marker="D" if primary else "o", ms=7 if primary else 5, capsize=3,
                     lw=0, elinewidth=2.4 if primary else 1.6)
    ax2.axvline(0, color=INK, lw=1.2, ls="--")
    ax2.set_yticks(y)
    ax2.set_yticklabels([n + ("  (primary)" if n == PRIMARY else "") for n in ct.index], fontsize=8)
    ax2.set_xlabel("first minus second, test log-loss (below 0: first arm better)", fontsize=8,
                   color=INK_SECONDARY)
    ax2.set_title("Contrasts with 95% interval (shared subject resamples)", fontsize=9, color=INK, loc="left")
    for ax in (ax1, ax2):
        style_axes(ax)
    ax2.yaxis.grid(False)
    for lab in ax2.get_yticklabels():  # after style_axes, which resets tick label colours
        if "primary" in lab.get_text():
            lab.set_color("#eb6834")
            lab.set_fontweight("bold")
    ax2.xaxis.grid(True, color="#e1e0d9", linewidth=0.6)
    fig.suptitle("DriftIRT ablation, matched parameters, OLMo-2 MMLU test block (checkpoints 75 to 93), h = 1",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.905, "L0 bar; L1 no skill state, flat prior; L2 no skill state, Fluid prior; L3 static subject "
             "ability; L3g static global ability;\nL4 full DriftIRT; L4g drifting global ability; L4s subjects shuffled "
             f"(seed 0). Bootstrap over the {N_SUBJECTS} subjects, {N_BOOT} replicates. "
             "Only L4 - L3 on log-loss is confirmatory.",
             fontsize=7.5, color=MUTED, va="top")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.13, wspace=0.42)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / "analysis.txt")
    print("DriftIRT ablation (docs/superpowers/specs/2026-11-03-driftirt-ablation.md, frozen before any arm was run)")
    print("OLMo-2 7B, MMLU, horizon 1, checkpoint clock. Blocks by target: train 0-55, val 56-74, test 75-93.")
    print("The test block had been examined before this follow-up was specified (the 2026-11-03 rescoring and the")
    print("October model selection both reported it). One confirmatory test: Analysis 1, L4 - L3, test log-loss.")
    print("Every other row is exploratory. Readings: a side is 'better' only when the 95% interval excludes zero,")
    print("otherwise 'no demonstrated difference'.")
    t1, c1, s1 = analysis1(out_dir)
    _, _, s2 = analysis2(out_dir)
    pd.concat([s1, s2], ignore_index=True).to_csv(out_dir / "shuffle_seeds.csv", index=False)
    plot(t1, c1, FIG)
    print(f"\nWrote {FIG.relative_to(REPO_ROOT)} and the CSVs in {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
