#!/usr/bin/env python
"""Rescore every learner model against the recency-weighted item mean, and compare clocks.

Usage:
    python scripts/analyze_bar.py [--out-dir docs/results/bar]

Part 1, the bar. Reads the item_mean_ewma grid at horizon 1 (models_ewma_grid_h1.parquet,
from run_models.py --predictors item_mean item_mean_ewma --rho ... --tag ewma_grid) and picks
rho by validation-block log-loss. That setting is the bar. It is then scored with the
selected learner models already on disk, without re-tuning them:

    T-SKIRT   tskirt_nu1_checkpoint_idx_recal_tau2_taua0.5   models_calibrated_h1, models_horizon_h{5,20}
    DriftIRT  drift_irt_qt0.01_qd0.01_pv1_fv0_zres          models_drift_grid_h1, models_drift_best_h{5,20}
    BKT       bkt_k0.1_r0.9_skill, bkt_k1_r0.9_fluid_lo-2_hi1  docs/results/bkt/models_bkt_{grid_h1,best_h5,best_h20}

plus item_mean and the chosen bar (models_ewma_grid_h1 / models_ewma_best_h{5,20}) and
copy_last (baseline_results_h{1,5,20}, OLMo-2 MMLU slices) for the ladder. Rows are
aligned on (model, origin_idx, target_idx, skill) across files; any slice not scored for
every predictor is dropped and reported. For h = 1, 5, 20: pooled / val / test scores with
cluster-bootstrap SEs over skills (n_boot=200), and paired differences (model minus the
bar) on log-loss and AUC with 95% intervals. The test block is the headline; the
validation block is the only one ever used to choose anything.

Look-ahead caveat. Blocks are assigned by target checkpoint and every setting (rho, q, nu)
was chosen on validation targets 56 to 74. At h = 20 every test forecast originates at
checkpoints 55 to 73, and at h = 5 the first four originate before 74, so tuning saw
observations later than those forecast origins (the test labels themselves stay held out).
The h = 5 and h = 20 tables are therefore retrospective; h = 1 is unaffected.

Sensitivity (optional, skipped when models_ewma_hl8_h1.parquet is missing). The recency
control with half-life 8 checkpoints (rho = 0.5 ** (1/8) = 0.917004; run_models.py
--predictors item_mean_ewma --rho 0.917004 --horizon 1 --tag ewma_hl8) against rho 0.9:
validation and test scores, DriftIRT minus each control on the test block at h = 1, and the
share of DriftIRT's test log-loss gain over the plain item mean that each control captures,
(item_mean - control) / (item_mean - DriftIRT).

Part 2, the clock. Reads the tokens-seen grids (models_clock_tskirt_grid_h1,
models_clock_drift_grid_h1; run_models.py --time-col tokens_seen_b), picks one T-SKIRT and
one DriftIRT setting by validation log-loss, and pairs them with their checkpoint-clock
counterparts at h = 1 (from the grids) and h = 5, 20 (models_clock_best_h{5,20}). These
rows are labelled "re-tuned on validation". The chosen DriftIRT tokens setting has a q_theta
ten times smaller than the rescaled checkpoint value, so they mix the clock with re-tuning.
At h = 1 only, rows labelled "matched parameters" pair each checkpoint-clock setting with
its exact rescaling on the tokens clock (DriftIRT q / 42, T-SKIRT nu / sqrt(42)), which
isolates the clock. The stored parameters are rounded to six significant figures.

Files read for part 2, exactly these names (the README shows how each is produced and
concatenated): models_clock_tskirt_grid_h1, models_clock_drift_grid_h1 (all tokens-clock
DriftIRT settings, one file), models_clock_best_h5 and models_clock_best_h20 (the chosen
T-SKIRT and DriftIRT tokens settings together in one file per horizon).

Writes to --out-dir: validation_rho.csv, h{1,5,20}_scores.csv, h{1,5,20}_paired_differences.csv,
alignment.csv, sensitivity_halflife8.csv, tskirt_calibrated_validation.csv, clock_validation.csv, clock_comparison.csv and
analysis.txt (the console report), and docs/figures/bar_rescoring.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from llm_kt.evaluate import align_slices, assign_blocks, paired_difference, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
BKT_DIR = REPO_ROOT / "docs" / "results" / "bkt"
FIG_DIR = REPO_ROOT / "docs" / "figures"
RESULTS_DIR = REPO_ROOT / "docs" / "results" / "bar"
FOCUS_MODEL = "olmo2-7b"
N_BOOT = 200
SCOPES = ("pooled", "val", "test")
HORIZONS = (1, 5, 20)

ITEM_MEAN = "item_mean"
COPY_LAST = "copy_last"
EWMA_PREFIX = "item_mean_ewma_r"
TSKIRT = "tskirt_nu1_checkpoint_idx_recal_tau2_taua0.5"
DRIFT = "drift_irt_qt0.01_qd0.01_pv1_fv0_zres"
BKT_SKILL = "bkt_k0.1_r0.9_skill"
BKT_FLUID = "bkt_k1_r0.9_fluid_lo-2_hi1"
TSKIRT_CAL = "tskirt_nu1_checkpoint_idx_recal_tau2_taua0.5_cal_temperaturek5"  # stored, not rescored
HL8 = "item_mean_ewma_r0.917004"  # half-life 8 checkpoints, rho = 0.5 ** (1/8); sensitivity only
# exact rescalings of the checkpoint-clock settings onto the tokens clock (42B tokens per checkpoint)
MATCHED = {"T-SKIRT": "tskirt_nu0.154303_tokens_seen_b_recal_tau2_taua0.5",  # nu 1 / sqrt(42)
           "DriftIRT": "drift_irt_qt0.000238095_qd0.000238095_pv1_fv0_zres_tokens_seen_b"}  # q 0.01 / 42
RETRO = ("Retrospective: hyperparameters were chosen on targets up to 74, later than these "
         "forecast origins.")
LABELS = {
    COPY_LAST: "copy last",
    ITEM_MEAN: "item mean",
    TSKIRT: "T-SKIRT",
    DRIFT: "DriftIRT",
    BKT_SKILL: "BKT, skill anchor",
    BKT_FLUID: "BKT, Fluid anchor",
}


def sources(h: int, bar: str) -> list[tuple[Path, list[str]]]:
    """(file, predictors to take from it) for horizon h. item_mean comes from the bar's file."""
    if h == 1:
        return [
            (PROC / "models_ewma_grid_h1.parquet", [ITEM_MEAN, bar]),
            (PROC / "baseline_results_h1.parquet", [COPY_LAST]),
            (PROC / "models_calibrated_h1.parquet", [TSKIRT]),
            (PROC / "models_drift_grid_h1.parquet", [DRIFT]),
            (BKT_DIR / "models_bkt_grid_h1.parquet", [BKT_SKILL, BKT_FLUID]),
        ]
    return [
        (PROC / f"models_ewma_best_h{h}.parquet", [ITEM_MEAN, bar]),
        (PROC / f"baseline_results_h{h}.parquet", [COPY_LAST]),
        (PROC / f"models_horizon_h{h}.parquet", [TSKIRT]),
        (PROC / f"models_drift_best_h{h}.parquet", [DRIFT]),
        (BKT_DIR / f"models_bkt_best_h{h}.parquet", [BKT_SKILL, BKT_FLUID]),
    ]


def with_blocks(df: pd.DataFrame, n_checkpoints: int) -> pd.DataFrame:
    """(Re)assign the chronological block of each target; baseline files carry none."""
    df = df.copy()
    df["block"] = "train"
    for name, (lo, hi) in assign_blocks(n_checkpoints).items():
        df.loc[df["target_idx"].between(lo, hi), "block"] = name
    return df


def load(files: list[tuple[Path, list[str]]], skills: set[str] | None = None) -> pd.DataFrame:
    """Concatenate the named predictors from each file, OLMo-2 only, restricted to `skills`."""
    frames = []
    for path, names in files:
        if not path.exists():
            print(f"(missing {path.relative_to(REPO_ROOT)}; {', '.join(names)} skipped)")
            continue
        df = pd.read_parquet(path)
        df = df[(df["model"] == FOCUS_MODEL) & df["predictor"].isin(names)]
        missing = set(names) - set(df["predictor"])
        if missing:
            raise ValueError(f"{path.name} lacks {sorted(missing)}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if skills is not None:
        df = df[df["skill"].isin(skills)]
    keep = ["predictor", "model", "origin_idx", "target_idx", "skill", "n",
            "log_loss", "brier", "accuracy", "auc"]
    return with_blocks(df[keep], n_checkpoints=94)


def _scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    return df if scope == "pooled" else df[df["block"] == scope]


def score_table(df: pd.DataFrame, order: list[str], scopes=SCOPES) -> pd.DataFrame:
    tables = []
    for scope in scopes:
        s = summarize(_scope(df, scope), by=["predictor"], n_boot=N_BOOT, cluster="skill")
        s.insert(0, "scope", scope)
        tables.append(s)
    t = pd.concat(tables, ignore_index=True)
    t["_s"] = t["scope"].map({s: i for i, s in enumerate(scopes)})
    t["_o"] = t["predictor"].map({p: i for i, p in enumerate(order)})
    return t.sort_values(["_s", "_o"]).drop(columns=["_s", "_o"]).reset_index(drop=True)


def paired_table(df: pd.DataFrame, pairs: list[tuple[str, str]], scopes=SCOPES) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        for scope in scopes:
            g = _scope(df, scope)
            for metric in ("log_loss", "auc"):
                d = paired_difference(g, a, b, by=[], metric=metric, n_boot=N_BOOT)
                d.insert(0, "metric", metric)
                d.insert(0, "scope", scope)
                d.insert(0, "b", b)
                d.insert(0, "a", a)
                rows.append(d)
    return pd.concat(rows, ignore_index=True)


def verdict(lo: float, hi: float, metric: str) -> str:
    """Reading of a paired difference a - b from its 95% interval."""
    if lo <= 0 <= hi:
        return "no demonstrated difference"
    better = hi < 0 if metric == "log_loss" else lo > 0
    return "a better" if better else "b better"


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


# -- part 1 --------------------------------------------------------------------------------


def choose_rho(out_dir: Path) -> str:
    df = pd.read_parquet(PROC / "models_ewma_grid_h1.parquet")
    df = with_blocks(df[df["model"] == FOCUS_MODEL], 94)
    val = summarize(df[df["block"] == "val"], by=["predictor"], n_boot=N_BOOT, cluster="skill")
    val["rho"] = val["predictor"].map(
        lambda p: float(p[len(EWMA_PREFIX):]) if p.startswith(EWMA_PREFIX) else 1.0)
    val = val.sort_values("log_loss").reset_index(drop=True)
    cols = ["predictor", "rho", "n", "log_loss", "log_loss_se", "brier", "accuracy", "auc", "auc_se"]
    val[cols].to_csv(out_dir / "validation_rho.csv", index=False)
    print("Validation block (checkpoints 56 to 74), horizon 1: item mean and the recency-weighted")
    print(f"item mean for each rho (item_mean is rho = 1). SEs: cluster bootstrap over skills, n_boot={N_BOOT}.")
    print(val[cols].set_index("predictor").round(4).to_string())
    bar = val[val["predictor"].str.startswith(EWMA_PREFIX)].iloc[0]["predictor"]
    print(f"\nChosen bar: {bar} (lowest validation log-loss)")
    return bar


def rescore(bar: str, out_dir: Path) -> pd.DataFrame:
    """Scores and paired differences at every horizon; returns all scores tagged with h."""
    order = [COPY_LAST, ITEM_MEAN, bar, TSKIRT, DRIFT, BKT_SKILL, BKT_FLUID]
    learners = [TSKIRT, DRIFT, BKT_SKILL, BKT_FLUID]
    all_scores, align_rows = [], []
    mmlu_skills = None
    for h in HORIZONS:
        raw = load(sources(h, bar))
        if mmlu_skills is None:
            mmlu_skills = set(raw.loc[raw["predictor"] == bar, "skill"])
        raw = raw[raw["skill"].isin(mmlu_skills)]
        df, dropped = align_slices(raw)
        dropped.insert(0, "h", h)
        align_rows.append(dropped)
        print(f"\n{'=' * 100}\nHorizon {h}\n{'=' * 100}")
        if h > 1:
            origins = sorted(set(raw.loc[raw["block"] == "test", "origin_idx"]))
            print(f"Test-block forecast origins {origins[0]} to {origins[-1]}. {RETRO}")
        print("Row alignment on (model, origin_idx, target_idx, skill):")
        print(dropped.set_index("predictor").drop(columns="h").to_string())
        if dropped["n_dropped"].sum() == 0:
            print("Identical row sets for every predictor; nothing dropped.")
        present = [p for p in order if p in set(df["predictor"])]

        scores = score_table(df, present)
        scores.to_csv(out_dir / f"h{h}_scores.csv", index=False)
        all_scores.append(scores.assign(h=h))
        cols = ["n", "log_loss", "log_loss_se", "auc", "auc_se"]
        print(f"\nh={h} scores (cluster-bootstrap SEs over skills, n_boot={N_BOOT})")
        print(scores.set_index(["scope", "predictor"])[cols].round(4).to_string())

        pairs = [(p, bar) for p in present if p != bar]
        diffs = paired_table(df, pairs)
        diffs["reading"] = [verdict(lo, hi, m) for lo, hi, m in
                            zip(diffs["ci_low"], diffs["ci_high"], diffs["metric"])]
        diffs.to_csv(out_dir / f"h{h}_paired_differences.csv", index=False)
        print(f"\nh={h} paired differences, model minus {bar} (95% bootstrap interval; "
              "negative log-loss / positive AUC favour the model)")
        for scope in ("test", "val", "pooled"):
            sub = diffs[diffs["scope"] == scope]
            print(f"\n  {scope} block" if scope != "pooled" else "\n  pooled over the run")
            print(sub.set_index(["a", "metric"])[["diff", "se", "ci_low", "ci_high", "n_slices", "reading"]]
                  .round(4).to_string())
        learner_test = diffs[(diffs["scope"] == "test") & diffs["a"].isin(learners)]
        print(f"\nh={h} test-block headline:" + (f" ({RETRO})" if h > 1 else ""))
        for a in learners:
            r = learner_test[learner_test["a"] == a].set_index("metric")
            if r.empty:
                continue
            print(f"  {LABELS[a]:<18} log-loss {r.loc['log_loss', 'diff']:+.4f} "
                  f"[{r.loc['log_loss', 'ci_low']:+.4f}, {r.loc['log_loss', 'ci_high']:+.4f}] "
                  f"({r.loc['log_loss', 'reading']}); AUC {r.loc['auc', 'diff']:+.4f} "
                  f"[{r.loc['auc', 'ci_low']:+.4f}, {r.loc['auc', 'ci_high']:+.4f}] ({r.loc['auc', 'reading']})")
    pd.concat(align_rows, ignore_index=True).to_csv(out_dir / "alignment.csv", index=False)
    return pd.concat(all_scores, ignore_index=True)


# -- sensitivity: half-life 8 control ---------------------------------------------------


def sensitivity(out_dir: Path, bar: str) -> None:
    """rho 0.917004 (half-life 8) against the chosen bar, and DriftIRT against each, at h = 1."""
    hl8_path = PROC / "models_ewma_hl8_h1.parquet"
    print(f"\n\n{'=' * 100}\nSensitivity: the half-life-8 recency control (rho 0.917004), horizon 1\n{'=' * 100}")
    if not hl8_path.exists():
        print(f"({hl8_path.relative_to(REPO_ROOT)} missing; section skipped)")
        return
    files = [(PROC / "models_ewma_grid_h1.parquet", [ITEM_MEAN, bar]),
             (hl8_path, [HL8]),
             (PROC / "models_drift_grid_h1.parquet", [DRIFT])]
    df, dropped = align_slices(load(files))
    if dropped["n_dropped"].sum():
        print(f"alignment dropped rows:\n{dropped.to_string(index=False)}")
    order = [ITEM_MEAN, bar, HL8, DRIFT]
    scores = score_table(df, order, scopes=("val", "test"))
    cols = ["n", "log_loss", "log_loss_se", "auc", "auc_se"]
    print(f"Scores (cluster-bootstrap SEs over skills, n_boot={N_BOOT}). rho {bar[len(EWMA_PREFIX):]} "
          "stays the bar: it is the best of the declared grid; half-life 8 is archived as sensitivity.")
    print(scores.set_index(["scope", "predictor"])[cols].round(4).to_string())
    diffs = paired_table(df, [(DRIFT, bar), (DRIFT, HL8)], scopes=("test",))
    diffs["reading"] = [verdict(lo, hi, m) for lo, hi, m in
                        zip(diffs["ci_low"], diffs["ci_high"], diffs["metric"])]
    print("\nPaired differences, test block, DriftIRT minus each control (95% bootstrap interval)")
    print(diffs.set_index(["b", "metric"])[["diff", "se", "ci_low", "ci_high", "reading"]].round(4).to_string())

    test = scores[scores["scope"] == "test"].set_index("predictor")["log_loss"]
    rows = [{"row": "score", "scope": r["scope"], "a": r["predictor"], "b": "", "metric": m,
             "value": r[m], "se": r[f"{m}_se"], "ci_low": None, "ci_high": None, "n": r["n"]}
            for _, r in scores.iterrows() for m in ("log_loss", "auc")]
    rows += [{"row": "paired_difference", "scope": r["scope"], "a": r["a"], "b": r["b"], "metric": r["metric"],
              "value": r["diff"], "se": r["se"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": r["n"]}
             for _, r in diffs.iterrows()]
    print("\nShare of DriftIRT's test log-loss gain over the plain item mean captured by recency,"
          "\n(item_mean - control) / (item_mean - DriftIRT):")
    gain = test[ITEM_MEAN] - test[DRIFT]
    for control in (bar, HL8):
        share = (test[ITEM_MEAN] - test[control]) / gain
        print(f"  {control:<26} {test[ITEM_MEAN] - test[control]:.4f} / {gain:.4f} = {share:.3f}")
        rows.append({"row": "recency_share", "scope": "test", "a": control, "b": DRIFT, "metric": "log_loss",
                     "value": share, "se": None, "ci_low": None, "ci_high": None, "n": None})
    pd.DataFrame(rows).to_csv(out_dir / "sensitivity_halflife8.csv", index=False)


def tskirt_calibrated(out_dir: Path, bar: str) -> None:
    """Validation log-loss of the stored temperature-calibrated T-SKIRT, which is not rescored."""
    files = [(PROC / "models_ewma_grid_h1.parquet", [bar]),
             (PROC / "models_calibrated_h1.parquet", [TSKIRT, TSKIRT_CAL])]
    df, _ = align_slices(load(files))
    val = summarize(df[df["block"] == "val"], by=["predictor"], n_boot=N_BOOT, cluster="skill")
    val["minus_bar"] = val["log_loss"] - val.loc[val["predictor"] == bar, "log_loss"].iloc[0]
    cols = ["predictor", "n", "log_loss", "log_loss_se", "minus_bar", "auc", "auc_se"]
    val[cols].to_csv(out_dir / "tskirt_calibrated_validation.csv", index=False)
    print("\nT-SKIRT with temperature calibration (stored in models_calibrated_h1, not rescored against the")
    print("bar), validation block, horizon 1. minus_bar is validation log-loss minus the bar's.")
    print(val[cols].set_index("predictor").round(4).to_string())


# -- part 2 --------------------------------------------------------------------------------


def clock(out_dir: Path, bar: str) -> None:
    grids = {"T-SKIRT": (PROC / "models_clock_tskirt_grid_h1.parquet", TSKIRT,
                         PROC / "models_calibrated_h1.parquet"),
             "DriftIRT": (PROC / "models_clock_drift_grid_h1.parquet", DRIFT,
                          PROC / "models_drift_grid_h1.parquet")}
    print(f"\n\n{'=' * 100}\nSection 2: the tokens-seen clock (time_col = tokens_seen_b)\n{'=' * 100}")
    if not all(p.exists() for p, _, _ in grids.values()):
        print("(clock grids missing; section skipped)")
        return
    val_rows, chosen = [], {}
    for family, (path, ckpt, ckpt_path) in grids.items():
        g = pd.read_parquet(path)
        ref = pd.read_parquet(ckpt_path)
        g = with_blocks(pd.concat([g, ref[ref["predictor"] == ckpt]], ignore_index=True), 94)
        val = summarize(g[g["block"] == "val"], by=["predictor"], n_boot=N_BOOT, cluster="skill")
        val = val.sort_values("log_loss").reset_index(drop=True)
        val.insert(0, "family", family)
        val["clock"] = ["checkpoint_idx" if p == ckpt else "tokens_seen_b" for p in val["predictor"]]
        val_rows.append(val)
        tok = val[val["clock"] == "tokens_seen_b"]
        chosen[family] = (tok.iloc[0]["predictor"], ckpt)
        print(f"\n{family}, validation block, horizon 1 (the checkpoint-clock setting for reference)")
        print(val.set_index("predictor")[["clock", "log_loss", "log_loss_se", "auc", "auc_se"]].round(4).to_string())
        print(f"Chosen on the tokens clock: {chosen[family][0]}")
    pd.concat(val_rows, ignore_index=True).to_csv(out_dir / "clock_validation.csv", index=False)

    rows = []
    for h in HORIZONS:
        rows += _clock_rows(h, bar, chosen, grids, "re-tuned on validation")
    # h = 1 only: the exact rescaling of each checkpoint-clock setting, so clock and tuning are not mixed
    matched = {family: (MATCHED[family], ckpt) for family, (_, ckpt) in chosen.items()}
    rows += _clock_rows(1, bar, matched, grids, "matched parameters")
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "clock_comparison.csv", index=False)
    show = comp[comp["scope"].isin(["test", "pooled"])]
    idx = ["tuning", "family", "comparison", "h", "scope", "metric"]
    cols = ["a_value", "b_value", "diff", "ci_low", "ci_high", "reading"]
    print("\nPaired differences (95% bootstrap interval), test block and pooled run.")
    print("Re-tuned on validation: each family's tokens-clock setting chosen on validation (DriftIRT's has a")
    print("q_theta ten times smaller than the rescaled checkpoint value, so clock and tuning are confounded).")
    print("Matched parameters: the tokens-clock setting is the exact rescaling of the checkpoint-clock setting")
    print("(DriftIRT q_theta = q_delta = 0.01/42, stored as 0.000238095; T-SKIRT nu = 1/sqrt(42), stored as")
    print("0.154303; rounding to six significant figures, no rerun warranted). The matched comparison exists")
    print("only at h = 1; it was not run at h = 5 or 20. For T-SKIRT the two coincide.")
    for tuning in ("re-tuned on validation", "matched parameters"):
        print(f"\n  {tuning}")
        print(show[show["tuning"] == tuning].set_index(idx)[cols].round(4).to_string())


def _clock_rows(h: int, bar: str, chosen: dict[str, tuple[str, str]], grids: dict, tuning: str) -> list[dict]:
    """Paired rows tokens vs checkpoint clock, and each vs the bar, for every family at horizon h."""
    files = [(PROC / ("models_ewma_grid_h1.parquet" if h == 1 else f"models_ewma_best_h{h}.parquet"), [bar])]
    for family, (tok, ckpt) in chosen.items():
        ckpt_file = next(p for p, names in sources(h, bar) if ckpt in names)
        tok_file = grids[family][0] if h == 1 else PROC / f"models_clock_best_h{h}.parquet"
        files += [(ckpt_file, [ckpt]), (tok_file, [tok])]
    if not all(p.exists() for p, _ in files):
        print(f"\n(missing clock_best file for h={h}; skipped)")
        return []
    df, dropped = align_slices(load(files))
    if dropped["n_dropped"].sum():
        print(f"\nh={h} alignment dropped rows:\n{dropped.to_string(index=False)}")
    scores = summarize(df, by=["predictor", "block"]).set_index(["predictor", "block"])
    pooled = summarize(df, by=["predictor"]).set_index("predictor")
    rows = []
    for family, (tok, ckpt) in chosen.items():
        for a, b, what in ((tok, ckpt, "tokens minus checkpoint clock"),
                           (tok, bar, "tokens clock minus bar"),
                           (ckpt, bar, "checkpoint clock minus bar")):
            d = paired_table(df, [(a, b)])
            for _, r in d.iterrows():
                src = pooled if r["scope"] == "pooled" else scores
                key = a if r["scope"] == "pooled" else (a, r["scope"])
                keyb = b if r["scope"] == "pooled" else (b, r["scope"])
                rows.append({"tuning": tuning, "h": h, "family": family, "comparison": what, **r.to_dict(),
                             "a_value": src.loc[key, r["metric"]], "b_value": src.loc[keyb, r["metric"]],
                             "reading": verdict(r["ci_low"], r["ci_high"], r["metric"])})
    return rows


# -- figure --------------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {TSKIRT: "#2a78d6", DRIFT: "#eb6834", BKT_SKILL: "#1baf7a", BKT_FLUID: "#eda100"}


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


def plot(scores: pd.DataFrame, bar: str, out_dir: Path, out: Path) -> None:
    """Left: test log-loss per model vs horizon, the bar as a dashed reference with its SE band.
    Right: paired difference model minus the bar with 95% intervals (0 = the bar)."""
    test = scores[scores["scope"] == "test"]
    diffs = pd.concat([pd.read_csv(out_dir / f"h{h}_paired_differences.csv").assign(h=h) for h in HORIZONS])
    diffs = diffs[(diffs["scope"] == "test") & (diffs["metric"] == "log_loss")]
    xpos = {h: i for i, h in enumerate(HORIZONS)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(SURFACE)

    b = test[test["predictor"] == bar].sort_values("h")
    xb = [xpos[h] for h in b["h"]]
    ax1.fill_between(xb, b["log_loss"] - b["log_loss_se"], b["log_loss"] + b["log_loss_se"],
                     color=INK_SECONDARY, alpha=0.10, linewidth=0)
    ax1.plot(xb, b["log_loss"], "--", color=INK, lw=1.5, label=f"the bar (recency item mean,\nrho {bar[len(EWMA_PREFIX):]}), band one SE")
    im = test[test["predictor"] == ITEM_MEAN].sort_values("h")
    ax1.plot([xpos[h] for h in im["h"]], im["log_loss"], ":", color=MUTED, lw=1.5, marker="o", ms=4,
             label="item mean (the old bar)")
    for k, (name, color) in enumerate(SERIES.items()):
        g = test[test["predictor"] == name].sort_values("h")
        x = [xpos[h] + (k - 1.5) * 0.06 for h in g["h"]]
        ax1.errorbar(x, g["log_loss"], yerr=g["log_loss_se"], color=color, lw=2.0, marker="o", ms=5,
                     capsize=2.5, label=LABELS[name])
        d = diffs[diffs["a"] == name].sort_values("h")
        xd = [xpos[h] + (k - 1.5) * 0.08 for h in d["h"]]
        ax2.errorbar(xd, d["diff"], yerr=[d["diff"] - d["ci_low"], d["ci_high"] - d["diff"]], color=color,
                     lw=0, elinewidth=2.0, marker="o", ms=6, capsize=3, label=LABELS[name])
    ax2.axhline(0, color=INK, lw=1.5, ls="--")
    ax2.text(-0.35, 0.002, "the bar", va="bottom", ha="left", fontsize=7.5, color=INK_SECONDARY)
    for ax, ylabel, title in (
            (ax1, "test-block log-loss (lower is better)", "Test-block log-loss, error bars one SE"),
            (ax2, "model minus the bar, log-loss (below 0 beats it)", "Paired difference with 95% interval")):
        style_axes(ax)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([str(h) for h in HORIZONS])
        ax.set_xlim(-0.4, len(HORIZONS) - 0.6)
        ax.set_xlabel("horizon (checkpoints ahead)", fontsize=8, color=INK_SECONDARY)
        ax.set_ylabel(ylabel, fontsize=8, color=INK_SECONDARY)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax1.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(2.34, 1.0), borderaxespad=0.0)
    fig.suptitle("Learner models against the recency-weighted bar, OLMo-2 MMLU test block (checkpoints 75 to 93)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.905, "Settings chosen on the validation block (targets 56 to 74) and rescored, not re-tuned. "
             "h = 5 and 20 are retrospective:\nsettings were chosen on targets up to 74, later than some (h = 5) or all (h = 20) of those "
             "forecast origins. "
             "SEs and intervals: bootstrap over the 57 subjects.",
             fontsize=8, color=INK_SECONDARY, va="top")
    fig.subplots_adjust(left=0.065, right=0.77, top=0.76, bottom=0.13, wspace=0.32)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--skip-clock", action="store_true", help="only part 1")
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / "analysis.txt")

    print("Section 1: every learner model rescored against the recency-weighted item mean")
    print("OLMo-2 7B, MMLU only. Blocks by target checkpoint: train 0-55, val 56-74, test 75-93. The learner")
    print("settings were chosen on validation earlier (against item_mean for T-SKIRT and DriftIRT); they are")
    print("rescored here, not re-tuned against the new bar.")
    print("Look-ahead: every hyperparameter (rho, q, nu) was chosen on validation targets 56-74. At h=20 every")
    print("test forecast originates at 55-73 and at h=5 the first four originate before 74, so tuning saw")
    print("observations later than those forecast origins (test labels stay held out). The h=5 and h=20")
    print("comparisons, paired ones included, are retrospective; h=1 is unaffected. Follow-up: select on")
    print("targets up to 54 and test from 75.")
    print("Readings: 'a better' / 'b better' when the 95% interval excludes zero, otherwise 'no demonstrated")
    print("difference'. These are measurements, not explanations: why a model wins is not tested here.\n")
    bar = choose_rho(out_dir)
    scores = rescore(bar, out_dir)
    fig = FIG_DIR / "bar_rescoring.png"
    plot(scores, bar, out_dir, fig)
    print(f"\nWrote {fig.relative_to(REPO_ROOT)}")
    sensitivity(out_dir, bar)
    tskirt_calibrated(out_dir, bar)
    if not args.skip_clock:
        clock(out_dir, bar)


if __name__ == "__main__":
    main()
