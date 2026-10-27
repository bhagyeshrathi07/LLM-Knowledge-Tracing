"""The baseline ladder: how much each level of granularity adds to prediction.

Three trivial predictors form a ladder of increasing granularity: `global_mean` knows
the model's overall accuracy, `skill_mean` knows its accuracy on the item's subject,
and `item_mean` knows the item's own history. This module aggregates their scores and
decomposes the variance of correctness to show where the predictive information lives.

Ladder aggregation. Pooled log-loss is the item-count-weighted mean of per-slice
log-loss and pooled AUC is the unweighted mean of per-slice AUC, exactly as in
`llm_kt.evaluate.summarize`. Two gaps are reported in log-loss:

    skill_gap = log_loss(global_mean) - log_loss(skill_mean)
    item_gap  = log_loss(skill_mean)  - log_loss(item_mean)

Positive gaps mean the finer predictor is better. Note that AUC is computed within
(model, checkpoint, skill) slices, so global_mean and skill_mean, which are constant
within a skill, score 0.5 by construction.

Variance decomposition. For one model at checkpoint t, let y_i in {0, 1} be the
correctness of item i (i = 1..N), s(i) its skill, N_s the item count of skill s,
p_s the mean of y over skill s and p the mean over all items. With population
(ddof=0) variances, the law of total variance holds exactly:

    total_var   = p (1 - p)                              = Var(y)
    between_var = sum_s (N_s / N) (p_s - p)^2            = Var(E[y | skill])
    within_var  = sum_s (N_s / N) p_s (1 - p_s)          = E[Var(y | skill)]
    total_var   = between_var + within_var
    between_frac = between_var / total_var

between_frac equals the R^2 of an OLS regression of y on skill indicators. It uses the
skill accuracies at checkpoint t itself, so it is an in-sample upper bound on what
skill identity can explain at t.

Item running mean. Let m_i(t) be the mean of item i's correctness over checkpoints
0..t-1 (strictly before t, so it never contains y_i(t); undefined at t = 0). We report

    item_mean_var = Var_i(m_i(t))                         (population variance)
    item_mean_r2  = corr_i(m_i(t), y_i(t))^2

item_mean_r2 is the R^2 of an OLS regression of y_i(t) on m_i(t) with an intercept,
computed over items with both values defined. Unlike between_frac it uses only past
checkpoints, so it is an out-of-time quantity. It is NaN when either variable is
constant across items.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from llm_kt.evaluate import summarize

LADDER = ("global_mean", "skill_mean", "item_mean")
ALL = "all"


def skill_to_benchmark(responses: pd.DataFrame) -> pd.Series:
    """Map each skill to its benchmark. Raises if a skill appears under two benchmarks."""
    pairs = responses[["skill", "benchmark"]].drop_duplicates()
    if pairs["skill"].duplicated().any():
        dup = pairs.loc[pairs["skill"].duplicated(), "skill"].tolist()
        raise ValueError(f"skills under more than one benchmark: {dup}")
    return pairs.set_index("skill")["benchmark"]


def add_gaps(wide: pd.DataFrame) -> pd.DataFrame:
    """Add skill_gap and item_gap columns (log-loss differences) to a wide ladder frame."""
    out = wide.copy()
    out["skill_gap"] = out["log_loss_global_mean"] - out["log_loss_skill_mean"]
    out["item_gap"] = out["log_loss_skill_mean"] - out["log_loss_item_mean"]
    return out


def pooled_ladder(results: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Pooled log-loss and AUC of the three ladder predictors per group, one row per group.

    `results` holds per-slice baseline scores (columns predictor, n, log_loss, auc, ...).
    Returns columns by..., n, log_loss_<predictor>, auc_<predictor> for each ladder
    predictor, skill_gap and item_gap. Raises if the predictors were scored on different
    item counts within a group, since the comparison would then not be paired.
    """
    sub = results[results["predictor"].isin(LADDER)]
    missing = set(LADDER) - set(sub["predictor"])
    if missing:
        raise ValueError(f"results lack ladder predictors: {sorted(missing)}")
    s = summarize(sub, by=[*by, "predictor"])
    n = s.pivot_table(index=by, columns="predictor", values="n", aggfunc="first")
    if not n.eq(n.iloc[:, 0], axis=0).all().all():
        raise ValueError("ladder predictors have different item counts within a group")
    wide = s.pivot_table(index=by, columns="predictor", values=["log_loss", "auc"])
    wide.columns = [f"{metric}_{pred}" for metric, pred in wide.columns]
    cols = [f"{m}_{p}" for m in ("log_loss", "auc") for p in LADDER]
    wide = wide[cols]
    wide.insert(0, "n", n.iloc[:, 0].astype(int))
    return add_gaps(wide.reset_index())


def ladder_table(results: pd.DataFrame, skill_bench: pd.Series) -> pd.DataFrame:
    """Ladder per (model, benchmark, horizon), with "all" rows for model and benchmark.

    Rows: every (model, benchmark, horizon); benchmark="all" per (model, horizon);
    model="all" per (benchmark, horizon); and model="all", benchmark="all" per horizon.
    """
    df = results[results["predictor"].isin(LADDER)].copy()
    df["benchmark"] = df["skill"].map(skill_bench)
    if df["benchmark"].isna().any():
        unknown = sorted(df.loc[df["benchmark"].isna(), "skill"].unique())
        raise ValueError(f"skills with no benchmark mapping: {unknown}")
    frames = []
    for model_all in (False, True):
        for bench_all in (False, True):
            d = df.copy()
            if model_all:
                d["model"] = ALL
            if bench_all:
                d["benchmark"] = ALL
            frames.append(pooled_ladder(d, by=["model", "benchmark", "horizon"]))
    out = pd.concat(frames, ignore_index=True)
    # put the "all" rows last within each sort key
    key_m = out["model"].eq(ALL)
    key_b = out["benchmark"].eq(ALL)
    out = out.assign(_m=key_m, _b=key_b).sort_values(
        ["horizon", "_m", "model", "_b", "benchmark"]
    )
    return out.drop(columns=["_m", "_b"]).reset_index(drop=True)


def ladder_over_training(results: pd.DataFrame) -> pd.DataFrame:
    """Ladder per (model, target checkpoint), pooled over the skills present in `results`.

    Adds target_norm = target_idx / the model's largest target_idx in `results`.
    """
    out = pooled_ladder(results, by=["model", "target_idx"])
    last = out.groupby("model")["target_idx"].transform("max")
    out.insert(2, "target_norm", out["target_idx"] / last)
    return out


def gap_trend(over_training: pd.DataFrame, column: str = "item_gap") -> pd.DataFrame:
    """Summarize how a gap column changes over training, per model.

    Reports the OLS slope of `column` on target_norm (change per full run), the
    Spearman rank correlation with target_idx, and the mean over the first and last
    third of targets (by target_norm).
    """
    rows = []
    for model, g in over_training.groupby("model", sort=True):
        x = g["target_norm"].to_numpy(dtype=float)
        y = g[column].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(g) > 1 else float("nan")
        rho = float(g["target_idx"].rank().corr(g[column].rank())) if len(g) > 1 else float("nan")
        rows.append(
            {
                "model": model,
                "n_targets": len(g),
                f"{column}_slope": slope,
                f"{column}_spearman": rho,
                f"{column}_first_third": float(y[x <= 1 / 3].mean()),
                f"{column}_last_third": float(y[x > 2 / 3].mean()),
            }
        )
    return pd.DataFrame(rows)


def _skill_decomposition(y: np.ndarray, skill_codes: np.ndarray) -> dict[str, float]:
    """Law-of-total-variance terms for one checkpoint's 0/1 vector over observed items."""
    n = len(y)
    p = y.mean()
    n_s = np.bincount(skill_codes)
    keep = n_s > 0
    p_s = np.bincount(skill_codes, weights=y)[keep] / n_s[keep]
    w_s = n_s[keep] / n
    total = p * (1 - p)
    between = float(np.sum(w_s * (p_s - p) ** 2))
    within = float(np.sum(w_s * p_s * (1 - p_s)))
    return {
        "n_items": n,
        "n_skills": int(keep.sum()),
        "accuracy": float(p),
        "total_var": float(total),
        "between_var": between,
        "within_var": within,
        "between_frac": between / total if total > 0 else float("nan"),
    }


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    """R^2 of an OLS fit of y on x with intercept (squared Pearson correlation)."""
    if len(x) < 2:
        return float("nan")
    dx = x - x.mean()
    dy = y - y.mean()
    sxx = float(dx @ dx)
    syy = float(dy @ dy)
    if sxx == 0 or syy == 0:
        return float("nan")
    return float((dx @ dy) ** 2 / (sxx * syy))


def variance_decomposition(responses: pd.DataFrame) -> pd.DataFrame:
    """Per (model, checkpoint_idx): skill decomposition and item running-mean R^2.

    `responses` has columns model, checkpoint_idx, skill, item_id, correct (filter to
    the benchmark of interest beforehand). Formulas are in the module docstring.
    Items missing at a checkpoint are left out of that checkpoint, and the running mean
    averages only the checkpoints where the item was observed.
    """
    rows = []
    for model, g in responses.groupby("model", sort=True):
        wide = g.pivot(index="item_id", columns="checkpoint_idx", values="correct").sort_index(
            axis=1
        )
        vals = wide.to_numpy(dtype=float)
        skill_of = g.drop_duplicates("item_id").set_index("item_id")["skill"]
        codes = pd.factorize(skill_of.loc[wide.index], sort=True)[0]
        observed = ~np.isnan(vals)
        cum_k = np.cumsum(np.nan_to_num(vals), axis=1)
        cum_n = np.cumsum(observed, axis=1)
        for j, t in enumerate(wide.columns):
            obs = observed[:, j]
            y = vals[obs, j]
            row = {"model": model, "checkpoint_idx": int(t)}
            row.update(_skill_decomposition(y, codes[obs]))
            if j == 0:
                row["item_mean_var"] = float("nan")
                row["item_mean_r2"] = float("nan")
            else:
                n_prev = cum_n[:, j - 1]
                ok = obs & (n_prev > 0)
                m = cum_k[ok, j - 1] / n_prev[ok]
                row["item_mean_var"] = float(m.var())
                row["item_mean_r2"] = _r2(m, vals[ok, j])
            rows.append(row)
    out = pd.DataFrame(rows)
    last = out.groupby("model")["checkpoint_idx"].transform("max")
    out.insert(2, "checkpoint_norm", out["checkpoint_idx"] / last)
    return out
