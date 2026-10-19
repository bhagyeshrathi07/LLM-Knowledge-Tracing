"""Rolling next-checkpoint evaluation.

For one model, walk forward through its checkpoints. At each origin t (with at least
`min_history` checkpoints observed), the predictor sees every response at checkpoints
<= t and must output P(correct) for every item at checkpoint t + horizon. Predictions
are scored against the observed correctness at that target checkpoint.

The split is strictly by time. Item identity is shared between history and target
(the same benchmark items are re-asked at every checkpoint), which is the point:
the question is whether the learner state inferred so far predicts the next reading.

Validation and test blocks. `assign_blocks` splits each model's target checkpoints
chronologically into "train", "val" and "test" (the last 20% of checkpoints are test,
the 20% before that are val). Every result row carries a `block` column naming the
block its target checkpoint falls in, computed per model since models have different
checkpoint counts. Choose hyperparameters and model variants on "val"
(`evaluate(..., blocks="val")`) and report final numbers on "test" only, once.

Blocks restrict which targets are scored, not what the predictor sees. An origin whose
target is in the test block still receives the full history up to that origin,
including the val-block checkpoints, because this is forecasting rather than a random
split: at deployment time a learner model would have seen everything before the
checkpoint it is asked about.

Uncertainty. Slices within one skill are strongly correlated across checkpoints, so
`summarize(..., n_boot=...)` and `paired_difference` use a cluster bootstrap that
resamples whole skills (by default) rather than individual rows.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import pandas as pd

from llm_kt.metrics import score

TARGET_COLUMNS = ["item_id", "benchmark", "skill", "checkpoint_idx", "step", "tokens_seen_b"]
BLOCK_NAMES = ("train", "val", "test")
METRICS = ("log_loss", "brier", "accuracy", "auc")


class Predictor(Protocol):
    name: str

    def predict(self, history: pd.DataFrame, target: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Return P(correct) for each row of `target`, in order.

        history: long-format rows for one model, all checkpoints <= origin.
        target:  rows at the target checkpoint with columns item_id, benchmark, skill,
                 checkpoint_idx, step, tokens_seen_b (no `correct` column). The step and
                 tokens_seen_b columns give the target's position in training where the
                 source release provides it (null otherwise), so a temporal model can
                 compute the elapsed gap from the last history checkpoint.
        horizon: number of checkpoints between the origin and the target (>= 1). The
                 trivial baselines ignore it.
        """
        ...


def rolling_origins(n_checkpoints: int, min_history: int, horizon: int) -> list[tuple[int, int]]:
    """(origin, target) checkpoint index pairs for a walk-forward evaluation."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1 (target strictly after origin), got {horizon}")
    if min_history < 1:
        raise ValueError(f"min_history must be >= 1, got {min_history}")
    return [
        (t, t + horizon)
        for t in range(min_history - 1, n_checkpoints - horizon)
    ]


def assign_blocks(
    n_checkpoints: int, test_frac: float = 0.2, val_frac: float = 0.2
) -> dict[str, tuple[int, int]]:
    """Chronological train/val/test blocks over target checkpoint indices.

    Returns inclusive (first, last) index ranges. Test is the last `test_frac` share of
    checkpoints, val the `val_frac` share immediately before it, train everything
    earlier. Block sizes are rounded up, so val and test are never empty; train must
    keep at least one checkpoint.
    """
    if not (0 < test_frac < 1 and 0 < val_frac < 1 and test_frac + val_frac < 1):
        raise ValueError(
            f"need 0 < test_frac, val_frac and test_frac + val_frac < 1, got {test_frac}, {val_frac}"
        )
    if n_checkpoints < 3:
        raise ValueError(f"need at least 3 checkpoints for three blocks, got {n_checkpoints}")
    # round() first so float noise (15 * 0.2 = 3.0000000000000004) does not bump the ceil
    n_test = math.ceil(round(n_checkpoints * test_frac, 9))
    n_val = math.ceil(round(n_checkpoints * val_frac, 9))
    n_train = n_checkpoints - n_test - n_val
    if n_train < 1:
        raise ValueError(
            f"{n_checkpoints} checkpoints leave no train block with test_frac={test_frac}, "
            f"val_frac={val_frac}"
        )
    return {
        "train": (0, n_train - 1),
        "val": (n_train, n_train + n_val - 1),
        "test": (n_train + n_val, n_checkpoints - 1),
    }


def _block_of(idx: int, block_ranges: dict[str, tuple[int, int]]) -> str:
    for name, (lo, hi) in block_ranges.items():
        if lo <= idx <= hi:
            return name
    raise ValueError(f"checkpoint index {idx} is outside every block")


def _normalize_blocks(blocks: str | list[str] | None) -> set[str] | None:
    if blocks is None:
        return None
    chosen = {blocks} if isinstance(blocks, str) else set(blocks)
    unknown = chosen - set(BLOCK_NAMES)
    if unknown or not chosen:
        raise ValueError(f"blocks must be drawn from {BLOCK_NAMES}, got {blocks!r}")
    return chosen


def evaluate_model(
    responses: pd.DataFrame,
    predictor: Predictor,
    min_history: int = 2,
    horizon: int = 1,
    by_skill: bool = True,
    blocks: str | list[str] | None = None,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
) -> pd.DataFrame:
    """Walk forward over one model's checkpoints and score the predictor at each target.

    Returns one row per (target_checkpoint, skill) if by_skill, else per target_checkpoint,
    with columns: predictor, model, origin_idx, target_idx, block, [skill], n, log_loss,
    brier, accuracy, auc. `block` is the `assign_blocks` block of target_idx. If `blocks`
    is given, origins whose target lies outside those blocks are skipped (the predictor
    is not called for them); history is never truncated.
    """
    models = responses["model"].unique()
    if len(models) != 1:
        raise ValueError(f"evaluate_model expects one model, got {list(models)}")
    model = models[0]

    chosen = _normalize_blocks(blocks)
    responses = responses.sort_values(["checkpoint_idx", "item_id"]).reset_index(drop=True)
    n_ckpt = int(responses["checkpoint_idx"].max()) + 1
    origins = rolling_origins(n_ckpt, min_history, horizon)
    block_ranges = assign_blocks(n_ckpt, test_frac, val_frac) if origins else {}
    rows = []

    for origin, target_idx in origins:
        block = _block_of(target_idx, block_ranges)
        if chosen is not None and block not in chosen:
            continue
        history = responses[responses["checkpoint_idx"] <= origin]
        target_rows = responses[responses["checkpoint_idx"] == target_idx]
        target = target_rows[TARGET_COLUMNS].reset_index(drop=True)
        y = target_rows["correct"].to_numpy()
        p = np.asarray(predictor.predict(history, target, horizon=horizon), dtype=float)
        if p.shape != y.shape:
            raise ValueError(f"{predictor.name}: predicted {p.shape}, expected {y.shape}")
        if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
            raise ValueError(f"{predictor.name}: predictions must be finite probabilities in [0, 1]")

        base = {
            "predictor": predictor.name,
            "model": model,
            "origin_idx": origin,
            "target_idx": target_idx,
            "block": block,
        }
        if by_skill:
            for skill, idx in target.groupby("skill").indices.items():
                rows.append({**base, "skill": skill, **score(y[idx], p[idx])})
        else:
            rows.append({**base, **score(y, p)})

    return pd.DataFrame(rows)


def evaluate(
    responses: pd.DataFrame,
    predictors: list[Predictor],
    min_history: int = 2,
    horizon: int = 1,
    by_skill: bool = True,
    blocks: str | list[str] | None = None,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
) -> pd.DataFrame:
    """Run every predictor on every model. Concatenated per-(model, target, skill) scores.

    `blocks` (e.g. "val", or ["val", "test"]) restricts scoring to targets in those
    chronological blocks, assigned per model; see `assign_blocks` and the module docstring.
    """
    _normalize_blocks(blocks)  # fail fast on a typo before running anything
    frames = []
    for model, group in responses.groupby("model", sort=True):
        for predictor in predictors:
            frames.append(
                evaluate_model(
                    group,
                    predictor,
                    min_history=min_history,
                    horizon=horizon,
                    by_skill=by_skill,
                    blocks=blocks,
                    test_frac=test_frac,
                    val_frac=val_frac,
                )
            )
    return pd.concat(frames, ignore_index=True)


def _cluster_ids(g: pd.DataFrame, cluster: str | None) -> np.ndarray:
    """Integer cluster label per row of g (each row its own cluster when cluster is None)."""
    if cluster is None:
        return np.arange(len(g))
    return pd.factorize(g[cluster], sort=True)[0]


def _boot_counts(rng: np.random.Generator, n_clusters: int, n_boot: int) -> np.ndarray:
    """(n_boot, n_clusters) matrix: how often each cluster is drawn in each replicate."""
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    counts = np.zeros((n_boot, n_clusters))
    np.add.at(counts, (np.arange(n_boot)[:, None], draws), 1)
    return counts


def _check_cluster(results: pd.DataFrame, cluster: str | None) -> None:
    if cluster is not None and cluster not in results.columns:
        raise ValueError(
            f"cluster column {cluster!r} not in results; pass cluster=None to resample rows"
        )


def summarize(
    results: pd.DataFrame,
    by: list[str] | None = None,
    cluster: str | None = "skill",
    n_boot: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate per-slice scores, weighting each slice by its item count.

    Default grouping is (predictor, model). AUC is averaged unweighted over slices
    where it is defined, since it does not pool across slices the way the others do.

    With n_boot > 0, adds log_loss_se, brier_se, accuracy_se, auc_se: the standard
    deviation of each statistic over n_boot cluster-bootstrap replicates. Within each
    summary group, clusters (default: skills; None: individual rows) are resampled with
    replacement and the statistics recomputed from all rows of the drawn clusters. A
    group with a single cluster gets SE 0, which means "not estimable", not "certain".
    """
    by = by or ["predictor", "model"]
    if n_boot > 0:
        _check_cluster(results, cluster)
    rng = np.random.default_rng(seed)
    w = results["n"]

    def agg(g: pd.DataFrame) -> pd.Series:
        wg = w.loc[g.index]
        out = {
            "n": int(wg.sum()),
            "log_loss": float(np.average(g["log_loss"], weights=wg)),
            "brier": float(np.average(g["brier"], weights=wg)),
            "accuracy": float(np.average(g["accuracy"], weights=wg)),
            "auc": float(g["auc"].mean(skipna=True)),
        }
        if n_boot > 0:
            out.update(_bootstrap_se(g, wg.to_numpy(dtype=float), _cluster_ids(g, cluster)))
        return pd.Series(out)

    def _bootstrap_se(g: pd.DataFrame, wg: np.ndarray, ids: np.ndarray) -> dict[str, float]:
        k = int(ids.max()) + 1
        n_c = np.bincount(ids, weights=wg, minlength=k)
        counts = _boot_counts(rng, k, n_boot)
        total_n = counts @ n_c
        se = {}
        for m in ("log_loss", "brier", "accuracy"):
            s_c = np.bincount(ids, weights=wg * g[m].to_numpy(dtype=float), minlength=k)
            se[f"{m}_se"] = float(np.std((counts @ s_c) / total_n, ddof=1))
        a = g["auc"].to_numpy(dtype=float)
        ok = ~np.isnan(a)
        auc_sum = np.bincount(ids[ok], weights=a[ok], minlength=k)
        auc_cnt = np.bincount(ids[ok], minlength=k).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            auc_boot = (counts @ auc_sum) / (counts @ auc_cnt)
        auc_boot = auc_boot[np.isfinite(auc_boot)]
        se["auc_se"] = float(np.std(auc_boot, ddof=1)) if len(auc_boot) > 1 else float("nan")
        return se

    out = results.groupby(by).apply(agg, include_groups=False).reset_index()
    out["n"] = out["n"].astype(int)
    return out


def paired_difference(
    results: pd.DataFrame,
    predictor_a: str,
    predictor_b: str,
    by: list[str] | None = None,
    metric: str = "log_loss",
    cluster: str | None = "skill",
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean metric difference (a minus b) per group, with a paired cluster bootstrap.

    Slices are paired on (model, origin_idx, target_idx, skill) (whichever of those exist,
    plus `block` and any `by` columns); slices scored for only one predictor are dropped.
    `by` defaults to ["model"]; pass [] to pool everything. The difference is the
    item-count-weighted mean of per-slice differences, except for auc, which is the
    unweighted mean over slices where both AUCs are defined (matching `summarize`).

    Clusters (default: skills; None: individual slices) are resampled with replacement
    within each group, and the same draw is applied to both predictors because each
    resampled unit is a paired slice difference. Returns columns: by..., n, n_slices,
    n_clusters, diff, se (bootstrap SD), ci_low, ci_high (95% percentile interval).
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    by = ["model"] if by is None else list(by)
    _check_cluster(results, cluster)
    keys = [c for c in ("model", "origin_idx", "target_idx", "skill", "block") if c in results]
    keys += [c for c in by if c not in keys]
    if cluster is not None and cluster not in keys:
        keys.append(cluster)

    a = results[results["predictor"] == predictor_a]
    b = results[results["predictor"] == predictor_b]
    if a.empty or b.empty:
        missing = predictor_a if a.empty else predictor_b
        raise ValueError(f"no results for predictor {missing!r}")
    for name, frame in (("a", a), ("b", b)):
        if frame.duplicated(keys).any():
            raise ValueError(f"predictor_{name} has duplicate slices on {keys}")
    paired = a[keys + ["n", metric]].merge(
        b[keys + ["n", metric]], on=keys, suffixes=("_a", "_b"), validate="one_to_one"
    )
    if not (paired["n_a"] == paired["n_b"]).all():
        raise ValueError("paired slices have different item counts; were they scored on the same data?")
    paired["d"] = paired[f"{metric}_a"] - paired[f"{metric}_b"]
    if metric == "auc":
        paired = paired[paired["d"].notna()].copy()
        paired["w"] = 1.0
    else:
        paired["w"] = paired["n_a"].astype(float)

    if paired.empty:
        raise ValueError(f"no slices shared by {predictor_a!r} and {predictor_b!r}")
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot}")

    rng = np.random.default_rng(seed)
    groups = paired.groupby(by, sort=True) if by else [((), paired)]
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        ids = _cluster_ids(g, cluster)
        k = int(ids.max()) + 1
        w_c = np.bincount(ids, weights=g["w"].to_numpy(), minlength=k)
        s_c = np.bincount(ids, weights=(g["w"] * g["d"]).to_numpy(), minlength=k)
        counts = _boot_counts(rng, k, n_boot)
        boot = (counts @ s_c) / (counts @ w_c)
        rows.append(
            {
                **dict(zip(by, key)),
                "n": int(g["n_a"].sum()),
                "n_slices": len(g),
                "n_clusters": k,
                "diff": float(s_c.sum() / w_c.sum()),
                "se": float(np.std(boot, ddof=1)),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
            }
        )
    return pd.DataFrame(rows)
