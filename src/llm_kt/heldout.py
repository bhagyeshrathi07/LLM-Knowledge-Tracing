"""Within-checkpoint held-out-item diagnosis and cross-benchmark transfer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from llm_kt.evaluate import assign_blocks
from llm_kt.metrics import score
from llm_kt.tskirt import TSKIRT


def split_items(
    responses: pd.DataFrame, holdout_frac: float = 0.5, seed: int = 0
) -> tuple[set[str], set[str]]:
    """Split items within each skill so both fit and held-out sets contain every skill."""
    if not 0 < holdout_frac < 1:
        raise ValueError(f"holdout_frac must be in (0, 1), got {holdout_frac}")
    item_skills = responses[["item_id", "skill"]].drop_duplicates()
    if item_skills["item_id"].duplicated().any():
        raise ValueError("each item_id must belong to exactly one skill")

    rng = np.random.default_rng(seed)
    fit_ids: set[str] = set()
    heldout_ids: set[str] = set()
    for _, group in item_skills.groupby("skill", sort=True):
        ids = np.asarray(sorted(group["item_id"].astype(str)))
        if len(ids) < 2:
            raise ValueError(f"each skill needs at least two items, got {len(ids)}")
        n_holdout = min(max(round(len(ids) * holdout_frac), 1), len(ids) - 1)
        shuffled = rng.permutation(ids)
        heldout_ids.update(shuffled[:n_holdout])
        fit_ids.update(shuffled[n_holdout:])
    return fit_ids, heldout_ids


def _block_by_checkpoint(n_checkpoints: int) -> dict[int, str]:
    ranges = assign_blocks(n_checkpoints)
    return {
        idx: name
        for name, (lo, hi) in ranges.items()
        for idx in range(lo, hi + 1)
    }


def evaluate_heldout_items(
    responses: pd.DataFrame,
    item_params: pd.DataFrame,
    model: str,
    nu_values: list[float],
    holdout_frac: float = 0.5,
    seed: int = 0,
    time_col: str = "checkpoint_idx",
) -> tuple[pd.DataFrame, set[str], set[str]]:
    """Fit T-SKIRT on one MMLU item split and score the same checkpoints on the other.

    This is within-checkpoint diagnosis, not a future-checkpoint forecast: the predictor
    sees fit-item responses through the checkpoint being scored, but never sees held-out
    item responses. Hyperparameters are selected on the chronological validation block.
    """
    mmlu = responses[(responses["model"] == model) & (responses["benchmark"] == "mmlu")]
    if mmlu.empty:
        raise ValueError(f"no MMLU responses for model {model!r}")
    fit_ids, heldout_ids = split_items(mmlu, holdout_frac=holdout_frac, seed=seed)
    n_checkpoints = int(mmlu["checkpoint_idx"].max()) + 1
    blocks = _block_by_checkpoint(n_checkpoints)
    params = item_params[item_params["item_id"].isin(mmlu["item_id"].unique())]
    rows = []

    for checkpoint_idx in sorted(mmlu["checkpoint_idx"].unique()):
        checkpoint_idx = int(checkpoint_idx)
        scope = blocks[checkpoint_idx]
        fit_history = mmlu[
            mmlu["item_id"].isin(fit_ids) & (mmlu["checkpoint_idx"] <= checkpoint_idx)
        ]
        fit_now = fit_history[fit_history["checkpoint_idx"] == checkpoint_idx]
        target_rows = mmlu[
            mmlu["item_id"].isin(heldout_ids)
            & (mmlu["checkpoint_idx"] == checkpoint_idx)
        ]
        if fit_now.empty or target_rows.empty:
            continue
        target = target_rows.drop(columns="correct").reset_index(drop=True)
        y = target_rows["correct"].to_numpy()
        skill_mean = fit_now.groupby("skill")["correct"].mean()
        global_mean = float(fit_now["correct"].mean())
        predictions = {
            "skill_mean": target["skill"].map(skill_mean).to_numpy(dtype=float),
            "global_mean": np.full(len(target), global_mean),
        }
        for nu in nu_values:
            predictor = TSKIRT(params, nu=nu, time_col=time_col)
            predictions[predictor.name] = predictor.predict(fit_history, target, horizon=1)

        for name, probability in predictions.items():
            for skill, indices in target.groupby("skill").indices.items():
                rows.append({
                    "model": model,
                    "benchmark": "mmlu",
                    "checkpoint_idx": checkpoint_idx,
                    "block": scope,
                    "skill": skill,
                    "predictor": name,
                    "nu": next(
                        (nu for nu in nu_values if name == f"tskirt_nu{nu:g}_{time_col}"),
                        np.nan,
                    ),
                    "n": len(indices),
                    **score(y[indices], probability[indices]),
                })
    return pd.DataFrame(rows), fit_ids, heldout_ids


def _fit_linear_map(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit an intercept and slope, failing explicitly on insufficient calibration data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        raise ValueError("cross-benchmark calibration needs at least two finite train checkpoints")
    design = np.column_stack((np.ones(valid.sum()), x[valid]))
    intercept, slope = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    return float(intercept), float(slope)


def evaluate_benchmark_transfer(
    responses: pd.DataFrame,
    item_params: pd.DataFrame,
    model: str,
    fit_item_ids: set[str],
    nu: float,
    benchmarks: list[str],
    time_col: str = "checkpoint_idx",
) -> pd.DataFrame:
    """Transfer an MMLU skill-trajectory summary to later same-checkpoint benchmark scores.

    A linear calibration from mean fitted MMLU ability to each target benchmark's
    checkpoint accuracy is learned on the chronological train block only. Test-block
    benchmark outcomes are used only for scoring. The MMLU mean-accuracy calibration and
    a train-block prevalence predictor are included as controls.
    """
    mmlu = responses[
        (responses["model"] == model)
        & (responses["benchmark"] == "mmlu")
        & responses["item_id"].isin(fit_item_ids)
    ]
    if mmlu.empty:
        raise ValueError(f"no MMLU fit-item responses for model {model!r}")
    n_checkpoints = int(responses[responses["model"] == model]["checkpoint_idx"].max()) + 1
    blocks = _block_by_checkpoint(n_checkpoints)
    params = item_params[item_params["item_id"].isin(mmlu["item_id"].unique())]
    trajectory_rows = []
    for checkpoint_idx in sorted(mmlu["checkpoint_idx"].unique()):
        checkpoint_idx = int(checkpoint_idx)
        history = mmlu[mmlu["checkpoint_idx"] <= checkpoint_idx]
        predictor = TSKIRT(params, nu=nu, time_col=time_col)
        origin_time = float(predictor._time(history).max())
        thetas = predictor.fit_skill_theta(history, origin_time)
        trajectory_rows.append({
            "checkpoint_idx": checkpoint_idx,
            "block": blocks[checkpoint_idx],
            "mean_theta": float(np.mean(list(thetas.values()))),
            "mmlu_accuracy": float(
                mmlu[mmlu["checkpoint_idx"] == checkpoint_idx]["correct"].mean()
            ),
        })
    trajectory = pd.DataFrame(trajectory_rows).set_index("checkpoint_idx")
    train_idx = trajectory.index[trajectory["block"] == "train"]
    if len(train_idx) < 2:
        raise ValueError("cross-benchmark transfer needs at least two train checkpoints")
    rows = []

    for benchmark in benchmarks:
        target = responses[
            (responses["model"] == model) & (responses["benchmark"] == benchmark)
        ]
        if target.empty:
            raise ValueError(f"no {benchmark!r} responses for model {model!r}")
        observed = target.groupby("checkpoint_idx")["correct"].agg(["mean", "count"])
        common = trajectory.index.intersection(observed.index)
        train_common = [idx for idx in train_idx if idx in common]
        target_mean = observed.loc[train_common, "mean"].to_numpy(dtype=float)
        baseline = float(np.average(
            target_mean,
            weights=observed.loc[train_common, "count"].to_numpy(dtype=float),
        ))
        transfer_models = {
            "mmlu_theta_transfer": "mean_theta",
            "mmlu_accuracy_transfer": "mmlu_accuracy",
        }
        for predictor_name, feature in transfer_models.items():
            intercept, slope = _fit_linear_map(
                trajectory.loc[train_common, feature].to_numpy(dtype=float), target_mean
            )
            for checkpoint_idx in common:
                scope = blocks[int(checkpoint_idx)]
                target_rows = target[target["checkpoint_idx"] == checkpoint_idx]
                y = target_rows["correct"].to_numpy()
                x = float(trajectory.loc[checkpoint_idx, feature])
                p = float(np.clip(intercept + slope * x, 1e-6, 1 - 1e-6))
                rows.append({
                    "model": model,
                    "benchmark": benchmark,
                    "checkpoint_idx": int(checkpoint_idx),
                    "block": scope,
                    "predictor": predictor_name,
                    "n": len(y),
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                    **score(y, np.full(len(y), p)),
                })
        for checkpoint_idx in common:
            target_rows = target[target["checkpoint_idx"] == checkpoint_idx]
            y = target_rows["correct"].to_numpy()
            rows.append({
                "model": model,
                "benchmark": benchmark,
                "checkpoint_idx": int(checkpoint_idx),
                "block": blocks[int(checkpoint_idx)],
                "predictor": "train_prevalence",
                "n": len(y),
                "calibration_intercept": np.nan,
                "calibration_slope": np.nan,
                **score(y, np.full(len(y), np.clip(baseline, 1e-6, 1 - 1e-6))),
            })
    return pd.DataFrame(rows)


def select_nu(validation_scores: pd.DataFrame) -> float:
    """Select T-SKIRT drift by item-weighted validation log-loss."""
    candidates = validation_scores[validation_scores["predictor"].str.startswith("tskirt_nu")]
    if candidates.empty:
        raise ValueError("no T-SKIRT validation scores to select from")
    losses = candidates.groupby("nu").apply(
        lambda g: np.average(g["log_loss"], weights=g["n"]), include_groups=False
    )
    return float(losses.idxmin())
