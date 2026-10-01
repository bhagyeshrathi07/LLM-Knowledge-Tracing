"""Rolling next-checkpoint evaluation.

For one model, walk forward through its checkpoints. At each origin t (with at least
`min_history` checkpoints observed), the predictor sees every response at checkpoints
<= t and must output P(correct) for every item at checkpoint t + horizon. Predictions
are scored against the observed correctness at that target checkpoint.

The split is strictly by time. Item identity is shared between history and target
(the same benchmark items are re-asked at every checkpoint), which is the point:
the question is whether the learner state inferred so far predicts the next reading.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from llm_kt.metrics import score

TARGET_COLUMNS = ["item_id", "benchmark", "skill", "checkpoint_idx", "step", "tokens_seen_b"]


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


def evaluate_model(
    responses: pd.DataFrame,
    predictor: Predictor,
    min_history: int = 2,
    horizon: int = 1,
    by_skill: bool = True,
) -> pd.DataFrame:
    """Walk forward over one model's checkpoints and score the predictor at each target.

    Returns one row per (target_checkpoint, skill) if by_skill, else per target_checkpoint,
    with columns: predictor, model, origin_idx, target_idx, [skill], n, log_loss, brier,
    accuracy, auc.
    """
    models = responses["model"].unique()
    if len(models) != 1:
        raise ValueError(f"evaluate_model expects one model, got {list(models)}")
    model = models[0]

    responses = responses.sort_values(["checkpoint_idx", "item_id"]).reset_index(drop=True)
    n_ckpt = int(responses["checkpoint_idx"].max()) + 1
    rows = []

    for origin, target_idx in rolling_origins(n_ckpt, min_history, horizon):
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
) -> pd.DataFrame:
    """Run every predictor on every model. Concatenated per-(model, target, skill) scores."""
    frames = []
    for model, group in responses.groupby("model", sort=True):
        for predictor in predictors:
            frames.append(
                evaluate_model(group, predictor, min_history=min_history, horizon=horizon, by_skill=by_skill)
            )
    return pd.concat(frames, ignore_index=True)


def summarize(results: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Aggregate per-slice scores, weighting each slice by its item count.

    Default grouping is (predictor, model). AUC is averaged unweighted over slices
    where it is defined, since it does not pool across slices the way the others do.
    """
    by = by or ["predictor", "model"]
    w = results["n"]

    def agg(g: pd.DataFrame) -> pd.Series:
        wg = w.loc[g.index]
        return pd.Series(
            {
                "n": int(wg.sum()),
                "log_loss": float(np.average(g["log_loss"], weights=wg)),
                "brier": float(np.average(g["brier"], weights=wg)),
                "accuracy": float(np.average(g["accuracy"], weights=wg)),
                "auc": float(g["auc"].mean(skipna=True)),
            }
        )

    out = results.groupby(by).apply(agg, include_groups=False).reset_index()
    out["n"] = out["n"].astype(int)
    return out
