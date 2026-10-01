"""Per-skill accuracy curves over training checkpoints.

This is the simplest possible baseline: raw accuracy per (model, skill, checkpoint)
with a Wilson confidence interval. No learner model. It exists to sanity-check that
the data has learning-curve structure before any knowledge tracing model is fit,
and it is the "per-skill accuracy curve" baseline the paper compares against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_KEYS = ["model", "benchmark", "skill", "checkpoint_idx"]
CHECKPOINT_META = ["checkpoint_label", "step", "tokens_seen_b"]


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for k successes out of n. Vectorised; n must be > 0."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # clip: floating-point noise can put k=0 or k=n a hair outside [0, 1]
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def skill_accuracy(responses: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the long response table to one row per (model, benchmark, skill, checkpoint).

    Columns: model, benchmark, skill, checkpoint_idx, checkpoint_label, step,
    tokens_seen_b, n, correct_n, acc, ci_lo, ci_hi.
    """
    g = responses.groupby(GROUP_KEYS + CHECKPOINT_META, dropna=False, sort=True)
    out = g["correct"].agg(n="size", correct_n="sum").reset_index()
    out["acc"] = out["correct_n"] / out["n"]
    out["ci_lo"], out["ci_hi"] = wilson_interval(out["correct_n"], out["n"])
    return out.sort_values(GROUP_KEYS).reset_index(drop=True)
