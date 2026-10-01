"""Trivial predictors every learner model has to beat.

All of them ignore the sequence structure beyond "what happened most recently".
If a KT model cannot beat CopyLast on next-checkpoint prediction, it has learned
nothing the raw data did not already say.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from llm_kt.evaluate import TARGET_COLUMNS


def _last_checkpoint(history: pd.DataFrame) -> pd.DataFrame:
    return history[history["checkpoint_idx"] == history["checkpoint_idx"].max()]


class GlobalMean:
    """P = overall accuracy at the most recent checkpoint. One number for everything."""

    name = "global_mean"

    def predict(self, history: pd.DataFrame, target: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        p = _last_checkpoint(history)["correct"].mean()
        return np.full(len(target), p)


class SkillMean:
    """P = accuracy of the item's skill at the most recent checkpoint."""

    name = "skill_mean"

    def predict(self, history: pd.DataFrame, target: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        last = _last_checkpoint(history)
        per_skill = last.groupby("skill")["correct"].mean()
        fallback = last["correct"].mean()
        return target["skill"].map(per_skill).fillna(fallback).to_numpy()


class ItemMean:
    """P = the item's own accuracy across the whole history, shrunk toward the global mean.

    (k + alpha * prior) / (n + alpha): the item's correct count over its observation
    count, with `alpha` pseudo-observations at the history-wide accuracy `prior`.
    This is shrinkage toward the global mean, not add-one Laplace smoothing.
    Inspired by checkpoint averaging (Heineman et al. 2025), not an implementation of it.
    """

    name = "item_mean"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def predict(self, history: pd.DataFrame, target: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        g = history.groupby("item_id")["correct"]
        k = g.sum()
        n = g.size()
        prior = history["correct"].mean()
        p = (k + self.alpha * prior) / (n + self.alpha)
        return target["item_id"].map(p).fillna(prior).to_numpy()


class CopyLast:
    """P = the item's correctness at the most recent checkpoint, softened.

    A right answer at t predicts P(right at t+1) = 1 - eps; wrong predicts eps.
    `eps` is the assumed flip rate; 0.5 would be a coin toss. Adjacent checkpoints
    are near-identical, so this is the baseline to beat.
    """

    name = "copy_last"

    def __init__(self, eps: float = 0.1):
        self.eps = eps

    def predict(self, history: pd.DataFrame, target: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        last = _last_checkpoint(history).set_index("item_id")["correct"]
        c = target["item_id"].map(last).fillna(0.5).to_numpy(dtype=float)
        return np.where(c == 1, 1 - self.eps, np.where(c == 0, self.eps, 0.5))


class CopyLastCalibrated:
    """CopyLast, but the flip rates are estimated per skill from the history.

    P(right at t+1 | right at t) and P(right at t+1 | wrong at t) are measured over
    all consecutive checkpoint pairs seen so far, per skill. This is a two-state
    Markov chain on observed correctness. The one-step transition is applied at every
    horizon; at h > 1 it is therefore a fixed predictor, not an h-step Markov forecast.
    """

    name = "copy_last_calibrated"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def predict(self, history: pd.DataFrame, target: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        wide = history.pivot(index="item_id", columns="checkpoint_idx", values="correct")
        wide = wide.sort_index(axis=1)
        skill_of = history.drop_duplicates("item_id").set_index("item_id")["skill"]

        prev = wide.iloc[:, :-1].to_numpy()
        nxt = wide.iloc[:, 1:].to_numpy()
        skills = skill_of.loc[wide.index].to_numpy()

        # per-skill transition counts over every consecutive pair
        df = pd.DataFrame(
            {
                "skill": np.repeat(skills, prev.shape[1]),
                "prev": prev.ravel(),
                "nxt": nxt.ravel(),
            }
        ).dropna()
        stats = df.groupby(["skill", "prev"])["nxt"].agg(["sum", "size"])
        base = df["nxt"].mean()
        p_trans = (stats["sum"] + self.alpha * base) / (stats["size"] + self.alpha)

        last = wide.iloc[:, -1]
        lookup = pd.DataFrame(
            {"skill": target["skill"].to_numpy(), "prev": target["item_id"].map(last).to_numpy()}
        )
        p = lookup.merge(
            p_trans.rename("p").reset_index(), on=["skill", "prev"], how="left"
        )["p"]
        return p.fillna(base).to_numpy(dtype=float)


DEFAULT_BASELINES = [GlobalMean(), SkillMean(), ItemMean(), CopyLast(), CopyLastCalibrated()]

__all__ = [
    "DEFAULT_BASELINES",
    "TARGET_COLUMNS",
    "CopyLast",
    "CopyLastCalibrated",
    "GlobalMean",
    "ItemMean",
    "SkillMean",
]
