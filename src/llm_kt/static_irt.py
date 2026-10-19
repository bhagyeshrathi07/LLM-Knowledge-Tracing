"""Static 2PL IRT baselines using Fluid Benchmarking's fitted item parameters.

Fluid Benchmarking (Hofmann et al., 2025) fit a logistic 2PL model to 102 Open LLM
Leaderboard models and released, for every item, a discrimination `a` and difficulty `b`:

    P(correct | theta) = sigmoid(a * (theta - b))

We keep the item parameters fixed and estimate one ability theta per (model, skill,
checkpoint) by MAP with a N(0, prior_var) prior. Two baselines are built on that:

  StaticIRTPersistence  forecast. theta is estimated from the last `window` checkpoints of
                        the history and carried forward unchanged to the target checkpoint.
  diagnose()            within-checkpoint diagnosis. theta is estimated on a random subset
                        of a checkpoint's items and scored on the held-out items of the same
                        checkpoint. Never scores an item that was used to fit theta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from llm_kt.ingest import HF_REPO
from llm_kt.metrics import score
from llm_kt.recalibrate import recal_suffix as _recal_suffix
from llm_kt.recalibrate import recalibrated_params

IRT_BENCHMARKS = ("arc_challenge", "gsm8k", "hellaswag", "mmlu", "winogrande")


def load_item_params(benchmarks: tuple[str, ...] = IRT_BENCHMARKS, cache_dir: Path | None = None) -> pd.DataFrame:
    """Download Fluid's per-benchmark IRT files and return one table: item_id, a, b."""
    from huggingface_hub import hf_hub_download

    frames = []
    for bench in benchmarks:
        path = hf_hub_download(HF_REPO, f"data/irt_models/{bench}.csv", repo_type="dataset", cache_dir=cache_dir)
        df = pd.read_csv(path).rename(columns={"Unnamed: 0": "item_id"})
        frames.append(df[["item_id", "a", "b"]])
    return pd.concat(frames, ignore_index=True)


def prob_logistic(theta: float | np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = np.asarray(a, dtype=float) * (theta - np.asarray(b, dtype=float))
    return expit(z)


def fit_theta_logistic(
    a: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    prior_var: float = 1.0,
    iters: int = 25,
    tol: float = 1e-6,
) -> float:
    """MAP ability under a logistic 2PL with fixed item parameters and a N(0, prior_var) prior.

    Newton's method on the log posterior. With no observations the MAP is the prior mean, 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(a) == 0:
        return 0.0
    theta = 0.0
    for _ in range(iters):
        p = prob_logistic(theta, a, b)
        grad = float(np.sum(a * (y - p)) - theta / prior_var)
        hess = float(-np.sum(a * a * p * (1 - p)) - 1.0 / prior_var)
        step = -grad / hess
        theta += step
        if abs(step) < tol:
            break
    return float(theta)


class StaticIRTPersistence:
    """P(correct at t+h) = sigmoid(a (theta_skill - b)), theta from the last `window` checkpoints."""

    def __init__(
        self,
        item_params: pd.DataFrame,
        window: int = 1,
        prior_var: float = 1.0,
        recalibrate: bool = False,
        tau: float = 1.0,
        tau_a: float | None = None,
    ):
        self.params = item_params.set_index("item_id")[["a", "b"]]
        self.window = window
        self.prior_var = prior_var
        self.recalibrate = recalibrate
        self.tau = tau
        self.tau_a = tau_a
        self.name = f"static_irt_persist_w{window}" + (_recal_suffix(tau, tau_a) if recalibrate else "")

    def _params_for(self, history: pd.DataFrame) -> pd.DataFrame:
        """Item parameters to use at this origin: Fluid's, or re-centred on this model's history."""
        if not self.recalibrate:
            return self.params
        return recalibrated_params(history, self.params, tau=self.tau, tau_a=self.tau_a)[["a", "b"]]

    def predict(self, history: pd.DataFrame, target: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        params = self._params_for(history)
        last_idx = history["checkpoint_idx"].max()
        recent = history[history["checkpoint_idx"] > last_idx - self.window]
        recent = recent.join(params, on="item_id")
        has = recent["a"].notna()

        thetas = {}
        for skill, g in recent[has].groupby("skill"):
            thetas[skill] = fit_theta_logistic(g["a"].to_numpy(), g["b"].to_numpy(), g["correct"].to_numpy(), self.prior_var)
        fallback = recent.groupby("skill")["correct"].mean()
        global_mean = float(recent["correct"].mean())

        t = target.join(params, on="item_id")
        theta_t = t["skill"].map(thetas)
        p = prob_logistic(theta_t.to_numpy(dtype=float), t["a"].to_numpy(dtype=float), t["b"].to_numpy(dtype=float))
        fb = t["skill"].map(fallback).fillna(global_mean).to_numpy(dtype=float)
        p = np.where(np.isnan(p), fb, p)
        return np.clip(p, 0.0, 1.0)


def diagnose(
    responses: pd.DataFrame,
    item_params: pd.DataFrame,
    model: str,
    checkpoint_idx: int,
    holdout_frac: float = 0.5,
    seed: int = 0,
    prior_var: float = 1.0,
) -> pd.DataFrame:
    """Within-checkpoint held-out-item diagnosis for one (model, checkpoint).

    For each skill: fit theta on a random (1 - holdout_frac) share of the items, predict the
    rest, score. Returns one row per skill with n_items, n_fit, n_holdout and the metrics.
    """
    rng = np.random.default_rng(seed)
    params = item_params.set_index("item_id")[["a", "b"]]
    sub = responses[(responses["model"] == model) & (responses["checkpoint_idx"] == checkpoint_idx)]
    sub = sub.join(params, on="item_id")
    sub = sub[sub["a"].notna()]

    rows = []
    for skill, g in sub.groupby("skill"):
        n = len(g)
        if n < 2:
            continue
        hold = rng.random(n) < holdout_frac
        if hold.all() or (~hold).all():
            hold[0] = not hold[0]
        fit, test = g[~hold], g[hold]
        theta = fit_theta_logistic(fit["a"].to_numpy(), fit["b"].to_numpy(), fit["correct"].to_numpy(), prior_var)
        p = prob_logistic(theta, test["a"].to_numpy(), test["b"].to_numpy())
        rows.append({"model": model, "checkpoint_idx": checkpoint_idx, "skill": skill, "theta": theta,
                     "n_items": n, "n_fit": int((~hold).sum()), "n_holdout": int(hold.sum()),
                     **{k: v for k, v in score(test["correct"].to_numpy(), p).items() if k != "n"}})
    return pd.DataFrame(rows)
