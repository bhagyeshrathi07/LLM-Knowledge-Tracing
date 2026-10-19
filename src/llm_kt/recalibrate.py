"""Per-model recalibration of item difficulty from a model's own checkpoint history.

Fluid Benchmarking's item parameters were fitted on 102 final models from the Open LLM
Leaderboard. Mid-training checkpoints of a single model are a different population, and
empirically a per-item running mean beats any predictor that holds those parameters fixed
(see scripts/run_models.py results, 2026-10-06). This module learns a per-item difficulty
offset for one model from history only:

    P(correct_{i,t}) = sigmoid(a_i * (theta_{s(i),t} - (b_i + delta_i)))

with theta_{s,t} one ability per (skill, checkpoint) and delta_i ~ N(0, tau^2) shrunk toward
Fluid's difficulty. Optionally (tau_a set) discrimination is recalibrated too:
a_i' = a_i * exp(gamma_i), gamma_i ~ N(0, tau_a^2), shrunk toward Fluid's a. Fitting alternates between the blocks of parameters (theta, delta, gamma), each a
vectorised Newton step with a Gaussian prior (MAP). Everything uses history up to the
origin, never the target checkpoint.

The offsets are then handed to any predictor that consumes (a, b) item parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit


def _theta_step(Y, M, a, b_adj, skill_idx, n_skills, theta, prior_var):
    """One Newton step on theta[s, t] for all skills and checkpoints at once."""
    z = a[:, None] * (theta[skill_idx, :] - b_adj[:, None])
    p = expit(z)
    g_item = np.where(M, a[:, None] * (Y - p), 0.0)           # items x T
    h_item = np.where(M, a[:, None] ** 2 * p * (1 - p), 0.0)  # items x T
    grad = np.zeros_like(theta)
    info = np.zeros_like(theta)
    np.add.at(grad, skill_idx, g_item)
    np.add.at(info, skill_idx, h_item)
    grad -= theta / prior_var
    info += 1.0 / prior_var
    return theta + grad / info


def _delta_step(Y, M, a, b, skill_idx, theta, delta, tau):
    """One Newton step on delta[i] for all items at once."""
    z = a[:, None] * (theta[skill_idx, :] - (b + delta)[:, None])
    p = expit(z)
    # d/d(delta) of log-lik: -a * (y - p)
    grad = np.where(M, -a[:, None] * (Y - p), 0.0).sum(axis=1) - delta / (tau * tau)
    info = np.where(M, a[:, None] ** 2 * p * (1 - p), 0.0).sum(axis=1) + 1.0 / (tau * tau)
    return delta + grad / info


def _gamma_step(Y, M, a, b_adj, skill_idx, theta, gamma, tau_a):
    """One Newton step on gamma[i] (log-discrimination multiplier) for all items at once.

    With z = a e^gamma (theta - b'), dz/dgamma = z, so for the 2PL log-lik
        d/d(gamma) [y log p + (1 - y) log(1 - p)] = (y - p) * z
    and the expected information is E[-d^2/dgamma^2] = p (1 - p) z^2 (the observed Hessian's
    extra (y - p) z term has expectation zero). Prior adds -gamma / tau_a^2 and 1 / tau_a^2.
    """
    z = (a * np.exp(gamma))[:, None] * (theta[skill_idx, :] - b_adj[:, None])
    p = expit(z)
    grad = np.where(M, (Y - p) * z, 0.0).sum(axis=1) - gamma / (tau_a * tau_a)
    info = np.where(M, p * (1 - p) * z * z, 0.0).sum(axis=1) + 1.0 / (tau_a * tau_a)
    return np.clip(gamma + grad / info, -2.0, 2.0)


def fit_item_offsets(
    Y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    skill_idx: np.ndarray,
    n_skills: int,
    tau: float = 1.0,
    theta_prior_var: float = 1.0,
    rounds: int = 3,
    newton_iters: int = 8,
    tau_a: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (delta[items], gamma[items], theta[skills, T]) for a model's history.

    Y: items x T matrix of 0/1 with NaN for unobserved cells. tau_a=None keeps discrimination
    fixed (gamma all zeros); otherwise gamma ~ N(0, tau_a^2) is learned, a' = a * exp(gamma).
    """
    Y = np.asarray(Y, dtype=float)
    M = ~np.isnan(Y)
    Y0 = np.where(M, Y, 0.0)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    T = Y.shape[1]
    theta = np.zeros((n_skills, T))
    delta = np.zeros(len(a))
    gamma = np.zeros(len(a))
    for _ in range(rounds):
        a_cur = a * np.exp(gamma)
        for _ in range(newton_iters):
            theta = _theta_step(Y0, M, a_cur, b + delta, skill_idx, n_skills, theta, theta_prior_var)
        for _ in range(newton_iters):
            delta = _delta_step(Y0, M, a_cur, b, skill_idx, theta, delta, tau)
        if tau_a is not None:
            for _ in range(newton_iters):
                gamma = _gamma_step(Y0, M, a, b + delta, skill_idx, theta, gamma, tau_a)
    return delta, gamma, theta


def recalibrated_params(
    history: pd.DataFrame, params: pd.DataFrame, tau: float = 1.0, rounds: int = 3, tau_a: float | None = None
) -> pd.DataFrame:
    """Per-item (a * exp(gamma), b + delta, delta, gamma) learned from one model's history.

    `params` indexed by item_id with a, b. gamma is zero unless tau_a is given.
    """
    wide = history.pivot(index="item_id", columns="checkpoint_idx", values="correct").sort_index(axis=1)
    wide = wide[wide.index.isin(params.index)]
    skill_of = history.drop_duplicates("item_id").set_index("item_id")["skill"].loc[wide.index]
    skills, skill_idx = np.unique(skill_of.to_numpy(), return_inverse=True)
    p = params.loc[wide.index]
    delta, gamma, _ = fit_item_offsets(wide.to_numpy(), p["a"].to_numpy(), p["b"].to_numpy(), skill_idx, len(skills),
                                       tau=tau, rounds=rounds, tau_a=tau_a)
    out = p[["a", "b"]].copy()
    out["a"] = out["a"] * np.exp(gamma)
    out["b"] = out["b"] + delta
    out["delta"] = delta
    out["gamma"] = gamma
    return out


def recal_suffix(tau: float, tau_a: float | None = None) -> str:
    """Predictor-name suffix, e.g. _recal_tau1 or _recal_tau1_taua0.5."""
    return f"_recal_tau{tau:g}" + (f"_taua{tau_a:g}" if tau_a is not None else "")
