"""Walk-forward probability calibration for any `Predictor`.

IRT-based predictors rank items well but are overconfident. `WalkForwardCalibrated`
wraps a base predictor and, at every origin, learns a calibration map from the base
predictor's own recent forecasts, using only the history it is given:

  1. For the last `k` origins o inside the history whose target o + horizon is also
     inside it, re-run the base predictor on the prefix (checkpoints <= o) and pair its
     forecasts with the observed outcomes at o + horizon.
  2. Fit a monotone map on those pairs: "temperature" (p' = sigmoid(logit(p) / T)) or
     "platt" (p' = sigmoid(A logit(p) + B)).
  3. Apply it to the base forecast for the real target.

Both maps are monotone increasing (T > 0; A is near 1 under its prior and positive in
practice), so per-slice ranking and AUC are unchanged unless A goes negative.

Cost: the base predictor is called k + 1 times per origin instead of once, so wrapping
multiplies evaluation time by about (k + 1). Wrap only at evaluation time and keep k small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from llm_kt.evaluate import TARGET_COLUMNS, Predictor

EPS = 1e-6
METHODS = ("temperature", "platt")
T_BOUNDS = (0.25, 4.0)


def _logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))


def _nll(z: np.ndarray, y: np.ndarray) -> float:
    """Mean Bernoulli negative log-likelihood of logits z (numerically stable)."""
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_temperature(p: np.ndarray, y: np.ndarray, bounds: tuple[float, float] = T_BOUNDS) -> float:
    """Scalar T minimising log-loss of sigmoid(logit(p) / T), searched on `bounds`."""
    z = _logit(p)
    y = np.asarray(y, dtype=float)
    res = minimize_scalar(lambda t: _nll(z / t, y), bounds=bounds, method="bounded")
    return float(res.x)


def fit_platt(
    p: np.ndarray, y: np.ndarray, prior_prec: float = 1.0, iters: int = 50, tol: float = 1e-8
) -> tuple[float, float]:
    """(A, B) for logit(p') = A logit(p) + B: logistic regression by Newton's method.

    A weak Gaussian prior with precision `prior_prec` centres (A, B) on the identity map
    (1, 0), which keeps the fit stable when the pairs are few or separable.
    """
    x = _logit(p)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([x, np.ones_like(x)])
    w0 = np.array([1.0, 0.0])
    w = w0.copy()
    for _ in range(iters):
        q = expit(X @ w)
        grad = X.T @ (y - q) - prior_prec * (w - w0)
        hess = (X * (q * (1 - q))[:, None]).T @ X + prior_prec * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w += step
        if np.max(np.abs(step)) < tol:
            break
    return float(w[0]), float(w[1])


class WalkForwardCalibrated:
    """Predictor: `base` followed by a calibration map fitted on its last `k` in-history forecasts.

    Passes the base forecast through unchanged when fewer than 2 origins can be
    reconstructed. Each call appends {history_max, n_origins, n_pairs, params} to `fits`.
    Costs about (k + 1) base-predictor calls per origin.
    """

    def __init__(self, base: Predictor, k: int = 5, method: str = "temperature"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.base = base
        self.k = k
        self.method = method
        self.name = f"{base.name}_cal_{method}k{k}"
        self.fits: list[dict] = []

    def _fit_origins(self, history: pd.DataFrame, horizon: int) -> list[int]:
        """The last k origins o in history with o + horizon also in history."""
        ckpts = {int(c) for c in history["checkpoint_idx"].unique()}
        origins = sorted(o for o in ckpts if o + horizon in ckpts)
        return origins[-self.k:]

    def _reconstruct(self, history: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, int]:
        origins = self._fit_origins(history, horizon)
        ps, ys = [], []
        for o in origins:
            sub_history = history[history["checkpoint_idx"] <= o]
            rows = history[history["checkpoint_idx"] == o + horizon]
            sub_target = rows[TARGET_COLUMNS].reset_index(drop=True)
            ps.append(np.asarray(self.base.predict(sub_history, sub_target, horizon=horizon), dtype=float))
            ys.append(rows["correct"].to_numpy(dtype=float))
        if not origins:
            return np.empty(0), np.empty(0), 0
        return np.concatenate(ps), np.concatenate(ys), len(origins)

    def predict(self, history: pd.DataFrame, target: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        p_fit, y_fit, n_origins = self._reconstruct(history, horizon)
        p = np.asarray(self.base.predict(history, target, horizon=horizon), dtype=float)
        record = {"history_max": int(history["checkpoint_idx"].max()), "n_origins": n_origins,
                  "n_pairs": len(y_fit), "params": None}
        self.fits.append(record)
        if n_origins < 2:
            return p
        z = _logit(p)
        if self.method == "temperature":
            T = fit_temperature(p_fit, y_fit)
            record["params"] = {"T": T}
            return expit(z / T)
        A, B = fit_platt(p_fit, y_fit)
        record["params"] = {"A": A, "B": B}
        return expit(A * z + B)
