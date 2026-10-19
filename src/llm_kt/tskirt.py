"""T-SKIRT: temporal IRT with a Wiener-process ability (Ekanadham & Karklin, 2017).

Ability for one (model, skill) follows a Wiener process in training time:

    theta(t + d) ~ N(theta(t), nu^2 d)

Under a probit (normal-ogive) 2PO link, a response observed `d` time units before the
origin can be marginalised over the ability drift in closed form, which turns into a
smaller *effective* discrimination for older responses:

    a_eff = a / sqrt(1 + a^2 nu^2 d)

So the MAP ability at the origin is an ordinary probit-IRT fit on all history with each
response's discrimination shrunk by its age. Forecasting h steps ahead adds the drift
variance nu^2 * d_forecast (plus the Laplace posterior variance of theta) the same way.

Item parameters come from Fluid Benchmarking's logistic 2PL fit and are converted to the
probit scale by dividing discrimination by 1.7 (difficulty is link-invariant). They stay
fixed; only theta is estimated. `nu` is a hyperparameter to be chosen on the validation
block (nu = 0 recovers static probit IRT on the full history).

Time is measured in `time_col` units (checkpoint index by default; `step` or
`tokens_seen_b` where the release provides them). The target's time is read from the same
column of the target frame, falling back to origin + horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from llm_kt.recalibrate import recal_suffix as _recal_suffix
from llm_kt.recalibrate import recalibrated_params

PROBIT_SCALE = 1.7


def to_probit(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logistic 2PL (a, b) -> probit 2PO (a / 1.7, b)."""
    return np.asarray(a, dtype=float) / PROBIT_SCALE, np.asarray(b, dtype=float)


def effective_discrimination(a: np.ndarray, nu: float, elapsed: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    elapsed = np.maximum(np.asarray(elapsed, dtype=float), 0.0)
    return a / np.sqrt(1.0 + a * a * nu * nu * elapsed)


def fit_theta_probit(
    a: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    prior_var: float = 1.0,
    iters: int = 30,
    tol: float = 1e-6,
) -> float:
    """MAP ability under probit 2PO with fixed (effective) item parameters and a N(0, prior_var) prior.

    Newton's method with the expected (Fisher) information, which keeps the step stable.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(a) == 0:
        return 0.0
    theta = 0.0
    for _ in range(iters):
        z = a * (theta - b)
        cdf = np.clip(norm.cdf(z), 1e-12, 1 - 1e-12)
        pdf = norm.pdf(z)
        grad = float(np.sum(a * pdf * (y / cdf - (1 - y) / (1 - cdf))) - theta / prior_var)
        info = float(np.sum(a * a * pdf * pdf / (cdf * (1 - cdf))) + 1.0 / prior_var)
        step = grad / info
        theta += step
        if abs(step) < tol:
            break
    return float(theta)


def posterior_var_probit(a: np.ndarray, b: np.ndarray, theta: float, prior_var: float) -> float:
    """Laplace posterior variance of theta at the MAP (inverse Fisher information)."""
    a = np.asarray(a, dtype=float)
    if len(a) == 0:
        return prior_var
    z = a * (theta - np.asarray(b, dtype=float))
    cdf = np.clip(norm.cdf(z), 1e-12, 1 - 1e-12)
    pdf = norm.pdf(z)
    info = float(np.sum(a * a * pdf * pdf / (cdf * (1 - cdf))) + 1.0 / prior_var)
    return 1.0 / info


class TSKIRT:
    """Predictor: Wiener-process probit IRT per skill with fixed Fluid item parameters."""

    def __init__(
        self,
        item_params: pd.DataFrame,
        nu: float = 0.1,
        prior_var: float = 1.0,
        time_col: str = "checkpoint_idx",
        include_posterior_var: bool = True,
        recalibrate: bool = False,
        tau: float = 1.0,
        tau_a: float | None = None,
    ):
        self.logistic_params = item_params.set_index("item_id")[["a", "b"]]
        self.params = self._to_probit_table(self.logistic_params)
        self.nu = nu
        self.prior_var = prior_var
        self.time_col = time_col
        self.include_posterior_var = include_posterior_var
        self.recalibrate = recalibrate
        self.tau = tau
        self.tau_a = tau_a
        self.name = f"tskirt_nu{nu:g}_{time_col}" + (_recal_suffix(tau, tau_a) if recalibrate else "")

    @staticmethod
    def _to_probit_table(logistic: pd.DataFrame) -> pd.DataFrame:
        a_p, b_p = to_probit(logistic["a"].to_numpy(), logistic["b"].to_numpy())
        return pd.DataFrame({"a": a_p, "b": b_p}, index=logistic.index)

    def _refresh_params(self, history: pd.DataFrame) -> None:
        """Re-centre item difficulty on this model's history (logistic space), then convert to probit."""
        if self.recalibrate:
            recal = recalibrated_params(history, self.logistic_params, tau=self.tau, tau_a=self.tau_a)[["a", "b"]]
            self.params = self._to_probit_table(recal)

    # -- fitting -------------------------------------------------------------------------

    def _time(self, df: pd.DataFrame) -> pd.Series:
        t = pd.to_numeric(df[self.time_col], errors="coerce")
        if t.isna().any():  # release lacks this clock for the model: fall back to checkpoint index
            t = df["checkpoint_idx"].astype(float)
        return t.astype(float)

    def fit_skill_theta(self, history: pd.DataFrame, origin_time: float) -> dict[str, float]:
        """MAP theta per skill at `origin_time`, using age-shrunk discriminations."""
        h = history.join(self.params, on="item_id")
        h = h[h["a"].notna()].copy()
        h["elapsed"] = origin_time - self._time(h)
        out = {}
        for skill, g in h.groupby("skill"):
            a_eff = effective_discrimination(g["a"].to_numpy(), self.nu, g["elapsed"].to_numpy())
            out[skill] = fit_theta_probit(a_eff, g["b"].to_numpy(), g["correct"].to_numpy(), self.prior_var)
        self._last_fit = h  # kept for posterior variance in predict
        return out

    def _posterior_vars(self, thetas: dict[str, float]) -> dict[str, float]:
        out = {}
        for skill, g in self._last_fit.groupby("skill"):
            a_eff = effective_discrimination(g["a"].to_numpy(), self.nu, g["elapsed"].to_numpy())
            out[skill] = posterior_var_probit(a_eff, g["b"].to_numpy(), thetas[skill], self.prior_var)
        return out

    # -- predicting ----------------------------------------------------------------------

    def predict(self, history: pd.DataFrame, target: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        origin_time = float(self._time(history).max())
        if self.time_col in target.columns and pd.to_numeric(target[self.time_col], errors="coerce").notna().all():
            target_time = float(pd.to_numeric(target[self.time_col]).iloc[0])
        else:
            target_time = origin_time + horizon
        d_forecast = max(target_time - origin_time, 0.0)

        self._refresh_params(history)
        thetas = self.fit_skill_theta(history, origin_time)
        post_var = self._posterior_vars(thetas) if self.include_posterior_var else {s: 0.0 for s in thetas}

        t = target.join(self.params, on="item_id")
        theta_t = t["skill"].map(thetas).to_numpy(dtype=float)
        var_t = t["skill"].map(post_var).to_numpy(dtype=float) + self.nu * self.nu * d_forecast
        a = t["a"].to_numpy(dtype=float)
        b = t["b"].to_numpy(dtype=float)
        a_fore = a / np.sqrt(1.0 + a * a * var_t)
        p = norm.cdf(a_fore * (theta_t - b))

        # items without parameters (or skills unseen in history): last-checkpoint skill mean
        last = history[history["checkpoint_idx"] == history["checkpoint_idx"].max()]
        fallback = t["skill"].map(last.groupby("skill")["correct"].mean()).fillna(last["correct"].mean())
        p = np.where(np.isnan(p), fallback.to_numpy(dtype=float), p)
        return np.clip(p, 0.0, 1.0)
