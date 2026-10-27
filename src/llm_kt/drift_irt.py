"""DriftIRT: state-space probit IRT with drifting per-item residuals.

T-SKIRT gives each skill one drifting ability and holds every item's difficulty fixed, so
it cannot follow an individual item whose correctness changes differently from the rest of
its skill. item_mean, the bar to beat, has one free number per item that updates at every
checkpoint. DriftIRT adds that degree of freedom inside the IRT model: each item carries a
difficulty residual that drifts over training on its own, shrunk toward Fluid's difficulty.

Model, for one (model, skill s) and item i in s, with t the checkpoint time:

    theta_s(t + d) = theta_s(t) + N(0, q_theta d)        theta_s(t0) ~ N(0, theta_prior_var)
    delta_i(t + d) = delta_i(t) + N(0, q_delta d)        delta_i(first seen) ~ N(0, delta_prior_var)
    P(y_i(t) = 1)  = Phi(z),  z = a_i (theta_s(t) - b_i) - c_i delta_i(t)

(a_i, b_i) are Fluid's logistic 2PL parameters converted to the probit scale exactly as in
`tskirt` (a / PROBIT_SCALE, b unchanged); discrimination is held fixed. The residual
loading c_i selects the residual's units:

    residual_scale="difficulty"  c_i = a_i, so z = a_i (theta_s - b_i - delta_i): delta is a
                                 shift of difficulty, as in `recalibrate`. Its effect on z is
                                 scaled by a_i, which is small for most MMLU items.
    residual_scale="probit"      c_i = 1, so delta is a shift of the probit linear predictor
                                 and moves every item's probability equally far regardless
                                 of a_i. Equivalent to the difficulty form with q_delta and
                                 delta_prior_var divided by a_i^2.

Time is measured in `time_col` units with the same fallback as T-SKIRT.

Inference is an assumed-density (Laplace) filter with a mean-field diagonal posterior.
Predict step between checkpoints d apart: var_theta += q_theta d, var_delta += q_delta d
(means unchanged). Update step at a checkpoint: block-coordinate Newton on the log
posterior given the predicted Gaussian priors. Each of `n_rounds` rounds runs
`newton_iters` Newton steps on every theta_s with deltas held fixed (vectorised over skills
with bincount), then `newton_iters` steps on every observed delta_i with abilities held
fixed (vectorised over items). Posterior variances are the inverse curvatures of each
block at the final point; the theta/delta posterior covariance is dropped, so variances
are somewhat underestimated. The probit log-likelihood is log-concave, so the observed
Hessian is used directly.

Assumed-density filtering is order dependent: with a very wide residual prior (tens of
probit units) the first observations of an item can push its residual far out where a
Gaussian approximation of the posterior is poor, and later checkpoints pull it back only
slowly. Keep delta_prior_var moderate (of order 1 on the probit scale).

The delta prior N(0, delta_prior_var) is the prior at an item's first observation (and for
items never observed). After that delta follows a pure random walk with no mean reversion.
This is a deliberate refinement of "stationary prior": an Ornstein-Uhlenbeck residual with
stationary variance P and innovation q has autocorrelation sqrt(1 - q / P) per step, so a
large q_delta would make it forget faster rather than track recent history, and the
h-step random-walk forecast variance h * q in the specification would not hold.

Forecast for a target time d_f after the origin: means are the last posterior means, and

    var_z = forecast_var_scale * (a^2 (v_theta + d_f q_theta) + c^2 (v_delta + d_f q_delta))
    p     = Phi(mean_z / sqrt(1 + var_z))

which is tskirt's include_posterior_var folding (a / sqrt(1 + a^2 var)) applied to both
latent terms. forecast_var_scale decouples the forecast spread from the filter's own
variances; 0 gives the plug-in forecast Phi(mean_z).

Evaluation calls predict at successive origins with a growing history. The filter state is
cached and only checkpoints newer than the last processed one are filtered, so a full
walk-forward run filters each checkpoint once. The cache is reset whenever the history does
not extend the cached one (different model, earlier origin, or different rows up to the
cached checkpoint, checked by row count and two checksums of `correct`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr

from llm_kt.tskirt import to_probit

RESIDUAL_SCALES = ("difficulty", "probit")
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
_MAX_STEP = 3.0


def _log_phi(z: np.ndarray) -> np.ndarray:
    return -0.5 * z * z - _LOG_SQRT_2PI


def probit_score_and_curvature(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """d/dz log P(y | z) and -d^2/dz^2 log P(y | z) under P(y = 1) = Phi(z).

    With x = s z and s = 2y - 1, log P = log Phi(x); the score is s * lambda(x) and the
    curvature lambda(x) (lambda(x) + x), lambda = phi / Phi, computed in log space. The
    curvature lies in (0, 1), which keeps Newton steps well conditioned.
    """
    s = 2.0 * np.asarray(y, dtype=float) - 1.0
    x = s * z
    lam = np.exp(_log_phi(x) - log_ndtr(x))
    return s * lam, np.clip(lam * (lam + x), 1e-12, 1.0)


def filter_update(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    skill_idx: np.ndarray,
    theta_m: np.ndarray,
    theta_v: np.ndarray,
    delta_m: np.ndarray,
    delta_v: np.ndarray,
    n_rounds: int = 3,
    newton_iters: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One checkpoint's Laplace update for all skills and the items observed at it.

    y, a, b, c, skill_idx, delta_m, delta_v are per observed item; theta_m, theta_v are
    per skill (the predicted priors). Returns posterior (theta_m, theta_v, delta_m, delta_v).
    Skills with no observed item keep their prior.
    """
    n_skills = len(theta_m)
    prior_tm, prior_tp = theta_m, 1.0 / theta_v
    prior_dm, prior_dp = delta_m, 1.0 / delta_v
    th = theta_m.copy()
    de = delta_m.copy()
    for _ in range(n_rounds):
        offset = a * b + c * de  # z = a * theta - offset
        for _ in range(newton_iters):
            g, h = probit_score_and_curvature(a * th[skill_idx] - offset, y)
            grad = np.bincount(skill_idx, a * g, n_skills) - (th - prior_tm) * prior_tp
            curv = np.bincount(skill_idx, a * a * h, n_skills) + prior_tp
            th = th + np.clip(grad / curv, -_MAX_STEP, _MAX_STEP)
        base = a * (th[skill_idx] - b)
        for _ in range(newton_iters):
            g, h = probit_score_and_curvature(base - c * de, y)
            grad = -c * g - (de - prior_dm) * prior_dp
            curv = c * c * h + prior_dp
            de = de + np.clip(grad / curv, -_MAX_STEP, _MAX_STEP)
    _, h = probit_score_and_curvature(a * (th[skill_idx] - b) - c * de, y)
    theta_v_post = 1.0 / (np.bincount(skill_idx, a * a * h, n_skills) + prior_tp)
    delta_v_post = 1.0 / (c * c * h + prior_dp)
    return th, theta_v_post, de, delta_v_post


def _ids(col: pd.Series) -> np.ndarray:
    """String keys as an object array, so Index.get_indexer uses a hash table (Arrow isin is slow)."""
    return col.to_numpy(dtype=object)


def _lookup(
    index: pd.Index, keys: np.ndarray, mean: np.ndarray, var: np.ndarray, prior_var: float
) -> tuple[np.ndarray, np.ndarray]:
    """State (mean, var) for each key; keys not in `index` get the prior N(0, prior_var)."""
    pos = index.get_indexer(keys)
    m = np.zeros(len(keys))
    v = np.full(len(keys), prior_var)
    hit = pos >= 0
    m[hit] = mean[pos[hit]]
    v[hit] = var[pos[hit]]
    return m, v


class DriftIRT:
    """Predictor: per-skill drifting ability plus per-item drifting residual, filtered forward."""

    def __init__(
        self,
        item_params: pd.DataFrame,
        q_theta: float = 0.01,
        q_delta: float = 0.01,
        delta_prior_var: float = 1.0,
        forecast_var_scale: float = 1.0,
        time_col: str = "checkpoint_idx",
        n_rounds: int = 3,
        newton_iters: int = 5,
        theta_prior_var: float = 1.0,
        residual_scale: str = "difficulty",
    ):
        if residual_scale not in RESIDUAL_SCALES:
            raise ValueError(f"residual_scale must be one of {RESIDUAL_SCALES}, got {residual_scale!r}")
        if min(q_theta, q_delta, forecast_var_scale) < 0:
            raise ValueError("q_theta, q_delta and forecast_var_scale must be >= 0")
        if delta_prior_var <= 0 or theta_prior_var <= 0:
            raise ValueError("delta_prior_var and theta_prior_var must be > 0")
        logistic = item_params.set_index("item_id")[["a", "b"]]
        a_p, b_p = to_probit(logistic["a"].to_numpy(), logistic["b"].to_numpy())
        self.params = pd.DataFrame({"a": a_p, "b": b_p}, index=pd.Index(logistic.index.to_numpy(dtype=object)))
        self.q_theta = q_theta
        self.q_delta = q_delta
        self.delta_prior_var = delta_prior_var
        self.forecast_var_scale = forecast_var_scale
        self.time_col = time_col
        self.n_rounds = n_rounds
        self.newton_iters = newton_iters
        self.theta_prior_var = theta_prior_var
        self.residual_scale = residual_scale
        self.name = (
            f"drift_irt_qt{q_theta:g}_qd{q_delta:g}_pv{delta_prior_var:g}_fv{forecast_var_scale:g}"
            + ("_zres" if residual_scale == "probit" else "")
            + ("" if time_col == "checkpoint_idx" else f"_{time_col}")
        )
        self._reset()

    # -- state ---------------------------------------------------------------------------

    def _reset(self) -> None:
        self._key = None            # (model, last checkpoint, n rows, sum correct, weighted sum)
        self._time_last = None      # time of the last filtered checkpoint
        self._items = pd.Index([], dtype=object)  # item ids in state, in insertion order
        self._item_skill = np.zeros(0, dtype=int)
        self._a = np.zeros(0)
        self._b = np.zeros(0)
        self._delta_m = np.zeros(0)
        self._delta_v = np.zeros(0)
        self._skills = pd.Index([], dtype=object)
        self._theta_m = np.zeros(0)
        self._theta_v = np.zeros(0)

    def _loading(self, a: np.ndarray) -> np.ndarray:
        return a if self.residual_scale == "difficulty" else np.ones_like(a)

    def _time(self, df: pd.DataFrame) -> pd.Series:
        t = pd.to_numeric(df[self.time_col], errors="coerce")
        if t.isna().any():  # release lacks this clock for the model: fall back to checkpoint index
            t = df["checkpoint_idx"].astype(float)
        return t.astype(float)

    @staticmethod
    def _fingerprint(history: pd.DataFrame, upto: int) -> tuple:
        h = history[history["checkpoint_idx"] <= upto]
        y = h["correct"].to_numpy(dtype=float)
        model = history["model"].iloc[0] if "model" in history.columns and len(history) else None
        return (model, upto, len(h), float(y.sum()), float((y * h["checkpoint_idx"].to_numpy()).sum()))

    def _add_skills(self, skills: np.ndarray) -> None:
        new = pd.Index(pd.unique(np.asarray(skills, dtype=object))).difference(self._skills)
        if len(new):
            self._skills = self._skills.append(new)
            self._theta_m = np.concatenate([self._theta_m, np.zeros(len(new))])
            self._theta_v = np.concatenate([self._theta_v, np.full(len(new), self.theta_prior_var)])

    def _add_items(self, rows: pd.DataFrame) -> None:
        rows = rows.drop_duplicates("item_id")
        rows = rows[self._items.get_indexer(_ids(rows["item_id"])) < 0]
        if rows.empty:
            return
        self._add_skills(rows["skill"].to_numpy())
        p = self.params.iloc[self.params.index.get_indexer(_ids(rows["item_id"]))]
        self._items = self._items.append(pd.Index(_ids(rows["item_id"])))
        self._item_skill = np.concatenate(
            [self._item_skill, self._skills.get_indexer(_ids(rows["skill"]))]
        )
        self._a = np.concatenate([self._a, p["a"].to_numpy()])
        self._b = np.concatenate([self._b, p["b"].to_numpy()])
        self._delta_m = np.concatenate([self._delta_m, np.zeros(len(rows))])
        self._delta_v = np.concatenate([self._delta_v, np.full(len(rows), self.delta_prior_var)])

    def _filter_checkpoint(self, rows: pd.DataFrame, t: float) -> None:
        if self._time_last is not None:
            d = max(t - self._time_last, 0.0)
            self._theta_v = self._theta_v + self.q_theta * d
            self._delta_v = self._delta_v + self.q_delta * d
        self._time_last = t
        rows = rows[self.params.index.get_indexer(_ids(rows["item_id"])) >= 0]
        if rows.empty:
            return
        self._add_items(rows)
        idx = self._items.get_indexer(_ids(rows["item_id"]))
        a = self._a[idx]
        th_m, th_v, de_m, de_v = filter_update(
            rows["correct"].to_numpy(dtype=float),
            a,
            self._b[idx],
            self._loading(a),
            self._item_skill[idx],
            self._theta_m,
            self._theta_v,
            self._delta_m[idx],
            self._delta_v[idx],
            n_rounds=self.n_rounds,
            newton_iters=self.newton_iters,
        )
        self._theta_m, self._theta_v = th_m, th_v
        self._delta_m[idx] = de_m
        self._delta_v[idx] = de_v

    def fit(self, history: pd.DataFrame) -> None:
        """Bring the filter state up to the last checkpoint of `history`, reusing the cache."""
        last = int(history["checkpoint_idx"].max())
        if self._key is not None:
            cached_last = self._key[1]
            if last < cached_last or self._fingerprint(history, cached_last) != self._key:
                self._reset()
        start = -np.inf if self._key is None else self._key[1]
        new = history[history["checkpoint_idx"] > start]
        times = self._time(new)
        for _, rows in new.groupby("checkpoint_idx", sort=True):
            self._filter_checkpoint(rows, float(times.loc[rows.index].iloc[0]))
        self._key = self._fingerprint(history, last)

    # -- predicting ----------------------------------------------------------------------

    def predict(self, history: pd.DataFrame, target: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        self.fit(history)
        origin_time = self._time_last
        tcol = pd.to_numeric(target[self.time_col], errors="coerce") if self.time_col in target else None
        if tcol is not None and tcol.notna().all() and len(tcol):
            target_time = float(tcol.iloc[0])
        else:
            target_time = origin_time + horizon
        d_f = max(target_time - origin_time, 0.0)

        t = target.join(self.params, on="item_id")
        a = t["a"].to_numpy(dtype=float)
        b = t["b"].to_numpy(dtype=float)
        c = self._loading(a)
        delta_m, delta_v = _lookup(
            self._items, _ids(t["item_id"]), self._delta_m, self._delta_v, self.delta_prior_var
        )
        theta_m, theta_v = _lookup(
            self._skills, _ids(t["skill"]), self._theta_m, self._theta_v, self.theta_prior_var
        )

        mean_z = a * (theta_m - b) - c * delta_m
        var_z = self.forecast_var_scale * (
            a * a * (theta_v + d_f * self.q_theta) + c * c * (delta_v + d_f * self.q_delta)
        )
        p = ndtr(mean_z / np.sqrt(1.0 + var_z))

        # items without parameters: last-checkpoint skill mean, as in T-SKIRT
        last = history[history["checkpoint_idx"] == history["checkpoint_idx"].max()]
        fallback = t["skill"].map(last.groupby("skill")["correct"].mean()).fillna(last["correct"].mean())
        p = np.where(np.isnan(p), fallback.to_numpy(dtype=float), p)
        return np.clip(p, 1e-6, 1.0 - 1e-6)
