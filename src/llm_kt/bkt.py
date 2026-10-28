"""Bayesian Knowledge Tracing per skill, in log space, with per-item guess and slip.

A two-state hidden Markov model per (model, skill). The state z_t is 0 (skill not learned) or
1 (learned) at checkpoint t; one transition happens between consecutive checkpoints; every
item observed at a checkpoint is a simultaneous noisy reading of the state (Decision 1 in
docs/project-notes.md: the checkpoint is the time step, observations are batched).

    initial      P(z_0 = 1) = pi
    transition   P(z_{t+1} = 1 | z_t = 0) = learn        P(z_{t+1} = 0 | z_t = 1) = forget
    emission     P(y_it = 1 | z_t = 1) = 1 - slip_i      P(y_it = 1 | z_t = 0) = guess_i
    batch        log P(y_t | z) = sum_i log P(y_it | z)  over the items observed at t

Why log space. pyBKT multiplies raw emission probabilities before normalising. With 100 to
1,534 items per checkpoint that product underflows to zero, so the recursions here run on log
probabilities with logsumexp throughout (`batch_log_emissions`, `log_forward_backward`).

Why per-item emissions. With one guess/slip pair per skill every item in a subject gets the
same probability at a checkpoint, and the within-subject AUC used by `llm_kt.evaluate` is 0.5
by construction. Each item therefore has its own guess and slip. Fitted freely they would
reproduce the item's own history (which is what the item-mean baseline already does) and
leave the skill state nothing to explain, so they are shrunk toward an anchor with a
pseudo-count `kappa` and, optionally, computed from recency-weighted counts (`rho < 1`) so
that they can move over training. The recency-weighted item step is a forgetting-factor
approximation, not an exact EM ascent step; the skill-level steps are exact MAP-EM.

The anchor is either the skill's shared guess and slip (anchor="skill", re-estimated by EM)
or the guess and slip implied by Fluid's 2PL item parameters at two global ability levels
(anchor="fluid": guess = P(theta_lo), 1 - slip = P(theta_hi)). `kappa -> inf` with the Fluid
anchor is the "borrowed item parameters" variant; with the skill anchor it is classic
per-skill BKT.

Design note: docs/superpowers/specs/2026-10-27-bkt-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from llm_kt.static_irt import prob_logistic

EPS = 1e-6
MIN_GAP = 0.05  # the learned state must be the better one: guess <= 1 - slip - MIN_GAP
ANCHORS = ("skill", "fluid")
COLD_START = {"pi": 0.2, "learn": 0.1, "forget": 0.02, "guess_s": 0.25, "slip_s": 0.3}
KAPPA_LIMIT = 1e5  # kappa at or above this pins every item exactly to its anchor (per-skill BKT)


def _clip_unit(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, EPS, 1.0 - EPS)


def batch_log_emissions(Y: np.ndarray, guess: np.ndarray, slip: np.ndarray) -> np.ndarray:
    """Per-checkpoint log P(y_t | z) for z = 0, 1, summed over the items observed at t.

    Y is (items, checkpoints) with NaN for unobserved cells; guess and slip are per item.
    Returns (checkpoints, 2): column 0 is the not-learned state, column 1 the learned state.
    """
    Y = np.asarray(Y, dtype=float)
    M = ~np.isnan(Y)
    Yz = np.where(M, Y, 0.0)
    g = _clip_unit(np.asarray(guess, dtype=float))[:, None]
    s = _clip_unit(np.asarray(slip, dtype=float))[:, None]
    log_z0 = Yz * np.log(g) + (1.0 - Yz) * np.log1p(-g)
    log_z1 = Yz * np.log1p(-s) + (1.0 - Yz) * np.log(s)
    out = np.empty((Y.shape[1], 2))
    out[:, 0] = np.where(M, log_z0, 0.0).sum(axis=0)
    out[:, 1] = np.where(M, log_z1, 0.0).sum(axis=0)
    return out


class FBResult(NamedTuple):
    gamma: np.ndarray  # (T, 2) smoothed P(z_t | y_{0..T-1})
    xi: np.ndarray  # (2, 2) expected transition counts summed over t: xi[i, j] for i -> j
    filtered: np.ndarray  # (T, 2) P(z_t | y_{0..t})
    loglik: float


def log_forward_backward(log_pi: np.ndarray, log_A: np.ndarray, log_emit: np.ndarray) -> FBResult:
    """Forward-backward for a two-state chain on log probabilities.

    log_pi: (2,) log initial distribution. log_A[i, j] = log P(z_{t+1} = j | z_t = i).
    log_emit: (T, 2) batched log emission likelihoods from `batch_log_emissions`.
    """
    log_pi = np.asarray(log_pi, dtype=float)
    log_A = np.asarray(log_A, dtype=float)
    log_emit = np.asarray(log_emit, dtype=float)
    T = log_emit.shape[0]
    log_alpha = np.empty((T, 2))
    log_beta = np.zeros((T, 2))
    log_alpha[0] = log_pi + log_emit[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_emit[t]
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(log_A + (log_emit[t + 1] + log_beta[t + 1])[None, :], axis=1)
    loglik = float(logsumexp(log_alpha[-1]))
    gamma = np.exp(log_alpha + log_beta - loglik)
    filtered = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
    if T > 1:
        pair = (
            log_alpha[:-1, :, None]
            + log_A[None, :, :]
            + (log_emit[1:] + log_beta[1:])[:, None, :]
            - loglik
        )
        xi = np.exp(logsumexp(pair, axis=0))
    else:
        xi = np.zeros((2, 2))
    return FBResult(gamma=gamma, xi=xi, filtered=filtered, loglik=loglik)


def clip_guess(guess: np.ndarray | float, slip: np.ndarray | float) -> np.ndarray | float:
    """Keep the learned state the better one: guess <= 1 - slip - MIN_GAP, both inside (0, 1)."""
    g = _clip_unit(np.asarray(guess, dtype=float))
    s = _clip_unit(np.asarray(slip, dtype=float))
    out = np.minimum(g, 1.0 - s - MIN_GAP)
    return _clip_unit(out) if out.ndim else float(_clip_unit(out))


def forecast_state(p1: np.ndarray | float, learn: float, forget: float, h: int) -> np.ndarray | float:
    """P(z = 1) after h transition steps from P(z = 1) = p1."""
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    p = np.asarray(p1, dtype=float)
    for _ in range(int(h)):
        p = p * (1.0 - forget) + (1.0 - p) * learn
    return p if p.ndim else float(p)


@dataclass
class SkillParams:
    pi: float
    learn: float
    forget: float
    guess_s: float
    slip_s: float
    guess: np.ndarray  # per item
    slip: np.ndarray  # per item
    n_iter: int = 0
    loglik: float = float("nan")
    filtered: np.ndarray | None = None  # P(z_t = 1 | y_{0..t}) per checkpoint, final params


def _rate(num: float, den: float, prior_mean: float, prior_count: float) -> float:
    return float(_clip_unit((num + prior_count * prior_mean) / (den + prior_count)))


def fit_skill(
    Y: np.ndarray,
    kappa: float,
    rho: float,
    anchor_guess: np.ndarray | None = None,
    anchor_slip: np.ndarray | None = None,
    init: SkillParams | None = None,
    max_iter: int = 50,
    tol: float = 1e-5,
    prior_count: float = 1.0,
) -> SkillParams:
    """MAP-EM for one skill on an (items, checkpoints) 0/1 matrix with NaN for unobserved cells.

    Skill-level parameters (pi, learn, forget, guess_s, slip_s) get exact EM updates with a
    pseudo-count prior of `prior_count` observations at the COLD_START values. Per-item guess
    and slip are shrunk toward the anchor with `kappa` pseudo-observations, using observations
    weighted by rho ** (T - 1 - t) so that recent checkpoints count more when rho < 1. With no
    anchor arrays the anchor is the skill's own (guess_s, slip_s), re-estimated each iteration.
    Stops when the mean log-likelihood per observation changes by less than `tol`.
    """
    Y = np.asarray(Y, dtype=float)
    N, T = Y.shape
    M = ~np.isnan(Y)
    Yz = np.where(M, Y, 0.0)
    n_obs = max(int(M.sum()), 1)
    w = rho ** np.arange(T - 1, -1, -1, dtype=float)  # (T,) recency weights, 1 at the origin
    fluid = anchor_guess is not None
    if fluid:  # NaN entries mean "no Fluid parameters for this item": use the skill anchor there
        a_guess_raw = np.asarray(anchor_guess, dtype=float)
        a_slip_raw = np.asarray(anchor_slip, dtype=float)

    def anchors(guess_s: float, slip_s: float) -> tuple[np.ndarray | float, np.ndarray | float]:
        if not fluid:
            return guess_s, slip_s
        s0 = _clip_unit(np.where(np.isnan(a_slip_raw), slip_s, a_slip_raw))
        g0 = clip_guess(np.where(np.isnan(a_guess_raw), guess_s, a_guess_raw), s0)
        return g0, s0

    pinned = kappa >= KAPPA_LIMIT
    m0 = COLD_START
    if init is None:
        pi, learn, forget, guess_s, slip_s = (
            m0["pi"], m0["learn"], m0["forget"], m0["guess_s"], m0["slip_s"])
        g0, s0 = anchors(guess_s, slip_s)
        guess, slip = np.broadcast_to(g0, N).copy(), np.broadcast_to(s0, N).copy()
    else:
        pi, learn, forget, guess_s, slip_s = (
            init.pi, init.learn, init.forget, init.guess_s, init.slip_s)
        guess, slip = init.guess.astype(float).copy(), init.slip.astype(float).copy()

    prev = -np.inf
    n_iter = 0
    loglik = float("nan")
    for n_iter in range(1, max_iter + 1):
        log_pi = np.log([1.0 - pi, pi])
        log_A = np.log([[1.0 - learn, learn], [forget, 1.0 - forget]])
        fb = log_forward_backward(log_pi, log_A, batch_log_emissions(Y, guess, slip))
        loglik = fb.loglik
        g1 = fb.gamma[:, 1]  # (T,)
        g0 = 1.0 - g1
        # skill-level M-step
        pi = _rate(g1[0], 1.0, m0["pi"], prior_count)
        learn = _rate(fb.xi[0, 1], fb.xi[0, 0] + fb.xi[0, 1], m0["learn"], prior_count)
        forget = _rate(fb.xi[1, 0], fb.xi[1, 0] + fb.xi[1, 1], m0["forget"], prior_count)
        obs0 = M * g0[None, :]
        obs1 = M * g1[None, :]
        guess_s = _rate((obs0 * Yz).sum(), obs0.sum(), m0["guess_s"], prior_count)
        slip_s = _rate((obs1 * (1.0 - Yz)).sum(), obs1.sum(), m0["slip_s"], prior_count)
        guess_s = clip_guess(guess_s, slip_s)
        # item-level M-step, recency weighted, shrunk toward the anchor
        g0_i, s0_i = anchors(guess_s, slip_s)
        if pinned:
            guess, slip = np.broadcast_to(g0_i, N).copy(), np.broadcast_to(s0_i, N).copy()
        else:
            w0 = obs0 * w[None, :]
            w1 = obs1 * w[None, :]
            guess = ((w0 * Yz).sum(axis=1) + kappa * g0_i) / (w0.sum(axis=1) + kappa)
            slip = ((w1 * (1.0 - Yz)).sum(axis=1) + kappa * s0_i) / (w1.sum(axis=1) + kappa)
            slip = _clip_unit(slip)
            guess = clip_guess(guess, slip)
        if abs(loglik - prev) / n_obs < tol:
            break
        prev = loglik
    # filtered state under the final parameters, for forecasting and the mastery curve
    log_pi = np.log([1.0 - pi, pi])
    log_A = np.log([[1.0 - learn, learn], [forget, 1.0 - forget]])
    fb = log_forward_backward(log_pi, log_A, batch_log_emissions(Y, guess, slip))
    return SkillParams(pi, learn, forget, guess_s, slip_s, guess, slip, n_iter, loglik,
                       filtered=fb.filtered[:, 1])


def _ids(col: pd.Series) -> np.ndarray:
    """String keys as an object array, so Index.get_indexer uses a hash table."""
    return col.to_numpy(dtype=object)


class BKT:
    """Predictor: per-skill two-state BKT with per-item guess and slip, refit at every origin."""

    def __init__(
        self,
        kappa: float = 5.0,
        rho: float = 1.0,
        anchor: str = "skill",
        item_params: pd.DataFrame | None = None,
        theta_lo: float = -1.0,
        theta_hi: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-5,
        prior_count: float = 1.0,
    ):
        if anchor not in ANCHORS:
            raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        if not 0 < rho <= 1:
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        if anchor == "fluid":
            if item_params is None:
                raise ValueError("anchor='fluid' needs item_params (item_id, a, b)")
            if not theta_lo < theta_hi:
                raise ValueError(f"need theta_lo < theta_hi, got {theta_lo}, {theta_hi}")
        self.kappa = kappa
        self.rho = rho
        self.anchor = anchor
        self.theta_lo = theta_lo
        self.theta_hi = theta_hi
        self.max_iter = max_iter
        self.tol = tol
        self.prior_count = prior_count
        self.anchors: pd.DataFrame | None = None
        if anchor == "fluid":
            ip = item_params.drop_duplicates("item_id")
            a = ip["a"].to_numpy(dtype=float)
            b = ip["b"].to_numpy(dtype=float)
            slip0 = 1.0 - prob_logistic(theta_hi, a, b)
            guess0 = prob_logistic(theta_lo, a, b)
            self.anchors = pd.DataFrame(
                {"guess0": guess0, "slip0": slip0}, index=pd.Index(_ids(ip["item_id"]))
            )
        self.name = f"bkt_k{kappa:g}_r{rho:g}_{anchor}" + (
            f"_lo{theta_lo:g}_hi{theta_hi:g}" if anchor == "fluid" else ""
        )
        self._reset()

    # -- state ---------------------------------------------------------------------------

    def _reset(self) -> None:
        self._key = None  # (model, last checkpoint, n rows, sum correct, weighted sum)
        self._state: dict[str, dict] = {}  # skill -> params, items (Index), ckpts
        self._global_last = float("nan")

    @staticmethod
    def _fingerprint(history: pd.DataFrame, upto: int) -> tuple:
        h = history[history["checkpoint_idx"] <= upto]
        y = h["correct"].to_numpy(dtype=float)
        model = history["model"].iloc[0] if "model" in history.columns and len(history) else None
        return (model, upto, len(h), float(y.sum()), float((y * h["checkpoint_idx"].to_numpy()).sum()))

    def _anchor_arrays(self, item_ids: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Fluid anchor values per item (NaN where the item has no parameters); None for skill anchor."""
        if self.anchors is None:
            return None, None
        pos = self.anchors.index.get_indexer(item_ids)
        g = np.full(len(item_ids), np.nan)
        s = np.full(len(item_ids), np.nan)
        hit = pos >= 0
        g[hit] = self.anchors["guess0"].to_numpy()[pos[hit]]
        s[hit] = self.anchors["slip0"].to_numpy()[pos[hit]]
        return g, s

    def fit(self, history: pd.DataFrame) -> None:
        """Fit every skill on `history`, warm-starting from the cache when it extends the cached run."""
        last = int(history["checkpoint_idx"].max())
        warm = False
        if self._key is not None:
            cached_last = self._key[1]
            if last >= cached_last and self._fingerprint(history, cached_last) == self._key:
                warm = True
            else:
                self._reset()
        wide = history.pivot(index="item_id", columns="checkpoint_idx", values="correct")
        wide.index = pd.Index(_ids(wide.index.to_series()))
        ckpts = wide.columns.to_numpy(dtype=int)
        item_skill = history.drop_duplicates("item_id").set_index("item_id")["skill"]
        new_state: dict[str, dict] = {}
        for skill, ids in item_skill.groupby(item_skill, sort=True).groups.items():
            items = pd.Index(np.sort(np.asarray(list(ids), dtype=object)))
            Y = wide.loc[items].to_numpy(dtype=float)
            a_g, a_s = self._anchor_arrays(items.to_numpy(dtype=object))
            init = None
            if warm and skill in self._state:
                old = self._state[skill]
                op = old["params"]
                pos = old["items"].get_indexer(items)
                hit = pos >= 0
                g0, s0 = (a_g, a_s) if a_g is not None else (op.guess_s, op.slip_s)
                guess = np.where(hit, op.guess[np.where(hit, pos, 0)], np.broadcast_to(g0, len(items)))
                slip = np.where(hit, op.slip[np.where(hit, pos, 0)], np.broadcast_to(s0, len(items)))
                guess = np.where(np.isnan(guess), op.guess_s, guess)
                slip = np.where(np.isnan(slip), op.slip_s, slip)
                init = SkillParams(op.pi, op.learn, op.forget, op.guess_s, op.slip_s, guess, slip)
            params = fit_skill(
                Y, self.kappa, self.rho, a_g, a_s, init=init,
                max_iter=self.max_iter, tol=self.tol, prior_count=self.prior_count,
            )
            new_state[skill] = {"params": params, "items": items, "ckpts": ckpts}
        self._state = new_state
        self._global_last = float(history.loc[history["checkpoint_idx"] == last, "correct"].mean())
        self._key = self._fingerprint(history, last)

    # -- predicting ----------------------------------------------------------------------

    def predict(self, history: pd.DataFrame, target: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        self.fit(history)
        p = np.full(len(target), self._global_last)
        items_all = _ids(target["item_id"])
        for skill, idx in target.groupby("skill").indices.items():
            st = self._state.get(skill)
            if st is None:
                continue  # skill never observed: last-checkpoint global accuracy
            sp = st["params"]
            z = forecast_state(sp.filtered[-1], sp.learn, sp.forget, horizon)
            items = items_all[idx]
            pos = st["items"].get_indexer(items)
            hit = pos >= 0
            a_g, a_s = self._anchor_arrays(items)
            if a_g is None:
                g0, s0 = np.full(len(items), sp.guess_s), np.full(len(items), sp.slip_s)
            else:
                s0 = _clip_unit(np.where(np.isnan(a_s), sp.slip_s, a_s))
                g0 = clip_guess(np.where(np.isnan(a_g), sp.guess_s, a_g), s0)
            guess = np.where(hit, sp.guess[np.where(hit, pos, 0)], g0)
            slip = np.where(hit, sp.slip[np.where(hit, pos, 0)], s0)
            p[idx] = z * (1.0 - slip) + (1.0 - z) * guess
        return np.clip(p, EPS, 1.0 - EPS)

    # -- interpretable output -------------------------------------------------------------

    def mastery(self) -> pd.DataFrame:
        """Filtered P(learned) per (skill, checkpoint) for the last fitted history."""
        frames = [
            pd.DataFrame({"skill": skill, "checkpoint_idx": st["ckpts"],
                          "p_learned": st["params"].filtered})
            for skill, st in self._state.items()
        ]
        if not frames:
            return pd.DataFrame(columns=["skill", "checkpoint_idx", "p_learned"])
        return pd.concat(frames, ignore_index=True)

    def skill_params(self) -> pd.DataFrame:
        """One row per skill: n_items, pi, learn, forget, guess_s, slip_s, n_iter."""
        rows = [
            {"skill": skill, "n_items": len(st["items"]), "pi": sp.pi, "learn": sp.learn,
             "forget": sp.forget, "guess_s": sp.guess_s, "slip_s": sp.slip_s, "n_iter": sp.n_iter}
            for skill, st in self._state.items()
            for sp in (st["params"],)
        ]
        cols = ["skill", "n_items", "pi", "learn", "forget", "guess_s", "slip_s", "n_iter"]
        return pd.DataFrame(rows, columns=cols)
