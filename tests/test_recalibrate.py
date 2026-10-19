import numpy as np
import pandas as pd
from scipy.special import expit

from llm_kt.recalibrate import fit_item_offsets, recalibrated_params


def simulate(n_items=600, T=30, n_skills=3, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.8, 2.0, n_items)
    b_fluid = rng.normal(0, 1, n_items)
    delta_true = rng.normal(0, 0.8, n_items)          # this model's items are easier/harder than Fluid says
    skill_idx = rng.integers(0, n_skills, n_items)
    theta_true = np.cumsum(rng.normal(0.08, 0.05, (n_skills, T)), axis=1) - 1.0  # rising abilities
    z = a[:, None] * (theta_true[skill_idx, :] - (b_fluid + delta_true)[:, None])
    Y = (rng.random((n_items, T)) < expit(z)).astype(float)
    return a, b_fluid, delta_true, skill_idx, theta_true, Y


class TestFitItemOffsets:
    def test_recovers_offsets_and_abilities(self):
        a, b, delta_true, skill_idx, theta_true, Y = simulate()
        delta, gamma, theta = fit_item_offsets(Y, a, b, skill_idx, n_skills=3, tau=1.0)
        assert np.allclose(gamma, 0.0)
        assert np.corrcoef(delta, delta_true)[0, 1] > 0.8
        assert np.corrcoef(theta.ravel(), theta_true.ravel())[0, 1] > 0.9

    def test_shrinks_to_zero_with_no_data(self):
        a = np.ones(5)
        b = np.zeros(5)
        Y = np.full((5, 4), np.nan)
        delta, gamma, theta = fit_item_offsets(Y, a, b, np.zeros(5, dtype=int), n_skills=1, tau_a=0.5)
        assert np.allclose(delta, 0.0)
        assert np.allclose(gamma, 0.0)
        assert np.allclose(theta, 0.0)

    def test_improves_held_out_log_loss(self):
        a, b, _delta_true, skill_idx, _theta_true, Y = simulate(T=31)
        hist, final = Y[:, :30], Y[:, 30]
        delta, _, theta = fit_item_offsets(hist, a, b, skill_idx, n_skills=3, tau=1.0)
        th = theta[skill_idx, -1]  # carry last ability forward
        p_recal = np.clip(expit(a * (th - (b + delta))), 1e-6, 1 - 1e-6)
        _, _, theta0 = fit_item_offsets(hist, a, b, skill_idx, n_skills=3, tau=1e-6)  # delta pinned ~0
        p_fixed = np.clip(expit(a * (theta0[skill_idx, -1] - b)), 1e-6, 1 - 1e-6)
        ll = lambda p: -np.mean(final * np.log(p) + (1 - final) * np.log(1 - p))
        assert ll(p_recal) < ll(p_fixed)

    def test_tau_a_none_matches_previous_implementation(self):
        # first five deltas from the delta-only implementation (before tau_a existed)
        a, b, _delta_true, skill_idx, _theta_true, Y = simulate()
        delta, gamma, _ = fit_item_offsets(Y, a, b, skill_idx, n_skills=3, tau=1.0)
        expected = [-0.7629929929518171, 0.3854605885047792, -0.10933113909619334,
                    0.19600037387814706, 0.9041360008567944]
        assert np.allclose(delta[:5], expected, rtol=0, atol=1e-12)
        assert np.all(gamma == 0.0)


def simulate_disc(n_items=600, T=31, n_skills=3, seed=1):
    """Supplied a is off by a log-normal factor (sd 0.5); supplied b is off by N(0, 0.8)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.8, 2.0, n_items)
    b_fluid = rng.normal(0, 1, n_items)
    delta_true = rng.normal(0, 0.8, n_items)
    log_factor = rng.normal(0, 0.5, n_items)
    skill_idx = rng.integers(0, n_skills, n_items)
    theta_true = np.cumsum(rng.normal(0.08, 0.05, (n_skills, T)), axis=1) - 1.0
    z = (a * np.exp(log_factor))[:, None] * (theta_true[skill_idx, :] - (b_fluid + delta_true)[:, None])
    Y = (rng.random((n_items, T)) < expit(z)).astype(float)
    return a, b_fluid, delta_true, log_factor, skill_idx, Y


class TestDiscriminationOffsets:
    def test_recovers_both_offsets(self):
        a, b, delta_true, log_factor, skill_idx, Y = simulate_disc()
        delta, gamma, _ = fit_item_offsets(Y[:, :30], a, b, skill_idx, n_skills=3, tau=1.0, tau_a=0.5)
        assert np.corrcoef(gamma, log_factor)[0, 1] > 0.6
        assert np.corrcoef(delta, delta_true)[0, 1] > 0.8
        assert np.all(np.abs(gamma) <= 2.0)

    def test_both_offsets_beat_delta_only_held_out(self):
        a, b, _delta_true, _log_factor, skill_idx, Y = simulate_disc()
        hist, final = Y[:, :30], Y[:, 30]
        ll = lambda p: -np.mean(final * np.log(p) + (1 - final) * np.log(1 - p))

        def held_out(tau_a):
            delta, gamma, theta = fit_item_offsets(hist, a, b, skill_idx, n_skills=3, tau=1.0, tau_a=tau_a)
            p = expit(a * np.exp(gamma) * (theta[skill_idx, -1] - (b + delta)))
            return ll(np.clip(p, 1e-6, 1 - 1e-6))

        assert held_out(0.5) < held_out(None)


class TestRecalibratedParams:
    def test_returns_adjusted_table(self):
        a, b, _delta_true, skill_idx, _theta_true, Y = simulate(n_items=50, T=5)
        items = [f"mmlu_s{skill_idx[i]}_{i}" for i in range(50)]
        rows = []
        for i, it in enumerate(items):
            for t in range(5):
                rows.append({"model": "m", "checkpoint_idx": t, "skill": f"s{skill_idx[i]}", "item_id": it, "correct": int(Y[i, t])})
        history = pd.DataFrame(rows)
        params = pd.DataFrame({"a": a, "b": b}, index=pd.Index(items, name="item_id"))
        out = recalibrated_params(history, params, tau=1.0)
        assert list(out.columns) == ["a", "b", "delta", "gamma"]
        assert len(out) == 50
        assert np.allclose(out["b"] - out["delta"], params.loc[out.index, "b"])
        assert np.allclose(out["a"], params.loc[out.index, "a"])
        assert np.allclose(out["gamma"], 0.0)

    def test_tau_a_scales_discrimination(self):
        a, b, _delta_true, skill_idx, _theta_true, Y = simulate(n_items=50, T=5)
        items = [f"mmlu_s{skill_idx[i]}_{i}" for i in range(50)]
        history = pd.DataFrame([{"model": "m", "checkpoint_idx": t, "skill": f"s{skill_idx[i]}", "item_id": it,
                                 "correct": int(Y[i, t])} for i, it in enumerate(items) for t in range(5)])
        params = pd.DataFrame({"a": a, "b": b}, index=pd.Index(items, name="item_id"))
        out = recalibrated_params(history, params, tau=1.0, tau_a=0.5)
        assert not np.allclose(out["gamma"], 0.0)
        assert np.allclose(out["a"], params.loc[out.index, "a"] * np.exp(out["gamma"]))

    def test_items_missing_from_params_are_ignored(self):
        history = pd.DataFrame({"model": "m", "checkpoint_idx": [0, 0, 1, 1], "skill": "s",
                                "item_id": ["x", "y", "x", "y"], "correct": [1, 0, 1, 0]})
        params = pd.DataFrame({"a": [1.0], "b": [0.0]}, index=pd.Index(["x"], name="item_id"))
        out = recalibrated_params(history, params)
        assert list(out.index) == ["x"]
