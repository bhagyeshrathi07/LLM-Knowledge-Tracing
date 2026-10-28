import numpy as np
import pytest
from scipy.special import logsumexp

from llm_kt.bkt import (
    COLD_START,
    EPS,
    SkillParams,
    batch_log_emissions,
    clip_guess,
    fit_skill,
    forecast_state,
    log_forward_backward,
)


def brute_force(log_pi, log_A, log_emit):
    T = log_emit.shape[0]
    paths = np.array(np.meshgrid(*[[0, 1]] * T, indexing="ij")).reshape(T, -1).T  # (2^T, T)
    lp = np.array(
        [
            log_pi[p[0]] + log_emit[0, p[0]]
            + sum(log_A[p[t - 1], p[t]] + log_emit[t, p[t]] for t in range(1, T))
            for p in paths
        ]
    )
    loglik = logsumexp(lp)
    w = np.exp(lp - loglik)
    gamma = np.stack([(w * (paths[:, t] == 1)).sum() for t in range(T)])
    xi = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            xi[i, j] = sum(
                (w * ((paths[:, t - 1] == i) & (paths[:, t] == j))).sum() for t in range(1, T)
            )
    return gamma, xi, loglik


class TestForwardBackward:
    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        T = 5
        log_pi = np.log(np.array([0.7, 0.3]))
        log_A = np.log(np.array([[0.8, 0.2], [0.1, 0.9]]))
        log_emit = rng.normal(0, 3, (T, 2))
        res = log_forward_backward(log_pi, log_A, log_emit)
        g, xi, ll = brute_force(log_pi, log_A, log_emit)
        assert res.loglik == pytest.approx(ll)
        assert res.gamma[:, 1] == pytest.approx(g)
        assert res.gamma.sum(axis=1) == pytest.approx(np.ones(T))
        assert res.xi == pytest.approx(xi)
        assert res.filtered[-1] == pytest.approx(res.gamma[-1])  # smoothing = filtering at the end
        assert res.filtered.sum(axis=1) == pytest.approx(np.ones(T))

    def test_no_underflow_with_many_items(self):
        rng = np.random.default_rng(1)
        N, T = 2000, 6
        Y = (rng.random((N, T)) < 0.6).astype(float)
        le = batch_log_emissions(Y, np.full(N, 0.25), np.full(N, 0.2))
        assert le.shape == (T, 2) and np.all(np.isfinite(le)) and (le < -100).all()
        res = log_forward_backward(np.log([0.5, 0.5]), np.log([[0.9, 0.1], [0.05, 0.95]]), le)
        assert np.all(np.isfinite(res.gamma)) and np.isfinite(res.loglik)

    def test_emissions_mask_nan_cells_and_clip(self):
        Y = np.array([[1.0, np.nan], [0.0, 1.0]])
        # guess 0 / slip 1 are clipped away from the boundary, so nothing is -inf
        le = batch_log_emissions(Y, np.array([0.0, 0.5]), np.array([1.0, 0.5]))
        assert np.all(np.isfinite(le))
        assert le[1, 0] == pytest.approx(np.log(0.5))  # only item 1 observed at t=1
        assert le[1, 1] == pytest.approx(np.log(0.5))


def simulate_skill(rng, N, T, pi, learn, forget, guess, slip):
    z = np.zeros(T, dtype=int)
    z[0] = rng.random() < pi
    for t in range(1, T):
        z[t] = (rng.random() < learn) if z[t - 1] == 0 else (rng.random() >= forget)
    p = np.where(z[None, :] == 1, 1 - slip[:, None], guess[:, None])
    return (rng.random((N, T)) < p).astype(float), z


class TestHelpers:
    def test_clip_guess_keeps_learned_state_better(self):
        g = clip_guess(np.array([0.9, 0.2, 0.0]), np.array([0.3, 0.3, 0.3]))
        assert g == pytest.approx([0.65, 0.2, 1e-6])

    def test_forecast_state_moves_toward_stationary(self):
        learn, forget = 0.1, 0.05
        stat = learn / (learn + forget)
        p = [forecast_state(0.0, learn, forget, h) for h in (1, 5, 20)]
        assert p[0] == pytest.approx(0.1) and p[0] < p[1] < p[2] < stat
        assert forecast_state(1.0, learn, forget, 1) == pytest.approx(0.95)

    def test_cold_start_values(self):
        assert COLD_START == {"pi": 0.2, "learn": 0.1, "forget": 0.02, "guess_s": 0.25, "slip_s": 0.3}


class TestFitSkill:
    def test_recovers_parameters_in_per_skill_limit(self):
        rng = np.random.default_rng(2)
        N, T = 400, 300
        Y, _ = simulate_skill(rng, N, T, 0.1, 0.08, 0.01, np.full(N, 0.25), np.full(N, 0.2))
        fp = fit_skill(Y, kappa=1e6, rho=1.0)
        assert isinstance(fp, SkillParams)
        assert fp.learn == pytest.approx(0.08, abs=0.03)
        assert fp.forget == pytest.approx(0.01, abs=0.015)
        assert fp.guess_s == pytest.approx(0.25, abs=0.03)
        assert fp.slip_s == pytest.approx(0.2, abs=0.03)
        assert np.ptp(fp.guess) < 1e-3 and np.ptp(fp.slip) < 1e-3  # items sit at the anchor
        assert 1 <= fp.n_iter <= 50 and np.isfinite(fp.loglik)

    def test_small_kappa_lets_items_move_and_rho_weights_recent(self):
        rng = np.random.default_rng(3)
        N, T = 60, 30
        Y, _ = simulate_skill(rng, N, T, 0.9, 0.1, 0.0, np.full(N, 0.25), np.full(N, 0.2))
        Y[0] = 1.0  # always right
        Y[1] = 0.0  # always wrong
        Y[2] = np.r_[np.zeros(T - 3), np.ones(3)]  # wrong, then right for the last 3
        fixed = fit_skill(Y, kappa=1.0, rho=1.0)
        recent = fit_skill(Y, kappa=1.0, rho=0.7)
        assert 1 - fixed.slip[0] > 0.9 and 1 - fixed.slip[1] < 0.1
        assert 1 - recent.slip[2] > 1 - fixed.slip[2] + 0.2

    def test_fluid_anchor_arrays_are_used(self):
        rng = np.random.default_rng(4)
        N, T = 50, 20
        Y, _ = simulate_skill(rng, N, T, 0.5, 0.1, 0.01, np.full(N, 0.3), np.full(N, 0.2))
        ag, asl = np.linspace(0.05, 0.4, N), np.linspace(0.1, 0.5, N)
        fp = fit_skill(Y, kappa=1e6, rho=1.0, anchor_guess=ag, anchor_slip=asl)
        assert fp.guess == pytest.approx(clip_guess(ag, asl), abs=1e-3)
        assert fp.slip == pytest.approx(asl, abs=1e-3)

    def test_degenerate_all_correct_skill_is_finite(self):
        fp = fit_skill(np.ones((30, 2)), kappa=5.0, rho=1.0)
        for v in (fp.pi, fp.learn, fp.forget, fp.guess_s, fp.slip_s):
            assert np.isfinite(v) and EPS <= v <= 1 - EPS
        assert np.all(np.isfinite(fp.guess)) and np.all(np.isfinite(fp.slip))

    def test_warm_start_from_init_converges_in_fewer_iterations(self):
        rng = np.random.default_rng(5)
        Y, _ = simulate_skill(rng, 200, 40, 0.1, 0.1, 0.02, np.full(200, 0.25), np.full(200, 0.2))
        cold = fit_skill(Y, kappa=5.0, rho=1.0)
        warm = fit_skill(Y, kappa=5.0, rho=1.0, init=cold)
        assert warm.n_iter < cold.n_iter
        assert warm.learn == pytest.approx(cold.learn, abs=1e-3)
