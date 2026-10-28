import numpy as np
import pytest
from scipy.special import logsumexp

from llm_kt.bkt import batch_log_emissions, log_forward_backward


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
