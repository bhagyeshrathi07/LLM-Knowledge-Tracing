import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, spearmanr

from llm_kt.baselines import ItemMean
from llm_kt.drift_irt import DriftIRT, filter_update, probit_score_and_curvature
from llm_kt.evaluate import TARGET_COLUMNS, evaluate
from llm_kt.tskirt import TSKIRT, to_probit


def rows_from_matrix(Y: np.ndarray, item_ids: list[str], skills: list[str], model: str = "m") -> pd.DataFrame:
    """Long-format history from an items x checkpoints 0/1 matrix."""
    n_items, T = Y.shape
    return pd.DataFrame(
        {
            "model": model,
            "checkpoint_idx": np.repeat(np.arange(T), n_items),
            "checkpoint_label": [f"step{t}" for t in np.repeat(np.arange(T), n_items)],
            "step": np.repeat(np.arange(T), n_items),
            "tokens_seen_b": pd.NA,
            "benchmark": "mmlu",
            "skill": np.tile(skills, T),
            "item_id": np.tile(item_ids, T),
            "correct": Y.T.ravel().astype(int),
        }
    )


def target_at(history: pd.DataFrame, idx: int) -> pd.DataFrame:
    last = history[history["checkpoint_idx"] == history["checkpoint_idx"].max()]
    return last[TARGET_COLUMNS].assign(checkpoint_idx=idx, step=idx).reset_index(drop=True)


@pytest.fixture
def world():
    """Two skills, 300 items, 12 checkpoints simulated from the probit model with rising ability."""
    rng = np.random.default_rng(0)
    n, T = 300, 12
    item_ids = [f"mmlu_{i}" for i in range(n)]
    skills = ["s0" if i < 150 else "s1" for i in range(n)]
    params = pd.DataFrame({"item_id": item_ids, "a": rng.uniform(1.0, 2.5, n), "b": rng.normal(0, 1, n)})
    a_p, b_p = to_probit(params["a"].to_numpy(), params["b"].to_numpy())
    theta = np.linspace(-0.5, 0.8, T)
    Y = (rng.random((n, T)) < norm.cdf(a_p[:, None] * (theta[None, :] - b_p[:, None]))).astype(float)
    return params, rows_from_matrix(Y, item_ids, skills), item_ids, skills


class TestProbitScore:
    def test_matches_finite_differences(self):
        z = np.linspace(-6, 6, 25)
        for y in (0.0, 1.0):
            g, h = probit_score_and_curvature(z, np.full_like(z, y))
            eps = 1e-5
            sign = 2 * y - 1

            def ll(x, sign=sign):
                return norm.logcdf(sign * x)

            assert g == pytest.approx((ll(z + eps) - ll(z - eps)) / (2 * eps), rel=1e-4, abs=1e-6)
            fd2 = (ll(z + eps) - 2 * ll(z) + ll(z - eps)) / eps**2
            assert h == pytest.approx(-fd2, rel=1e-2, abs=1e-4)
            assert ((h > 0) & (h <= 1)).all()

    def test_stable_far_in_the_tails(self):
        z = np.array([-40.0, 40.0])
        g, h = probit_score_and_curvature(z, np.array([1.0, 0.0]))
        assert np.all(np.isfinite(g)) and np.all(np.isfinite(h))


class TestFilterUpdate:
    def test_recovers_ability_with_many_items(self):
        rng = np.random.default_rng(1)
        n = 4000
        a = rng.uniform(0.5, 1.5, n)
        b = rng.normal(0, 1, n)
        skill_idx = (np.arange(n) >= n // 2).astype(int)
        true = np.array([-0.7, 0.9])
        y = (rng.random(n) < norm.cdf(a * (true[skill_idx] - b))).astype(float)
        th, th_v, de, de_v = filter_update(
            y, a, b, a, skill_idx, np.zeros(2), np.full(2, 10.0), np.zeros(n), np.full(n, 1e-6)
        )
        assert th == pytest.approx(true, abs=0.1)
        assert (th_v < 0.01).all()
        assert np.abs(de).max() < 1e-3
        assert (de_v <= 1e-6).all()


class TestPredictor:
    def test_shape_alignment_and_range(self, world):
        params, hist, _, _ = world
        m = DriftIRT(params, q_theta=0.01, q_delta=0.01, delta_prior_var=1.0)
        target = target_at(hist, 12)
        p = m.predict(hist, target, horizon=1)
        assert p.shape == (len(target),)
        assert np.all(np.isfinite(p)) and ((p > 0) & (p < 1)).all()
        # shuffling target rows permutes predictions the same way
        perm = np.random.default_rng(3).permutation(len(target))
        p_shuf = DriftIRT(params, 0.01, 0.01, 1.0).predict(hist, target.iloc[perm].reset_index(drop=True))
        assert p_shuf == pytest.approx(p[perm])

    @pytest.mark.parametrize("residual_scale", ["difficulty", "probit"])
    def test_always_right_beats_always_wrong_in_same_skill(self, residual_scale):
        T = 8
        ids = ["mmlu_right", "mmlu_wrong"] + [f"mmlu_x{i}" for i in range(40)]
        skills = ["s"] * len(ids)
        rng = np.random.default_rng(4)
        Y = np.vstack([np.ones(T), np.zeros(T), (rng.random((40, T)) < 0.5).astype(float)])
        # identical Fluid parameters for the two items of interest
        params = pd.DataFrame({"item_id": ids, "a": 1.5, "b": 0.0})
        hist = rows_from_matrix(Y, ids, skills)
        m = DriftIRT(params, 0.01, 0.05, 1.0, residual_scale=residual_scale)
        p = pd.Series(m.predict(hist, target_at(hist, T)), index=ids)
        assert p["mmlu_right"] > 0.5 > p["mmlu_wrong"]

    def test_longer_horizon_pulls_toward_half_only_with_forecast_variance(self, world):
        params, hist, _, _ = world
        target_near, target_far = target_at(hist, 12), target_at(hist, 31)
        m = DriftIRT(params, 0.05, 0.05, 1.0, forecast_var_scale=1.0)
        near = m.predict(hist, target_near, horizon=1)
        far = m.predict(hist, target_far, horizon=20)
        assert np.mean(np.abs(far - 0.5)) < np.mean(np.abs(near - 0.5))
        assert np.all(np.abs(far - 0.5) <= np.abs(near - 0.5) + 1e-12)
        plug = DriftIRT(params, 0.05, 0.05, 1.0, forecast_var_scale=0.0)
        assert plug.predict(hist, target_far, horizon=20) == pytest.approx(
            plug.predict(hist, target_near, horizon=1)
        )

    def test_no_residual_limit_is_per_skill(self, world):
        params, hist, _, _ = world
        q = 0.01
        m = DriftIRT(params, q_theta=q, q_delta=0.0, delta_prior_var=1e-8)
        target = target_at(hist, 12)
        p = m.predict(hist, target)
        assert np.abs(m._delta_m).max() < 1e-6
        # with no residual, a prediction is a function of the item's (a, b) and its skill only
        t = target.join(params.set_index("item_id"), on="item_id")
        a_p, b_p = to_probit(t["a"].to_numpy(), t["b"].to_numpy())
        theta = pd.Series(m._theta_m, index=m._skills)[t["skill"]].to_numpy()
        theta_v = pd.Series(m._theta_v, index=m._skills)[t["skill"]].to_numpy() + q
        assert p == pytest.approx(norm.cdf(a_p * (theta - b_p) / np.sqrt(1 + a_p**2 * theta_v)), abs=1e-6)

    def test_no_drift_no_residual_matches_static_tskirt(self, world):
        # q_theta = q_delta = 0 and delta_prior_var -> 0: the filter is a sequential Laplace fit of
        # one static ability per skill, which T-SKIRT with nu = 0 fits in batch
        params, hist, _, _ = world
        target = target_at(hist, 12)
        p = DriftIRT(params, q_theta=0.0, q_delta=0.0, delta_prior_var=1e-8).predict(hist, target)
        p_ts = TSKIRT(params, nu=0.0).predict(hist, target)
        assert np.abs(p - p_ts).max() < 0.01

    def test_large_item_drift_tracks_recent_history(self):
        T = 20
        ids = ["mmlu_flip"] + [f"mmlu_x{i}" for i in range(60)]
        skills = ["s"] * len(ids)
        rng = np.random.default_rng(5)
        flip = np.r_[np.zeros(T - 3), np.ones(3)]  # wrong for 17 checkpoints, then right for 3
        Y = np.vstack([flip, (rng.random((60, T)) < 0.5).astype(float)])
        params = pd.DataFrame({"item_id": ids, "a": 1.7, "b": 0.0})
        hist = rows_from_matrix(Y, ids, skills)
        target = target_at(hist, T)

        def p_flip(**kw):
            m = DriftIRT(params, residual_scale="probit", **kw)
            return pd.Series(m.predict(hist, target), index=ids)["mmlu_flip"]

        tracking = p_flip(q_theta=1e-4, q_delta=2.0, delta_prior_var=4.0, forecast_var_scale=0.0)
        static = p_flip(q_theta=1e-4, q_delta=0.0, delta_prior_var=4.0, forecast_var_scale=0.0)
        item_mean = (3 + 0.5) / (T + 1)
        assert tracking > 0.5            # follows the last few checkpoints
        assert static < 0.35             # q_delta = 0: a static offset, near the running mean
        assert abs(static - item_mean) < abs(tracking - item_mean)

    def test_static_offset_limit_ranks_items_like_item_mean(self, world):
        params, hist, _, _ = world
        target = target_at(hist, 12)
        m = DriftIRT(params, 1e-4, 0.0, 1.0, forecast_var_scale=0.0, residual_scale="probit")
        rho = spearmanr(m.predict(hist, target), ItemMean().predict(hist, target)).statistic
        assert rho > 0.97

    def test_unseen_item_uses_prior_residual(self, world):
        params, hist, item_ids, _ = world
        m = DriftIRT(params, 0.02, 0.03, 0.7, forecast_var_scale=1.0)
        target = target_at(hist, 12)
        m.predict(hist[hist["item_id"] != item_ids[0]], target)
        p = m.predict(hist[hist["item_id"] != item_ids[0]], target)[0]
        a, b = to_probit(params["a"].iloc[0], params["b"].iloc[0])
        th, th_v = m._theta_m[0], m._theta_v[0]
        expected = norm.cdf(a * (th - b) / np.sqrt(1 + a * a * (th_v + 0.02) + a * a * (0.7 + 0.03)))
        assert p == pytest.approx(expected)

    def test_items_without_params_fall_back_to_skill_mean(self, world):
        params, hist, _, _ = world
        m = DriftIRT(params.iloc[1:], 0.01, 0.01, 1.0)
        target = target_at(hist, 12)
        p = m.predict(hist, target)
        last = hist[hist["checkpoint_idx"] == 11]
        assert p[0] == pytest.approx(last[last["skill"] == "s0"]["correct"].mean())

    def test_incremental_cache_matches_fresh_fit_and_resets(self, world):
        params, hist, _, _ = world
        target = target_at(hist, 12)
        m = DriftIRT(params, 0.02, 0.02, 1.0)
        for t in (3, 7):
            m.predict(hist[hist["checkpoint_idx"] <= t], target)
        assert m.predict(hist, target) == pytest.approx(DriftIRT(params, 0.02, 0.02, 1.0).predict(hist, target))
        # a different history (flipped answers) must not reuse the cached state
        other = hist.assign(correct=1 - hist["correct"])
        assert m.predict(other, target) == pytest.approx(DriftIRT(params, 0.02, 0.02, 1.0).predict(other, target))
        # going back to an earlier origin also resets
        short = hist[hist["checkpoint_idx"] <= 4]
        assert m.predict(short, target) == pytest.approx(DriftIRT(params, 0.02, 0.02, 1.0).predict(short, target))

    def test_runs_in_harness(self, world):
        params, hist, _, _ = world
        res = evaluate(hist, [DriftIRT(params, 0.01, 0.01, 1.0), ItemMean()], horizon=2)
        assert set(res["predictor"]) == {"drift_irt_qt0.01_qd0.01_pv1_fv1", "item_mean"}
        assert res["log_loss"].notna().all()

    def test_name_and_validation(self):
        params = pd.DataFrame({"item_id": ["x"], "a": [1.0], "b": [0.0]})
        assert DriftIRT(params, 0.05, 0.001, 0.5, 0.0, residual_scale="probit").name == (
            "drift_irt_qt0.05_qd0.001_pv0.5_fv0_zres"
        )
        with pytest.raises(ValueError):
            DriftIRT(params, residual_scale="logit")
        with pytest.raises(ValueError):
            DriftIRT(params, delta_prior_var=0.0)
