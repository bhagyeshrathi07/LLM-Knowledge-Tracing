import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, spearmanr

from llm_kt.baselines import ItemMean
from llm_kt.drift_irt import DriftIRT, filter_update, flat_item_params, probit_score_and_curvature
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


# -- skill_state=False and the flat item table (DriftIRT ablation) ------------------------------

TOY_IDS = [f"mmlu_{i}" for i in range(12)]
TOY_SKILLS = ["s0"] * 4 + ["s1"] * 4 + ["s2"] * 4
TOY_PARAMS = pd.DataFrame({
    "item_id": TOY_IDS,
    "a": [2.048, 1.378, 2.217, 1.895, 0.688, 2.451, 2.022, 2.072, 0.756, 1.401, 1.242, 2.354],
    "b": [0.066, 1.127, 0.468, -0.859, 0.369, -0.959, 0.878, -0.05, -0.185, -0.681, 1.223, -0.155],
})
TOY_Y = np.array([
    [0, 1, 1, 1, 1, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 0, 1, 0, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 1, 1, 0, 1, 1], [1, 0, 1, 1, 1, 0, 0, 0, 1, 0], [0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
], dtype=float)


def item_only_reference(y: np.ndarray, a: float, b: float, q_delta: float, prior_var: float,
                        forecast_var_scale: float, d_f: float = 1.0) -> float:
    """One item, no ability: z = -a b - delta, delta a Gaussian random walk, Laplace update per
    checkpoint (Newton to convergence), forecast Phi(mean / sqrt(1 + scale * (v + d_f q)))."""
    a_p, b_p = a / 1.7, b
    m, v = 0.0, prior_var
    for t, yt in enumerate(y):
        if t > 0:
            v += q_delta
        s = 2.0 * yt - 1.0
        x = m
        for _ in range(100):
            u = s * (-a_p * b_p - x)
            lam = np.exp(norm.logpdf(u) - norm.logcdf(u))
            grad = -s * lam - (x - m) / v
            curv = lam * (lam + u) + 1.0 / v
            x += grad / curv
        u = s * (-a_p * b_p - x)
        lam = np.exp(norm.logpdf(u) - norm.logcdf(u))
        m, v = x, 1.0 / (lam * (lam + u) + 1.0 / v)
    mean_z = -a_p * b_p - m
    return float(norm.cdf(mean_z / np.sqrt(1.0 + forecast_var_scale * (v + d_f * q_delta))))


class TestNoSkillState:
    @pytest.fixture
    def toy(self):
        return rows_from_matrix(TOY_Y, TOY_IDS, TOY_SKILLS)

    def _model(self, **kw):
        kw = {"q_theta": 0.0, "q_delta": 0.01, "delta_prior_var": 1.0, "forecast_var_scale": 0.0,
              "residual_scale": "probit", "skill_state": False, **kw}
        return DriftIRT(TOY_PARAMS, **kw)

    def test_theta_is_exactly_zero(self, toy):
        m = self._model()
        m.predict(toy, target_at(toy, 10))
        assert len(m._theta_m) == 3          # skills are still recorded
        assert np.all(m._theta_m == 0.0)
        assert np.all(m._theta_v == m.theta_prior_var)

    @pytest.mark.parametrize("fv", [0.0, 1.0])
    def test_matches_item_only_reference(self, toy, fv):
        p = self._model(q_delta=0.03, forecast_var_scale=fv).predict(toy, target_at(toy, 10))
        ref = [item_only_reference(TOY_Y[i], TOY_PARAMS["a"][i], TOY_PARAMS["b"][i], 0.03, 1.0, fv)
               for i in range(len(TOY_IDS))]
        assert np.all(TOY_PARAMS["a"] * TOY_PARAMS["b"] != 0)  # the item offset -a b matters
        assert p == pytest.approx(ref, abs=1e-9)

    def test_invariant_to_skill_labels(self, toy):
        target = target_at(toy, 10)
        p = self._model().predict(toy, target)
        one = toy.assign(skill="all")
        p_one = self._model().predict(one, target.assign(skill="all"))
        rng = np.random.default_rng(1)
        labels = dict(zip(TOY_IDS, rng.choice(["x", "y", "z", "w"], len(TOY_IDS))))
        relabelled = toy.assign(skill=toy["item_id"].map(labels))
        p_rel = self._model().predict(relabelled, target.assign(skill=target["item_id"].map(labels)))
        assert np.array_equal(p, p_one) and np.array_equal(p, p_rel)

    def test_invariant_to_other_items_outcomes(self, toy):
        target = target_at(toy, 10)
        p = self._model().predict(toy, target)
        other = toy.copy()
        flip = other["item_id"] != TOY_IDS[5]
        other.loc[flip, "correct"] = 1 - other.loc[flip, "correct"]
        p_other = self._model().predict(other, target)
        assert p_other[5] == p[5]
        # the same flip moves item 5 when the skill state is on (sanity check of the test)
        on = DriftIRT(TOY_PARAMS, 0.01, 0.01, 1.0, 0.0, residual_scale="probit")
        on2 = DriftIRT(TOY_PARAMS, 0.01, 0.01, 1.0, 0.0, residual_scale="probit")
        assert on.predict(toy, target)[5] != on2.predict(other, target)[5]

    def test_q_theta_must_be_zero_and_name(self):
        with pytest.raises(ValueError, match="q_theta"):
            DriftIRT(TOY_PARAMS, q_theta=0.01, skill_state=False)
        assert self._model().name == "drift_irt_qt0_qd0.01_pv1_fv0_zres_noskill"

    def test_flat_item_table_is_a1_b0_on_probit_scale(self):
        flat = flat_item_params(["x", "y", "z"])
        assert list(flat["item_id"]) == ["x", "y", "z"]
        m = DriftIRT(flat, q_theta=0.0, skill_state=False)
        assert np.all(m.params["a"] == 1.0) and np.all(m.params["b"] == 0.0)


class TestDefaultPathRegression:
    """The default skill_state=True path reproduces predictions recorded from the code before the
    ablation options were added (archived setting drift_irt_qt0.01_qd0.01_pv1_fv0_zres)."""

    def test_archived_setting(self):
        hist = rows_from_matrix(TOY_Y, TOY_IDS, TOY_SKILLS)
        m = DriftIRT(TOY_PARAMS, 0.01, 0.01, 1.0, 0.0, residual_scale="probit")
        assert m.name == "drift_irt_qt0.01_qd0.01_pv1_fv0_zres"
        expected = [0.5688533741217846, 0.5981018940957763, 0.44732178382485954, 0.626861815709494,
                    0.7517832169401983, 0.573746660181722, 0.6959125554313003, 0.4477130453683395,
                    0.7337843248483624, 0.5942068355302763, 0.8333686586045127, 0.7657353919385629]
        assert m.predict(hist, target_at(hist, 10)) == pytest.approx(expected, rel=0, abs=1e-14)

    def test_difficulty_scale_with_forecast_variance(self):
        hist = rows_from_matrix(TOY_Y, TOY_IDS, TOY_SKILLS)
        m = DriftIRT(TOY_PARAMS, 0.01, 0.01, 1.0, 1.0, residual_scale="difficulty")
        expected = [0.5616973905392539, 0.5732990271235537, 0.46158887967509604, 0.6049826243738433,
                    0.6481554926894979, 0.5301846427182518, 0.708584406035187, 0.4630865295783158,
                    0.674735729476615, 0.6027636286139273, 0.7623036168948383, 0.7246492447260762]
        assert m.predict(hist, target_at(hist, 12), horizon=3) == pytest.approx(expected, rel=0, abs=1e-14)
