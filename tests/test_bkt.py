import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from llm_kt.baselines import ItemMean
from llm_kt.bkt import (
    ANCHORS,
    BKT,
    COLD_START,
    EPS,
    MIN_GAP,
    SkillParams,
    batch_log_emissions,
    clip_guess,
    fit_skill,
    forecast_state,
    log_forward_backward,
)
from llm_kt.evaluate import TARGET_COLUMNS, evaluate
from llm_kt.static_irt import prob_logistic


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


def _state_forecast(m: BKT, skill: str, h: int = 1) -> float:
    sp = m.skill_params().set_index("skill").loc[skill]
    p_last = m.mastery().query("skill == @skill")["p_learned"].iloc[-1]
    return forecast_state(p_last, sp["learn"], sp["forget"], h)


@pytest.fixture
def world():
    """Two skills, 200 items, 15 checkpoints; skill s0 learned early, s1 late."""
    rng = np.random.default_rng(0)
    N, T = 200, 15
    ids = [f"mmlu_{i}" for i in range(N)]
    skills = ["s0" if i < 100 else "s1" for i in range(N)]
    guess, slip = rng.uniform(0.1, 0.4, N), rng.uniform(0.1, 0.4, N)
    z0 = (np.arange(T) >= 3).astype(int)
    z1 = (np.arange(T) >= 10).astype(int)
    z = np.where(np.array(skills)[:, None] == "s0", z0[None, :], z1[None, :])
    Y = (rng.random((N, T)) < np.where(z == 1, 1 - slip[:, None], guess[:, None])).astype(float)
    params = pd.DataFrame({"item_id": ids, "a": rng.uniform(0.5, 2.0, N), "b": rng.normal(0, 1, N)})
    return params, rows_from_matrix(Y, ids, skills), ids, skills


class TestPredictor:
    def test_shape_alignment_and_range(self, world):
        _, hist, _, _ = world
        target = target_at(hist, 15)
        p = BKT(kappa=5.0).predict(hist, target, horizon=1)
        assert p.shape == (len(target),) and np.all(np.isfinite(p)) and ((p > 0) & (p < 1)).all()
        perm = np.random.default_rng(3).permutation(len(target))
        shuffled = target.iloc[perm].reset_index(drop=True)
        assert BKT(kappa=5.0).predict(hist, shuffled) == pytest.approx(p[perm])

    def test_per_skill_limit_is_constant_within_skill(self, world):
        _, hist, _, skills = world
        p = BKT(kappa=1e6).predict(hist, target_at(hist, 15))
        for s in ("s0", "s1"):
            assert np.ptp(p[np.array(skills) == s]) < 1e-6

    def test_per_item_ranks_right_above_wrong(self):
        T = 10
        ids = ["mmlu_right", "mmlu_wrong"] + [f"mmlu_x{i}" for i in range(40)]
        rng = np.random.default_rng(4)
        Y = np.vstack([np.ones(T), np.zeros(T), (rng.random((40, T)) < 0.5).astype(float)])
        hist = rows_from_matrix(Y, ids, ["s"] * len(ids))
        p = pd.Series(BKT(kappa=1.0).predict(hist, target_at(hist, T)), index=ids)
        assert p["mmlu_right"] > 0.6 and p["mmlu_wrong"] < 0.4

    def test_rho_tracks_recent_flip(self):
        T = 20
        ids = ["mmlu_flip"] + [f"mmlu_x{i}" for i in range(60)]
        rng = np.random.default_rng(5)
        flip = np.r_[np.zeros(T - 3), np.ones(3)]
        Y = np.vstack([flip, (rng.random((60, T)) < 0.5).astype(float)])
        hist = rows_from_matrix(Y, ids, ["s"] * len(ids))
        target = target_at(hist, T)
        p = {
            r: pd.Series(BKT(kappa=1.0, rho=r).predict(hist, target), index=ids)["mmlu_flip"]
            for r in (1.0, 0.7)
        }
        assert p[0.7] > p[1.0] + 0.2 and p[1.0] < 0.4

    def test_longer_horizon_moves_toward_stationary(self, world):
        _, hist, _, _ = world
        m = BKT(kappa=5.0)
        target = target_at(hist, 15)
        p1 = m.predict(hist, target, horizon=1)
        p20 = m.predict(hist, target, horizon=20)
        sp = m.skill_params().set_index("skill")
        state1 = m.mastery().groupby("skill")["p_learned"].last()
        assert set(sp.index) == {"s0", "s1"} and (state1 > 0.5).all()  # both learned by t=14
        assert np.all(np.isfinite(p20)) and np.mean(np.abs(p20 - p1)) > 0  # horizon matters
        for s in ("s0", "s1"):
            st = sp.loc[s, "learn"] / (sp.loc[s, "learn"] + sp.loc[s, "forget"])
            assert abs(_state_forecast(m, s, 20) - st) <= abs(_state_forecast(m, s, 1) - st)

    def test_fluid_anchor_unseen_item_uses_2pl_anchor(self, world):
        params, hist, ids, _ = world
        m = BKT(kappa=1e6, anchor="fluid", item_params=params, theta_lo=-1.0, theta_hi=1.0)
        target = target_at(hist, 15)
        p = m.predict(hist[hist["item_id"] != ids[0]], target)[0]
        a, b = params["a"].iloc[0], params["b"].iloc[0]
        g, hi = prob_logistic(-1.0, a, b), prob_logistic(1.0, a, b)
        g = min(g, hi - MIN_GAP)
        z = _state_forecast(m, "s0", 1)
        assert p == pytest.approx(z * hi + (1 - z) * g, abs=1e-6)

    def test_fluid_anchor_item_without_params_uses_skill_anchor(self, world):
        params, hist, _, _ = world
        m = BKT(kappa=1e6, anchor="fluid", item_params=params.iloc[1:])
        p = m.predict(hist, target_at(hist, 15))
        sp = m.skill_params().set_index("skill").loc["s0"]
        z = _state_forecast(m, "s0", 1)
        assert p[0] == pytest.approx(z * (1 - sp["slip_s"]) + (1 - z) * sp["guess_s"], abs=1e-6)

    def test_unseen_skill_falls_back_to_global_mean(self, world):
        _, hist, _, _ = world
        target = target_at(hist, 15).copy()
        target.loc[0, "skill"] = "never_seen"
        target.loc[0, "item_id"] = "mmlu_new"
        p = BKT().predict(hist, target)
        last = hist[hist["checkpoint_idx"] == 14]
        assert p[0] == pytest.approx(last["correct"].mean())

    def test_two_checkpoint_history_is_finite(self, world):
        _, hist, _, _ = world
        short = hist[hist["checkpoint_idx"] <= 1]
        p = BKT(kappa=5.0, rho=0.7).predict(short, target_at(short, 2))
        assert np.all(np.isfinite(p)) and ((p > 0) & (p < 1)).all()

    def test_missing_cells_are_ignored(self, world):
        _, hist, ids, _ = world
        hole = (hist["item_id"] == ids[0]) & (hist["checkpoint_idx"].isin([5, 6, 7]))
        holed = hist[~hole]
        p = BKT(kappa=5.0).predict(holed, target_at(hist, 15))
        assert np.all(np.isfinite(p))
        full = BKT(kappa=5.0).predict(hist, target_at(hist, 15))
        assert abs(p[0] - full[0]) < 0.2 and p[1:] == pytest.approx(full[1:], abs=0.05)

    def test_missing_cells_are_not_counted_as_wrong_in_the_m_step(self):
        # item 0 is right at every observed checkpoint; blanking three learned-phase cells
        # must leave its fitted slip where it was, while zero-filling them must raise it
        rng = np.random.default_rng(1)
        N, T = 40, 16
        z = (np.arange(T) >= 4).astype(float)
        Y = (rng.random((N, T)) < np.where(z[None, :] == 1, 0.85, 0.25)).astype(float)
        Y[0] = 1.0
        holed, zeroed = Y.copy(), Y.copy()
        holed[0, 8:11] = np.nan
        zeroed[0, 8:11] = 0.0
        slip_full = fit_skill(Y, kappa=1.0, rho=1.0).slip[0]
        slip_holed = fit_skill(holed, kappa=1.0, rho=1.0).slip[0]
        slip_zeroed = fit_skill(zeroed, kappa=1.0, rho=1.0).slip[0]
        assert slip_holed == pytest.approx(slip_full, abs=0.02)
        assert slip_zeroed > slip_holed + 0.05

    def test_cache_warm_start_close_to_cold_and_resets(self, world):
        _, hist, _, _ = world
        target = target_at(hist, 15)
        m = BKT(kappa=5.0)
        for t in (4, 9):
            m.predict(hist[hist["checkpoint_idx"] <= t], target)
        assert m.predict(hist, target) == pytest.approx(BKT(kappa=5.0).predict(hist, target), abs=0.02)
        other = hist.assign(correct=1 - hist["correct"])
        assert m.predict(other, target) == pytest.approx(BKT(kappa=5.0).predict(other, target))
        short = hist[hist["checkpoint_idx"] <= 4]
        assert m.predict(short, target) == pytest.approx(BKT(kappa=5.0).predict(short, target))

    def test_mastery_and_skill_params_shapes(self, world):
        _, hist, _, _ = world
        m = BKT(kappa=5.0)
        m.fit(hist)
        ms = m.mastery()
        assert list(ms.columns) == ["skill", "checkpoint_idx", "p_learned"] and len(ms) == 2 * 15
        assert ms["p_learned"].between(0, 1).all()
        sp = m.skill_params()
        assert list(sp.columns) == [
            "skill", "n_items", "pi", "learn", "forget", "guess_s", "slip_s", "n_iter"
        ]
        assert sp["n_items"].tolist() == [100, 100]

    def test_runs_in_harness(self, world):
        _, hist, _, _ = world
        res = evaluate(hist, [BKT(kappa=5.0), ItemMean()], horizon=2)
        assert set(res["predictor"]) == {"bkt_k5_r1_skill", "item_mean"}
        assert res["log_loss"].notna().all()

    def test_name_and_validation(self):
        params = pd.DataFrame({"item_id": ["x"], "a": [1.0], "b": [0.0]})
        assert BKT(kappa=20, rho=0.9).name == "bkt_k20_r0.9_skill"
        fluid = BKT(kappa=1e6, anchor="fluid", item_params=params, theta_lo=-2, theta_hi=1)
        assert fluid.name == "bkt_k1e+06_r1_fluid_lo-2_hi1"
        bad = (
            {"anchor": "logit"}, {"kappa": 0}, {"rho": 0}, {"rho": 1.5}, {"anchor": "fluid"},
            {"anchor": "fluid", "item_params": params, "theta_lo": 1, "theta_hi": 1},
        )
        for kw in bad:
            with pytest.raises(ValueError):
                BKT(**kw)
        assert ANCHORS == ("skill", "fluid")
