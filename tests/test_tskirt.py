import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from llm_kt.tskirt import (
    PROBIT_SCALE,
    TSKIRT,
    effective_discrimination,
    fit_theta_probit,
    to_probit,
)


class TestConversion:
    def test_logistic_to_probit_divides_discrimination_only(self):
        a, b = to_probit(np.array([1.7, 3.4]), np.array([0.5, -0.5]))
        assert a == pytest.approx([1.0, 2.0])
        assert b == pytest.approx([0.5, -0.5])
        assert PROBIT_SCALE == pytest.approx(1.7)


class TestEffectiveDiscrimination:
    def test_no_elapsed_time_leaves_a_unchanged(self):
        a = np.array([0.5, 1.0, 2.0])
        assert effective_discrimination(a, nu=0.3, elapsed=np.zeros(3)) == pytest.approx(a)

    def test_older_responses_are_down_weighted(self):
        a = np.array([1.0, 1.0, 1.0])
        eff = effective_discrimination(a, nu=0.3, elapsed=np.array([0.0, 10.0, 100.0]))
        assert eff[0] > eff[1] > eff[2] > 0

    def test_zero_drift_means_no_down_weighting(self):
        a = np.array([1.0, 1.0])
        assert effective_discrimination(a, nu=0.0, elapsed=np.array([0.0, 1e6])) == pytest.approx(a)


class TestFitThetaProbit:
    def test_recovers_ability(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(0.5, 1.5, 3000)
        b = rng.normal(0, 1, 3000)
        for true_theta in (-1.0, 0.8):
            y = (rng.random(3000) < norm.cdf(a * (true_theta - b))).astype(int)
            est = fit_theta_probit(a, b, y, prior_var=100.0)
            assert est == pytest.approx(true_theta, abs=0.12)

    def test_no_data_returns_prior_mean(self):
        assert fit_theta_probit(np.array([]), np.array([]), np.array([])) == 0.0


def make_history(theta_by_ckpt: dict[int, float], items: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t, theta in theta_by_ckpt.items():
        a_p, b_p = to_probit(items["a"].to_numpy(), items["b"].to_numpy())
        p = norm.cdf(a_p * (theta - b_p))
        y = (rng.random(len(items)) < p).astype(int)
        for item, c in zip(items["item_id"], y):
            rows.append({"model": "m", "checkpoint_idx": t, "checkpoint_label": f"step{t}", "step": t,
                         "tokens_seen_b": pd.NA, "benchmark": "mmlu", "skill": "s", "item_id": item, "correct": int(c)})
    return pd.DataFrame(rows)


@pytest.fixture
def items():
    rng = np.random.default_rng(2)
    return pd.DataFrame({"item_id": [f"mmlu_s_{i}" for i in range(400)], "a": rng.uniform(1.0, 2.5, 400), "b": rng.normal(0, 1, 400)})


class TestTSKIRTPredictor:
    def test_predict_shape_and_range(self, items):
        hist = make_history({0: -1.0, 1: -0.5, 2: 0.0}, items)
        target = hist[hist.checkpoint_idx == 2].drop(columns="correct").assign(checkpoint_idx=3).reset_index(drop=True)
        m = TSKIRT(items, nu=0.3)
        p = m.predict(hist, target, horizon=1)
        assert p.shape == (len(target),)
        assert ((p >= 0) & (p <= 1)).all()

    def test_drift_tracks_recent_ability_better_than_static(self, items):
        # ability jumps from -1.5 (many old checkpoints) to +1.5 (one recent checkpoint)
        hist = make_history({t: -1.5 for t in range(10)} | {10: 1.5}, items)
        static = TSKIRT(items, nu=0.0)   # nu = 0: every response weighted equally -> pulled toward -1.5
        drift = TSKIRT(items, nu=1.0)    # moderate drift: old responses discounted (a / sqrt(1 + a^2 nu^2 d))
        strong = TSKIRT(items, nu=20.0)  # strong drift: old responses nearly ignored -> near +1.5
        theta_static = static.fit_skill_theta(hist, origin_time=10.0)["s"]
        theta_drift = drift.fit_skill_theta(hist, origin_time=10.0)["s"]
        theta_strong = strong.fit_skill_theta(hist, origin_time=10.0)["s"]
        # the paper's discount is a / sqrt(1 + a^2 nu^2 d): the score of an old response shrinks
        # only like 1/sqrt(d), so a moderate nu moves theta a little and a large nu a lot
        assert theta_static < -1.0
        assert theta_drift > theta_static
        assert theta_strong > 0.5

    def test_forecast_uncertainty_pulls_probabilities_toward_half(self, items):
        hist = make_history({0: 1.5, 1: 1.5, 2: 1.5}, items)
        base = hist[hist.checkpoint_idx == 2].drop(columns="correct").reset_index(drop=True)
        m = TSKIRT(items, nu=0.5)
        p_near = m.predict(hist, base.assign(checkpoint_idx=3), horizon=1)
        p_far = m.predict(hist, base.assign(checkpoint_idx=23), horizon=21)
        # farther ahead => more Wiener variance => predictions less extreme
        assert np.mean(np.abs(p_far - 0.5)) < np.mean(np.abs(p_near - 0.5))

    def test_items_without_params_fall_back_to_skill_mean(self, items):
        hist = make_history({0: 0.0, 1: 0.0}, items)
        target = hist[hist.checkpoint_idx == 1].drop(columns="correct").assign(checkpoint_idx=2).reset_index(drop=True)
        m = TSKIRT(items.iloc[1:], nu=0.3)
        p = pd.Series(m.predict(hist, target, horizon=1), index=target.item_id)
        skill_mean = hist[hist.checkpoint_idx == 1]["correct"].mean()
        assert p[items.item_id.iloc[0]] == pytest.approx(skill_mean)


class TestRecalibrateFlag:
    def test_recalibrated_tskirt_runs_and_updates_params(self, items):
        hist = make_history({0: 0.0, 1: 0.2, 2: 0.4}, items)
        target = hist[hist.checkpoint_idx == 2].drop(columns="correct").assign(checkpoint_idx=3).reset_index(drop=True)
        m = TSKIRT(items, nu=0.5, recalibrate=True, tau=1.0)
        before = m.params["b"].to_numpy().copy()
        p = m.predict(hist, target, horizon=1)
        assert m.name.endswith("_recal_tau1")
        assert p.shape == (len(target),) and ((p >= 0) & (p <= 1)).all()
        assert not np.allclose(m.params["b"].to_numpy(), before)

    def test_tau_a_changes_discrimination_and_name(self, items):
        hist = make_history({0: 0.0, 1: 0.2, 2: 0.4}, items)
        target = hist[hist.checkpoint_idx == 2].drop(columns="correct").assign(checkpoint_idx=3).reset_index(drop=True)
        m = TSKIRT(items, nu=0.5, recalibrate=True, tau=1.0, tau_a=0.5)
        before = m.params["a"].to_numpy().copy()
        p = m.predict(hist, target, horizon=1)
        assert m.name.endswith("_recal_tau1_taua0.5")
        assert p.shape == (len(target),) and ((p >= 0) & (p <= 1)).all()
        assert not np.allclose(m.params["a"].to_numpy(), before)
