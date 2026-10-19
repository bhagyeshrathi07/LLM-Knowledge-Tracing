import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from llm_kt.calibrate import WalkForwardCalibrated, fit_platt, fit_temperature
from llm_kt.evaluate import TARGET_COLUMNS, evaluate_model
from llm_kt.metrics import log_loss


def overconfident_sample(n: int, seed: int, scale: float = 2.0):
    rng = np.random.default_rng(seed)
    p_true = expit(rng.normal(0.0, 1.0, n))
    y = (rng.random(n) < p_true).astype(float)
    p = expit(logit(p_true) * scale)
    return p, y


def make_responses(n_ckpt: int = 8, n_items: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_ckpt):
        for i in range(n_items):
            rows.append({
                "model": "m", "item_id": f"i{i}", "benchmark": "b", "skill": f"s{i % 3}",
                "checkpoint_idx": c, "step": c * 10, "tokens_seen_b": float(c),
                "correct": int(rng.random() < 0.3 + 0.05 * c),
            })
    return pd.DataFrame(rows)


class Constant:
    name = "const"

    def __init__(self, p: float = 0.9):
        self.p = p

    def predict(self, history, target, horizon=1):
        return np.full(len(target), self.p)


class Spy:
    """Records every (history, target) it is asked about; predicts an overconfident 0.95."""

    name = "spy"

    def __init__(self):
        self.calls = []

    def predict(self, history, target, horizon=1):
        self.calls.append((history.copy(), target.copy(), horizon))
        return np.full(len(target), 0.95)


class TestFitTemperature:
    def test_recovers_overconfidence_factor(self):
        p, y = overconfident_sample(20000, seed=0, scale=2.0)
        assert fit_temperature(p, y) == pytest.approx(2.0, abs=0.3)

    def test_calibrated_beats_uncalibrated_on_fresh_data(self):
        p, y = overconfident_sample(5000, seed=1)
        T = fit_temperature(p, y)
        p2, y2 = overconfident_sample(5000, seed=2)
        p2_cal = expit(logit(p2) / T)
        assert log_loss(y2, p2_cal) < log_loss(y2, p2)

    def test_handles_extreme_probabilities(self):
        T = fit_temperature(np.array([0.0, 1.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0, 0.0]))
        assert 0.25 <= T <= 4.0 and np.isfinite(T)


class TestFitPlatt:
    def test_recovers_slope_and_intercept(self):
        p, y = overconfident_sample(20000, seed=3, scale=2.0)
        A, B = fit_platt(p, y)
        assert A == pytest.approx(0.5, abs=0.1)
        assert B == pytest.approx(0.0, abs=0.1)

    def test_calibrated_beats_uncalibrated_on_fresh_data(self):
        p, y = overconfident_sample(5000, seed=4)
        A, B = fit_platt(p, y)
        p2, y2 = overconfident_sample(5000, seed=5)
        p2_cal = expit(A * logit(np.clip(p2, 1e-6, 1 - 1e-6)) + B)
        assert log_loss(y2, p2_cal) < log_loss(y2, p2)


class TestWrapper:
    def test_names(self):
        assert WalkForwardCalibrated(Constant(), k=5).name == "const_cal_temperaturek5"
        assert WalkForwardCalibrated(Constant(), k=3, method="platt").name == "const_cal_plattk3"

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError):
            WalkForwardCalibrated(Constant(), method="isotonic")

    def test_passes_through_without_two_origins(self):
        r = make_responses(n_ckpt=3)
        wrapped = WalkForwardCalibrated(Constant(0.9), k=5)
        # history checkpoints {0, 1}, horizon 1: only origin 0 -> target 1 is reconstructable
        history = r[r["checkpoint_idx"] <= 1]
        target = r[r["checkpoint_idx"] == 2][TARGET_COLUMNS]
        assert wrapped.predict(history, target, horizon=1) == pytest.approx(np.full(len(target), 0.9))
        # horizon 2 with the same history: no origin at all
        assert wrapped.predict(history, target, horizon=2) == pytest.approx(np.full(len(target), 0.9))

    def test_calibrates_overconfident_constant(self):
        r = make_responses(n_ckpt=8)
        history = r[r["checkpoint_idx"] <= 6]
        target = r[r["checkpoint_idx"] == 7][TARGET_COLUMNS]
        p = WalkForwardCalibrated(Constant(0.99), k=5).predict(history, target)
        # base rate in the last checkpoints is ~0.6, so the 0.99 forecast must be pulled down
        assert np.all(p < 0.95)
        assert len(WalkForwardCalibrated(Constant(0.99)).fits) == 0  # fresh object, no fits

    def test_records_fits(self):
        r = make_responses(n_ckpt=8)
        w = WalkForwardCalibrated(Constant(0.99), k=3)
        w.predict(r[r["checkpoint_idx"] <= 6], r[r["checkpoint_idx"] == 7][TARGET_COLUMNS])
        assert len(w.fits) == 1
        assert w.fits[0]["n_origins"] == 3 and w.fits[0]["history_max"] == 6

    @pytest.mark.parametrize("horizon", [1, 2])
    def test_reconstruction_never_looks_past_history(self, horizon):
        r = make_responses(n_ckpt=8).sort_values(["checkpoint_idx", "item_id"]).reset_index(drop=True)
        spy = Spy()
        wrapped = WalkForwardCalibrated(spy, k=3)
        hmax = 5
        history = r[r["checkpoint_idx"] <= hmax]
        target = r[r["checkpoint_idx"] == hmax + horizon][TARGET_COLUMNS].reset_index(drop=True)
        wrapped.predict(history, target, horizon=horizon)

        fitting_calls, real_calls = spy.calls[:-1], spy.calls[-1:]
        assert len(fitting_calls) == 3
        for sub_hist, sub_target, h in fitting_calls:
            assert h == horizon
            sub_max = sub_hist["checkpoint_idx"].max()
            # strict prefix of the outer history: every row at checkpoints <= sub_max, nothing else
            assert sub_max < hmax
            expected = history[history["checkpoint_idx"] <= sub_max]
            pd.testing.assert_frame_equal(sub_hist.reset_index(drop=True), expected.reset_index(drop=True))
            assert len(sub_hist) < len(history)
            # target used for fitting is inside the outer history and carries no label
            assert "correct" not in sub_target.columns
            assert sub_target["checkpoint_idx"].nunique() == 1
            tgt = int(sub_target["checkpoint_idx"].iloc[0])
            assert tgt <= hmax and tgt == sub_max + horizon
        # the last call is the real forecast on the full outer history
        pd.testing.assert_frame_equal(real_calls[0][0], history)
        pd.testing.assert_frame_equal(real_calls[0][1], target)

    def test_runs_in_harness(self):
        r = make_responses(n_ckpt=8)
        out = evaluate_model(r, WalkForwardCalibrated(Constant(0.9), k=2), by_skill=False)
        assert len(out) == 6 and out["predictor"].iloc[0] == "const_cal_temperaturek2"
