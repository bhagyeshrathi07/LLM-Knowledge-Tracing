import numpy as np
import pandas as pd
import pytest

from llm_kt.static_irt import (
    StaticIRTPersistence,
    diagnose,
    fit_theta_logistic,
    prob_logistic,
)

rng = np.random.default_rng(0)


def simulate(theta: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = prob_logistic(theta, a, b)
    return (rng.random(len(a)) < p).astype(int)


class TestFitTheta:
    def test_recovers_ability_on_many_items(self):
        a = rng.uniform(0.8, 2.0, 2000)
        b = rng.normal(0, 1, 2000)
        for true_theta in (-1.0, 0.0, 1.5):
            y = simulate(true_theta, a, b)
            est = fit_theta_logistic(a, b, y, prior_var=100.0)
            assert est == pytest.approx(true_theta, abs=0.15)

    def test_prior_shrinks_toward_zero_with_no_data(self):
        assert fit_theta_logistic(np.array([]), np.array([]), np.array([])) == 0.0

    def test_all_correct_is_finite(self):
        a = np.ones(20)
        b = np.zeros(20)
        est = fit_theta_logistic(a, b, np.ones(20, dtype=int), prior_var=1.0)
        assert np.isfinite(est) and est > 0


@pytest.fixture
def item_params():
    items = [f"mmlu_anatomy_{i}" for i in range(6)] + [f"arc_challenge_{i}" for i in range(4)]
    return pd.DataFrame({"item_id": items, "a": 1.0, "b": np.linspace(-1, 1, 10)})


@pytest.fixture
def responses(item_params):
    rows = []
    for t in range(3):
        theta = -1.0 + t  # ability rises each checkpoint
        for _, r in item_params.iterrows():
            p = prob_logistic(theta, r.a, r.b)
            rows.append(
                {
                    "model": "m",
                    "checkpoint_idx": t,
                    "checkpoint_label": f"step{t}",
                    "step": t,
                    "tokens_seen_b": pd.NA,
                    "benchmark": "mmlu" if r.item_id.startswith("mmlu") else "arc_challenge",
                    "skill": "anatomy" if r.item_id.startswith("mmlu") else "arc_challenge",
                    "item_id": r.item_id,
                    "correct": int(p > 0.5),
                }
            )
    return pd.DataFrame(rows)


class TestPersistence:
    def test_returns_probabilities_for_every_target_row(self, item_params, responses):
        pred = StaticIRTPersistence(item_params)
        history = responses[responses.checkpoint_idx <= 1]
        target = responses[responses.checkpoint_idx == 2].drop(columns="correct").reset_index(drop=True)
        p = pred.predict(history, target, horizon=1)
        assert p.shape == (len(target),)
        assert ((p >= 0) & (p <= 1)).all()

    def test_easier_items_get_higher_probability(self, item_params, responses):
        pred = StaticIRTPersistence(item_params)
        history = responses[responses.checkpoint_idx <= 1]
        target = responses[responses.checkpoint_idx == 2].drop(columns="correct").reset_index(drop=True)
        p = pd.Series(pred.predict(history, target, horizon=1), index=target.item_id)
        assert p["mmlu_anatomy_0"] > p["mmlu_anatomy_5"]  # b = -1 vs b ~ +0.1

    def test_item_without_params_falls_back_to_skill_mean(self, item_params, responses):
        pred = StaticIRTPersistence(item_params.iloc[1:])  # drop params for mmlu_anatomy_0
        history = responses[responses.checkpoint_idx <= 1]
        target = responses[responses.checkpoint_idx == 2].drop(columns="correct").reset_index(drop=True)
        p = pd.Series(pred.predict(history, target, horizon=1), index=target.item_id)
        last = history[(history.checkpoint_idx == 1) & (history.skill == "anatomy")]["correct"].mean()
        assert p["mmlu_anatomy_0"] == pytest.approx(last)


class TestDiagnose:
    def test_one_row_per_skill_and_scores_in_range(self, item_params, responses):
        out = diagnose(responses, item_params, model="m", checkpoint_idx=2, holdout_frac=0.5, seed=0)
        assert set(out["skill"]) == {"anatomy", "arc_challenge"}
        assert (out["n_fit"] + out["n_holdout"] == out["n_items"]).all()
        assert (out["log_loss"] >= 0).all()

    def test_fit_items_never_scored(self, item_params, responses):
        out = diagnose(responses, item_params, model="m", checkpoint_idx=2, holdout_frac=0.5, seed=0)
        assert (out["n_holdout"] > 0).all()
        assert (out["n_fit"] > 0).all()


class TestRecalibrateFlag:
    def test_recalibrated_predictor_runs_and_is_named(self, item_params, responses):
        pred = StaticIRTPersistence(item_params, recalibrate=True, tau=1.0)
        assert pred.name.endswith("_recal_tau1")
        history = responses[responses.checkpoint_idx <= 1]
        target = responses[responses.checkpoint_idx == 2].drop(columns="correct").reset_index(drop=True)
        p = pred.predict(history, target, horizon=1)
        assert p.shape == (len(target),) and ((p >= 0) & (p <= 1)).all()

    def test_tau_a_predictor_runs_and_is_named(self, item_params, responses):
        pred = StaticIRTPersistence(item_params, recalibrate=True, tau=1.0, tau_a=0.5)
        assert pred.name.endswith("_recal_tau1_taua0.5")
        history = responses[responses.checkpoint_idx <= 1]
        target = responses[responses.checkpoint_idx == 2].drop(columns="correct").reset_index(drop=True)
        p = pred.predict(history, target, horizon=1)
        assert p.shape == (len(target),) and ((p >= 0) & (p <= 1)).all()
