import numpy as np
import pandas as pd
import pytest

from llm_kt.baselines import (
    CopyLast,
    CopyLastCalibrated,
    GlobalMean,
    ItemMean,
    SkillMean,
)
from llm_kt.evaluate import evaluate, evaluate_model, rolling_origins, summarize


def make_responses(model: str, matrix: dict[str, list[int]], skill_of: dict[str, str]) -> pd.DataFrame:
    """matrix: item_id -> correctness per checkpoint (all same length)."""
    rows = []
    for item, seq in matrix.items():
        for t, c in enumerate(seq):
            rows.append(
                {
                    "model": model,
                    "checkpoint_idx": t,
                    "checkpoint_label": f"step{t}",
                    "step": t,
                    "tokens_seen_b": pd.NA,
                    "benchmark": "mmlu",
                    "skill": skill_of[item],
                    "item_id": item,
                    "correct": c,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def responses():
    # skill a: learned at t=2 and stays; skill b: never learned
    matrix = {
        "a0": [0, 0, 1, 1, 1],
        "a1": [0, 1, 1, 1, 1],
        "b0": [0, 0, 0, 0, 0],
        "b1": [1, 0, 0, 0, 1],
    }
    skill_of = {"a0": "a", "a1": "a", "b0": "b", "b1": "b"}
    return make_responses("m", matrix, skill_of)


class TestRollingOrigins:
    def test_default(self):
        assert rolling_origins(n_checkpoints=5, min_history=2, horizon=1) == [(1, 2), (2, 3), (3, 4)]

    def test_horizon_two(self):
        assert rolling_origins(n_checkpoints=5, min_history=2, horizon=2) == [(1, 3), (2, 4)]

    def test_too_short_gives_nothing(self):
        assert rolling_origins(n_checkpoints=2, min_history=2, horizon=1) == []

    def test_horizon_below_one_rejected(self):
        with pytest.raises(ValueError):
            rolling_origins(n_checkpoints=5, min_history=2, horizon=0)
        with pytest.raises(ValueError):
            rolling_origins(n_checkpoints=5, min_history=2, horizon=-1)


class TestEvaluateModel:
    def test_one_row_per_target_and_skill(self, responses):
        out = evaluate_model(responses, GlobalMean(), min_history=2)
        assert len(out) == 3 * 2  # 3 targets x 2 skills
        assert set(out["skill"]) == {"a", "b"}

    def test_overall_rows_when_not_by_skill(self, responses):
        out = evaluate_model(responses, GlobalMean(), min_history=2, by_skill=False)
        assert len(out) == 3
        assert "skill" not in out.columns
        assert (out["n"] == 4).all()

    def test_rejects_multiple_models(self, responses):
        two = pd.concat([responses, responses.assign(model="other")])
        with pytest.raises(ValueError):
            evaluate_model(two, GlobalMean())

    def test_predictor_only_sees_history(self, responses):
        seen = []

        class Spy:
            name = "spy"

            def predict(self, history, target, horizon=1):
                seen.append(int(history["checkpoint_idx"].max()))
                return np.full(len(target), 0.5)

        evaluate_model(responses, Spy(), min_history=2)
        assert seen == [1, 2, 3]

    def test_target_has_no_correct_column_but_has_time_metadata(self, responses):
        class Spy:
            name = "spy"

            def predict(self, history, target, horizon=1):
                assert "correct" not in target.columns
                for col in ("checkpoint_idx", "step", "tokens_seen_b"):
                    assert col in target.columns
                assert (target["checkpoint_idx"] == history["checkpoint_idx"].max() + horizon).all()
                return np.full(len(target), 0.5)

        evaluate_model(responses, Spy(), min_history=2)

    def test_rejects_out_of_range_probabilities(self, responses):
        class Bad:
            name = "bad"

            def predict(self, history, target, horizon=1):
                return np.full(len(target), 1.5)

        with pytest.raises(ValueError):
            evaluate_model(responses, Bad(), min_history=2)


class TestBaselines:
    def test_global_mean_is_last_checkpoint_accuracy(self, responses):
        history = responses[responses.checkpoint_idx <= 2]
        target = responses[responses.checkpoint_idx == 3][["item_id", "benchmark", "skill"]]
        p = GlobalMean().predict(history, target)
        assert np.allclose(p, 0.5)  # at t=2: a0=1,a1=1,b0=0,b1=0

    def test_skill_mean(self, responses):
        history = responses[responses.checkpoint_idx <= 2]
        target = responses[responses.checkpoint_idx == 3][["item_id", "benchmark", "skill"]].reset_index(drop=True)
        p = SkillMean().predict(history, target)
        assert p[target.skill == "a"].tolist() == [1.0, 1.0]
        assert p[target.skill == "b"].tolist() == [0.0, 0.0]

    def test_item_mean_uses_full_history(self, responses):
        history = responses[responses.checkpoint_idx <= 2]
        target = responses[responses.checkpoint_idx == 3][["item_id", "benchmark", "skill"]].reset_index(drop=True)
        p = ItemMean(alpha=0.0).predict(history, target)
        assert p[target.item_id == "a1"][0] == pytest.approx(2 / 3)

    def test_copy_last(self, responses):
        history = responses[responses.checkpoint_idx <= 2]
        target = responses[responses.checkpoint_idx == 3][["item_id", "benchmark", "skill"]].reset_index(drop=True)
        p = CopyLast(eps=0.1).predict(history, target)
        assert p[target.item_id == "a0"][0] == pytest.approx(0.9)
        assert p[target.item_id == "b0"][0] == pytest.approx(0.1)

    def test_copy_last_calibrated_learns_transition_rates(self, responses):
        history = responses[responses.checkpoint_idx <= 3]
        target = responses[responses.checkpoint_idx == 4][["item_id", "benchmark", "skill"]].reset_index(drop=True)
        p = CopyLastCalibrated(alpha=0.0).predict(history, target)
        # skill a, prev=1: transitions (1->1) x4 => P=1
        assert p[target.item_id == "a0"][0] == pytest.approx(1.0)
        # skill b, prev=0: transitions 0->0 x5 => P=0
        assert p[target.item_id == "b0"][0] == pytest.approx(0.0)

    def test_all_baselines_return_probabilities(self, responses):
        history = responses[responses.checkpoint_idx <= 2]
        target = responses[responses.checkpoint_idx == 3][["item_id", "benchmark", "skill"]].reset_index(drop=True)
        for b in [GlobalMean(), SkillMean(), ItemMean(), CopyLast(), CopyLastCalibrated()]:
            p = b.predict(history, target)
            assert p.shape == (4,)
            assert ((p >= 0) & (p <= 1)).all()


class TestEvaluateAndSummarize:
    def test_evaluate_runs_all_predictors_on_all_models(self, responses):
        two = pd.concat([responses, responses.assign(model="other")], ignore_index=True)
        out = evaluate(two, [GlobalMean(), CopyLast()], min_history=2)
        assert set(out["predictor"]) == {"global_mean", "copy_last"}
        assert set(out["model"]) == {"m", "other"}

    def test_summarize_weights_by_n(self):
        results = pd.DataFrame(
            {
                "predictor": ["p", "p"],
                "model": ["m", "m"],
                "skill": ["a", "b"],
                "n": [1, 3],
                "log_loss": [1.0, 0.0],
                "brier": [1.0, 0.0],
                "accuracy": [0.0, 1.0],
                "auc": [np.nan, 0.8],
            }
        )
        s = summarize(results)
        assert len(s) == 1
        assert s.loc[0, "log_loss"] == pytest.approx(0.25)
        assert s.loc[0, "accuracy"] == pytest.approx(0.75)
        assert s.loc[0, "auc"] == pytest.approx(0.8)
        assert s.loc[0, "n"] == 4
