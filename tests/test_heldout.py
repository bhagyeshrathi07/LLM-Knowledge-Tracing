import numpy as np
import pandas as pd
import pytest

from llm_kt.heldout import (
    evaluate_benchmark_transfer,
    evaluate_heldout_items,
    select_nu,
    split_items,
)


def make_responses(n_checkpoints=8, n_items=12):
    rows = []
    for checkpoint_idx in range(n_checkpoints):
        for item_idx in range(n_items):
            rows.append({
                "model": "m",
                "checkpoint_idx": checkpoint_idx,
                "checkpoint_label": f"step{checkpoint_idx}",
                "step": checkpoint_idx,
                "tokens_seen_b": pd.NA,
                "benchmark": "mmlu",
                "skill": "subject",
                "item_id": f"mmlu_subject_{item_idx}",
                "correct": int((item_idx + checkpoint_idx) % 3 != 0),
            })
    return pd.DataFrame(rows)


def make_transfer_responses():
    rows = []
    for checkpoint_idx in range(8):
        for item_idx in range(12):
            rows.append({
                "model": "m",
                "checkpoint_idx": checkpoint_idx,
                "checkpoint_label": f"step{checkpoint_idx}",
                "step": checkpoint_idx,
                "tokens_seen_b": pd.NA,
                "benchmark": "mmlu",
                "skill": f"subject_{item_idx // 6}",
                "item_id": f"mmlu_subject_{item_idx // 6}_{item_idx % 6}",
                "correct": int((item_idx + checkpoint_idx) % 3 != 0),
            })
        for item_idx in range(4):
            rows.append({
                "model": "m",
                "checkpoint_idx": checkpoint_idx,
                "checkpoint_label": f"step{checkpoint_idx}",
                "step": checkpoint_idx,
                "tokens_seen_b": pd.NA,
                "benchmark": "arc_challenge",
                "skill": "arc_challenge",
                "item_id": f"arc_challenge_{item_idx}",
                "correct": int(item_idx < (3 if checkpoint_idx < 4 else 1)),
            })
    return pd.DataFrame(rows)


class TestItemSplit:
    def test_is_repeatable_stratified_and_disjoint(self):
        responses = make_responses()
        fit, heldout = split_items(responses, seed=12)
        assert fit == split_items(responses, seed=12)[0]
        assert fit.isdisjoint(heldout)
        assert fit | heldout == set(responses["item_id"].unique())

    def test_rejects_skills_without_two_items(self):
        responses = make_responses(n_items=1)
        with pytest.raises(ValueError, match="at least two items"):
            split_items(responses)


class TestHeldoutEvaluation:
    def test_test_scores_use_only_heldout_items(self):
        responses = make_responses()
        items = responses[["item_id"]].drop_duplicates()
        params = pd.DataFrame({"item_id": items.item_id, "a": 1.0, "b": 0.0})
        scores, fit, heldout = evaluate_heldout_items(
            responses, params, model="m", nu_values=[0.0, 0.1], seed=2
        )
        test = scores[(scores["block"] == "test") & (scores["predictor"] == "skill_mean")]
        assert len(test) > 0
        assert test["n"].sum() == len(heldout) * 2
        assert fit.isdisjoint(heldout)
        assert select_nu(scores[scores["block"] == "val"]) in {0.0, 0.1}

    def test_test_benchmark_outcomes_do_not_fit_transfer_calibration(self):
        responses = make_transfer_responses()
        fit, _ = split_items(responses[responses["benchmark"] == "mmlu"], seed=3)
        params = pd.DataFrame({
            "item_id": sorted(fit),
            "a": 1.0,
            "b": 0.0,
        })
        first = evaluate_benchmark_transfer(
            responses, params, "m", fit, 0.1, ["arc_challenge"]
        )
        changed = responses.copy()
        test = (changed["benchmark"] == "arc_challenge") & changed["checkpoint_idx"].isin([6, 7])
        changed.loc[test, "correct"] = 1
        second = evaluate_benchmark_transfer(
            changed, params, "m", fit, 0.1, ["arc_challenge"]
        )
        first_test = first[first["block"] == "test"].sort_values(["checkpoint_idx", "predictor"])
        second_test = second[second["block"] == "test"].sort_values(["checkpoint_idx", "predictor"])
        transfer = first_test["predictor"] != "train_prevalence"
        assert first_test.loc[transfer, "calibration_intercept"].to_numpy() == pytest.approx(
            second_test.loc[transfer, "calibration_intercept"].to_numpy()
        )
        assert first_test.loc[transfer, "calibration_slope"].to_numpy() == pytest.approx(
            second_test.loc[transfer, "calibration_slope"].to_numpy()
        )
        assert not np.allclose(first_test["log_loss"], second_test["log_loss"])
