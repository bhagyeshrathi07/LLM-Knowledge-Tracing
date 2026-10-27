import numpy as np
import pandas as pd
import pytest

from llm_kt.ladder import (
    gap_trend,
    ladder_over_training,
    ladder_table,
    pooled_ladder,
    skill_to_benchmark,
    variance_decomposition,
)


def make_results() -> pd.DataFrame:
    """Two targets x two skills x three predictors, with hand-picked scores.

    Skill "algebra" (mmlu) has n=10 per slice, skill "gsm8k" (gsm8k) has n=30.
    """
    ll = {
        "global_mean": {"algebra": 0.70, "gsm8k": 0.60},
        "skill_mean": {"algebra": 0.65, "gsm8k": 0.50},
        "item_mean": {"algebra": 0.40, "gsm8k": 0.30},
    }
    auc = {"global_mean": 0.5, "skill_mean": 0.5, "item_mean": 0.8}
    n = {"algebra": 10, "gsm8k": 30}
    rows = []
    for pred in ll:
        for target in (2, 3):
            for skill in ("algebra", "gsm8k"):
                # the second target is 0.1 worse for every predictor
                bump = 0.1 if target == 3 else 0.0
                rows.append(
                    {
                        "predictor": pred,
                        "model": "m",
                        "origin_idx": target - 1,
                        "target_idx": target,
                        "skill": skill,
                        "n": n[skill],
                        "log_loss": ll[pred][skill] + bump,
                        "brier": 0.2,
                        "accuracy": 0.7,
                        "auc": auc[pred] if pred != "item_mean" or skill == "algebra" else 0.6,
                        "horizon": 1,
                    }
                )
    return pd.DataFrame(rows)


SKILL_BENCH = pd.Series({"algebra": "mmlu", "gsm8k": "gsm8k"})


def test_pooled_ladder_weights_by_n_and_adds_gaps():
    out = pooled_ladder(make_results(), by=["model"])
    assert len(out) == 1
    r = out.iloc[0]
    assert r["n"] == 80
    # n-weighted: (10*0.70 + 30*0.60) / 40 averaged over targets with +0.05 mean bump
    assert r["log_loss_global_mean"] == pytest.approx((10 * 0.70 + 30 * 0.60) / 40 + 0.05)
    assert r["log_loss_skill_mean"] == pytest.approx((10 * 0.65 + 30 * 0.50) / 40 + 0.05)
    assert r["log_loss_item_mean"] == pytest.approx((10 * 0.40 + 30 * 0.30) / 40 + 0.05)
    assert r["skill_gap"] == pytest.approx(r["log_loss_global_mean"] - r["log_loss_skill_mean"])
    assert r["item_gap"] == pytest.approx(r["log_loss_skill_mean"] - r["log_loss_item_mean"])
    assert r["skill_gap"] == pytest.approx(0.625 - 0.5375)
    assert r["item_gap"] == pytest.approx(0.5375 - 0.325)
    # AUC is the unweighted slice mean: item_mean has two 0.8 and two 0.6 slices
    assert r["auc_item_mean"] == pytest.approx(0.7)
    assert r["auc_global_mean"] == pytest.approx(0.5)


def test_pooled_ladder_requires_all_predictors():
    res = make_results()
    with pytest.raises(ValueError, match="item_mean"):
        pooled_ladder(res[res["predictor"] != "item_mean"], by=["model"])


def test_pooled_ladder_rejects_unpaired_counts():
    res = make_results()
    res.loc[res["predictor"] == "item_mean", "n"] += 1
    with pytest.raises(ValueError, match="item counts"):
        pooled_ladder(res, by=["model"])


def test_ladder_table_has_per_benchmark_and_all_rows():
    out = ladder_table(make_results(), SKILL_BENCH).set_index(["model", "benchmark", "horizon"])
    assert set(out.index) == {
        ("m", "gsm8k", 1),
        ("m", "mmlu", 1),
        ("m", "all", 1),
        ("all", "gsm8k", 1),
        ("all", "mmlu", 1),
        ("all", "all", 1),
    }
    mmlu = out.loc[("m", "mmlu", 1)]
    assert mmlu["n"] == 20
    assert mmlu["log_loss_skill_mean"] == pytest.approx(0.70)
    assert mmlu["item_gap"] == pytest.approx(0.25)
    assert mmlu["skill_gap"] == pytest.approx(0.05)
    assert out.loc[("m", "all", 1), "n"] == 80
    # with a single model, the model="all" rows equal the per-model rows
    pd.testing.assert_series_equal(
        out.loc[("all", "all", 1)], out.loc[("m", "all", 1)], check_names=False
    )


def test_ladder_table_rejects_unmapped_skill():
    with pytest.raises(ValueError, match="gsm8k"):
        ladder_table(make_results(), SKILL_BENCH.drop("gsm8k"))


def test_ladder_over_training_and_trend():
    res = make_results()
    out = ladder_over_training(res[res["skill"] == "algebra"])
    assert list(out["target_idx"]) == [2, 3]
    assert list(out["target_norm"]) == pytest.approx([2 / 3, 1.0])
    assert list(out["log_loss_item_mean"]) == pytest.approx([0.40, 0.50])
    assert list(out["item_gap"]) == pytest.approx([0.25, 0.25])


def test_gap_trend_slope_and_thirds():
    ot = pd.DataFrame(
        {
            "model": "m",
            "target_idx": [1, 2, 3, 4, 5, 6],
            "target_norm": [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0],
            "item_gap": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    t = gap_trend(ot).iloc[0]
    assert t["n_targets"] == 6
    assert t["item_gap_slope"] == pytest.approx(0.6)
    assert t["item_gap_spearman"] == pytest.approx(1.0)
    assert t["item_gap_first_third"] == pytest.approx(0.15)
    assert t["item_gap_last_third"] == pytest.approx(0.55)


def test_skill_to_benchmark():
    df = pd.DataFrame({"skill": ["a", "a", "b"], "benchmark": ["mmlu", "mmlu", "gsm8k"]})
    assert skill_to_benchmark(df).to_dict() == {"a": "mmlu", "b": "gsm8k"}
    with pytest.raises(ValueError, match="more than one"):
        skill_to_benchmark(pd.DataFrame({"skill": ["a", "a"], "benchmark": ["x", "y"]}))


def make_responses(matrix: dict[str, list[int]], skill_of: dict[str, str]) -> pd.DataFrame:
    rows = [
        {"model": "m", "checkpoint_idx": t, "skill": skill_of[i], "item_id": i, "correct": c}
        for i, seq in matrix.items()
        for t, c in enumerate(seq)
    ]
    return pd.DataFrame(rows)


def test_variance_decomposition_hand_computed():
    # checkpoint 0: A = {1, 1}, B = {1, 0, 0, 0}
    #   p = 0.5, total = 0.25, p_A = 1, p_B = 0.25
    #   between = (2/6)(0.5)^2 + (4/6)(0.25)^2 = 0.125
    #   within  = (2/6)(0) + (4/6)(0.25)(0.75) = 0.125
    # checkpoint 1: every item correct except b4, so y = [1, 1, 1, 1, 1, 0]
    #   running mean m = y(0) = [1, 1, 1, 0, 0, 0], Var(m) = 0.25
    matrix = {
        "a1": [1, 1],
        "a2": [1, 1],
        "b1": [1, 1],
        "b2": [0, 1],
        "b3": [0, 1],
        "b4": [0, 0],
    }
    skill_of = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "b3": "B", "b4": "B"}
    out = variance_decomposition(make_responses(matrix, skill_of)).set_index("checkpoint_idx")

    r0 = out.loc[0]
    assert r0["n_items"] == 6 and r0["n_skills"] == 2
    assert r0["total_var"] == pytest.approx(0.25)
    assert r0["between_var"] == pytest.approx(0.125)
    assert r0["within_var"] == pytest.approx(0.125)
    assert r0["between_frac"] == pytest.approx(0.5)
    assert np.isnan(r0["item_mean_var"]) and np.isnan(r0["item_mean_r2"])

    r1 = out.loc[1]
    # y = [1,1,1,1,1,0]: p = 5/6; p_A = 1, p_B = 3/4
    p = 5 / 6
    assert r1["total_var"] == pytest.approx(p * (1 - p))
    between = (2 / 6) * (1 - p) ** 2 + (4 / 6) * (0.75 - p) ** 2
    within = (4 / 6) * 0.75 * 0.25
    assert r1["between_var"] == pytest.approx(between)
    assert r1["within_var"] == pytest.approx(within)
    assert r1["between_var"] + r1["within_var"] == pytest.approx(r1["total_var"])
    # m = [1,1,1,0,0,0], y = [1,1,1,1,1,0]
    # dx = [0.5]*3 + [-0.5]*3, dy = [1/6]*5 + [-5/6]
    # sum dx dy = 0.25 - 1/6 + 5/12 = 0.5; sxx = 1.5; syy = 5/36 + 25/36 = 5/6
    assert r1["item_mean_var"] == pytest.approx(0.25)
    assert r1["item_mean_r2"] == pytest.approx(0.5**2 / (1.5 * (5 / 6)))
    assert list(out["checkpoint_norm"]) == pytest.approx([0.0, 1.0])


def test_variance_decomposition_running_mean_excludes_current_checkpoint():
    # Item running means at t=2 are over checkpoints 0 and 1 only.
    matrix = {"x": [1, 1, 0], "y": [0, 0, 1], "z": [1, 0, 1]}
    skill_of = {"x": "S", "y": "S", "z": "S"}
    out = variance_decomposition(make_responses(matrix, skill_of)).set_index("checkpoint_idx")
    m = np.array([1.0, 0.0, 0.5])
    y = np.array([0.0, 1.0, 1.0])
    assert out.loc[2, "item_mean_var"] == pytest.approx(m.var())
    assert out.loc[2, "item_mean_r2"] == pytest.approx(np.corrcoef(m, y)[0, 1] ** 2)
    # one skill: nothing is between-skill
    assert out.loc[2, "between_var"] == pytest.approx(0.0)
    assert out.loc[2, "between_frac"] == pytest.approx(0.0)


def test_variance_decomposition_constant_checkpoint_gives_nan_fraction():
    matrix = {"x": [1, 1], "y": [1, 0]}
    out = variance_decomposition(make_responses(matrix, {"x": "S", "y": "T"}))
    assert np.isnan(out.loc[out["checkpoint_idx"] == 0, "between_frac"].iloc[0])
