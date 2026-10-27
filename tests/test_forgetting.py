import numpy as np
import pandas as pd
import pytest

from llm_kt.forgetting import (
    chance_check,
    chance_proximity,
    empirical_percentile,
    flip_summary,
    letter_check,
    letter_expected_accuracy,
    letter_r2,
    match_mmlu_gold,
    mcnemar_one_sided,
    paired_drop,
    peak_to_final_table,
)


def make_responses(model: str, matrix: dict[str, list[int]], skill_of: dict[str, str]) -> pd.DataFrame:
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


class TestMcNemar:
    def test_five_forgets_no_recoveries(self):
        # under H0 b ~ Bin(5, 0.5); P(b >= 5) = 1/32
        assert mcnemar_one_sided(b=5, c=0) == pytest.approx(1 / 32)

    def test_balanced_discordance_is_not_significant(self):
        assert mcnemar_one_sided(b=3, c=3) > 0.5

    def test_no_discordant_pairs_gives_one(self):
        assert mcnemar_one_sided(b=0, c=0) == 1.0

    def test_more_recoveries_than_forgets_gives_large_p(self):
        assert mcnemar_one_sided(b=1, c=6) > 0.9


class TestPairedDrop:
    def test_counts_and_effect(self):
        before = np.array([1, 1, 1, 1, 0, 0])
        after = np.array([0, 0, 1, 1, 1, 0])
        d = paired_drop(before, after)
        assert d["n"] == 6
        assert d["b_1to0"] == 2
        assert d["c_0to1"] == 1
        assert d["drop"] == pytest.approx((4 - 3) / 6)
        assert d["se"] > 0
        assert 0 < d["p_mcnemar"] <= 1


class TestPeakToFinal:
    @pytest.fixture
    def responses(self):
        # skill "forget": all 6 items correct at t=2, then 5 of them wrong at the end
        forget = {f"f{i}": [0, 1, 1, 1, 0, 0] if i < 5 else [0, 1, 1, 1, 1, 1] for i in range(6)}
        # skill "learn": monotone, peak is the final checkpoint
        learn = {f"l{i}": [0, 0, 1, 1, 1, 1] for i in range(6)}
        skill_of = {**{k: "forget" for k in forget}, **{k: "learn" for k in learn}}
        return make_responses("m", {**forget, **learn}, skill_of)

    def test_one_row_per_model_skill(self, responses):
        out = peak_to_final_table(responses)
        assert set(out["skill"]) == {"forget", "learn"}
        assert len(out) == 2

    def test_forgetting_skill_detected(self, responses):
        out = peak_to_final_table(responses).set_index("skill")
        row = out.loc["forget"]
        assert row["peak_idx"] in (1, 2, 3)
        assert row["final_idx"] == 5
        assert row["drop"] == pytest.approx(5 / 6)
        assert row["b_1to0"] == 5 and row["c_0to1"] == 0
        assert row["p_mcnemar"] == pytest.approx(1 / 32)

    def test_monotone_skill_has_zero_drop(self, responses):
        out = peak_to_final_table(responses).set_index("skill")
        row = out.loc["learn"]
        assert row["drop"] == 0
        assert row["p_mcnemar"] == 1.0

    def test_adjusted_p_values_are_more_conservative(self, responses):
        out = peak_to_final_table(responses)
        assert (out["p_holm"] >= out["p_mcnemar"]).all()
        assert (out["p_selection"] >= out["p_mcnemar"]).all()
        assert (out["p_selection"] <= 1).all()

    def test_selection_multiplier_counts_skills_and_candidate_peaks(self, responses):
        out = peak_to_final_table(responses).set_index("skill")
        # 2 skills x 5 candidate peak positions (checkpoints 0..4, final excluded) = 10
        assert out.loc["forget", "p_selection"] == pytest.approx(min(1.0, (1 / 32) * 10))

    def test_min_items_filter(self, responses):
        out = peak_to_final_table(responses, min_items=7)
        assert out.empty

    def test_min_peak_idx_excludes_early_checkpoints(self, responses):
        # with the first three checkpoints excluded, the forgetting skill's peak is 3 and the
        # candidate-peak count in the selection multiplier shrinks accordingly
        out = peak_to_final_table(responses, min_peak_idx=3).set_index("skill")
        assert out.loc["forget", "peak_idx"] == 3
        assert out.loc["forget", "p_selection"] == pytest.approx(min(1.0, out.loc["forget", "p_mcnemar"] * 2 * 2))


class TestChanceProximity:
    def test_at_chance_is_inside_interval(self):
        out = chance_proximity(k_peak=40, k_final=25, n=100)
        assert out["acc_peak_over_chance"] == pytest.approx(1.6)
        assert out["acc_final_over_chance"] == pytest.approx(1.0)
        assert out["final_chance_in_ci"]
        assert not out["final_below_chance"]
        # Wilson upper bound for 40/100 is about 0.497, lower about 0.309: chance excluded
        assert not out["peak_chance_in_ci"]

    def test_well_below_chance_is_flagged(self):
        # Wilson 95% upper bound for 10/100 is about 0.174
        out = chance_proximity(k_peak=40, k_final=10, n=100)
        assert out["final_ci_hi"] == pytest.approx(0.1744, abs=1e-3)
        assert out["final_below_chance"]
        assert not out["final_chance_in_ci"]


class TestEmpiricalPercentile:
    def test_ties_count_half(self):
        assert empirical_percentile(0.3, np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(0.625)

    def test_extremes(self):
        assert empirical_percentile(1.0, np.array([0.1, 0.2])) == 1.0
        assert empirical_percentile(-1.0, np.array([0.1, 0.2])) == 0.0

    def test_empty_is_nan(self):
        assert np.isnan(empirical_percentile(0.3, np.array([])))


class TestFlipSummary:
    def test_rates(self):
        out = flip_summary(np.array([1, 1, 1, 0]), np.array([0, 0, 1, 1]))
        assert out["flip_asym"] == pytest.approx(0.25)
        assert out["flip_asym"] == pytest.approx(out["drop"])
        assert out["disc_rate"] == pytest.approx(0.75)
        assert out["forget_share"] == pytest.approx(2 / 3)

    def test_no_changes(self):
        out = flip_summary(np.array([1, 0]), np.array([1, 0]))
        assert out["disc_rate"] == 0
        assert np.isnan(out["forget_share"])


class TestChanceCheck:
    @pytest.fixture
    def responses(self):
        # checkpoints 0, 1 (peak), 2 (final)
        matrix = {
            "s0": [0, 1, 0], "s1": [0, 1, 0], "s2": [1, 1, 0], "s3": [1, 1, 1],  # drop 0.75
            "a0": [0, 1, 1], "a1": [0, 1, 0], "a2": [0, 0, 0], "a3": [0, 0, 0],  # drop 0.25
            "b0": [0, 0, 1], "b1": [0, 0, 1], "b2": [0, 0, 0], "b3": [0, 0, 0],  # drop -0.5
        }
        skill_of = {k: {"s": "s", "a": "o1", "b": "o2"}[k[0]] for k in matrix}
        return make_responses("m", matrix, skill_of)

    def test_against_hand_computed_values(self, responses):
        flagged = pd.DataFrame([{"model": "m", "skill": "s", "peak_idx": 1, "final_idx": 2}])
        row = chance_check(responses, flagged).iloc[0]
        assert row["n"] == 4
        assert row["acc_peak"] == 1.0 and row["acc_final"] == 0.25
        assert row["acc_before_peak"] == 0.5
        assert row["drop"] == pytest.approx(0.75)
        assert row["n_other_skills"] == 2
        assert row["other_drop_median"] == pytest.approx(-0.125)
        assert row["pct_same_ckpts"] == 1.0
        # pooled rest: 8 items, 2 correct at the peak, 3 at the final; b = 1 (a1), c = 2 (b0, b1)
        assert row["rest_n"] == 8
        assert row["rest_drop"] == pytest.approx(-0.125)
        assert row["rest_flip_asym"] == pytest.approx(-0.125)
        assert row["rest_disc_rate"] == pytest.approx(3 / 8)
        assert row["rest_forget_share"] == pytest.approx(1 / 3)
        assert row["excess_drop"] == pytest.approx(0.875)
        assert row["final_chance_in_ci"]

    def test_own_peak_percentile_uses_table(self, responses):
        table = peak_to_final_table(responses)
        flagged = table[table["skill"] == "s"]
        row = chance_check(responses, flagged, table=table).iloc[0]
        # o1 peaks at 1 (drop 0.25), o2 at 0 (drop -0.5, earliest tie); s at 1 with drop 0.75
        assert row["pct_own_peak"] == 1.0

    def test_min_items_excludes_small_other_skills(self, responses):
        flagged = pd.DataFrame([{"model": "m", "skill": "s", "peak_idx": 1, "final_idx": 2}])
        row = chance_check(responses, flagged, min_items=5).iloc[0]
        assert row["n_other_skills"] == 0
        assert np.isnan(row["pct_same_ckpts"])


class TestLetterHelpers:
    def test_r2_one_when_correctness_is_the_letter(self):
        letters = np.array([0, 1, 2, 3, 3, 3])
        assert letter_r2((letters == 3).astype(int), letters) == pytest.approx(1.0)

    def test_r2_zero_when_independent_of_letter(self):
        assert letter_r2(np.array([1, 0, 1, 0]), np.array([0, 0, 1, 1])) == pytest.approx(0.0)

    def test_r2_constant_is_nan(self):
        assert np.isnan(letter_r2(np.array([1, 1]), np.array([0, 1])))

    def test_expected_accuracy_of_always_d(self):
        share = np.array([0.2, 0.2, 0.2, 0.4])
        assert letter_expected_accuracy(np.array([0, 0, 0, 1.0]), share) == pytest.approx(0.4)

    def test_expected_accuracy_without_letter_dependence_is_quarter(self):
        share = np.array([0.1, 0.2, 0.3, 0.4])
        assert letter_expected_accuracy(np.full(4, 0.7), share) == pytest.approx(0.25)
        assert np.isnan(letter_expected_accuracy(np.zeros(4), share))


class TestMatchMmluGold:
    def test_matches_ignoring_whitespace_and_handles_duplicates(self):
        item_map = {
            "mmlu_x_0": {"example": "Q one?\nA. a\nB. b\nC. c\nD. d\nAnswer:"},
            "mmlu_x_1": {"example": "Q  one?\nA. a\nB. b\nC. c\nD.  d\nAnswer:"},
            "mmlu_y_0": {"example": "Q two?\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer:"},
            "arc_challenge_0": {"example": "ignored"},
        }
        test = pd.DataFrame({
            "question": ["Q one?", "Q one?", "Q two? "],
            "subject": ["x", "x", "y"],
            "choices": [["a", "b", "c", "d"], ["a", "b", "c", "d"], ["1", "2", "3", "4"]],
            "answer": [2, 2, 0],
        })
        gold = match_mmlu_gold(item_map, test)
        assert gold.to_dict() == {"mmlu_x_0": 2, "mmlu_x_1": 2, "mmlu_y_0": 0}

    def test_unmatched_raises(self):
        item_map = {"mmlu_x_0": {"example": "Missing?\nA. a\nB. b\nC. c\nD. d\nAnswer:"}}
        test = pd.DataFrame({"question": ["Other?"], "subject": ["x"],
                             "choices": [["a", "b", "c", "d"]], "answer": [1]})
        with pytest.raises(ValueError, match="no cais/mmlu match"):
            match_mmlu_gold(item_map, test)

    def test_conflicting_answers_raise(self):
        item_map = {"mmlu_x_0": {"example": "Q?\nA. a\nB. b\nC. c\nD. d\nAnswer:"}}
        test = pd.DataFrame({"question": ["Q?", "Q?"], "subject": ["x", "x"],
                             "choices": [["a", "b", "c", "d"]] * 2, "answer": [1, 3]})
        with pytest.raises(ValueError, match="ambiguous"):
            match_mmlu_gold(item_map, test)


class TestLetterCheck:
    def test_pure_letter_switch_is_position_bias(self):
        # skill "s": gold D, D, A, B. Peak (t=0) answers D everywhere, final (t=1) answers A.
        matrix = {"s0": [1, 0], "s1": [1, 0], "s2": [0, 1], "s3": [0, 0],
                  "o0": [1, 1], "o1": [1, 1], "o2": [1, 1], "o3": [1, 1]}
        gold = pd.Series({"s0": 3, "s1": 3, "s2": 0, "s3": 1, "o0": 0, "o1": 1, "o2": 2, "o3": 3})
        skill_of = {k: k[0] for k in matrix}
        responses = make_responses("m", matrix, skill_of)
        flagged = pd.DataFrame([{"model": "m", "skill": "s", "peak_idx": 0, "final_idx": 1}])
        row = letter_check(responses, gold, flagged).iloc[0]
        assert row["drop"] == pytest.approx(0.25)
        assert [row[f"gold_share_{c}"] for c in "ABCD"] == [0.25, 0.25, 0.0, 0.5]
        assert row["letter_r2_peak"] == pytest.approx(1.0)
        assert row["letter_exp_peak"] == pytest.approx(0.5)
        assert row["letter_exp_final"] == pytest.approx(0.25)
        assert row["letter_drop"] == pytest.approx(0.25)
        assert row["residual_drop"] == pytest.approx(0.0)
        # the other skill is always right, so its letter profile predicts no drop
        assert row["rest_letter_drop"] == pytest.approx(0.0)
        assert 0 < row["p_letter_change"] <= 1
        assert row["position_bias"]

    def test_letter_independent_drop_is_not_position_bias(self):
        # 16 items, 4 per gold letter: 3 of 4 right per letter at the peak, 1 of 4 at the final
        matrix, gold = {}, {}
        for letter in range(4):
            for j in range(4):
                key = f"s{letter}{j}"
                matrix[key] = [int(j < 3), int(j < 1)]
                gold[key] = letter
        matrix["o0"], gold["o0"] = [1, 1], 0
        for letter in range(1, 4):
            matrix[f"o{letter}"], gold[f"o{letter}"] = [0, 0], letter
        responses = make_responses("m", matrix, {k: k[0] for k in matrix})
        flagged = pd.DataFrame([{"model": "m", "skill": "s", "peak_idx": 0, "final_idx": 1}])
        row = letter_check(responses, pd.Series(gold), flagged).iloc[0]
        assert row["drop"] == pytest.approx(0.5)
        assert row["letter_r2_peak"] == pytest.approx(0.0)
        assert row["letter_exp_peak"] == pytest.approx(0.25)
        assert row["letter_drop"] == pytest.approx(0.0)
        assert row["residual_drop"] == pytest.approx(0.5)
        assert not row["position_bias"]
