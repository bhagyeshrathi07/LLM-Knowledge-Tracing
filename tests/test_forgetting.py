import numpy as np
import pandas as pd
import pytest

from llm_kt.forgetting import mcnemar_one_sided, paired_drop, peak_to_final_table


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
