import numpy as np
import pandas as pd
import pytest

from llm_kt.curves import skill_accuracy, wilson_interval


class TestWilsonInterval:
    def test_half_is_centred(self):
        lo, hi = wilson_interval(np.array([50]), np.array([100]))
        assert lo < 0.5 < hi
        assert abs((lo + hi) / 2 - 0.5) < 1e-9

    def test_bounds_stay_in_unit_interval(self):
        lo, hi = wilson_interval(np.array([0, 10]), np.array([10, 10]))
        assert lo[0] >= 0 and hi[1] <= 1

    def test_wider_for_smaller_n(self):
        lo_s, hi_s = wilson_interval(np.array([5]), np.array([10]))
        lo_l, hi_l = wilson_interval(np.array([500]), np.array([1000]))
        assert (hi_s - lo_s) > (hi_l - lo_l)


@pytest.fixture
def responses():
    rows = []
    for ckpt, label in [(0, "step0"), (1, "step100")]:
        for item, skill, correct in [
            ("mmlu_anatomy_0", "anatomy", 0),
            ("mmlu_anatomy_1", "anatomy", 1 if ckpt else 0),
            ("arc_challenge_0", "arc_challenge", 1),
        ]:
            rows.append(
                {
                    "model": "m",
                    "checkpoint_idx": ckpt,
                    "checkpoint_label": label,
                    "step": ckpt * 100,
                    "tokens_seen_b": pd.NA,
                    "benchmark": "mmlu" if skill == "anatomy" else "arc_challenge",
                    "skill": skill,
                    "item_id": item,
                    "correct": correct,
                }
            )
    df = pd.DataFrame(rows)
    df["step"] = df["step"].astype("Int64")
    df["tokens_seen_b"] = df["tokens_seen_b"].astype("Int64")
    return df


class TestSkillAccuracy:
    def test_one_row_per_model_skill_checkpoint(self, responses):
        out = skill_accuracy(responses)
        assert len(out) == 4  # 2 skills x 2 checkpoints

    def test_accuracy_and_counts(self, responses):
        out = skill_accuracy(responses).set_index(["skill", "checkpoint_idx"])
        assert out.loc[("anatomy", 0), "n"] == 2
        assert out.loc[("anatomy", 0), "acc"] == 0.0
        assert out.loc[("anatomy", 1), "acc"] == 0.5
        assert out.loc[("arc_challenge", 1), "acc"] == 1.0

    def test_checkpoint_metadata_carried_through(self, responses):
        out = skill_accuracy(responses)
        row = out[(out.skill == "anatomy") & (out.checkpoint_idx == 1)].iloc[0]
        assert row["checkpoint_label"] == "step100"
        assert row["step"] == 100
        assert pd.isna(row["tokens_seen_b"])

    def test_null_tokens_do_not_drop_rows(self, responses):
        # groupby with dropna=False must keep rows whose tokens_seen_b is NA
        out = skill_accuracy(responses)
        assert out["n"].sum() == len(responses)

    def test_ci_brackets_acc(self, responses):
        out = skill_accuracy(responses)
        assert (out["ci_lo"] <= out["acc"]).all()
        assert (out["acc"] <= out["ci_hi"]).all()
