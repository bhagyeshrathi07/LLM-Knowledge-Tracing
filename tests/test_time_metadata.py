import pandas as pd
import pytest

from llm_kt.time_metadata import (
    PYTHIA_TOKENS_PER_STEP,
    fill_time_metadata,
    tokens_from_ckpt_label,
    tokens_from_step,
)


class TestTokensFromStep:
    def test_pythia_batch_is_1024_by_2048(self):
        assert PYTHIA_TOKENS_PER_STEP == 2_097_152

    def test_pythia_final_step_matches_published_300b(self):
        assert tokens_from_step("pythia-7b", 143_000) == 300  # 299.9B rounds to 300

    def test_step_zero_is_zero(self):
        assert tokens_from_step("pythia-3b", 0) == 0

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            tokens_from_step("olmo2-7b", 10)


class TestTokensFromCkptLabel:
    def test_amber_last_checkpoint_is_full_run(self):
        assert tokens_from_ckpt_label("amber-7b", "ckpt_359") == 1259

    def test_amber_first_checkpoint_is_one_chunk(self):
        assert tokens_from_ckpt_label("amber-7b", "ckpt_000") == 3

    def test_k2_last_stage1_checkpoint_is_1300b(self):
        assert tokens_from_ckpt_label("k2-65b", "ckpt_360") == 1300

    def test_k2_midpoint(self):
        assert tokens_from_ckpt_label("k2-65b", "ckpt_180") == 650

    def test_monotone_in_index(self):
        a = [tokens_from_ckpt_label("amber-7b", f"ckpt_{k:03d}") for k in range(0, 360, 37)]
        assert a == sorted(a) and len(set(a)) == len(a)

    def test_bad_label_raises(self):
        with pytest.raises(ValueError):
            tokens_from_ckpt_label("amber-7b", "step100")


class TestFillTimeMetadata:
    @pytest.fixture
    def table(self):
        rows = [
            ("olmo2-7b", "stage1-step5000-tokens21B", 5000, 21),
            ("olmo2-7b", "stage1-step5000-tokens21B", 5000, 21),
            ("pythia-7b", "step143000", 143000, pd.NA),
            ("pythia-7b", "step0", 0, pd.NA),
            ("amber-7b", "ckpt_359", pd.NA, pd.NA),
            ("k2-65b", "ckpt_360", pd.NA, pd.NA),
        ]
        df = pd.DataFrame(rows, columns=["model", "checkpoint_label", "step", "tokens_seen_b"])
        df["step"] = df["step"].astype("Int64")
        df["tokens_seen_b"] = df["tokens_seen_b"].astype("Int64")
        df["item_id"] = "arc_challenge_0"
        df["correct"] = 1
        return df

    def test_published_values_untouched(self, table):
        out = fill_time_metadata(table)
        olmo = out[out.model == "olmo2-7b"]
        assert (olmo.tokens_seen_b == 21).all()
        assert (olmo.tokens_seen_source == "published").all()

    def test_pythia_derived(self, table):
        out = fill_time_metadata(table).set_index("checkpoint_label")
        assert out.loc["step143000", "tokens_seen_b"] == 300
        assert out.loc["step0", "tokens_seen_b"] == 0
        assert out.loc["step143000", "tokens_seen_source"] == "derived"

    def test_llm360_estimated(self, table):
        out = fill_time_metadata(table).set_index("checkpoint_label")
        assert out.loc["ckpt_359", "tokens_seen_b"] == 1259
        assert out.loc["ckpt_360", "tokens_seen_b"] == 1300
        assert out.loc["ckpt_359", "tokens_seen_source"] == "estimated"

    def test_no_nulls_remain_and_dtype_kept(self, table):
        out = fill_time_metadata(table)
        assert out.tokens_seen_b.isna().sum() == 0
        assert str(out.tokens_seen_b.dtype) == "Int64"

    def test_row_order_and_count_preserved(self, table):
        out = fill_time_metadata(table)
        assert len(out) == len(table)
        assert list(out.checkpoint_label) == list(table.checkpoint_label)

    def test_input_not_mutated(self, table):
        before = table.copy()
        fill_time_metadata(table)
        pd.testing.assert_frame_equal(table, before)

    def test_unknown_model_left_null(self):
        df = pd.DataFrame({"model": ["mystery"], "checkpoint_label": ["ckpt_1"],
                           "step": pd.array([pd.NA], dtype="Int64"),
                           "tokens_seen_b": pd.array([pd.NA], dtype="Int64")})
        out = fill_time_metadata(df)
        assert out.tokens_seen_b.isna().all()
        assert out.tokens_seen_source.isna().all()
