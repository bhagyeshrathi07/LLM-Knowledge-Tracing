import pandas as pd
import pytest

from llm_kt.ingest import (
    parse_checkpoint,
    parse_item_id,
    wide_to_long,
)


class TestParseCheckpoint:
    def test_pythia_step_only(self):
        assert parse_checkpoint("step143000") == (143000, None)

    def test_pythia_step_zero(self):
        assert parse_checkpoint("step0") == (0, None)

    def test_olmo1_step_and_tokens(self):
        assert parse_checkpoint("step5000-tokens20B") == (5000, 20)

    def test_olmo2_stage_prefix(self):
        assert parse_checkpoint("stage1-step928646-tokens3896B") == (928646, 3896)

    def test_llm360_ckpt_index_has_no_step_or_tokens(self):
        assert parse_checkpoint("ckpt_093") == (None, None)

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            parse_checkpoint("epoch3")


class TestParseItemId:
    def test_mmlu_has_subject_as_skill(self):
        assert parse_item_id("mmlu_abstract_algebra_0") == ("mmlu", "abstract_algebra")

    def test_mmlu_multiword_subject(self):
        assert parse_item_id("mmlu_high_school_computer_science_12") == (
            "mmlu",
            "high_school_computer_science",
        )

    def test_non_mmlu_benchmark_is_its_own_skill(self):
        assert parse_item_id("arc_challenge_7") == ("arc_challenge", "arc_challenge")
        assert parse_item_id("gsm8k_1318") == ("gsm8k", "gsm8k")
        assert parse_item_id("hellaswag_0") == ("hellaswag", "hellaswag")
        assert parse_item_id("winogrande_42") == ("winogrande", "winogrande")

    def test_truthfulqa(self):
        assert parse_item_id("truthfulqa_mc2_3") == ("truthfulqa_mc2", "truthfulqa_mc2")


class TestWideToLong:
    @pytest.fixture
    def wide(self):
        return pd.DataFrame(
            {
                "Unnamed: 0": ["arc_challenge_0", "mmlu_anatomy_0", "truthfulqa_mc2_0"],
                "step0": [0.0, 1.0, 0.31],
                "step2000": [1.0, 1.0, 0.62],
            }
        )

    def test_columns(self, wide):
        long = wide_to_long(wide, model="pythia-3b")
        assert list(long.columns) == [
            "model",
            "checkpoint_idx",
            "checkpoint_label",
            "step",
            "tokens_seen_b",
            "benchmark",
            "skill",
            "item_id",
            "correct",
        ]

    def test_truthfulqa_dropped_by_default(self, wide):
        long = wide_to_long(wide, model="pythia-3b")
        assert "truthfulqa_mc2" not in set(long["benchmark"])
        assert len(long) == 4  # 2 items x 2 checkpoints

    def test_truthfulqa_kept_when_requested_is_thresholded(self, wide):
        long = wide_to_long(wide, model="pythia-3b", drop_benchmarks=())
        assert len(long) == 6
        tqa = long[long.benchmark == "truthfulqa_mc2"].set_index("checkpoint_label")["correct"]
        assert tqa["step0"] == 0  # 0.31 < 0.5
        assert tqa["step2000"] == 1  # 0.62 >= 0.5

    def test_checkpoint_idx_follows_column_order(self, wide):
        long = wide_to_long(wide, model="pythia-3b")
        idx = long.drop_duplicates("checkpoint_label").set_index("checkpoint_label")["checkpoint_idx"]
        assert idx["step0"] == 0
        assert idx["step2000"] == 1

    def test_values(self, wide):
        long = wide_to_long(wide, model="pythia-3b")
        row = long[(long.item_id == "arc_challenge_0") & (long.checkpoint_label == "step2000")].iloc[0]
        assert row["model"] == "pythia-3b"
        assert row["step"] == 2000
        assert pd.isna(row["tokens_seen_b"])
        assert row["benchmark"] == "arc_challenge"
        assert row["skill"] == "arc_challenge"
        assert row["correct"] == 1

    def test_correct_is_integer_binary(self, wide):
        long = wide_to_long(wide, model="pythia-3b")
        assert set(long["correct"].unique()) <= {0, 1}
        assert long["correct"].dtype.kind in "iu"

    def test_non_binary_value_in_kept_benchmark_raises(self):
        wide = pd.DataFrame({"Unnamed: 0": ["arc_challenge_0"], "step0": [0.5]})
        with pytest.raises(ValueError):
            wide_to_long(wide, model="x")
