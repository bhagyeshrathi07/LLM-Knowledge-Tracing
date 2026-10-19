import numpy as np
import pandas as pd
import pytest

from llm_kt.baselines import GlobalMean
from llm_kt.evaluate import assign_blocks, evaluate, evaluate_model


def make_responses(model: str, n_ckpt: int, n_items: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for t in range(n_ckpt):
        for i in range(n_items):
            rows.append(
                {
                    "model": model,
                    "checkpoint_idx": t,
                    "checkpoint_label": f"step{t}",
                    "step": t,
                    "tokens_seen_b": pd.NA,
                    "benchmark": "mmlu",
                    "skill": "a" if i % 2 == 0 else "b",
                    "item_id": f"i{i}",
                    "correct": int(rng.integers(0, 2)),
                }
            )
    return pd.DataFrame(rows)


def check_partition(blocks: dict[str, tuple[int, int]], n: int) -> None:
    assert list(blocks) == ["train", "val", "test"]
    assert blocks["train"][0] == 0
    assert blocks["train"][1] + 1 == blocks["val"][0]
    assert blocks["val"][1] + 1 == blocks["test"][0]
    assert blocks["test"][1] == n - 1
    for lo, hi in blocks.values():
        assert lo <= hi


class TestAssignBlocks:
    def test_n5(self):
        b = assign_blocks(5)
        check_partition(b, 5)
        assert b == {"train": (0, 2), "val": (3, 3), "test": (4, 4)}

    def test_n10(self):
        b = assign_blocks(10)
        check_partition(b, 10)
        assert b == {"train": (0, 5), "val": (6, 7), "test": (8, 9)}

    def test_n94(self):
        b = assign_blocks(94)
        check_partition(b, 94)
        assert b["test"] == (75, 93)  # 19 checkpoints
        assert b["val"] == (56, 74)  # 19 checkpoints
        assert b["train"] == (0, 55)

    @pytest.mark.parametrize("n", range(5, 200))
    def test_partition_and_nonempty_test(self, n):
        check_partition(assign_blocks(n), n)

    def test_float_fraction_not_overcounted(self):
        # 15 * 0.2 is 3.0000000000000004 in floating point; must still give 3
        b = assign_blocks(15)
        assert b["test"] == (12, 14)
        assert b["val"] == (9, 11)

    def test_bad_args(self):
        with pytest.raises(ValueError):
            assign_blocks(2)
        with pytest.raises(ValueError):
            assign_blocks(10, test_frac=0.0)
        with pytest.raises(ValueError):
            assign_blocks(10, test_frac=0.5, val_frac=0.5)


class TestBlockColumn:
    def test_block_labels_follow_target_idx(self):
        out = evaluate_model(make_responses("m", 10), GlobalMean(), min_history=2)
        assert list(out.columns[:5]) == ["predictor", "model", "origin_idx", "target_idx", "block"]
        expected = {t: "train" for t in range(6)} | {6: "val", 7: "val", 8: "test", 9: "test"}
        assert (out["block"] == out["target_idx"].map(expected)).all()
        assert set(out["block"]) == {"train", "val", "test"}

    def test_blocks_are_per_model(self):
        responses = pd.concat(
            [make_responses("short", 5), make_responses("long", 10)], ignore_index=True
        )
        out = evaluate(responses, [GlobalMean()], min_history=2)
        short = out[out.model == "short"]
        long = out[out.model == "long"]
        assert set(short.loc[short.block == "test", "target_idx"]) == {4}
        assert set(short.loc[short.block == "val", "target_idx"]) == {3}
        assert set(long.loc[long.block == "test", "target_idx"]) == {8, 9}
        assert set(long.loc[long.block == "val", "target_idx"]) == {6, 7}


class TestBlockFilter:
    def spy(self):
        class Spy:
            name = "spy"

            def __init__(self):
                self.seen = []

            def predict(self, history, target, horizon=1):
                self.seen.append(int(history["checkpoint_idx"].max()))
                return np.full(len(target), 0.5)

        return Spy()

    def test_val_only_calls_predictor_for_val_targets(self):
        spy = self.spy()
        out = evaluate(make_responses("m", 10), [spy], min_history=2, blocks="val")
        assert spy.seen == [5, 6]  # targets 6 and 7
        assert set(out["block"]) == {"val"}
        assert set(out["target_idx"]) == {6, 7}

    def test_test_origins_see_val_history(self):
        spy = self.spy()
        evaluate(make_responses("m", 10), [spy], min_history=2, blocks="test")
        assert spy.seen == [7, 8]  # history includes val-block checkpoints 6, 7

    def test_list_of_blocks(self):
        spy = self.spy()
        out = evaluate(make_responses("m", 10), [spy], min_history=2, blocks=["val", "test"])
        assert spy.seen == [5, 6, 7, 8]
        assert set(out["block"]) == {"val", "test"}

    def test_horizon_two_filters_by_target(self):
        spy = self.spy()
        out = evaluate(make_responses("m", 10), [spy], min_history=2, horizon=2, blocks="val")
        assert spy.seen == [4, 5]
        assert set(out["target_idx"]) == {6, 7}

    def test_unknown_block_rejected(self):
        with pytest.raises(ValueError):
            evaluate(make_responses("m", 10), [GlobalMean()], blocks="validation")

    def test_none_scores_everything(self):
        spy = self.spy()
        evaluate(make_responses("m", 10), [spy], min_history=2)
        assert spy.seen == list(range(1, 9))
