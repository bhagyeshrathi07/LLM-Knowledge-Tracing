"""Recency-weighted item mean: the control for BKT's recency gain."""

import numpy as np
import pandas as pd
import pytest

from llm_kt.baselines import ItemMean, ItemMeanEWMA


def _history(correct_by_checkpoint: list[int], item_id: str = "q1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "checkpoint_idx": range(len(correct_by_checkpoint)),
            "item_id": item_id,
            "skill": "s",
            "correct": correct_by_checkpoint,
        }
    )


def _target(item_id: str = "q1") -> pd.DataFrame:
    return pd.DataFrame({"item_id": [item_id], "skill": ["s"]})


def test_name_carries_rho():
    assert ItemMeanEWMA(rho=0.9).name == "item_mean_ewma_r0.9"


def test_rho_one_matches_item_mean():
    hist = _history([1, 0, 1, 1, 0, 1])
    expected = ItemMean(alpha=1.0).predict(hist, _target())
    got = ItemMeanEWMA(rho=1.0, alpha=1.0).predict(hist, _target())
    assert got == pytest.approx(expected)


def test_recent_checkpoints_weigh_more():
    # same counts, opposite order: the EWMA must follow the recent answers
    rising = _history([0, 0, 0, 1, 1, 1])
    falling = _history([1, 1, 1, 0, 0, 0])
    m = ItemMeanEWMA(rho=0.5, alpha=0.0)
    assert m.predict(rising, _target())[0] > 0.8
    assert m.predict(falling, _target())[0] < 0.2
    assert ItemMean(alpha=0.0).predict(rising, _target())[0] == pytest.approx(0.5)


def test_exact_weighted_mean_with_shrinkage():
    hist = _history([1, 0, 1])  # ages 2, 1, 0 at the last checkpoint
    rho, alpha = 0.5, 1.0
    w = np.array([rho**2, rho**1, rho**0])
    prior = hist["correct"].mean()
    expected = (np.dot(w, [1, 0, 1]) + alpha * prior) / (w.sum() + alpha)
    got = ItemMeanEWMA(rho=rho, alpha=alpha).predict(hist, _target())[0]
    assert got == pytest.approx(expected)


def test_unseen_item_gets_the_prior():
    hist = _history([1, 1, 0])
    got = ItemMeanEWMA(rho=0.9).predict(hist, _target("never_seen"))[0]
    assert got == pytest.approx(hist["correct"].mean())
