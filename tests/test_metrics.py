import numpy as np
import pytest

from llm_kt.metrics import accuracy, auc, brier, log_loss, score


def test_log_loss_perfect_is_near_zero():
    assert log_loss(np.array([1, 0]), np.array([1.0, 0.0])) < 1e-5


def test_log_loss_coin_flip_is_ln2():
    assert log_loss(np.array([1, 0, 1]), np.array([0.5, 0.5, 0.5])) == pytest.approx(np.log(2))


def test_brier():
    assert brier(np.array([1, 0]), np.array([0.8, 0.3])) == pytest.approx((0.04 + 0.09) / 2)


def test_accuracy_threshold_at_half():
    assert accuracy(np.array([1, 0, 1, 0]), np.array([0.5, 0.49, 0.9, 0.6])) == 0.75


def test_auc_perfect_ranking():
    assert auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_auc_reversed_ranking():
    assert auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auc_ties_are_half():
    assert auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_auc_single_class_is_nan():
    assert np.isnan(auc(np.array([1, 1]), np.array([0.2, 0.9])))


def test_score_keys():
    s = score(np.array([1, 0]), np.array([0.7, 0.2]))
    assert set(s) == {"n", "log_loss", "brier", "accuracy", "auc"}
    assert s["n"] == 2
