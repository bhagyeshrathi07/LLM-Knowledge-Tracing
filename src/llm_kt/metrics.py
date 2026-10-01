"""Scoring for binary correctness predictions. No sklearn dependency."""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((np.asarray(p) >= 0.5) == (y == 1)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC via the rank statistic (Mann-Whitney U). NaN if y has one class."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(p)
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given the average rank."""
    order = np.argsort(x, kind="mergesort")
    _, inverse, counts = np.unique(x[order], return_inverse=True, return_counts=True)
    first_rank = np.cumsum(counts) - counts + 1
    avg_rank = first_rank + (counts - 1) / 2
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = avg_rank[inverse]
    return ranks


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "n": len(y),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "accuracy": accuracy(y, p),
        "auc": auc(y, p),
    }
