"""Run a predictor under different skill labels while the harness keeps scoring real subjects.

Used by the DriftIRT ablation to replace the MMLU subject grouping inside a model: one
constant label (a single global ability) or a seeded random partition of the items with the
same group sizes. `RemappedSkills` relabels `skill` in copies of the history and target that
it hands to the inner predictor and returns the predictions in the target's row order. The
harness never sees the new labels, so slices, bootstrap clusters and within-skill AUC are
always computed on the original subjects.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

GLOBAL_LABEL = "all"


def constant_skill(items: Iterable[str]) -> dict[str, str]:
    """Every item to one label, so a skill-state model has a single shared ability."""
    return {item: GLOBAL_LABEL for item in pd.unique(np.asarray(list(items), dtype=object))}


def shuffled_skills(items_by_subject: Mapping[str, Iterable[str]], seed: int) -> dict[str, str]:
    """A seeded random partition of the items with every subject's size preserved.

    The subject labels, one per item, are permuted across items: each item gets exactly one
    label and each label keeps its original count. Items and subjects are sorted first, so the
    result depends only on the partition and the seed, not on input order.
    """
    owner: dict[str, str] = {}
    for subject, items in items_by_subject.items():
        for item in items:
            if item in owner and owner[item] != subject:
                raise ValueError(f"item {item!r} belongs to more than one subject")
            owner[item] = subject
    items = sorted(owner)
    labels = np.array([owner[i] for i in items], dtype=object)
    perm = np.random.default_rng(seed).permutation(len(items))
    return dict(zip(items, labels[perm]))


def items_by_subject(responses: pd.DataFrame) -> dict[str, list[str]]:
    """{skill: sorted item ids} from a long response table."""
    pairs = responses[["item_id", "skill"]].drop_duplicates()
    return {s: sorted(g["item_id"]) for s, g in pairs.groupby("skill", sort=True)}


class RemappedSkills:
    """Wrap a predictor so it sees skill labels from `mapping` (item_id to label)."""

    def __init__(self, predictor, mapping: Mapping[str, str], suffix: str):
        self.predictor = predictor
        self.mapping = pd.Series(mapping, dtype=object)
        self.name = predictor.name + suffix

    def _relabel(self, df: pd.DataFrame) -> pd.DataFrame:
        ids = df["item_id"].to_numpy(dtype=object)
        pos = self.mapping.index.get_indexer(ids)
        if (pos < 0).any():
            missing = pd.unique(ids[pos < 0])[:5]
            raise ValueError(f"no mapping for items {list(missing)}")
        out = df.copy()
        out["skill"] = self.mapping.to_numpy()[pos]
        return out

    def predict(self, history: pd.DataFrame, target: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        p = self.predictor.predict(self._relabel(history), self._relabel(target), horizon=horizon)
        return np.asarray(p, dtype=float)
