"""Paired forgetting test on item-level checkpoint responses.

Following Kübler et al. (2026, "When LLMs get significantly worse"): for two checkpoints
of the same model answering the same items, the only informative items are the
discordant ones. Let b = items correct at the earlier checkpoint and wrong at the later
one (1 -> 0), c = the reverse (0 -> 1). Under the null of no change b ~ Binomial(b + c, 1/2);
the exact one-sided McNemar p-value is P(B >= b).

The comparison we report is peak-to-final per (model, skill). Because the peak is chosen
after inspecting the whole curve and many skills are screened, the raw p-value is
anti-conservative. Two adjustments are reported alongside it:

  p_holm       Holm step-down across skills within a model (controls family-wise error
               over the skills, but not over the choice of peak).
  p_selection  Bonferroni over (number of skills) x (number of candidate peak positions,
               i.e. every checkpoint except the final one). Deliberately conservative: it
               treats every possible peak as a separate comparison.

A significant paired drop is behavioural evidence that accuracy fell on these items; it
does not by itself identify latent forgetting (item sampling noise and benchmark
non-monotonicity are competing explanations, see Fluid Benchmarking §6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(B >= b) for B ~ Binomial(b + c, 1/2). Returns 1.0 when there are no discordant pairs."""
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    return float(binom.sf(b - 1, n_disc, 0.5))


def paired_drop(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    """Accuracy drop from `before` to `after` on the same items, with paired SE and McNemar p."""
    before = np.asarray(before, dtype=int)
    after = np.asarray(after, dtype=int)
    if before.shape != after.shape:
        raise ValueError("before and after must align item-for-item")
    n = len(before)
    b = int(((before == 1) & (after == 0)).sum())
    c = int(((before == 0) & (after == 1)).sum())
    drop = (b - c) / n
    # Var(drop) = [P_b + P_c - (P_b - P_c)^2] / n  (paired-difference variance)
    var = (b / n + c / n - drop**2) / n
    return {
        "n": n,
        "acc_before": float(before.mean()),
        "acc_after": float(after.mean()),
        "drop": float(drop),
        "se": float(np.sqrt(max(var, 0.0))),
        "b_1to0": b,
        "c_0to1": c,
        "p_mcnemar": mcnemar_one_sided(b, c),
    }


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def peak_to_final_table(responses: pd.DataFrame, min_items: int = 1, min_peak_idx: int = 0) -> pd.DataFrame:
    """Peak-vs-final paired drop for every (model, skill).

    The peak is the checkpoint with the highest skill accuracy (earliest on ties) among
    checkpoints with index >= `min_peak_idx`; the final is the last checkpoint. Skills with
    fewer than `min_items` items are skipped. Use `min_peak_idx` to exclude the first few
    checkpoints, where a near-random model can score above chance by positional bias (for
    example Pythia step 0), so that a "peak" there is not mistaken for learned-then-forgotten.
    """
    rows = []
    for model, group in responses.groupby("model", sort=True):
        wide = group.pivot(index="item_id", columns="checkpoint_idx", values="correct").sort_index(axis=1)
        skill_of = group.drop_duplicates("item_id").set_index("item_id")["skill"]
        final_idx = int(wide.columns[-1])
        eligible = [c for c in wide.columns[:-1] if c >= min_peak_idx]
        if not eligible:
            continue
        n_candidate_peaks = len(eligible)
        model_rows = []
        for skill, items in skill_of.groupby(skill_of).groups.items():
            sub = wide.loc[list(items)]
            if len(sub) < min_items:
                continue
            acc = sub[eligible].mean(axis=0)
            peak_idx = int(acc.idxmax())
            d = paired_drop(sub[peak_idx].to_numpy(), sub[final_idx].to_numpy())
            model_rows.append(
                {
                    "model": model,
                    "skill": skill,
                    "peak_idx": peak_idx,
                    "final_idx": final_idx,
                    **d,
                }
            )
        if not model_rows:
            continue
        df = pd.DataFrame(model_rows)
        df["p_holm"] = _holm(df["p_mcnemar"].to_numpy())
        df["p_selection"] = np.minimum(1.0, df["p_mcnemar"] * len(df) * max(n_candidate_peaks, 1))
        rows.append(df)
    if not rows:
        return pd.DataFrame(
            columns=["model", "skill", "peak_idx", "final_idx", "n", "acc_before", "acc_after",
                     "drop", "se", "b_1to0", "c_0to1", "p_mcnemar", "p_holm", "p_selection"]
        )
    out = pd.concat(rows, ignore_index=True)
    return out.rename(columns={"acc_before": "acc_peak", "acc_after": "acc_final"})
