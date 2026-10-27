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

Chance-adjusted check (`chance_check`). A model near chance on 4-option multiple choice
can lose accuracy on one subject without losing knowledge: if its preferred answer letter
changes between two checkpoints, subjects whose correct letters are unevenly distributed
swing up or down. The Fluid release stores only correctness (no predicted letter, no gold
letter), so the letter distribution cannot be observed directly. The check therefore uses
correctness alone:

  chance proximity  accuracy at peak and final against the chance rate, with Wilson 95%
                    intervals; a final interval entirely below chance is itself a sign of
                    a systematic answer preference rather than missing knowledge.
  null comparison   the drop on every other skill of the same benchmark, measured between
                    the same two checkpoints, and the drop pooled over all of those items.
                    A drop matched by broad drops elsewhere is a global shift (letter bias,
                    a training instability), not subject-specific forgetting.

Two percentiles are reported. `pct_same_ckpts` places the flagged drop among the other
skills' drops at the flagged skill's peak and final checkpoints. Because that peak was
chosen to maximise the flagged skill's accuracy, this percentile is biased upward (other
skills are not evaluated at their own best checkpoint). `pct_own_peak` instead compares
against the other skills' own selected peak-to-final drops, which share the same selection.

Letter check (`letter_check`). Fluid has no predicted letters, but the gold answer letter of
every MMLU item can be recovered by matching the Fluid item text against the cais/mmlu test
split (`match_mmlu_gold`). For a model that guesses with letter preference q, P(correct |
gold = L) = q_L, so per-gold-letter accuracy within a skill reveals the letter preference
without observing the picks. Reported per flagged pair:

  letter_r2_*       share of the variance in item correctness explained by the gold letter
                    (eta squared). Near 1 means correctness is fixed by position alone.
  p_letter_change   G-test that the distribution of correct answers over gold letters
                    differs between peak and final (treats the two checkpoints as
                    independent samples, which ignores the pairing).
  letter_exp_*      accuracy a pure guesser with the implied preference would score,
                    sum_L q_L g_L with q_L = acc_L / sum_L acc_L and g_L the skill's gold
                    letter shares. Equals 0.25 for a model whose accuracy does not depend on
                    the gold letter, so it attributes no drop to position in that case.
  letter_drop       letter_exp_peak - letter_exp_final; residual_drop = drop - letter_drop.
  rest_letter_exp_* sum_L acc_L(rest) g_L, using per-letter accuracy on all other MMLU
                    items: the skill's accuracy if it shared the model's global letter
                    profile. Tests whether a global (not skill-specific) preference explains
                    the drop.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2_contingency, norm

from llm_kt.curves import wilson_interval


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


def chance_proximity(k_peak: int, k_final: int, n: int, chance: float = 0.25) -> dict[str, float]:
    """Accuracy at peak and final relative to `chance`, with Wilson 95% intervals.

    `k_peak` and `k_final` are the numbers of correct items out of `n` at each checkpoint.
    `*_in_ci` is True when the Wilson interval contains `chance`; `final_below_chance` is
    True when the whole final interval lies below it.
    """
    lo, hi = wilson_interval(np.array([k_peak, k_final]), np.array([n, n]))
    return {
        "chance": chance,
        "acc_peak_over_chance": k_peak / n / chance,
        "acc_final_over_chance": k_final / n / chance,
        "peak_ci_lo": float(lo[0]),
        "peak_ci_hi": float(hi[0]),
        "final_ci_lo": float(lo[1]),
        "final_ci_hi": float(hi[1]),
        "peak_chance_in_ci": bool(lo[0] <= chance <= hi[0]),
        "final_chance_in_ci": bool(lo[1] <= chance <= hi[1]),
        "final_below_chance": bool(hi[1] < chance),
    }


def empirical_percentile(value: float, others: np.ndarray) -> float:
    """Share of `others` below `value`, counting ties as half. NaN when `others` is empty."""
    others = np.asarray(others, dtype=float)
    if len(others) == 0:
        return float("nan")
    return float(((others < value).sum() + 0.5 * (others == value).sum()) / len(others))


def flip_summary(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    """Flip rates between two checkpoints on the same items.

    `flip_asym` = (right-to-wrong minus wrong-to-right) / n, which equals the accuracy drop.
    `disc_rate` = share of items that changed, and `forget_share` = share of the changed
    items that went right-to-wrong (0.5 under no net change; NaN with no changes).
    """
    d = paired_drop(before, after)
    n, b, c = d["n"], d["b_1to0"], d["c_0to1"]
    return {
        "n": n,
        "drop": d["drop"],
        "se": d["se"],
        "flip_asym": (b - c) / n,
        "disc_rate": (b + c) / n,
        "forget_share": b / (b + c) if b + c else float("nan"),
    }


def chance_check(
    responses: pd.DataFrame,
    flagged: pd.DataFrame,
    table: pd.DataFrame | None = None,
    chance: float = 0.25,
    min_items: int = 1,
) -> pd.DataFrame:
    """Chance-adjusted check for flagged (model, skill) peak-to-final drops.

    `flagged` needs columns model, skill, peak_idx, final_idx (rows of `peak_to_final_table`).
    For each row, the comparison set is every other skill of the same benchmark with at least
    `min_items` items. `table` is the full peak-to-final table; when given, `pct_own_peak` is
    computed against the other skills' own selected drops in it.

    Output columns, beyond the identifiers and `chance_proximity` fields:
      n, acc_peak, acc_final, drop, se, flip_asym, disc_rate, forget_share  (flagged skill)
      acc_before_peak, acc_after_peak  accuracy at the checkpoints adjacent to the peak; a
                           peak far above both is a one-checkpoint excursion
      n_other_skills, other_drop_median, other_drop_q90, other_drop_max,
      pct_same_ckpts       percentile of `drop` among other skills at the same checkpoints
      pct_own_peak         percentile of `drop` among other skills' own peak-to-final drops
      rest_n, rest_drop, rest_se, rest_flip_asym, rest_disc_rate, rest_forget_share
                           the same quantities pooled over all other-skill items
      excess_drop, p_excess  drop minus rest_drop, with a one-sided z-test (independent items)
    """
    rows = []
    for model, fl in flagged.groupby("model", sort=False):
        group = responses[responses["model"] == model]
        wide = group.pivot(index="item_id", columns="checkpoint_idx", values="correct")
        meta = group.drop_duplicates("item_id").set_index("item_id")[["benchmark", "skill"]]
        meta = meta.loc[wide.index]
        for rec in fl.itertuples(index=False):
            p_idx, f_idx = int(rec.peak_idx), int(rec.final_idx)
            in_skill = (meta["skill"] == rec.skill).to_numpy()
            bench = meta.loc[in_skill, "benchmark"].iloc[0]
            sub = wide[in_skill]
            target = flip_summary(sub[p_idx].to_numpy(), sub[f_idx].to_numpy())
            n = target["n"]
            k_peak, k_final = int(sub[p_idx].sum()), int(sub[f_idx].sum())

            same_bench = (meta["benchmark"] == bench) & ~in_skill
            sizes = meta.loc[same_bench, "skill"].value_counts()
            other_skills = sizes[sizes >= min_items].index
            rest_mask = same_bench & meta["skill"].isin(other_skills)
            rest = wide[rest_mask.to_numpy()]
            other_drops = (
                (rest[p_idx] - rest[f_idx]).groupby(meta.loc[rest.index, "skill"]).mean().to_numpy()
                if len(rest)
                else np.array([])
            )
            out = {
                "model": model,
                "skill": rec.skill,
                "benchmark": bench,
                "peak_idx": p_idx,
                "final_idx": f_idx,
                "acc_peak": k_peak / n,
                "acc_final": k_final / n,
                "acc_before_peak": float(sub[p_idx - 1].mean()) if p_idx - 1 in sub else np.nan,
                "acc_after_peak": float(sub[p_idx + 1].mean()) if p_idx + 1 in sub else np.nan,
                **target,
                **chance_proximity(k_peak, k_final, n, chance=chance),
                "n_other_skills": len(other_drops),
                "other_drop_median": float(np.median(other_drops)) if len(other_drops) else np.nan,
                "other_drop_q90": float(np.quantile(other_drops, 0.9)) if len(other_drops) else np.nan,
                "other_drop_max": float(other_drops.max()) if len(other_drops) else np.nan,
                "pct_same_ckpts": empirical_percentile(target["drop"], other_drops),
                "pct_own_peak": np.nan,
            }
            if table is not None:
                own = table[(table["model"] == model) & table["skill"].isin(other_skills)]["drop"]
                out["pct_own_peak"] = empirical_percentile(target["drop"], own.to_numpy())
            if len(rest):
                r = flip_summary(rest[p_idx].to_numpy(), rest[f_idx].to_numpy())
                out.update({f"rest_{k}": v for k, v in r.items()})
                excess = target["drop"] - r["drop"]
                se = np.sqrt(target["se"] ** 2 + r["se"] ** 2)
                out["excess_drop"] = excess
                out["p_excess"] = float(norm.sf(excess / se)) if se > 0 else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


LETTERS = "ABCD"
_WS = re.compile(r"\s+")
_MMLU_ID = re.compile(r"^mmlu_(.*)_\d+$")


def _mmlu_key(subject: str, text: str) -> tuple[str, str]:
    return subject, _WS.sub("", text)


def match_mmlu_gold(item_map: dict[str, dict], mmlu_test: pd.DataFrame) -> pd.Series:
    """Gold answer index (0..3 for A..D) of each Fluid MMLU item, keyed by item_id.

    `item_map` is Fluid's id_to_item_map.json (item_id -> {"example", "choices"}), whose
    "example" is "<question>\nA. <c0>\nB. <c1>\nC. <c2>\nD. <c3>\nAnswer:". `mmlu_test` is
    the cais/mmlu "all" test split (columns question, subject, choices, answer). Items are
    matched on subject plus the prompt text with all whitespace removed. Raises if a Fluid
    item has no match or matches rows with different answers.
    """
    lookup: dict[tuple[str, str], set[int]] = {}
    for row in mmlu_test.itertuples(index=False):
        body = row.question.strip() + "".join(
            f"\n{letter}. {str(c).strip()}" for letter, c in zip(LETTERS, row.choices)
        )
        lookup.setdefault(_mmlu_key(row.subject, body + "\nAnswer:"), set()).add(int(row.answer))
    gold = {}
    for item_id, item in item_map.items():
        m = _MMLU_ID.match(item_id)
        if not m:
            continue
        answers = lookup.get(_mmlu_key(m.group(1), item["example"]))
        if not answers:
            raise ValueError(f"no cais/mmlu match for {item_id}")
        if len(answers) > 1:
            raise ValueError(f"ambiguous cais/mmlu match for {item_id}: {sorted(answers)}")
        gold[item_id] = next(iter(answers))
    return pd.Series(gold, name="gold", dtype="int8")


def letter_r2(correct: np.ndarray, letter: np.ndarray) -> float:
    """Share of the variance in binary `correct` explained by `letter` (eta squared).

    NaN when `correct` is constant.
    """
    y = np.asarray(correct, dtype=float)
    g = np.asarray(letter)
    total = ((y - y.mean()) ** 2).sum()
    if total == 0:
        return float("nan")
    between = sum((g == k).sum() * (y[g == k].mean() - y.mean()) ** 2 for k in np.unique(g))
    return float(between / total)


def letter_expected_accuracy(acc_by_letter: np.ndarray, gold_share: np.ndarray) -> float:
    """Accuracy of a pure guesser whose pick distribution is acc_by_letter normalised to 1.

    NaN when every per-letter accuracy is zero.
    """
    acc = np.asarray(acc_by_letter, dtype=float)
    total = acc.sum()
    if total == 0:
        return float("nan")
    return float((acc / total * np.asarray(gold_share, dtype=float)).sum())


def _g_test_profile(k_peak: np.ndarray, k_final: np.ndarray) -> float:
    """G-test p-value that correct counts are distributed alike over letters at both times."""
    table = np.vstack([k_peak, k_final]).astype(float)
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2 or (table.sum(axis=1) == 0).any():
        return float("nan")
    return float(chi2_contingency(table, lambda_="log-likelihood")[1])


def letter_check(
    responses: pd.DataFrame,
    gold: pd.Series,
    flagged: pd.DataFrame,
    r2_threshold: float = 0.5,
    explained_share: float = 0.5,
) -> pd.DataFrame:
    """Gold-letter decomposition of flagged peak-to-final drops (see module docstring).

    `gold` maps item_id to the gold answer index 0..3; only items present in it are used.
    `position_bias` is True when the gold letter explains at least `r2_threshold` of the
    correctness variance at the peak or the final checkpoint and the within-skill letter
    model accounts for at least `explained_share` of the observed drop.
    """
    rows = []
    for model, fl in flagged.groupby("model", sort=False):
        group = responses[(responses["model"] == model) & responses["item_id"].isin(gold.index)]
        wide = group.pivot(index="item_id", columns="checkpoint_idx", values="correct")
        skill = group.drop_duplicates("item_id").set_index("item_id")["skill"].loc[wide.index]
        letter = gold.loc[wide.index].to_numpy()
        for rec in fl.itertuples(index=False):
            p_idx, f_idx = int(rec.peak_idx), int(rec.final_idx)
            ins = (skill == rec.skill).to_numpy()
            lt = letter[ins]
            y_p = wide[p_idx].to_numpy()[ins]
            y_f = wide[f_idx].to_numpy()[ins]
            n = int(ins.sum())
            n_l = np.array([(lt == k).sum() for k in range(4)])
            share = n_l / n
            k_p = np.array([y_p[lt == k].sum() for k in range(4)])
            k_f = np.array([y_f[lt == k].sum() for k in range(4)])
            with np.errstate(invalid="ignore", divide="ignore"):
                acc_p, acc_f = k_p / n_l, k_f / n_l
            acc_p, acc_f = np.nan_to_num(acc_p), np.nan_to_num(acc_f)

            rest = ~ins
            rest_l = letter[rest]
            rest_p = wide[p_idx].to_numpy()[rest]
            rest_f = wide[f_idx].to_numpy()[rest]
            racc_p = np.array([rest_p[rest_l == k].mean() for k in range(4)])
            racc_f = np.array([rest_f[rest_l == k].mean() for k in range(4)])

            drop = float(y_p.mean() - y_f.mean())
            exp_p = letter_expected_accuracy(acc_p, share)
            exp_f = letter_expected_accuracy(acc_f, share)
            r2_p, r2_f = letter_r2(y_p, lt), letter_r2(y_f, lt)
            rexp_p, rexp_f = float(racc_p @ share), float(racc_f @ share)
            out = {
                "model": model,
                "skill": rec.skill,
                "peak_idx": p_idx,
                "final_idx": f_idx,
                "n": n,
                "acc_peak": float(y_p.mean()),
                "acc_final": float(y_f.mean()),
                "drop": drop,
                **{f"gold_share_{c}": float(v) for c, v in zip(LETTERS, share)},
                **{f"acc_peak_{c}": float(v) for c, v in zip(LETTERS, acc_p)},
                **{f"acc_final_{c}": float(v) for c, v in zip(LETTERS, acc_f)},
                **{f"rest_acc_peak_{c}": float(v) for c, v in zip(LETTERS, racc_p)},
                **{f"rest_acc_final_{c}": float(v) for c, v in zip(LETTERS, racc_f)},
                "letter_r2_peak": r2_p,
                "letter_r2_final": r2_f,
                "p_letter_change": _g_test_profile(k_p, k_f),
                "letter_exp_peak": exp_p,
                "letter_exp_final": exp_f,
                "letter_drop": exp_p - exp_f,
                "residual_drop": drop - (exp_p - exp_f),
                "rest_letter_exp_peak": rexp_p,
                "rest_letter_exp_final": rexp_f,
                "rest_letter_drop": rexp_p - rexp_f,
                "rest_residual_drop": drop - (rexp_p - rexp_f),
            }
            max_r2 = np.nanmax([r2_p, r2_f]) if not np.isnan([r2_p, r2_f]).all() else np.nan
            out["position_bias"] = bool(
                max_r2 >= r2_threshold
                and drop > 0
                and out["letter_drop"] >= explained_share * drop
            )
            rows.append(out)
    return pd.DataFrame(rows)
