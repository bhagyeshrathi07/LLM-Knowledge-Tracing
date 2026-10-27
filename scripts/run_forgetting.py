#!/usr/bin/env python
"""Peak-to-final paired forgetting test per (model, skill).

Writes data/processed/forgetting_peak_to_final.parquet and prints the skills whose
selection-adjusted p-value is below alpha, plus the seven candidate subjects named in
the project notes regardless of significance.

With --chance-check, also runs the chance-adjusted check (llm_kt.forgetting.chance_check)
on every pair with p_selection < alpha, plus the gold-letter check
(llm_kt.forgetting.letter_check), and writes data/processed/forgetting_chance_check.parquet.
The letter check needs MMLU gold answers, which Fluid does not include: they are taken from
the cais/mmlu test split on HuggingFace (about 3.5 MB, downloaded once into the HF cache) and
matched to Fluid items by prompt text. Pass --skip-letter to run without them.

Usage:
    python scripts/run_forgetting.py [--models m1 m2 ...] [--alpha 0.05] [--min-items 50]
                                     [--chance-check] [--chance 0.25] [--skip-letter]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llm_kt.forgetting import chance_check, letter_check, match_mmlu_gold, peak_to_final_table
from llm_kt.ingest import HF_REPO, MODELS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "data" / "processed" / "fluid_responses.parquet"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "forgetting_peak_to_final.parquet"
DEFAULT_CHANCE_OUT = REPO_ROOT / "data" / "processed" / "forgetting_chance_check.parquet"

CANDIDATES = [  # OLMo-2 MMLU subjects with peak-minus-final accuracy > 0.08 (project notes, 2026-09-30)
    "machine_learning", "abstract_algebra", "global_facts", "public_relations",
    "college_mathematics", "econometrics", "business_ethics",
]


def load_mmlu_gold() -> pd.Series:
    """Gold answer index per Fluid MMLU item_id, from Fluid's item map and cais/mmlu."""
    from huggingface_hub import hf_hub_download

    item_map_path = hf_hub_download(HF_REPO, "data/id_to_item_map.json", repo_type="dataset")
    mmlu_path = hf_hub_download("cais/mmlu", "all/test-00000-of-00001.parquet", repo_type="dataset")
    with open(item_map_path) as f:
        item_map = json.load(f)
    return match_mmlu_gold(item_map, pd.read_parquet(mmlu_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-items", type=int, default=50)
    parser.add_argument("--min-peak-idx", type=int, default=5, help="ignore peaks in the first N checkpoints")
    parser.add_argument("--chance-check", action="store_true",
                        help="run the chance-adjusted check on pairs with p_selection < alpha")
    parser.add_argument("--chance", type=float, default=0.25, help="chance accuracy (4-option MMLU)")
    parser.add_argument("--chance-out", type=Path, default=DEFAULT_CHANCE_OUT)
    parser.add_argument("--skip-letter", action="store_true",
                        help="skip the gold-letter check (no cais/mmlu download)")
    args = parser.parse_args()

    responses = pd.read_parquet(args.inp)
    responses = responses[responses["model"].isin(args.models)]
    table = peak_to_final_table(responses, min_items=args.min_items, min_peak_idx=args.min_peak_idx)
    table.to_parquet(args.out, index=False)
    print(f"Wrote {len(table):,} rows to {args.out}\n")

    pd.set_option("display.width", 200)
    cols = ["model", "skill", "peak_idx", "final_idx", "n", "acc_peak", "acc_final", "drop", "se",
            "b_1to0", "c_0to1", "p_mcnemar", "p_holm", "p_selection"]

    sig = table[table["p_selection"] < args.alpha].sort_values("p_selection")
    print(f"Skills with selection-adjusted p < {args.alpha}: {len(sig)}")
    if not sig.empty:
        print(sig[cols].round(4).to_string(index=False))

    print("\nOLMo-2 candidate subjects (from the curve inspection):")
    cand = table[(table["model"] == "olmo2-7b") & (table["skill"].isin(CANDIDATES))]
    print(cand[cols].round(4).to_string(index=False))

    if args.chance_check:
        check = chance_check(responses, sig, table=table, chance=args.chance,
                             min_items=args.min_items)
        letters = None
        if not args.skip_letter and not check.empty:
            letters = letter_check(responses, load_mmlu_gold(), sig)
            keys = ["model", "skill", "peak_idx", "final_idx"]
            extra = [c for c in letters.columns if c not in check.columns or c in keys]
            check = check.merge(letters[extra], on=keys, how="left")
        check.to_parquet(args.chance_out, index=False)
        print(f"\nChance-adjusted check on {len(check)} pairs, wrote {args.chance_out}")
        if check.empty:
            return
        ccols = ["model", "skill", "n", "peak_idx", "final_idx", "acc_before_peak", "acc_peak",
                 "acc_after_peak", "acc_final",
                 "final_ci_lo", "final_ci_hi", "final_chance_in_ci", "final_below_chance",
                 "n_other_skills", "other_drop_median", "pct_same_ckpts", "pct_own_peak"]
        print(check[ccols].round(3).to_string(index=False))
        fcols = ["model", "skill", "drop", "flip_asym", "disc_rate", "forget_share",
                 "rest_n", "rest_drop", "rest_disc_rate", "rest_forget_share",
                 "excess_drop", "p_excess"]
        print()
        print(check[fcols].round(4).to_string(index=False))
        if letters is not None:
            print("\nGold-letter check (per-gold-letter accuracy within the skill, A/B/C/D):")
            for r in check.itertuples(index=False):
                gs = "/".join(f"{getattr(r, f'gold_share_{c}'):.2f}" for c in "ABCD")
                ap = "/".join(f"{getattr(r, f'acc_peak_{c}'):.2f}" for c in "ABCD")
                af = "/".join(f"{getattr(r, f'acc_final_{c}'):.2f}" for c in "ABCD")
                print(f"  {r.model:10s} {r.skill:22s} gold {gs}  peak {ap}  final {af}")
            lcols = ["model", "skill", "drop", "letter_r2_peak", "letter_r2_final",
                     "p_letter_change", "letter_exp_peak", "letter_exp_final", "letter_drop",
                     "residual_drop", "rest_letter_drop", "rest_residual_drop", "position_bias"]
            print()
            print(check[lcols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
