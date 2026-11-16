#!/usr/bin/env python
"""Score learner models (static IRT, T-SKIRT, DriftIRT, BKT) with the walk-forward harness.

Usage:
    python scripts/run_models.py --predictors tskirt static_irt --nu 0.1 0.3 --horizon 1 \
        --models olmo2-7b --benchmarks mmlu
    python scripts/run_models.py --predictors drift_irt item_mean --q-theta 0.01 0.05 \
        --q-delta 0.001 0.01 --delta-prior-var 1 --forecast-var-scale 0 1 \
        --residual-scale probit --models olmo2-7b --benchmarks mmlu --tag drift

    python scripts/run_models.py --predictors bkt item_mean skill_mean --kappa 1 5 20 100 1e6 \
        --rho 1.0 0.9 0.7 --anchor skill fluid --theta-lo -1 -2 --theta-hi 1 2 \
        --models olmo2-7b --benchmarks mmlu --tag bkt_grid
    python scripts/run_models.py --predictors item_mean_ewma --rho 0.9 --horizon 1 \
        --models olmo2-7b --benchmarks mmlu --tag bkt_ewma   # recency-weighted item mean control

DriftIRT ablation switches (docs/superpowers/specs/2026-11-03-driftirt-ablation.md), applied to
every DriftIRT setting of the run:
    --no-skill-state    no per-skill ability (theta held at 0; needs --q-theta 0); name + _noskill
    --flat-items        item table a = PROBIT_SCALE, b = 0 for every item (a = 1, b = 0 on the probit
                        scale) instead of Fluid's 2PL; name + _flat
    --global-skill      one ability shared by all subjects (skill relabelled to a constant inside a
                        wrapper; the harness still scores the real subjects); name + _gskill
    --shuffle-skills S  subjects replaced by a random size-preserving partition of the items, one
                        per seed S; name + _shuf<S>
Skill mappings are built once per model from that model's full response table.

DriftIRT and BKT dials are given as lists; one predictor is built per combination. BKT
combinations that give an identical predictor (theta values under the skill anchor, rho at
kappa >= 1e5 where items are pinned to the anchor) are built once.

Writes data/processed/models_<tag>_h<horizon>.parquet and prints a per-model summary.
item_mean from the baseline run is included for reference when its results file exists.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import pandas as pd

from llm_kt.baselines import ItemMean, ItemMeanEWMA, SkillMean
from llm_kt.bkt import ANCHORS, BKT, KAPPA_LIMIT
from llm_kt.drift_irt import RESIDUAL_SCALES, DriftIRT, flat_item_params
from llm_kt.evaluate import evaluate, summarize
from llm_kt.ingest import MODELS
from llm_kt.skill_remap import RemappedSkills, constant_skill, items_by_subject, shuffled_skills
from llm_kt.static_irt import StaticIRTPersistence, load_item_params
from llm_kt.tskirt import TSKIRT

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "data" / "processed" / "fluid_responses.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--benchmarks", nargs="+", default=None, help="restrict to these benchmarks (e.g. mmlu)")
    parser.add_argument("--predictors", nargs="+", default=["static_irt", "tskirt"],
                        choices=["static_irt", "tskirt", "drift_irt", "bkt", "item_mean", "item_mean_ewma", "skill_mean"])
    parser.add_argument("--nu", nargs="+", type=float, default=[0.1], help="T-SKIRT drift values to try")
    parser.add_argument("--window", type=int, default=1, help="static IRT: checkpoints used to fit theta")
    parser.add_argument("--time-col", default="checkpoint_idx", choices=["checkpoint_idx", "step", "tokens_seen_b"])
    parser.add_argument("--recalibrate", action="store_true", help="learn per-item difficulty offsets from history")
    parser.add_argument("--tau", nargs="+", type=float, default=[1.0], help="prior sd(s) of the per-item difficulty offset")
    parser.add_argument("--tau-a", type=float, default=None,
                        help="prior sd of the per-item log-discrimination offset (default: discrimination fixed)")
    parser.add_argument("--q-theta", nargs="+", type=float, default=[0.01],
                        help="DriftIRT: per-step variance of the skill ability random walk")
    parser.add_argument("--q-delta", nargs="+", type=float, default=[0.01],
                        help="DriftIRT: per-step variance of the item residual random walk")
    parser.add_argument("--delta-prior-var", nargs="+", type=float, default=[1.0],
                        help="DriftIRT: prior variance of an item residual at first observation")
    parser.add_argument("--forecast-var-scale", nargs="+", type=float, default=[1.0],
                        help="DriftIRT: multiplier on the forecast variance (0 = plug-in forecast)")
    parser.add_argument("--residual-scale", nargs="+", default=["difficulty"], choices=RESIDUAL_SCALES,
                        help="DriftIRT: residual in difficulty units (z = a(theta - b - delta)) "
                             "or probit units (z = a(theta - b) - delta)")
    parser.add_argument("--no-skill-state", action="store_true",
                        help="DriftIRT ablation: remove the per-skill ability (requires --q-theta 0)")
    parser.add_argument("--flat-items", action="store_true",
                        help="DriftIRT ablation: a = 1, b = 0 (probit scale) for every item instead of Fluid's 2PL")
    parser.add_argument("--global-skill", action="store_true",
                        help="DriftIRT ablation: one ability shared by every subject")
    parser.add_argument("--shuffle-skills", nargs="+", type=int, default=None, metavar="SEED",
                        help="DriftIRT ablation: random size-preserving partition of items into subjects, per seed")
    parser.add_argument("--kappa", nargs="+", type=float, default=[5.0],
                        help="BKT: pseudo-count shrinking per-item guess/slip toward the anchor")
    parser.add_argument("--rho", nargs="+", type=float, default=[1.0],
                        help="BKT: recency weight per checkpoint for the per-item step (1 = none)")
    parser.add_argument("--anchor", nargs="+", default=["skill"], choices=ANCHORS,
                        help="BKT: anchor for per-item guess/slip (skill-shared or Fluid 2PL)")
    parser.add_argument("--theta-lo", nargs="+", type=float, default=[-1.0],
                        help="BKT fluid anchor: ability whose 2PL probability is the item's guess")
    parser.add_argument("--theta-hi", nargs="+", type=float, default=[1.0],
                        help="BKT fluid anchor: ability whose 2PL probability is 1 - slip")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-history", type=int, default=2)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    # filter while reading: the full table needs about 3 GB, one model's MMLU rows under 1 GB
    filters = [("model", "in", list(args.models))]
    if args.benchmarks:
        filters.append(("benchmark", "in", list(args.benchmarks)))
    responses = pd.read_parquet(args.inp, filters=filters)
    responses = responses[responses["model"].isin(args.models)]
    if args.benchmarks:
        responses = responses[responses["benchmark"].isin(args.benchmarks)]

    params = load_item_params()
    predictors = []
    if "item_mean" in args.predictors:
        predictors.append(ItemMean())
    if "item_mean_ewma" in args.predictors:
        predictors += [ItemMeanEWMA(rho=rho) for rho in args.rho]
    if "skill_mean" in args.predictors:
        predictors.append(SkillMean())
    # one predictor per tau when recalibrating (names carry tau and tau_a); tau is unused otherwise
    taus = args.tau if args.recalibrate else args.tau[:1]
    recal = {"recalibrate": args.recalibrate, "tau_a": args.tau_a}
    if "static_irt" in args.predictors:
        predictors += [StaticIRTPersistence(params, window=args.window, tau=tau, **recal) for tau in taus]
    if "tskirt" in args.predictors:
        predictors += [TSKIRT(params, nu=nu, time_col=args.time_col, tau=tau, **recal) for nu in args.nu for tau in taus]
    drift_settings = []
    if "drift_irt" in args.predictors:
        drift_params = flat_item_params(responses["item_id"]) if args.flat_items else params
        drift_settings = list(itertools.product(
            args.q_theta, args.q_delta, args.delta_prior_var, args.forecast_var_scale, args.residual_scale))

    def make_drift(qt, qd, pv, fv, rs):
        m = DriftIRT(drift_params, q_theta=qt, q_delta=qd, delta_prior_var=pv, forecast_var_scale=fv,
                     time_col=args.time_col, residual_scale=rs, skill_state=not args.no_skill_state)
        if args.flat_items:
            m.name += "_flat"
        return m

    remap = args.global_skill or args.shuffle_skills is not None
    if not remap:
        predictors += [make_drift(*s) for s in drift_settings]

    if "bkt" in args.predictors:
        seen = set()
        for kappa, rho, anchor, lo, hi in itertools.product(
                args.kappa, args.rho, args.anchor, args.theta_lo, args.theta_hi):
            if kappa >= KAPPA_LIMIT:
                rho = 1.0  # items are pinned to the anchor; rho has no effect
            if anchor == "skill":
                lo, hi = -1.0, 1.0  # thetas unused
            m = BKT(kappa=kappa, rho=rho, anchor=anchor, item_params=params, theta_lo=lo, theta_hi=hi)
            if m.name not in seen:
                seen.add(m.name)
                predictors.append(m)

    t0 = time.time()
    if not remap:
        results = evaluate(responses, predictors, min_history=args.min_history, horizon=args.horizon)
    else:
        frames = []
        for model, group in responses.groupby("model", sort=True):
            # mappings from this model's full response table, so history and target always agree
            mappings = []
            if args.global_skill:
                mappings.append((constant_skill(group["item_id"]), "_gskill"))
            for seed in args.shuffle_skills or []:
                mappings.append((shuffled_skills(items_by_subject(group), seed), f"_shuf{seed}"))
            wrapped = [RemappedSkills(make_drift(*s), mapping, suffix)  # a fresh filter per wrapper
                       for mapping, suffix in mappings for s in drift_settings]
            frames.append(evaluate(group, predictors + wrapped, min_history=args.min_history,
                                   horizon=args.horizon))
        results = pd.concat(frames, ignore_index=True)
    results["horizon"] = args.horizon
    tag = args.tag or ("_".join(args.benchmarks) if args.benchmarks else "all")
    out = REPO_ROOT / "data" / "processed" / f"models_{tag}_h{args.horizon}.parquet"
    results.to_parquet(out, index=False)
    print(f"Wrote {len(results):,} rows to {out} in {time.time() - t0:.0f}s\n")

    pd.set_option("display.width", 200)
    print(f"horizon={args.horizon}, benchmarks={args.benchmarks or 'all'} (lower log_loss better, higher auc better)")
    print(summarize(results, by=["model", "predictor"]).set_index(["model", "predictor"]).round(4).to_string())
    print("\nPooled over models")
    print(summarize(results, by=["predictor"]).set_index("predictor").round(4).to_string())


if __name__ == "__main__":
    main()
