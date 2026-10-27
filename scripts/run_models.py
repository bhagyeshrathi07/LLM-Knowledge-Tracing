#!/usr/bin/env python
"""Score IRT-based predictors (static persistence, T-SKIRT, DriftIRT) with the walk-forward harness.

Usage:
    python scripts/run_models.py --predictors tskirt static_irt --nu 0.1 0.3 --horizon 1 \
        --models olmo2-7b --benchmarks mmlu
    python scripts/run_models.py --predictors drift_irt item_mean --q-theta 0.01 0.05 \
        --q-delta 0.001 0.01 --delta-prior-var 1 --forecast-var-scale 0 1 \
        --residual-scale probit --models olmo2-7b --benchmarks mmlu --tag drift

DriftIRT dials are given as lists; one predictor is built per combination.

Writes data/processed/models_<tag>_h<horizon>.parquet and prints a per-model summary.
item_mean from the baseline run is included for reference when its results file exists.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import pandas as pd

from llm_kt.baselines import ItemMean
from llm_kt.drift_irt import RESIDUAL_SCALES, DriftIRT
from llm_kt.evaluate import evaluate, summarize
from llm_kt.ingest import MODELS
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
                        choices=["static_irt", "tskirt", "drift_irt", "item_mean"])
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
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-history", type=int, default=2)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    responses = pd.read_parquet(args.inp)
    responses = responses[responses["model"].isin(args.models)]
    if args.benchmarks:
        responses = responses[responses["benchmark"].isin(args.benchmarks)]

    params = load_item_params()
    predictors = []
    if "item_mean" in args.predictors:
        predictors.append(ItemMean())
    # one predictor per tau when recalibrating (names carry tau and tau_a); tau is unused otherwise
    taus = args.tau if args.recalibrate else args.tau[:1]
    recal = {"recalibrate": args.recalibrate, "tau_a": args.tau_a}
    if "static_irt" in args.predictors:
        predictors += [StaticIRTPersistence(params, window=args.window, tau=tau, **recal) for tau in taus]
    if "tskirt" in args.predictors:
        predictors += [TSKIRT(params, nu=nu, time_col=args.time_col, tau=tau, **recal) for nu in args.nu for tau in taus]
    if "drift_irt" in args.predictors:
        predictors += [
            DriftIRT(params, q_theta=qt, q_delta=qd, delta_prior_var=pv, forecast_var_scale=fv,
                     time_col=args.time_col, residual_scale=rs)
            for qt, qd, pv, fv, rs in itertools.product(
                args.q_theta, args.q_delta, args.delta_prior_var, args.forecast_var_scale,
                args.residual_scale)
        ]

    t0 = time.time()
    results = evaluate(responses, predictors, min_history=args.min_history, horizon=args.horizon)
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
