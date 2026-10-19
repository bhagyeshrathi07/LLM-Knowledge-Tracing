#!/usr/bin/env python
"""Walk-forward calibration of recalibrated static IRT and T-SKIRT, against item_mean.

Usage:
    python scripts/run_calibrated.py --models olmo2-7b --benchmarks mmlu --horizon 1 \
        --k 5 --method temperature

Predictors: item_mean, static IRT persistence (window 1, recalibrated with tau
--static-tau, discrimination fixed), T-SKIRT (--nu, --tau, --tau-a, recalibrated), and
each IRT predictor wrapped in WalkForwardCalibrated(k, method). The wrapped versions call
their base predictor about (k + 1) times per origin, so this run is slow.

Writes data/processed/models_calibrated_h<horizon>.parquet and prints summaries (pooled
and per block, cluster-bootstrap SEs), the fitted calibration parameters, and paired
differences (calibrated T-SKIRT minus item_mean, test block).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from llm_kt.baselines import ItemMean
from llm_kt.calibrate import METHODS, WalkForwardCalibrated
from llm_kt.evaluate import assign_blocks, evaluate, paired_difference, summarize
from llm_kt.ingest import MODELS
from llm_kt.static_irt import StaticIRTPersistence, load_item_params
from llm_kt.tskirt import TSKIRT

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "data" / "processed" / "fluid_responses.parquet"


class Timed:
    """Pass-through wrapper that prints progress every `every` calls."""

    def __init__(self, inner, every: int = 10):
        self.inner, self.every, self.name = inner, every, inner.name
        self.calls, self.t0 = 0, None

    def predict(self, history, target, horizon=1):
        if self.t0 is None:
            self.t0 = time.time()
        p = self.inner.predict(history, target, horizon=horizon)
        self.calls += 1
        if self.calls % self.every == 0:
            el = time.time() - self.t0
            print(f"  {self.name}: {self.calls} origins, {el:.0f}s ({el / self.calls:.1f}s/origin)", flush=True)
        return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--benchmarks", nargs="+", default=None, help="restrict to these benchmarks (e.g. mmlu)")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-history", type=int, default=2)
    parser.add_argument("--nu", type=float, default=1.0, help="T-SKIRT drift")
    parser.add_argument("--tau", type=float, default=2.0, help="T-SKIRT prior sd of the difficulty offset")
    parser.add_argument("--tau-a", type=float, default=0.5, help="T-SKIRT prior sd of the log-discrimination offset")
    parser.add_argument("--static-tau", type=float, default=1.0, help="static IRT prior sd of the difficulty offset")
    parser.add_argument("--k", type=int, default=5, help="past origins used to fit the calibration map")
    parser.add_argument("--method", default="temperature", choices=METHODS)
    args = parser.parse_args()

    responses = pd.read_parquet(args.inp)
    responses = responses[responses["model"].isin(args.models)]
    if args.benchmarks:
        responses = responses[responses["benchmark"].isin(args.benchmarks)]

    params = load_item_params()
    static = StaticIRTPersistence(params, window=1, recalibrate=True, tau=args.static_tau)
    tskirt = TSKIRT(params, nu=args.nu, recalibrate=True, tau=args.tau, tau_a=args.tau_a)
    cal = [WalkForwardCalibrated(base, k=args.k, method=args.method) for base in (static, tskirt)]
    predictors = [ItemMean(), static, tskirt, *cal]

    t0 = time.time()
    results = evaluate(responses, [Timed(p) for p in predictors], min_history=args.min_history, horizon=args.horizon)
    results["horizon"] = args.horizon
    out = REPO_ROOT / "data" / "processed" / f"models_calibrated_h{args.horizon}.parquet"
    results.to_parquet(out, index=False)
    print(f"\nWrote {len(results):,} rows to {out} in {time.time() - t0:.0f}s\n")

    pd.set_option("display.width", 220)
    print("Fitted calibration parameters across origins (all models pooled)")
    for w in cal:
        fitted = [f["params"] for f in w.fits if f["params"] is not None]
        print(f"  {w.name}: {len(fitted)}/{len(w.fits)} origins calibrated")
        for key in (fitted[0] if fitted else {}):
            v = np.array([f[key] for f in fitted])
            print(f"    {key}: median {np.median(v):.3f}, IQR [{np.percentile(v, 25):.3f}, "
                  f"{np.percentile(v, 75):.3f}], min {v.min():.3f}, max {v.max():.3f}")

    # blocks are assigned per model from that model's checkpoint count (as in evaluate_model)
    n_ckpt = responses.groupby("model")["checkpoint_idx"].max() + 1
    ranges = {m: assign_blocks(int(n)) for m, n in n_ckpt.items()}
    results["block"] = [
        next(b for b, (lo, hi) in ranges[m].items() if lo <= t <= hi)
        for m, t in zip(results["model"], results["target_idx"])
    ]

    cols = ["n", "log_loss", "log_loss_se", "brier", "brier_se", "auc", "auc_se", "accuracy"]
    print(f"\nhorizon={args.horizon}, benchmarks={args.benchmarks or 'all'}, k={args.k}, method={args.method}")
    print("\nPooled over all targets")
    print(summarize(results, by=["model", "predictor"], n_boot=300)
          .set_index(["model", "predictor"])[cols].round(4).to_string())
    for block in ("train", "val", "test"):
        print(f"\nBlock: {block}")
        print(summarize(results[results["block"] == block], by=["model", "predictor"], n_boot=300)
              .set_index(["model", "predictor"])[cols].round(4).to_string())

    test = results[results["block"] == "test"]
    print(f"\nPaired difference, test block: {cal[1].name} minus item_mean")
    for metric in ("log_loss", "auc"):
        d = paired_difference(test, cal[1].name, "item_mean", metric=metric, n_boot=1000)
        print(f"  {metric}:\n{d.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
