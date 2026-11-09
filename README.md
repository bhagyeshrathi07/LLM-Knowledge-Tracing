# Knowledge Tracing for Large Language Models

CMPE 295A/B master's project, SJSU. Bhagyesh Rathi, Ronak Patel, Aniket Sarrin. Advisor: Dr. Mahima Suresh Agumbe.

We apply knowledge tracing, the learner model used inside intelligent tutoring systems, to LLMs across their training checkpoints. Each checkpoint is a moment in the learner's history; the model's answers to skill-tagged benchmark items at that checkpoint are evidence of what it knows. The output is a per-skill mastery estimate over training, validated by predicting held-out checkpoints.

See `docs/project-notes.md` for background, literature review, design decisions and roadmap. See `docs/abstract-v2.md` for the current abstract.

## Layout

```
src/llm_kt/       package code (data ingest, models, evaluation)
scripts/          command-line entry points
tests/            pytest suite
data/raw/         downloaded source data (gitignored)
data/processed/   canonical Parquet tables (gitignored)
docs/             notes, abstract, reports
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Phase 1 data

Source: AI2 Fluid Benchmarking release, `allenai/fluid-benchmarking` on HuggingFace. Per-item, per-checkpoint correctness for six models (Amber-7B, K2-65B, OLMo1-7B, OLMo2-7B, Pythia-2.8B, Pythia-6.9B) on six benchmarks.

Build the canonical table:

```bash
python scripts/ingest_fluid.py
```

Writes `data/processed/fluid_responses.parquet`, long format, one row per (model, checkpoint, item). TruthfulQA is dropped because its scores are continuous rather than 0/1.

## Baselines and evaluation

Per-skill accuracy curves (raw accuracy with a Wilson interval, no learner model):

```bash
python scripts/plot_skill_curves.py --model olmo2-7b --skill high_school_biology
```

Writes `data/processed/skill_curves.parquet` and two figures under `docs/figures/`.

Next-checkpoint prediction harness. Walks forward through each model's checkpoints; at each origin the predictor sees all responses up to that checkpoint and predicts P(correct) for every item at the next one (`--horizon` for further ahead). Scored with log-loss, Brier, accuracy and AUC per (model, target checkpoint, skill).

```bash
python scripts/run_baselines.py --horizon 1
```

Trivial predictors in `src/llm_kt/baselines.py`: `global_mean`, `skill_mean`, `item_mean`, `copy_last`, `copy_last_calibrated`. The strongest is `item_mean` (pooled h=1: log-loss 0.360, within-skill AUC 0.80; MMLU only: 0.483 / 0.79). A learner model must beat it on the same benchmark subset and horizon to have learned something the raw data did not already say. AUC is computed within (model, checkpoint, skill) slices, so any predictor that is constant within a skill scores 0.50 by construction. To add a model, implement the `Predictor` protocol in `src/llm_kt/evaluate.py` (it receives the forecast `horizon`) and pass it to `evaluate()`. Results are written to `data/processed/baseline_results_h{horizon}.parquet`.

## Learner models

`scripts/run_models.py` scores the learner models with the same harness: static IRT persistence, T-SKIRT, DriftIRT, BKT (`src/llm_kt/bkt.py`, a per-skill two-state model with one transition per checkpoint, batched log-space emissions and per-item guess/slip shrunk toward an anchor). Learner models run on MMLU only; the other benchmarks stay for the ladder, transfer tests and controls.

```bash
# the h = 1 grid as run on 2026-10-27 (about 10 min per setting; split --kappa / --anchor over
# several processes with different --tag values and concatenate into models_bkt_grid_h1.parquet)
python scripts/run_models.py --models olmo2-7b --benchmarks mmlu \
    --predictors bkt item_mean skill_mean --kappa 0.03 0.1 0.3 1 5 20 100 1e6 --rho 1.0 0.9 0.7 \
    --anchor skill fluid --theta-lo -1 -2 --theta-hi 1 2 --horizon 1 --tag bkt_grid
# the selected settings further ahead (models_bkt_best_h{5,20}.parquet; about 70 min per setting at h = 20)
for h in 5 20; do
  python scripts/run_models.py --models olmo2-7b --benchmarks mmlu --predictors bkt item_mean skill_mean \
      --anchor skill --kappa 0.1 1 1e6 --rho 0.9 --horizon $h --tag bkt_bestS
  python scripts/run_models.py --models olmo2-7b --benchmarks mmlu --predictors bkt \
      --anchor fluid --theta-lo -2 --theta-hi 1 --kappa 1 --rho 0.9 --horizon $h --tag bkt_bestF
done   # then concatenate bkt_bestS and bkt_bestF into models_bkt_best_h{5,20}.parquet
python scripts/run_models.py --models olmo2-7b --benchmarks mmlu \
    --predictors item_mean_ewma --rho 0.9 --horizon 1 --tag bkt_ewma   # recency-weighted item mean, the control (also at --horizon 5 and 20)
python scripts/analyze_bkt.py --mastery   # selects on validation, compares with item mean and the control,
                                          # writes docs/results/bkt/ and the four figures in docs/figures/bkt_*.png
python scripts/analyze_bkt.py --replot-mastery   # redraw the two mastery figures from the stored CSVs (no refit)
```

Rescoring against the recency-weighted bar (from Sprint 5 the bar is `item_mean_ewma` with `rho` chosen on validation) and the tokens-seen clock:

```bash
# the bar: rho grid at h = 1 (about 1.5 min), then the chosen rho (0.9) further ahead
python scripts/run_models.py --models olmo2-7b --benchmarks mmlu --predictors item_mean item_mean_ewma \
    --rho 0.5 0.7 0.8 0.9 0.95 0.99 --horizon 1 --tag ewma_grid
for h in 5 20; do
  python scripts/run_models.py --models olmo2-7b --benchmarks mmlu --predictors item_mean item_mean_ewma \
      --rho 0.9 --horizon $h --tag ewma_best
done
# sensitivity control, half-life 8 checkpoints (rho = 0.5 ** (1/8)); not the bar, which is the best of the grid above
python scripts/run_models.py --models olmo2-7b --benchmarks mmlu --predictors item_mean_ewma \
    --rho 0.917004 --horizon 1 --tag ewma_hl8
# tokens-seen clock, h = 1 grids. T-SKIRT's nu is a standard deviation per sqrt(time), so a 42x longer
# clock rescales it by 1/sqrt(42); DriftIRT's q values are variances per unit time, rescaled by 1/42.
# analyze_bar.py reads exactly models_clock_tskirt_grid_h1, models_clock_drift_grid_h1 and
# models_clock_best_h{5,20}.parquet; the concatenation steps below produce the last three.
# T-SKIRT: about 4.5 min per nu, all seven in one process (about 30 min)
python scripts/run_models.py --models olmo2-7b --benchmarks mmlu --predictors tskirt --recalibrate \
    --tau 2 --tau-a 0.5 --time-col tokens_seen_b --nu 0.0119048 0.0238095 0.047619 0.0952381 0.0771517 0.154303 0.308607 \
    --horizon 1 --tag clock_tskirt_grid
# DriftIRT: the 3 x 3 grid, then the extra point added when the optimum sat at the lower q_theta edge
python scripts/run_models.py --models olmo2-7b --benchmarks mmlu --predictors drift_irt --time-col tokens_seen_b \
    --q-theta 0.0000714286 0.000238095 0.000714286 --q-delta 0.0000714286 0.000238095 0.000714286 \
    --delta-prior-var 1 --forecast-var-scale 0 --residual-scale probit --horizon 1 --tag clock_drift_grid9
python scripts/run_models.py --models olmo2-7b --benchmarks mmlu --predictors drift_irt --time-col tokens_seen_b \
    --q-theta 0.0000238095 --q-delta 0.000238095 \
    --delta-prior-var 1 --forecast-var-scale 0 --residual-scale probit --horizon 1 --tag clock_drift_extra
python -c "import pandas as pd; d = 'data/processed/'; pd.concat([pd.read_parquet(d + f'models_{t}_h1.parquet') \
    for t in ('clock_drift_grid9', 'clock_drift_extra')]).to_parquet(d + 'models_clock_drift_grid_h1.parquet', index=False)"
# the settings chosen on validation, at h = 5 and 20, one file per horizon
for h in 5 20; do
  python scripts/run_models.py --models olmo2-7b --benchmarks mmlu --predictors tskirt --recalibrate \
      --tau 2 --tau-a 0.5 --nu 0.154303 --time-col tokens_seen_b --horizon $h --tag clock_best_tskirt
  python scripts/run_models.py --models olmo2-7b --benchmarks mmlu --predictors drift_irt --time-col tokens_seen_b \
      --q-theta 0.0000238095 --q-delta 0.000238095 --delta-prior-var 1 --forecast-var-scale 0 \
      --residual-scale probit --horizon $h --tag clock_best_drift
  python -c "import pandas as pd; d = 'data/processed/'; pd.concat([pd.read_parquet(d + f'models_{t}_h$h.parquet') \
      for t in ('clock_best_tskirt', 'clock_best_drift')]).to_parquet(d + 'models_clock_best_h$h.parquet', index=False)"
done
python scripts/analyze_bar.py   # chooses rho and the clock settings on validation, rescores every learner model
                                # against the bar, adds the half-life-8 sensitivity and the matched-parameter
                                # clock rows; writes docs/results/bar/ and docs/figures/bar_rescoring.png
```

DriftIRT ablation (`docs/superpowers/specs/2026-11-03-driftirt-ablation.md`): which ingredient carries DriftIRT's lead over the bar. OLMo-2 MMLU, h = 1; each arm is one `run_models.py` call with a DriftIRT ablation switch, writing `models_abl_<arm>_h1.parquet` (logs in `data/processed/logs/abl_<arm>_h1.log`). Each grid covers both analyses: the matched setting (`q_delta` 0.01, `q_theta` 0.01 or 0) is in every grid. Run as five parallel processes, about 1.5 min per arm without a state and 5 to 6 min for the nine-setting grids; one process needs about 1.6 GB.

```bash
D="python scripts/run_models.py --models olmo2-7b --benchmarks mmlu --predictors drift_irt \
   --delta-prior-var 1 --forecast-var-scale 0 --residual-scale probit --horizon 1"
$D --q-theta 0 --q-delta 0.003 0.01 0.03 --no-skill-state --flat-items --tag abl_L1   # L1, names end _noskill_flat
$D --q-theta 0 --q-delta 0.003 0.01 0.03 --no-skill-state --tag abl_L2                # L2, _noskill
$D --q-theta 0 --q-delta 0.003 0.01 0.03 --tag abl_L3                                 # L3, static subject ability
$D --q-theta 0 --q-delta 0.003 0.01 0.03 --global-skill --tag abl_L3g                 # L3g, _gskill
$D --q-theta 0.003 0.01 0.03 --q-delta 0.003 0.01 0.03 --tag abl_L4                   # L4 re-tuning grid
$D --q-theta 0.003 0.01 0.03 --q-delta 0.003 0.01 0.03 --global-skill --tag abl_L4g   # L4g
$D --q-theta 0.003 0.01 0.03 --q-delta 0.003 0.01 0.03 --shuffle-skills 0 --tag abl_L4s   # L4s, _shuf0
$D --q-theta 0.01 --q-delta 0.01 --shuffle-skills 1 2 3 --tag abl_L4s123              # shuffle seeds 1-3, matched
$D --q-theta 0.003 --q-delta 0.01 --shuffle-skills 1 2 3 --tag abl_L4s123_retuned     # seeds 1-3 at seed 0's
                                                                                       # validation winner
python scripts/analyze_ablation.py   # matched (primary) and re-tuned analyses, writes docs/results/ablation/
                                     # and docs/figures/driftirt_ablation.png
```

The archived L4 (`drift_irt_qt0.01_qd0.01_pv1_fv0_zres` in `models_drift_grid_h1.parquet`) and the bar (`item_mean_ewma_r0.9` in `models_ewma_grid_h1.parquet`) are read from the earlier runs. Seed 0's validation winner for L4s was `q_theta` 0.003, `q_delta` 0.01, so seeds 1 to 3 needed the second shuffle run above (45 configurations in all, 1.5 min for that run). Logs: `data/processed/logs/abl_*_h1.log`.

## Tests

```bash
pytest
```
