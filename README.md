# Knowledge Tracing for Large Language Models

CMPE 295A/B master's project, SJSU. Bhagyesh Rathi, Ronak Patel, Aniket Sarrin. Advisor: Dr. Mahima Suresh Agumbe.

We apply knowledge tracing, the learner model used inside intelligent tutoring systems, to LLMs across their training checkpoints. Each checkpoint is a moment in the learner's history; the model's answers to skill-tagged benchmark items at that checkpoint are evidence of what it knows. The output is a per-skill mastery estimate over training, validated by predicting held-out checkpoints.

Start with `docs/team-briefing.md` (what we found, in plain language, with the status of every action item), then `docs/project-notes.md` (background, literature review, design decisions, the BKT write-up in §9, roadmap). `docs/abstract-v2.md` is the current abstract.

## Layout

```
src/llm_kt/       package code: ingest, curves, metrics, evaluation harness, baselines,
                  static_irt, tskirt, recalibrate, calibrate, drift_irt, bkt, forgetting, ladder
scripts/          command-line entry points (one docstring each with its own usage)
tests/            pytest suite
data/raw/         downloaded source data (gitignored)
data/processed/   canonical Parquet tables and per-run result tables (gitignored, rebuilt by the scripts)
docs/             notes, briefing, abstract, lit review, workbook
docs/figures/     committed figures (skill curves, ladder, BKT)
docs/results/     committed result tables per experiment (currently docs/results/bkt/)
docs/superpowers/ design specs and implementation plans (BKT: specs/2026-10-27-bkt-design.md)
meetings/         one note per advisor or team meeting
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Every command below assumes the virtualenv is active and is run from the repository root.

## Phase 1 data

Source: AI2 Fluid Benchmarking release, `allenai/fluid-benchmarking` on HuggingFace. Per-item, per-checkpoint correctness for six models (Amber-7B, K2-65B, OLMo1-7B, OLMo2-7B, Pythia-2.8B, Pythia-6.9B) on six benchmarks.

Build the canonical table:

```bash
python scripts/ingest_fluid.py
```

Writes `data/processed/fluid_responses.parquet`, long format, one row per (model, checkpoint, item): 452 checkpoints, 27,842 items, 12.6 million rows. TruthfulQA is dropped because its scores are continuous rather than 0/1. MMLU skills are the bare subject names (`anatomy`, `high_school_biology`, ...).

## Baselines and evaluation

Per-skill accuracy curves (raw accuracy with a Wilson interval, no learner model):

```bash
python scripts/plot_skill_curves.py --model olmo2-7b --skill high_school_biology
```

Writes `data/processed/skill_curves.parquet` and two figures under `docs/figures/`.

Next-checkpoint prediction harness (`src/llm_kt/evaluate.py`). Walks forward through each model's checkpoints; at each origin the predictor sees all responses up to that checkpoint and predicts P(correct) for every item `--horizon` checkpoints ahead. Scored with log-loss, Brier, accuracy and AUC per (model, target checkpoint, skill); each model's checkpoints are split into chronological train / validation / test blocks (OLMo-2: 0–55 / 56–74 / 75–93), and settings are always chosen on validation and reported on test. `summarize` gives subject-clustered bootstrap standard errors; `paired_difference` gives a difference with a 95% interval between any two predictors.

```bash
python scripts/run_baselines.py --horizon 1
```

No-learning predictors in `src/llm_kt/baselines.py`: `global_mean`, `skill_mean`, `item_mean`, `item_mean_ewma` (recency-weighted item mean, `rho` per checkpoint of age), `copy_last`, `copy_last_calibrated`. The bar every learner model has to clear is `item_mean`: pooled over all six models and benchmarks, h = 1 log-loss 0.360; OLMo-2 on MMLU, 0.457 log-loss / 0.862 AUC. AUC is computed within (model, checkpoint, skill) slices, so any predictor that is constant within a skill scores 0.50 by construction. To add a model, implement the `Predictor` protocol in `src/llm_kt/evaluate.py` (it receives the forecast `horizon`) and pass it to `evaluate()`. Results go to `data/processed/baseline_results_h{horizon}.parquet`.

Baseline-ladder tables and figures (how much knowing the subject vs. the question is worth): `python scripts/analyze_ladder.py`. Forgetting test with the chance-adjusted check: `python scripts/run_forgetting.py --chance-check`.

## Learner models

`scripts/run_models.py` scores the learner models with the same harness: static IRT persistence (`static_irt`), T-SKIRT (`tskirt`, with `--recalibrate` for per-model item parameters), DriftIRT (`drift_irt`) and BKT (`bkt`). Learner models run on MMLU only (Decision 6); the other benchmarks stay for the ladder, transfer tests and controls. Output: `data/processed/models_<tag>_h<horizon>.parquet`. Companion analysis scripts: `analyze_sweep.py` and `analyze_horizons.py` (T-SKIRT), `run_calibrated.py` (calibration layer), `analyze_drift_irt.py`, `analyze_bkt.py`. Each script's docstring has its full usage.

### BKT (`src/llm_kt/bkt.py`)

A per-skill two-state hidden Markov model: one learn / forget transition per checkpoint, all items at a checkpoint read as one batch in log space (so 1,500 items do not underflow), and per-item guess and slip pulled toward an anchor by a pseudo-count `kappa` with recency weights `rho^age`. Anchors: `skill` (the subject's shared pair, learned by EM) or `fluid` (Fluid's published 2PL parameters at `--theta-lo` / `--theta-hi`). `kappa` at or above 1e5 pins every item to the anchor, which is classic per-skill BKT.

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
# the control: recency-weighted item mean with the same rho, at h = 1, 5 and 20
for h in 1 5 20; do
  python scripts/run_models.py --models olmo2-7b --benchmarks mmlu \
      --predictors item_mean_ewma --rho 0.9 --horizon $h --tag bkt_ewma
done
# selection on validation, comparisons with item mean and the control, all tables and figures
python scripts/analyze_bkt.py --mastery          # ~10 min: also fits the two mastery settings on the full history
python scripts/analyze_bkt.py --replot-mastery   # seconds: redraw the mastery figures from the stored CSVs
```

`analyze_bkt.py` writes everything to `docs/results/bkt/` (validation table, scores and paired differences at h = 1, 5, 20, mastery curves and skill parameters, the per-slice result parquets, and `analysis.txt`, the console report) and four figures to `docs/figures/`: `bkt_validation_grid.png`, `bkt_horizons.png`, `bkt_mastery_olmo2-7b.png`, `bkt_mastery_olmo2-7b_k1e+06_r1.png`.

## Results so far (OLMo-2 on MMLU, one checkpoint ahead)

| Predictor                                     | Test log-loss | Test AUC |
| --------------------------------------------- | ------------- | -------- |
| Item mean (the bar)                           | 0.375         | 0.916    |
| T-SKIRT, recalibrated                         | 0.375         | 0.912    |
| Recency-weighted item mean (control)          | 0.317         | 0.933    |
| BKT, per-item emissions, chosen on validation | 0.310         | 0.933    |
| BKT, classic per-skill                        | 0.623         | 0.500    |

BKT beats the bar at every horizon, but a recency-weighted item mean ties it on the test block, so most of the gain is recency rather than the per-skill state. The full reading, with intervals, is in `docs/team-briefing.md` §5 and `docs/project-notes.md` §9.

## Tests and lint

```bash
pytest
ruff check .
```
