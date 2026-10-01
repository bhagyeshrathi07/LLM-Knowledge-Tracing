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

## Tests

```bash
pytest
```
