# Design: DriftIRT ablation, what the per-skill state contributes

Date: 2026-11-03, revised 2026-11-03 after two reviews (revision 3). Owner: Ronak. Status:
frozen before any arm result was inspected; implementation follows this text. Follow-up 2 of `docs/project-notes.md` §10,
which withdrew the Workbook 2 claim that DriftIRT's gain over the recency-weighted item
mean comes from the per-skill state.

Changes in revision 3: interaction-row and interpretability wording narrowed; static-arm
grid stated explicitly; `_flat` suffix for L1; four-arm bootstrap cancellation test; L1 − L0
reading narrowed; configuration count corrected; spec frozen.

Changes in revision 2: skill remapping moved inside a predictor wrapper so the harness
always scores the original subjects (§3); theta is removed by an explicit `skill_state`
switch instead of a tiny prior (§2.1, §3); two analyses, matched-parameter (primary) and
re-tuned (secondary), replace the single mixed one (§2.3); the global-state comparison is a
difference-in-differences with a joint bootstrap (§2.2); one primary test, everything else
exploratory, and the universal detection threshold is gone (§2.4); shuffle seeds are a
sensitivity row with a stated averaging rule (§2.4); L0's shrinkage and L1 − L0's two
changed ingredients are stated (§2.1, §2.2); `align_slices` is required to drop nothing
(§2.4).

## 1. Question

On OLMo-2 7B, MMLU, horizon 1, DriftIRT beats the bar (recency-weighted item mean, `rho`
0.9) by 0.011 test log-loss, 95% interval [−0.014, −0.008], with no demonstrated AUC
difference (`docs/results/bar/h1_paired_differences.csv`). DriftIRT differs from the bar
in four ways at once: a probit link with a Gaussian filter instead of a weighted average;
a per-item prior centred on Fluid's 2PL difficulty instead of shrinkage toward the running
accuracy; a random-walk residual update instead of exponential weighting; and a per-skill
ability shared by the items of a subject. The question is which of these carries the
0.011, and in particular whether the per-skill state, the object this project is about,
carries any of it.

This is a pre-specified follow-up on a test block that has already been examined (the §10
rescoring and the October model selection both reported it). The design below fixes the
primary test before any arm is run, but the test block is not fresh and the write-up will
say so.

## 2. Pre-registered design

### 2.1 Arms

All arms are DriftIRT (`src/llm_kt/drift_irt.py`) with `residual_scale="probit"`,
`forecast_var_scale=0`, `delta_prior_var=1`, checkpoint clock, horizon 1, OLMo-2 7B, MMLU.
Each arm removes or neutralises one ingredient; read top to bottom as a ladder from the bar
to the full model.

| Arm | What it is | Skill state | Item prior | How it is built |
|---|---|---|---|---|
| L0 | the bar: `item_mean_ewma_r0.9` | none | one pseudo-observation at the history-wide accuracy (`alpha` 1) | exists |
| L1 | item-only probit random walk, flat item prior | removed (`skill_state=False`) | N(0, 1) on the probit shift, around a = 1, b = 0 | item table with a = PROBIT_SCALE, b = 0 for every item (so a / PROBIT_SCALE = 1) |
| L2 | item-only probit random walk, Fluid prior | removed | N(0, 1) around Fluid's 2PL (a_i, b_i) | Fluid item table |
| L3 | static per-skill ability | per skill, `q_theta` 0 | Fluid 2PL | `q_theta` 0, `theta_prior_var` 1 |
| L3g | static global ability | one state shared by all subjects, `q_theta` 0 | Fluid 2PL | skill remapped to one constant inside the wrapper |
| L4 | full DriftIRT, archived setting | per skill, `q_theta` 0.01 | Fluid 2PL | exists: `drift_irt_qt0.01_qd0.01_pv1_fv0_zres` |
| L4g | drifting global ability | one shared state, `q_theta` 0.01 | Fluid 2PL | skill remapped to one constant |
| L4s | full model, subject labels shuffled | per "subject" after a random permutation of items across subjects, sizes preserved | Fluid 2PL | skill remapped by a seeded permutation (seed 0 primary; 1, 2, 3 sensitivity) |

`skill_state=False` is an explicit mode of `DriftIRT`: theta is held at exactly 0, the
theta Newton block and the theta predict step are skipped, and the forecast variance drops
the theta term. A tiny prior was considered and rejected: with `theta_prior_var` 1e-8 and
1,000 always-correct items, theta reaches 5e-6 at the first checkpoint and 6e-5 by the
94th, so "pinned" would have been false even though the prediction effect is negligible.

### 2.2 What each contrast isolates

Primary contrast in bold. All contrasts are reported as (first arm minus second arm) on
test-block log-loss, matched parameters (§2.3).

| Contrast | Ingredient isolated | Reading if the interval excludes zero (negative = first arm better) |
|---|---|---|
| L1 − L0 | link and filter, *and* the prior treatment (Gaussian N(0,1) on a probit shift versus one pseudo-observation at the running accuracy). Two things change; this step is descriptive only. | the probit random-walk filter with a flat prior beats, matches or loses to the EWMA |
| L2 − L1 | Fluid 2PL prior centre on the item | the borrowed difficulty helps or hurts once the item has its own history |
| L3 − L2 | a static shared offset per subject | a per-subject level, with no drift, adds information beyond the item's own history |
| **L4 − L3** | **drift in the per-skill ability** | **the dynamic part of the skill state, the part the project is about, adds information** |
| L3 − L3g | subject-specific static level versus one global level | static subject differences carry information |
| (L4 − L3) − (L4g − L3g) | whether allowing drift helps the subject model more than the global model | allowing drift provides a larger predictive benefit in the subject model than in the global model, at these matched settings; this does not identify the underlying learning process. Bootstrapped jointly on identical subject draws across the four arms |
| L4 − L4s | the real subject grouping versus a random grouping of equal sizes | the MMLU subject labels carry information beyond what a random partition of the same sizes gives |

The contrasts are not additive (ingredients interact), so the report presents the ladder
and the contrasts, never a percentage decomposition of the 0.011. L4 − L4g alone is not
reported as isolating dynamics, because it changes static subject differences and drift at
once; the difference-in-differences row replaces it.

### 2.3 Two analyses

**Analysis 1, matched parameters (primary).** Every arm uses the archived L4 setting
wherever the ingredient exists: `q_delta` 0.01, `delta_prior_var` 1, `forecast_var_scale`
0; `q_theta` 0.01 for drifting states (L4, L4g, L4s), 0 for static states (L3, L3g), state
removed for L1 and L2. No selection anywhere. Each contrast is then a single intervention
on the archived model and answers "what does removing this ingredient from the model we
reported cost".

**Analysis 2, re-tuned (secondary, exploratory).** Each arm gets its own grid on the
validation block: `q_delta` in {0.003, 0.01, 0.03} for every arm; only the drifting arms
L4, L4g and L4s also search `q_theta` in {0.003, 0.01, 0.03}; L1 and L2 have no theta and
L3 and L3g keep `q_theta` 0 (their theta is learned but does not drift);
`delta_prior_var` and `forecast_var_scale` fixed as above.
Selection by validation log-loss only. L4 follows the same rule (its grid contains the
archived setting; whichever wins on validation is the Analysis 2 L4, and if it differs from
the archived setting the table says so). This answers "how competitive can each reduced
model become after re-tuning", and its contrasts are not single-variable interventions when
the chosen `q_delta` values differ, which the table will show.

Horizon 1 only. At h = 1 the hyperparameters are chosen on targets 56 to 74 and every test
forecast originates at 74 to 92, so no tuning observation is later than a forecast origin.
Horizons 5 and 20 are not part of this ablation. Blocks: targets 0 to 55 train, 56 to 74
validation, 75 to 93 test (`assign_blocks`).

### 2.4 Statistics

- **Primary test:** Analysis 1, L4 − L3, test-block log-loss, paired difference with a
  cluster bootstrap over the 57 subjects, 2,000 replicates, percentile 95% interval
  (`paired_difference` in `src/llm_kt/evaluate.py`). One confirmatory claim; its reading
  is fixed in §2.2. Its own bootstrap SE and interval are what determine what it can
  detect; no detection threshold is assumed in advance. For orientation only, a paired SE
  of 0.0017 (the archived L4 − L0 value) would give 80% power at two-sided 5% for a true
  difference of about 0.0048; the L4 − L3 SE will differ and is reported.
- **Everything else is exploratory:** the other contrasts, both AUC columns, Analysis 2,
  validation and pooled scopes. No multiplicity control is applied; exploratory rows are
  labelled as such in every table and are not used for confirmatory wording.
- **Joint bootstrap:** the difference-in-differences row and every contrast in one table
  are computed on the same 2,000 subject resamples (one seed, one set of draws, applied to
  all arms), so rows can be compared with each other.
- **Slices:** all arms are scored on identical (model, origin, target, skill) slices.
  `align_slices` intersects; the script asserts that its `dropped` table shows zero
  dropped rows for every arm and that per-slice item counts `n` are identical across
  arms, and stops otherwise.
- **Shuffle seeds:** seed 0 is the pre-registered L4s in both analyses. Seeds 1, 2 and 3
  reuse seed 0's parameters (in Analysis 1 all seeds share the archived setting by
  construction; in Analysis 2 seeds 1 to 3 reuse seed 0's validation winner, so 42
  settings are run in total rather than 66). The sensitivity row reports the four L4 − L4s
  differences individually plus their mean and range, averaging differences in log-loss,
  never averaging predictions. A non-significant L4 − L4s supports "no demonstrated
  advantage of the subject labels over these tested partitions", not that groupings are
  interchangeable.
- Wording rules as in §10: "beats" only when the interval excludes zero; otherwise "no
  demonstrated difference"; never "equivalent", never "adds nothing".

### 2.5 Predictions, written before running

- L1 versus L0: expected small; either direction plausible. A loss would mean the EWMA
  weighting with its running-accuracy shrinkage is a better forgetting curve than a
  Gaussian random walk with a fixed N(0, 1) prior at these sample sizes. More precisely,
  a loss establishes only that the evaluated EWMA configuration predicts better than the
  evaluated L1 configuration; the two differ in prior treatment as well as in the update.
- L2 − L1: expected near zero on the test block (by checkpoint 75 every item has 70+
  observations and the prior centre has washed out); a difference pooled over the run
  (early checkpoints) is possible.
- L3 − L2 and L4 − L3: this is the question. The §10 verdict stands either way; the
  purpose is to replace a withdrawn claim with a measured one.
- L4 − L4s: no directional prediction beyond "if the subject labels carry information,
  L4s should not beat L4". The design does not assume that a shuffle penalty must exceed
  the drift benefit.

### 2.6 What this ablation does not do

- It does not re-tune the bar (`rho` is fixed at 0.9, the declared-grid winner; the
  half-life-8 sensitivity control is not repeated here).
- It does not test the forecast-variance choice or the residual scale; both were chosen
  on validation in October and are held fixed.
- It does not run other models or benchmarks (Decision 6).
- A positive primary result would show that the skill drift *predicts*, not that the
  fitted ability curves are *interpretable*. Interpretation of the mastery curves and
  their uncertainty needs its own validation; the held-out transfer test (Aniket) speaks
  to generalisation, not to that.

## 3. Implementation

Small changes, all behind options that default to current behaviour.

- `src/llm_kt/drift_irt.py`: `skill_state: bool = True`. When False: `_add_skills` still
  records skills (for bookkeeping) but theta means stay 0 and are never updated;
  `filter_update` is called with a flag that skips the theta Newton block; the predict
  step skips theta variance; the forecast variance drops the theta term; the predictor name
  gains `_noskill`. `q_theta` must be 0 when `skill_state` is False (raise otherwise).
- `src/llm_kt/drift_irt.py` or a new `src/llm_kt/skill_remap.py`: a predictor wrapper
  `RemappedSkills(predictor, mapping)` whose `predict(history, target, horizon)` copies
  both frames, replaces `skill` through `mapping` (a dict item_id to new label, or the
  constant for the global arms), calls the inner predictor and returns predictions in the
  target's original row order. The harness never sees the remapped labels, so slices,
  bootstrap clusters and within-skill AUC are computed on the real subjects for every arm.
  Two mappings: `constant_skill(items)` and `shuffled_skills(items_by_subject, seed)`, the
  latter a seeded permutation of items across subjects that preserves every subject's size
  (a bijection on items). The wrapper's name is the inner name plus `_gskill` or `_shuf<seed>`.
- `scripts/run_models.py`: `--flat-items` (replace the Fluid table with a = PROBIT_SCALE,
  b = 0 for every item in the responses; the predictor name gains `_flat`, so L1 is
  `..._noskill_flat` and L2 is `..._noskill`, distinct in every result file),
  `--no-skill-state`, `--global-skill`, `--shuffle-skills SEED [SEED ...]`. Mappings are built once per model from the full
  response table before `evaluate`, so history and target always agree.
- `scripts/analyze_ablation.py`: reads the arm files; Analysis 1 table (ladder and
  contrasts, matched parameters) and Analysis 2 table (validation selection per arm,
  ladder and contrasts, with each arm's chosen setting shown); the difference-in-
  differences row with a joint bootstrap (a small `contrast_bootstrap(results, weights,
  ...)` helper in `evaluate.py` taking a dict predictor to coefficient, sharing one set of
  subject draws); the shuffle-seed sensitivity row; the zero-drop and equal-`n` assertions.
  Writes `docs/results/ablation/{matched_ladder.csv, matched_contrasts.csv,
  retuned_validation.csv, retuned_ladder.csv, retuned_contrasts.csv, shuffle_seeds.csv,
  analysis.txt}` and `docs/figures/driftirt_ablation.png` (Analysis 1 ladder of test
  log-loss with SEs on the left, contrasts with intervals on the right, primary contrast
  marked; same style as `bar_rescoring.png`).
- Tests:
  - `skill_state=False`: theta means are exactly 0 after filtering; predictions equal an
    independent item-only reference computed in the test (a one-item probit random-walk
    filter written out by hand) with nonzero Fluid parameters, so the reference keeps the
    item offset, z = −a_i b_i − delta_i; predictions for an item are invariant to the
    skill labels and to other items' outcomes; and the default `skill_state=True` path
    reproduces the archived `drift_irt_qt0.01_qd0.01_pv1_fv0_zres` predictions on a fixed
    toy table (regression test).
  - Flat item table gives a = 1, b = 0 after the probit conversion.
  - `shuffled_skills`: bijection on items, every subject keeps its size, deterministic
    for a seed, different across seeds.
  - `RemappedSkills`: the inner predictor sees remapped labels (checked with a recording
    stub), the wrapper returns predictions in the target's original order, and the harness
    metrics computed through the wrapper on a toy table equal a manual calculation on the
    original subjects (log-loss and within-skill AUC, not just the labels).
  - `contrast_bootstrap`: with weights {a: 1, b: −1} it reproduces `paired_difference`
    exactly for the same seed; with four arms and weights (1, −1, −1, 1) it returns the
    difference-in-differences point estimate; and on a synthetic table where the four-arm
    contrast cancels within every slice, every bootstrap replicate, the SE and both
    interval ends are exactly zero (this fails if the arms are resampled independently).

Runtime: a DriftIRT setting at h = 1 on OLMo-2 MMLU takes about 20 s. Analysis 1 is 8
settings (L4 exists); Analysis 2 adds up to 45 unique configurations (42 if seed 0's
validation winner is the archived pair, since seeds 1 to 3 then share one fit for both
analyses, otherwise seeds 1 to 3 need a matched and a re-tuned fit each); under 20
minutes in total, parallelisable by arm.

## 4. Outputs and write-up

- `docs/results/ablation/` as above; `docs/project-notes.md` §10 gets a short "Ablation
  result" paragraph replacing the "pending an ablation" sentence, with the Analysis 1
  ladder, the primary contrast with its interval, the difference-in-differences row and
  the shuffle row, each labelled confirmatory or exploratory; `docs/team-briefing.md` §5.1
  gets one sentence.
- The headline is written after the numbers, in the metric-specific form used in §10, and
  states that the test block had been examined before this follow-up was specified.

## 5. Review questions for the reviewer

1. Is the explicit `skill_state=False` mode plus the three invariance tests sufficient to
   call L1 and L2 "no skill state", or is an external item-only reference implementation
   also wanted?
2. Analysis 1 uses `q_delta` 0.01 for every arm. Is there a reason to prefer each arm's
   own validation winner as the primary, given the single-intervention argument?
3. Is one primary test (L4 − L3, log-loss) with all else exploratory the right split, or
   should L3 − L2 also be confirmatory with a Holm correction over two tests?
4. Anything in §2.2's readings that still overreaches?

## Implementation notes (2026-11-06)

Added after the runs; the design sections above are unchanged.

- Seed 0's L4s validation winner was `q_theta` 0.003, `q_delta` 0.01, not the archived pair, so
  seeds 1 to 3 were run twice (matched and re-tuned): 45 configurations, the upper count in §3.
- L4 in Analysis 1 is the archived file; the re-tuning grid reran the archived setting, and its
  per-slice scores are identical to the archive (largest difference 0).
- `contrast_bootstrap` takes `by` (default pooled) and `return_boot` besides the arguments named
  in §3, and `paired_difference` now calls it with weights {a: 1, b: −1}, so the two share one
  code path; a regression test pins `paired_difference` to values recorded before the change.
- `RemappedSkills` takes the name suffix as an argument (`_gskill`, `_shuf<seed>`).
- `run_models.py` now filters the response table while reading it (model and benchmark), which
  cuts a process from about 3 GB to 1.6 GB so arms could run in parallel; results are unchanged.
- Runtime: about 7 min wall clock for the eight arm runs in five parallel processes, plus 1.5 min
  for the re-tuned shuffle seeds; the analysis script takes a few seconds.
