# Design: log-space BKT with per-item emissions

Date: 2026-10-27. Owner: Bhagyesh. Action item from the 2026-10-19 advisor meeting
(`meetings/2026-10-19-advisor.md`): "Implement log-space BKT with forgetting and per-item
emissions (`src/llm_kt/bkt.py` + tests); score against the item-mean bar at h = 1 and h = 5 on
OLMo-2 MMLU."

## 1. Intent

Bayesian Knowledge Tracing (BKT) is the second of the two primary model families adopted in
Decision 2 (`docs/project-notes.md` §6): a per-skill hidden Markov model with an explicit learn
rate, forget rate, guess and slip. T-SKIRT and DriftIRT cover the temporal-IRT family; BKT is
the family the paper's forgetting story rests on, and nothing in the repository implements it
yet. The question it answers: does a two-state learner with forgetting, read once per
checkpoint from a batch of items, predict the next checkpoints as well as the item-mean bar,
and what does it say about when each MMLU subject was learned?

## 2. Why these choices (the "why" for Ronak and Aniket)

- **One transition per checkpoint, batched observations.** Decision 1. The model does not
  learn from the test; all items at a checkpoint are simultaneous noisy readings of the
  state at that checkpoint.
- **Log space.** pyBKT's E-step multiplies raw emission probabilities before normalising.
  With 100 to 1,534 items per checkpoint the product underflows to zero, so pyBKT cannot be
  used (`docs/lit-review-2026-09-30.md` §9b). We implement the forward-backward recursion
  on log probabilities with `logsumexp`.
- **Forgetting.** A forget rate is a native BKT parameter and is what makes the state able
  to go down. The hidden state is a Markov chain, so old evidence fades on its own; the
  item mean never forgets. Ronak reports that recency carried most of DriftIRT's gain (his
  analysis, not yet recorded in the repository), so BKT may do better than the AUC argument
  below suggests.
- **Per-item guess and slip.** With one guess/slip pair per skill, every item in a subject
  receives the same probability at a given checkpoint, and our within-subject AUC is 0.50
  by construction (ties get average rank). It would still beat the skill-mean baseline on
  log-loss, but it would lose to item mean on both metrics because it carries no per-item
  information. So each item gets its own guess and slip.
- **Shrinkage, because free per-item emissions overfit.** 14,042 MMLU items times two
  parameters is 28,084 numbers, each seen at most 94 times for one model. Fitted freely
  they reproduce each item's own history (which is what item mean does) and the skill state
  has nothing left to explain; the state and the item parameters also trade off, so it is
  an identifiability problem. Each item's guess and slip therefore start at an anchor and
  move away only as evidence accumulates, controlled by a pseudo-count `kappa`.
- **Time-varying per-item emissions.** Recalibration (`recalibrate.py`) fits fixed per-item
  offsets, refitted at every origin; DriftIRT lets them drift. Ronak's reported recency
  finding makes time-varying the stronger bet, so the per-item update uses
  recency weights `rho^(origin - t)`. `rho = 1` recovers the fixed (recalibration-style)
  variant for free, so both are one knob apart and both are scored.
- **Two anchors.** `anchor = "skill"`: the item starts at its skill's shared guess and slip
  (learned by EM). `anchor = "fluid"`: the item starts at the guess and slip implied by
  Fluid's published 2PL parameters at two global ability levels. The Fluid anchor with
  `kappa` large is the cheap "borrowed parameters" variant (option 1 in the discussion); the
  skill anchor with finite `kappa` is option 2. Scoring both shows what learning from the
  model's own history adds, the same comparison that mattered for T-SKIRT.
- **MMLU only, OLMo-2 first.** Decision 6 (Ronak, 2026-10-27, on `ronak_branch`, not yet
  merged to `main`): learner
  models run on MMLU only; the other benchmarks stay for the ladder, transfer tests and
  controls. The run uses `--benchmarks mmlu`, as every learner-model run so far has.

## 3. The model

For one model and one skill `s`, checkpoints `t = 0..T` (checkpoint index is the clock;
`step` and `tokens_seen_b` are not used, see §7), items `i` in `s`:

```
state        z_t in {0 (not learned), 1 (learned)}
initial      P(z_0 = 1) = pi_s
transition   P(z_{t+1} = 1 | z_t = 0) = learn_s        P(z_{t+1} = 0 | z_t = 1) = forget_s
emission     P(y_it = 1 | z_t = 1) = 1 - slip_i        P(y_it = 1 | z_t = 0) = guess_i
batch        log P(y_t | z) = sum_i log P(y_it | z)     over the items observed at t
```

Prediction for a target `h` checkpoints after the origin: propagate the filtered
`P(z_origin | y_{0..origin})` through the transition matrix `h` times, then

```
p_i = P(z = 1) (1 - slip_i) + P(z = 0) guess_i
```

The h-step propagation moves `P(z = 1)` toward its stationary value, so longer horizons
are automatically less certain. No separate forecast-variance knob.

Constraint: the learned state must be the better one, `guess < 1 - slip`, enforced at
skill level and item level by clipping `guess <= 1 - slip - 0.05`.

## 4. Fitting

At each origin the predictor sees the history up to that checkpoint and runs MAP-EM per
skill, warm-started from the previous origin's parameters (an incremental cache keyed by the
same fingerprint DriftIRT uses; reset when the history is not an extension of the cached
one).

- **E-step.** Log-space forward-backward on the batched emissions gives `gamma_t = P(z_t = 1 |
  y)` and the pairwise transition posteriors.
- **M-step, skill level.** `pi`, `learn`, `forget`, and the skill's shared `guess_s`,
  `slip_s` from expected counts, each with a weak Beta(1, 1)-style pseudo-count prior so a
  two-checkpoint history cannot drive a rate to exactly 0 or 1.
- **M-step, item level.** With recency weight `w_t = rho^(origin - t)` and anchor values
  `(g_i^0, s_i^0)`:

  ```
  guess_i = (sum_t w_t (1 - gamma_t) y_it     + kappa g_i^0) / (sum_t w_t (1 - gamma_t) + kappa)
  slip_i  = (sum_t w_t gamma_t (1 - y_it)     + kappa s_i^0) / (sum_t w_t gamma_t       + kappa)
  ```

  `kappa -> inf` gives per-skill BKT (every item at its anchor). `rho = 1` gives fixed
  per-item emissions. The recency-weighted M-step is a forgetting-factor approximation, not
  exact EM; it is documented as such.
- **Anchors.** `skill`: `g_i^0 = guess_s`, `s_i^0 = slip_s` (re-estimated each EM
  iteration). `fluid`: `g_i^0 = P_2PL(theta_lo; a_i, b_i)`, `1 - s_i^0 = P_2PL(theta_hi; a_i,
  b_i)` with global `theta_lo < theta_hi` chosen on validation; items without Fluid
  parameters use the skill anchor.
- **Convergence.** Up to 50 EM iterations, stop when the mean log-likelihood per observation
  changes by less than 1e-5. Warm start keeps this to a few iterations per origin after the
  first.
- **Initial values (cold start).** `pi = 0.2, learn = 0.1, forget = 0.02, guess_s = 0.25,
  slip_s = 0.3` (4-option chance for guess).

Fallbacks follow T-SKIRT and DriftIRT: an item unseen in the history uses its anchor values;
a skill unseen in the history uses the last checkpoint's global accuracy.

## 5. Interpretable output

`BKT.mastery()` returns the filtered `P(z_t = 1)` per (skill, checkpoint) for the last fitted
history, plus the skill parameters. `scripts/analyze_bkt.py` plots it for OLMo-2 as the
per-subject mastery figure Decision 2 promised ("binary mastery state with probability; easy
to visualize").

## 6. Evaluation

- Harness: `llm_kt.evaluate` unchanged. OLMo-2, MMLU, `min_history = 2`.
- Grid on `h = 1`, selected on the validation block only: `kappa` in {1, 5, 20, 100, 1e6},
  `rho` in {1.0, 0.9, 0.7}, anchor in {skill, fluid}; for the Fluid anchor `(theta_lo,
  theta_hi)` in {(-1, 1), (-2, 1), (-1, 2)}. The grid is run once; if it is too slow the
  Fluid grid is cut first.
- Report, for the best skill-anchored and best Fluid-anchored setting: pooled, validation
  and test log-loss and AUC with skill-clustered bootstrap SEs, and paired differences
  against `item_mean` with 95% intervals, at `h = 1`, `5` and `20` (the same layout as
  `analyze_drift_irt.py`). Test numbers are reported, never ranked.
- The per-skill BKT limit (`kappa = 1e6`, skill anchor) is reported as well, because its
  within-skill AUC of 0.50 and its log-loss against skill mean are the two facts the
  per-item design was motivated by.

## 7. Non-goals and anti-goals

- No alternative clocks (`step`, `tokens_seen_b`): the transition is per checkpoint. Uneven
  spacing is noted as a limitation; T-SKIRT and DriftIRT can use the other clocks if needed.
- No multi-model fitting: parameters are per (model, skill), as for the other learner
  models. Pooling across the six models is a possible follow-up, not this build.
- No deep KT, no pyBKT dependency.
- Anti-goal: a result that beats item mean only because `kappa` is tiny and the item
  parameters have memorised the item histories. The write-up must state the selected
  `kappa` and show the per-skill mastery curves, so a reader can see how much work the state
  is doing.

## 8. Deliverables

1. `src/llm_kt/bkt.py`: `log_forward_backward`, `fit_skill` (MAP-EM), class `BKT`
   implementing the `Predictor` protocol with `name` like
   `bkt_k{kappa}_r{rho}_{anchor}[_lo{theta_lo}_hi{theta_hi}]`, incremental cache,
   `mastery()`.
2. `tests/test_bkt.py`: forward-backward against brute-force enumeration on a tiny chain;
   no underflow with 2,000 items per checkpoint; EM recovers learn/forget/guess/slip on a
   simulated skill; per-skill limit gives identical probabilities within a skill; per-item
   version ranks an always-right item above an always-wrong one; `rho < 1` tracks a recent
   flip while `rho = 1` sits near the running mean; longer horizon moves predictions toward
   the stationary value; unseen item and unseen skill fallbacks; cache reset on a different
   history; runs in `evaluate`; name and argument validation.
3. `scripts/run_models.py`: `bkt` predictor with `--kappa`, `--rho`, `--anchor`,
   `--theta-lo`, `--theta-hi` list flags, one predictor per combination.
4. `scripts/analyze_bkt.py`: selection on validation, comparison tables, mastery figure
   `docs/figures/bkt_mastery_olmo2-7b.png`.
5. Write-up: `docs/project-notes.md` new section (design, why, results, what it means for
   the "per-item state is what matters" question); `docs/team-briefing.md` results
   subsection, repo map, status table; README run command.

## 9. Builder's calls (made after the builder check, 2026-10-27)

1. Selection and reporting mirror `analyze_drift_irt.py`: rank on validation log-loss at
   h = 1; reuse the selected settings for h = 5 and h = 20; n_boot = 200; paired
   differences vs `item_mean`. The h = 5 and h = 20 runs contain only the selected
   skill-anchored setting, the selected Fluid-anchored setting, the per-skill limit,
   `item_mean` and `skill_mean`.
2. `skill_mean` is added as a `run_models.py` predictor option so the per-skill limit can be
   compared with it on identical rows (no baseline results parquet is committed).
3. Fluid anchors use the logistic 2PL directly (`static_irt.prob_logistic`); `theta_lo`,
   `theta_hi` are on Fluid's logistic scale. The `guess <= 1 - slip - 0.05` rule clips guess
   downward, at both the anchor and the estimate. (On MMLU the clip binds on 2 to 8% of the
   Fluid anchors depending on the theta pair; guess never exceeds 1 - slip.)
4. At `kappa = 1e6` the `rho` values give identical predictors; the grid is deduplicated to
   one per anchor / theta pair.
5. EM runs forward-backward over the full history at each origin; the warm start reuses
   parameters only. Because the recency-weighted item step is not an exact EM ascent step,
   the convergence test may oscillate at `rho < 1`; the 50-iteration cap stands and
   iteration counts are logged.
6. The mastery figure uses the selected skill-anchored setting fitted once on the full
   OLMo-2 history, with a per-subject learn / forget table alongside.
7. Facts verified on the data: OLMo-2 MMLU is fully dense (14,042 items at all 94
   checkpoints, 57 subjects at every checkpoint), and all MMLU items have Fluid parameters,
   so the unseen-item, unseen-skill and no-Fluid-parameter fallbacks never fire on this data.
   They are still implemented and tested.
8. The new project-notes section is appended at the end of the file to avoid conflicting
   with Decision 6 on `ronak_branch`; Decision 6 is cited as pending merge.

## 10. Left to the builder

Exact vectorisation of the per-skill E/M steps (items x checkpoints matrices per skill are
the obvious layout); how the warm-start cache stores per-skill parameters; the plotting
layout of the mastery figure; test tolerances.

## 11. Results (2026-10-27)

Written up in `docs/project-notes.md` §9 and `docs/team-briefing.md` §5.8. Tables and the
console report are stored under `docs/results/bkt/`; the figures, all drawn by
`scripts/analyze_bkt.py`, are `docs/figures/bkt_validation_grid.png` (the grid),
`bkt_horizons.png` (further ahead), `bkt_mastery_olmo2-7b.png` (selected setting) and
`bkt_mastery_olmo2-7b_k1e+06_r1.png` (per-skill limit). One addition to the
plan: a recency-weighted item mean (`ItemMeanEWMA` in `baselines.py`, `--predictors
item_mean_ewma`) was run as the control the anti-goal in §7 asks for, and `analyze_bkt.py`
pairs the selected setting against it.
