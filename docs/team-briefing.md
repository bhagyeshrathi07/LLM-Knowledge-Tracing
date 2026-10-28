# Team Briefing: Where We Are and What We Do Next

Written 2026-10-19 for Bhagyesh and Aniket, right after the advisor meeting. This replaces the 2026-10-01 briefing. It covers everything in the repository as of today (Sprint 3 code and the Workbook 1 package were pushed to `main` this morning), what the experiments showed, what the advisor said, and what each of us does next. Reference documents are listed at the end. Read this first; go to those for detail.

**Updated 2026-10-27:** the repository map (§4), the action-item status (§7) and the head commit (§9) now reflect the current `main`; the OLMo-2 global-mean log-loss in §5.1 was corrected to 0.69. The results in §5 are otherwise as presented on Oct 19.

---

## 1. The project in plain language

Large language models learn in stages. During training, developers save snapshots (checkpoints), sometimes hundreds. Today each checkpoint gets one benchmark score. That number cannot tell you which skills the model has, when it got each one, whether it later lost any, or how sure anyone should be.

Tutoring software for human students solved this with **knowledge tracing (KT)**: keep a hidden estimate of how well a student knows each skill, update it after every answer, and prove the estimate is any good by predicting the student's next answer before seeing it.

We apply that to LLMs. Each checkpoint is one moment in the model's learning history. Each benchmark question, tagged with a skill, is evidence of what the model knows at that moment. Output: a per-skill mastery curve across training, with uncertainty, validated by predicting later checkpoints.

Two decisions shape everything (adopted 2026-09-29, `docs/project-notes.md` §6):

- **The time step is the checkpoint, not the question.** The model does not learn from the exam; it learns from training data between checkpoints. All questions at one checkpoint are read together as one batch of evidence. Off-the-shelf KT tools assume the opposite, so we adapt them.
- **Simple interpretable models are primary; neural KT is a comparison only.** We have six "students" (six models), not thousands.

---

## 2. Prior work, in one table

Two literature passes (Sep 22 and Sep 30), 49 references, about 45 read in full, two independent audits of the workbook.

| Group of papers                                                                               | What they do                                                                              | What they lack                                                                                 |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Test-scoring statistics applied to AI models (Fluid Benchmarking, Item Response Scaling Laws) | Score each checkpoint like a student taking one exam; one group published the data we use | Each checkpoint scored alone; nothing carries forward, so no notion of a skill learned or lost |
| Studies of models during training (MathCAMPS, Implicit Curriculum, Harmon et al.)             | Show skills appear in a stable order; count questions that flip right-to-wrong            | Charts and counts only; no model of the student, no prediction test                            |
| Student-tracking methods (education, political science)                                       | Built for a few learners watched over many periods, exactly our shape                     | Never applied to an AI model's checkpoints                                                     |

Our claim, worded carefully: *in the sources reviewed through September 30, 2026, we did not identify a method that jointly models a per-skill latent state across an LLM's pretraining checkpoints, treats each checkpoint's responses as a batch, and evaluates item-level predictions on later held-out checkpoints with calibrated uncertainty.* Not "nobody has done this"; a search cannot prove that.

---

## 3. Data

**Phase 1 (now):** `allenai/fluid-benchmarking` on HuggingFace. Six models (Amber-6.7B, K2-65B, OLMo-1 7B, OLMo-2 7B, Pythia-2.8B, Pythia-6.9B), 452 checkpoints, 27,842 items after dropping TruthfulQA (continuous scores), 12,584,584 right/wrong answers. Skills: the 57 MMLU subjects; ARC-Challenge, GSM8K, HellaSwag and WinoGrande count as one skill each. The release also ships pre-fitted 2PL IRT item parameters (difficulty, discrimination) estimated on 102 finished models. No GPU needed. Step and token counts are missing for Amber, K2 and Pythia; checkpoint index is the clock.

**Phase 2 (spring):** OLMo-2 checkpoints on MathCAMPS (real skill tags). Needs GPU; Aniket owns the pilot and cost estimate (Nov 25) and the go/no-go (Dec 12). No-GPU fallbacks: `allenai/DataDecide-eval-instances` and the Implicit Curriculum repository.

---

## 4. What is in the repository

Everything below is on `main` as of 2026-10-27, plus the BKT work on `bhagyesh_branch` (same day). 405 tests pass (`pytest`), `ruff` clean. (At the Oct 19 briefing: 328 tests.)

### Library, `src/llm_kt/`

| Module           | What it does                                                                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ingest.py`      | Six Fluid CSVs to one long Parquet table (columns in the notes below)                                                                                  |
| `curves.py`      | Per-skill accuracy per checkpoint with Wilson intervals                                                                                                |
| `metrics.py`     | Log-loss, Brier, accuracy, AUC                                                                                                                         |
| `evaluate.py`    | Walk-forward harness; chronological train/validation/test blocks; clustered bootstrap SEs; paired differences                                          |
| `baselines.py`   | Six no-learning predictors: global mean, skill mean, item mean, recency-weighted item mean (Oct 27), copy-last, copy-last-calibrated                   |
| `static_irt.py`  | Static IRT persistence: one ability per skill re-estimated from past checkpoints and carried forward                                                   |
| `tskirt.py`      | T-SKIRT: temporal IRT with a drifting ability, probit link, optional forecast variance                                                                 |
| `recalibrate.py` | Per-model recalibration of item difficulty (δ) and discrimination (γ) from the model's own history                                                     |
| `calibrate.py`   | Walk-forward temperature / Platt calibration wrapper                                                                                                   |
| `forgetting.py`  | Peak-to-final McNemar test with Holm and selection adjustment; chance and letter checks (added Oct 27)                                                 |
| `ladder.py`      | Baseline ladder tables and skill-vs-item variance decomposition (added Oct 27)                                                                         |
| `drift_irt.py`   | DriftIRT: per-skill ability plus drifting per-item residuals (added Oct 27)                                                                            |
| `bkt.py`         | BKT: per-skill two-state model, one transition per checkpoint, log-space batched emissions, per-item guess/slip shrunk toward an anchor (added Oct 27) |

Notes on the heavier modules:

- `ingest.py` columns: `model, checkpoint_idx, checkpoint_label, step, tokens_seen_b, benchmark, skill, item_id, correct`.
- `evaluate.py` sees everything up to checkpoint t and predicts every item at t+h. Blocks per model (OLMo-2: checkpoints 0–55 / 56–74 / 75–93). `summarize` gives subject-clustered bootstrap standard errors; `paired_difference` gives a difference with a 95% interval between any two predictors.
- `static_irt.py` uses Fluid's 2PL item parameters and a window of past responses.
- `tskirt.py` lets older answers count less (discrimination discounted by the time gap); the forecast can include posterior variance.
- `recalibrate.py` learns δ and γ from the model's own past checkpoints instead of borrowing Fluid's population values; Newton MAP with shrinkage (`tau`, `tau_a`).
- `forgetting.py`: `min_peak_idx` guards against early noisy peaks. `chance_check` compares accuracy at peak and final with chance (Wilson intervals) and with the rest of MMLU at the same checkpoints. `letter_check` recovers the answer-letter preference from accuracy split by gold letter, G-tests for a change in that profile, and reports the residual drop after removing the letter effect.
- `ladder.py`: pooled ladder with skill and item gaps, per-checkpoint ladder, between-skill vs. within-skill variance, item-history R².
- `drift_irt.py`: state-space probit IRT; each item's residual drifts on its own, shrunk toward Fluid's difficulty; diagonal Gaussian filter; forecast variance decoupled from fitting drift.
- `bkt.py`: forward-backward on log-probabilities (`logsumexp`), MAP-EM per subject at every origin with a warm start; per-item guess and slip pulled toward an anchor by a pseudo-count `kappa`, with recency weights `rho^age`; anchors `skill` (the subject's shared pair, learned) or `fluid` (Fluid's 2PL at two ability levels); `kappa >= 1e5` pins items to the anchor exactly (the per-skill limit). `mastery()` returns the filtered P(learned) per subject and checkpoint.

### Scripts, `scripts/`

`ingest_fluid.py`, `plot_skill_curves.py`, `run_baselines.py`, `run_models.py` (`--predictors item_mean static_irt tskirt drift_irt`, `--nu`, `--recalibrate`, `--tau`, `--tau-a`, `--q-theta`, `--q-delta`, `--horizon`), `run_calibrated.py`, `run_forgetting.py` (`--chance-check` adds the chance-adjusted table), `analyze_sweep.py`, `analyze_horizons.py`, `analyze_ladder.py` (ladder tables and the two ladder figures in `docs/figures/`), `analyze_drift_irt.py` (selects a DriftIRT setting on validation; paired differences vs. item mean at h = 1, 5, 20), `analyze_bkt.py` (selects the BKT settings on validation; paired differences vs. item mean and vs. the recency-weighted item mean at h = 1, 5, 20; `--mastery` plots per-subject mastery; stores every table under `docs/results/bkt/`). `run_models.py` also takes `--predictors bkt item_mean_ewma` with `--kappa`, `--rho`, `--anchor`, `--theta-lo`, `--theta-hi`. Each writes to `data/processed/` (gitignored). The README has the setup, ingest and baseline commands; each script's docstring has its own.

### Documents

`docs/workbook/Workbook1.pdf` (the submission, 36 pages) and its two review reports; `docs/project-notes.md` (decisions, data, roadmap); `docs/lit-review-2026-09-30.md` (§9b has the corrections); `docs/research-brief.md`; `docs/presentation-script.md`; `docs/figures/`; `meetings/` (Aug 31 to Oct 19).

---

## 5. What the experiments showed (OLMo-2 on MMLU unless stated)

Metrics: **log-loss** (quality of the predicted probabilities, lower is better) and **AUC** (ranking quality within a subject, 0.5 is a coin flip, 1.0 perfect). "Pooled" = averaged over every checkpoint in the run. "Test block" = the last 19 checkpoints, never used for tuning; the honest number. "±" is a bootstrap standard error over the 57 subjects.

### 5.1 The bar

The strongest predictor with no learning in it: for each question, average how the model did on it at all previous checkpoints (item mean). One checkpoint ahead: log-loss 0.457, AUC 0.862 pooled; 0.375 ± 0.015 / 0.916 ± 0.006 on the test block.

**The result the advisor cared most about.** The baseline ladder, pooled log-loss one checkpoint ahead:

| What the predictor knows                       | OLMo-2, MMLU | All six models, all benchmarks |
| ---------------------------------------------- | ------------ | ------------------------------ |
| The model's overall score so far (global mean) | 0.69         | 0.661                          |
| Its score on that subject (skill mean)         | 0.63         | 0.582                          |
| That question's own history (item mean)        | 0.46         | 0.360                          |

Knowing the subject helps little; knowing the question helps a lot. Any learner model must carry per-question difficulty or it loses.

Reproduced 2026-10-27 from `run_baselines.py` (all benchmarks) and `run_models.py --benchmarks mmlu`. The item mean shrinks toward the global mean of the benchmarks included in the run, so the OLMo-2 MMLU item-mean value is 0.457 in the MMLU-only run and 0.454 in the all-benchmark run; the other values agree to the digits shown.

### 5.2 Borrowed item difficulties lose badly

T-SKIRT using Fluid's published item parameters, best drift setting: pooled log-loss 0.561 vs. 0.457, AUC 0.804 vs. 0.862. Those parameters were fitted on 102 finished models and do not describe one model's trajectory.

### 5.3 Recalibration closes most of the gap

| One checkpoint ahead                                       | Pooled log-loss | Pooled AUC | Test log-loss | Test AUC      |
| ---------------------------------------------------------- | --------------- | ---------- | ------------- | ------------- |
| Item mean (the bar)                                        | 0.457           | 0.862      | 0.375 ± 0.015 | 0.916 ± 0.006 |
| T-SKIRT, recalibrated (settings chosen on validation only) | 0.446           | 0.858      | 0.375 ± 0.024 | 0.912 ± 0.006 |

Paired difference on the test block (T-SKIRT minus item mean): log-loss −0.0003, 95% interval [−0.020, +0.025], a tie; AUC −0.003 [−0.006, −0.001], item mean slightly but reliably better at ranking.

### 5.4 Further ahead

All values log-loss on the test block (pooled in brackets).

| Horizon | Item mean     | T-SKIRT       | Paired difference                       |
| ------- | ------------- | ------------- | --------------------------------------- |
| 1       | 0.375 [0.457] | 0.375 [0.446] | −0.000 [−0.020, +0.025], tie            |
| 5       | 0.382 [0.484] | 0.382 [0.463] | −0.001 [−0.014, +0.012], tie            |
| 10      | 0.392 [0.506] | 0.399 [0.483] | +0.007 [−0.003, +0.015], tie            |
| 20      | 0.414 [0.545] | 0.437 [0.518] | +0.024 [+0.016, +0.031], item mean wins |

Pooled over the whole run, T-SKIRT leads by 0.01–0.03 log-loss, growing with horizon. Reading: the temporal model wins during the early, fast-changing phase of training and ties or loses on the late test checkpoints, where learning has plateaued and its forecast uncertainty only spreads predictions out.

### 5.5 Tuning and calibration do not change the picture

- Sweep of `tau`, `tau_a` and `nu`, selected on validation only: validation log-loss plateaus at 0.377 vs. item mean's 0.386. The family is maxed out.
- Calibration layer: fixes static IRT's under-confidence (test log-loss 0.393 to 0.382, AUC unchanged at 0.901); does nothing for T-SKIRT (0.375 to 0.375). The remaining gap is ranking, not confidence.

### 5.6 Forgetting

McNemar peak-to-final test on every (model, subject): did more questions flip right-to-wrong than wrong-to-right between the subject's best checkpoint and its last? P-values adjusted for screening 57 subjects and for picking the peak after looking.

- The seven OLMo-2 subjects that looked like forgetting by eye do not survive: raw p 0.002–0.049 become 1.0 after adjustment.
- Five (model, subject) pairs across all six models survive at adjusted p < 0.05. One looks real: OLMo-1, moral_scenarios, 895 questions, accuracy 0.37 to 0.24 between checkpoint 79 and the final checkpoint 82, adjusted p < 0.0001. The three Pythia hits (0.40 to 0.18–0.28 on security_studies and professional_medicine) look like answer-position bias on a near-chance model; a chance-adjusted check is the next step (implemented Oct 27 as `run_forgetting.py --chance-check`; its results are not yet recorded here).

### 5.7 Honest verdict

With per-model recalibration, a per-skill temporal model matches the strongest no-learning predictor at short range, beats it while skills are being acquired, and loses at long range. What remains is ranking within a skill, and the cause is structural: the item mean has one free number per question that updates every checkpoint; our model has one ability per skill plus a fixed per-question offset. No tuning closes that.

### 5.8 BKT with per-item emissions (Bhagyesh, Oct 27)

What it is: for each subject a hidden state, learned or not, that can switch once per checkpoint (learn rate up, forget rate down). Every question at a checkpoint is one noisy reading of the state, with its own guess and slip. Readings are combined in log space so 1,500 questions do not underflow. Each question's guess and slip are pulled toward an anchor by a pseudo-count `kappa` and weigh recent checkpoints more by `rho` per checkpoint of age. Settings were chosen on the validation block only; full write-up and the "why" in `docs/project-notes.md` §9, tables in `docs/results/bkt/`.

Chosen: skill anchor, `kappa` 0.1 (each question essentially free), `rho` 0.9 (half-life about 6.6 checkpoints). One checkpoint ahead:

| Predictor                                       | Pooled log-loss | Pooled AUC | Test log-loss | Test AUC      |
| ----------------------------------------------- | --------------- | ---------- | ------------- | ------------- |
| Item mean (the bar)                             | 0.457           | 0.862      | 0.375 ± 0.015 | 0.916 ± 0.006 |
| Recency-weighted item mean, `rho` 0.9 (control) | 0.403           | 0.880      | 0.317 ± 0.019 | 0.933 ± 0.006 |
| BKT, per-item emissions, chosen setting         | 0.411           | 0.880      | 0.310 ± 0.023 | 0.933 ± 0.006 |
| BKT, Fluid-anchored (`kappa` 1, `rho` 0.9)      | 0.388           | 0.882      | 0.316 ± 0.022 | 0.929 ± 0.007 |
| BKT, classic per-skill (one guess/slip pair)    | 0.637           | 0.500      | 0.623 ± 0.011 | 0.500         |

Test-block log-loss further ahead (pooled in brackets), with the paired difference BKT minus the other:

| Horizon | Item mean     | BKT           | BKT minus item mean     | Recency item mean | BKT minus recency item mean  |
| ------- | ------------- | ------------- | ----------------------- | ----------------- | ---------------------------- |
| 1       | 0.375 [0.457] | 0.310 [0.411] | −0.065 [−0.078, −0.045] | 0.317 [0.403]     | −0.007 [−0.013, +0.002], tie |
| 5       | 0.382 [0.484] | 0.323 [0.450] | −0.059 [−0.071, −0.041] | 0.326 [0.430]     | −0.002 [−0.009, +0.008], tie |
| 20      | 0.414 [0.545] | 0.359 [0.525] | −0.055 [−0.067, −0.039] | 0.355 [0.487]     | +0.003 [−0.002, +0.011], tie |

Reading, in five lines:

- BKT beats the bar on both metrics at every horizon, including horizon 20 where T-SKIRT lost. The gain is 0.06 log-loss and 0.017 AUC on the test block.
- The whole gain is recency. With `rho` 1 no BKT setting beats the item mean; a recency-weighted item mean (one line, same `rho`) ties BKT on the test block at every horizon and beats it pooled.
- Classic BKT with one guess/slip pair per subject is the skill mean: log-loss 0.62, AUC 0.50 by construction. Per-question numbers are not optional.
- The Fluid-anchored variant is the only one that beats the control anywhere: pooled over the run (which includes early training, when histories are short) it leads by 0.015 to 0.051 log-loss; on the test block it ties.
- At the chosen setting the mastery state is not interpretable: it sits at 0 or 1 on 98.5% of checkpoints, seven subjects never leave "not learned", and some subjects' two states are near-identical. Prediction wants small `kappa`; interpretation wants large. The per-skill-limit figure (`docs/figures/bkt_mastery_olmo2-7b_k1e+06_r1.png`) is the one to read for when each subject was learned.

Verdict: the three learner models share one finding, sharper than open question 1 put it: recent per-item state is what matters. Next: decide whether the recency-weighted item mean becomes the bar (T-SKIRT and DriftIRT have not been scored against it), and pool `learn`/`forget` across the six models as the one way a per-skill state could still add information.

---

## 6. Advisor meeting, 2026-10-19 (11:20–11:40)

Full note in `meetings/2026-10-19-advisor.md`. Three things came out of it:

1. **Expand the baseline ladder.** The advisor was most interested in the subject-vs-item finding in 5.1 and asked us to look at it from more angles: per model, per benchmark, per horizon; how much response variance is between skills vs. between items; whether the gap changes over training. This is cheap, uses what we have, and may become a paper section of its own.
2. **Test the trajectory on a held-out dataset.** Having fitted a knowledge trajectory during training, check whether it predicts the model's performance on a separate test set at the same checkpoints (held-out items within MMLU; other benchmarks such as ARC or HellaSwag). This is a different validation from predicting the next checkpoint and would strengthen the claim that the per-skill state means something.
3. **A later direction, not this semester:** knowledge tracing on chain-of-thought responses. Inputs and chain-of-thought outputs as the dataset; track or reason about what changes over a conversation that makes the model hallucinate. Record it as future work.

General direction: keep experimenting while the foundation and baselines are being built, expand into more directions, uncover new insights from the models we train, and implement BKT.

---

## 7. What each of us does next

Owners follow the track split from Sep 29 (Ronak: data, harness, visualization; Bhagyesh: learner models; Aniket: analyses, statistics, Phase 2). Due dates are in the meeting note.

**Ronak**
- Expand the baseline-ladder analysis (per model, per benchmark, per horizon; between-skill vs. between-item variance decomposition; gap over training). Due Oct 26.
- Build the drifting per-item residual model: each question's difficulty offset drifts slowly, shrunk toward its skill; forecast variance decoupled from fitting drift; chosen on validation, reported on test with intervals. Due Nov 2. This is the smallest change that gives the model what the item mean has while keeping the per-skill state the project is about.

**Bhagyesh**
- Log-space BKT with forgetting and per-item emissions in `src/llm_kt/bkt.py`, with tests, scored at h = 1 and h = 5 against the item-mean bar. Why log-space: pyBKT's E-step multiplies raw emissions and underflows at hundreds of items per checkpoint. Why per-item emissions: with one guess/slip pair per skill, within-skill AUC is 0.50 by construction. Due Nov 2.
- Write the chain-of-thought / hallucination direction into `docs/project-notes.md` as future work, with the data it would need. Due Oct 26.

**Aniket**
- Chance-adjusted forgetting check (answer-letter distribution at peak vs. final), then re-run `scripts/run_forgetting.py`. Due Oct 26.
- Design and run the advisor's held-out test: fit per-skill trajectories on one item set, test whether they predict the same checkpoints' performance on held-out items and on other benchmarks. Due Nov 2.
- Phase 2 GPU pilot and cost estimate stay on the calendar (Nov 25 / Dec 12).

**Status as of 2026-10-27** (what is on `main`; "not recorded" means the code exists but no numbers are in the repo yet):

| Action                                                | Owner    | Due            | Status                                                                                                                                                            |
| ----------------------------------------------------- | -------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Expand the baseline-ladder analysis                   | Ronak    | Oct 26         | Code and figures on `main` (`ladder.py`, `analyze_ladder.py`, `docs/figures/ladder_over_training.png`, `variance_decomposition.png`); findings not yet written up |
| Drifting per-item residual model                      | Ronak    | Nov 2          | Code on `main` (`drift_irt.py`, `analyze_drift_irt.py`); results not recorded                                                                                     |
| Log-space BKT                                         | Bhagyesh | Nov 2          | Done Oct 27 on `bhagyesh_branch` (`src/llm_kt/bkt.py`, `scripts/analyze_bkt.py`, `docs/project-notes.md` §9, §5.8 below)                                          |
| Chain-of-thought direction in `docs/project-notes.md` | Bhagyesh | Oct 26         | Done Oct 27 (`docs/project-notes.md` §8)                                                                                                                          |
| Chance-adjusted forgetting check                      | Aniket   | Oct 26         | Code on `main` (`chance_check`, `letter_check`, `run_forgetting.py --chance-check`); re-run not recorded                                                          |
| Held-out test of fitted trajectories                  | Aniket   | Nov 2          | Not started                                                                                                                                                       |
| Phase 2 GPU pilot and cost estimate; go/no-go         | Aniket   | Nov 25; Dec 12 | On the calendar                                                                                                                                                   |

**Everyone:** read `meetings/2026-10-19-advisor.md` and correct anything you remember differently before the next team meeting.

---

## 8. Open questions

1. BKT vs. the bar: answered Oct 27 (§5.8). BKT with per-item emissions beats item mean, but a recency-weighted item mean does the same; the shared finding is "recent per-item state is what matters". Do we present the three models as that family, with the Fluid-anchored BKT's early-training lead as the one place a prior beyond the item's own history helped?
2. The held-out test: which item split? Random half of each subject (easy) or whole benchmarks (harder, more convincing)?
3. GPU access for Phase 2 is still unresolved; decide by Dec 12 whether to switch to the fallback datasets.

---

## 9. Where everything lives

| Document                                                                                      | What it is                                                                                                                                                                                    |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docs/project-notes.md`                                                                       | Decisions 1–5 with pros/cons, data verification, lit-review consequences, roadmap, risks, future work (§8), BKT design and results (§9)                                                       |
| `docs/superpowers/specs/2026-10-27-bkt-design.md`, `docs/superpowers/plans/2026-10-27-bkt.md` | The BKT design document (the why, the model, the grid) and the task-by-task plan it was built from                                                                                            |
| `docs/results/bkt/`                                                                           | Every BKT table: validation ranking, scores and paired differences at h = 1, 5, 20, mastery curves and skill parameters, the per-slice result parquets, and the console report `analysis.txt` |
| `docs/abstract-v2.md`                                                                         | Revised abstract with change log                                                                                                                                                              |
| `docs/research-brief.md`                                                                      | The brief given to the research agents; also records measured findings                                                                                                                        |
| `docs/lit-review-2026-09-30.md`                                                               | Five-priority literature review; §9b has the corrections and supersedes earlier recommendations                                                                                               |
| `docs/workbook/Workbook1.pdf`                                                                 | The submission; `build/content.py` (gitignored) is its source                                                                                                                                 |
| `docs/workbook/Workbook1-review-2026-09-30.md`, `Workbook1-followup-review-2026-09-30.md`     | The two independent audits                                                                                                                                                                    |
| `docs/presentation-script.md`                                                                 | Speaker notes for the Workbook 1 presentation                                                                                                                                                 |
| `meetings/`                                                                                   | One note per meeting, Aug 31 to Oct 19                                                                                                                                                        |
| `README.md`                                                                                   | Setup, ingest, plot, run baselines                                                                                                                                                            |
| `data/processed/` (gitignored, rebuild with the scripts)                                      | `fluid_responses.parquet`, baseline and model result tables, ladder tables (`ladder_table.csv`, `ladder_over_training.csv`, `variance_decomposition.csv`), forgetting and chance-check tables |

Repo: `github.com/bhagyeshrathi07/LLM-Knowledge-Tracing`, branch `main`, head `9135fc0` (2026-10-27; `1b032cf` at the Oct 19 briefing).
