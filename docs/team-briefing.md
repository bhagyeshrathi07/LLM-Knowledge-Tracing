# Team Briefing: Where We Are and What We Do Next

Written 2026-10-19 for Bhagyesh and Aniket, right after the advisor meeting. This replaces the 2026-10-01 briefing. It covers everything in the repository as of today (Sprint 3 code and the Workbook 1 package were pushed to `main` this morning), what the experiments showed, what the advisor said, and what each of us does next. Reference documents are listed at the end. Read this first; go to those for detail.

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

| Group of papers | What they do | What they lack |
|---|---|---|
| Test-scoring statistics applied to AI models (Fluid Benchmarking, Item Response Scaling Laws) | Score each checkpoint like a student taking one exam; one group published the data we use | Each checkpoint scored alone; nothing carries forward, so no notion of a skill learned or lost |
| Studies of models during training (MathCAMPS, Implicit Curriculum, Harmon et al.) | Show skills appear in a stable order; count questions that flip right-to-wrong | Charts and counts only; no model of the student, no prediction test |
| Student-tracking methods (education, political science) | Built for a few learners watched over many periods, exactly our shape | Never applied to an AI model's checkpoints |

Our claim, worded carefully: *in the sources reviewed through September 30, 2026, we did not identify a method that jointly models a per-skill latent state across an LLM's pretraining checkpoints, treats each checkpoint's responses as a batch, and evaluates item-level predictions on later held-out checkpoints with calibrated uncertainty.* Not "nobody has done this"; a search cannot prove that.

---

## 3. Data

**Phase 1 (now):** `allenai/fluid-benchmarking` on HuggingFace. Six models (Amber-6.7B, K2-65B, OLMo-1 7B, OLMo-2 7B, Pythia-2.8B, Pythia-6.9B), 452 checkpoints, 27,842 items after dropping TruthfulQA (continuous scores), 12,584,584 right/wrong answers. Skills: the 57 MMLU subjects; ARC-Challenge, GSM8K, HellaSwag and WinoGrande count as one skill each. The release also ships pre-fitted 2PL IRT item parameters (difficulty, discrimination) estimated on 102 finished models. No GPU needed. Step and token counts are missing for Amber, K2 and Pythia; checkpoint index is the clock.

**Phase 2 (spring):** OLMo-2 checkpoints on MathCAMPS (real skill tags). Needs GPU; Aniket owns the pilot and cost estimate (Nov 25) and the go/no-go (Dec 12). No-GPU fallbacks: `allenai/DataDecide-eval-instances` and the Implicit Curriculum repository.

---

## 4. What is in the repository

Everything below is on `main`. 328 tests pass (`pytest`), `ruff` clean.

### Library, `src/llm_kt/`

| Module | What it does |
|---|---|
| `ingest.py` | Six Fluid CSVs to one Parquet table: `model, checkpoint_idx, checkpoint_label, step, tokens_seen_b, benchmark, skill, item_id, correct` |
| `curves.py` | Per-skill accuracy per checkpoint with Wilson intervals |
| `metrics.py` | Log-loss, Brier, accuracy, AUC |
| `evaluate.py` | Walk-forward harness: see everything up to checkpoint t, predict every item at t+h. Chronological train/validation/test blocks per model (OLMo-2: checkpoints 0–55 / 56–74 / 75–93). `summarize` gives subject-clustered bootstrap standard errors; `paired_difference` gives a difference with a 95% interval between any two predictors |
| `baselines.py` | Five no-learning predictors: global mean, per-skill mean, item-history mean (shrunk toward global), copy-last, copy-last-calibrated |
| `static_irt.py` | Static IRT persistence: at each checkpoint re-estimate one ability per skill from a window of past responses, using 2PL item parameters, and carry it forward |
| `tskirt.py` | T-SKIRT (temporal IRT): ability drifts over training, so older answers count less (discrimination discounted by the time gap); probit link; optional posterior variance in the forecast |
| `recalibrate.py` | Per-model recalibration: re-learn each item's difficulty offset (δ) and discrimination multiplier (γ) from the model's own past checkpoints instead of borrowing Fluid's population values; Newton MAP with shrinkage (`tau`, `tau_a`) |
| `calibrate.py` | Walk-forward temperature / Platt calibration wrapper: refit the confidence mapping on past checkpoints only |
| `forgetting.py` | Peak-to-final McNemar test per (model, subject) with Holm and selection adjustment; `min_peak_idx` guards against early noisy peaks |

### Scripts, `scripts/`

`ingest_fluid.py`, `plot_skill_curves.py`, `run_baselines.py`, `run_models.py` (`--predictors item_mean static_irt tskirt`, `--nu`, `--recalibrate`, `--tau`, `--tau-a`, `--horizon`), `run_calibrated.py`, `run_forgetting.py`, `analyze_sweep.py`, `analyze_horizons.py`. Each writes to `data/processed/` (gitignored); the README has the commands.

### Documents

`docs/workbook/Workbook1.pdf` (the submission, 36 pages) and its two review reports; `docs/project-notes.md` (decisions, data, roadmap); `docs/lit-review-2026-09-30.md` (§9b has the corrections); `docs/research-brief.md`; `docs/presentation-script.md`; `docs/figures/`; `meetings/` (Aug 31 to Oct 19).

---

## 5. What the experiments showed (OLMo-2 on MMLU unless stated)

Metrics: **log-loss** (quality of the predicted probabilities, lower is better) and **AUC** (ranking quality within a subject, 0.5 is a coin flip, 1.0 perfect). "Pooled" = averaged over every checkpoint in the run. "Test block" = the last 19 checkpoints, never used for tuning; the honest number. "±" is a bootstrap standard error over the 57 subjects.

### 5.1 The bar

The strongest predictor with no learning in it: for each question, average how the model did on it at all previous checkpoints (item mean). One checkpoint ahead: log-loss 0.457, AUC 0.862 pooled; 0.375 ± 0.015 / 0.916 ± 0.006 on the test block.

**The result the advisor cared most about.** Same slice, same horizon, log-loss: guessing from the overall score so far gives 0.70; from the model's score on that subject, 0.63; from that specific question's own history, 0.46. Knowing the subject helps little; knowing the question helps a lot. Any learner model must carry per-question difficulty or it loses. (Across all six models and all benchmarks, the same ladder is 0.661 / 0.582 / 0.360 at h = 1.)

### 5.2 Borrowed item difficulties lose badly

T-SKIRT using Fluid's published item parameters, best drift setting: pooled log-loss 0.561 vs. 0.457, AUC 0.804 vs. 0.862. Those parameters were fitted on 102 finished models and do not describe one model's trajectory.

### 5.3 Recalibration closes most of the gap

| One checkpoint ahead | Pooled log-loss | Pooled AUC | Test log-loss | Test AUC |
|---|---|---|---|---|
| Item mean (the bar) | 0.457 | 0.862 | 0.375 ± 0.015 | 0.916 ± 0.006 |
| T-SKIRT, recalibrated (settings chosen on validation only) | 0.446 | 0.858 | 0.375 ± 0.024 | 0.912 ± 0.006 |

Paired difference on the test block (T-SKIRT minus item mean): log-loss −0.0003, 95% interval [−0.020, +0.025], a tie; AUC −0.003 [−0.006, −0.001], item mean slightly but reliably better at ranking.

### 5.4 Further ahead

All values log-loss on the test block (pooled in brackets).

| Horizon | Item mean | T-SKIRT | Paired difference |
|---|---|---|---|
| 1 | 0.375 [0.457] | 0.375 [0.446] | −0.000 [−0.020, +0.025], tie |
| 5 | 0.382 [0.484] | 0.382 [0.463] | −0.001 [−0.014, +0.012], tie |
| 10 | 0.392 [0.506] | 0.399 [0.483] | +0.007 [−0.003, +0.015], tie |
| 20 | 0.414 [0.545] | 0.437 [0.518] | +0.024 [+0.016, +0.031], item mean wins |

Pooled over the whole run, T-SKIRT leads by 0.01–0.03 log-loss, growing with horizon. Reading: the temporal model wins during the early, fast-changing phase of training and ties or loses on the late test checkpoints, where learning has plateaued and its forecast uncertainty only spreads predictions out.

### 5.5 Tuning and calibration do not change the picture

- Sweep of `tau`, `tau_a` and `nu`, selected on validation only: validation log-loss plateaus at 0.377 vs. item mean's 0.386. The family is maxed out.
- Calibration layer: fixes static IRT's under-confidence (test log-loss 0.393 to 0.382, AUC unchanged at 0.901); does nothing for T-SKIRT (0.375 to 0.375). The remaining gap is ranking, not confidence.

### 5.6 Forgetting

McNemar peak-to-final test on every (model, subject): did more questions flip right-to-wrong than wrong-to-right between the subject's best checkpoint and its last? P-values adjusted for screening 57 subjects and for picking the peak after looking.

- The seven OLMo-2 subjects that looked like forgetting by eye do not survive: raw p 0.002–0.049 become 1.0 after adjustment.
- Five (model, subject) pairs across all six models survive at adjusted p < 0.05. One looks real: OLMo-1, moral_scenarios, 895 questions, accuracy 0.37 to 0.24 between checkpoint 79 and the final checkpoint 82, adjusted p < 0.0001. The three Pythia hits (0.40 to 0.18–0.28 on security_studies and professional_medicine) look like answer-position bias on a near-chance model; a chance-adjusted check is the next step.

### 5.7 Honest verdict

With per-model recalibration, a per-skill temporal model matches the strongest no-learning predictor at short range, beats it while skills are being acquired, and loses at long range. What remains is ranking within a skill, and the cause is structural: the item mean has one free number per question that updates every checkpoint; our model has one ability per skill plus a fixed per-question offset. No tuning closes that.

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

**Everyone:** read `meetings/2026-10-19-advisor.md` and correct anything you remember differently before the next team meeting.

---

## 8. Open questions

1. BKT vs. the bar: if BKT with per-item emissions also only ties item mean, do we present the three models as a family with a shared finding ("per-item state is what matters, per-skill state adds value early in training")?
2. The held-out test: which item split? Random half of each subject (easy) or whole benchmarks (harder, more convincing)?
3. GPU access for Phase 2 is still unresolved; decide by Dec 12 whether to switch to the fallback datasets.

---

## 9. Where everything lives

| Document | What it is |
|---|---|
| `docs/project-notes.md` | Decisions 1–5 with pros/cons, data verification, lit-review consequences, roadmap, risks |
| `docs/abstract-v2.md` | Revised abstract with change log |
| `docs/research-brief.md` | The brief given to the research agents; also records measured findings |
| `docs/lit-review-2026-09-30.md` | Five-priority literature review; §9b has the corrections and supersedes earlier recommendations |
| `docs/workbook/Workbook1.pdf` | The submission; `build/content.py` (gitignored) is its source |
| `docs/workbook/Workbook1-review-2026-09-30.md`, `Workbook1-followup-review-2026-09-30.md` | The two independent audits |
| `docs/presentation-script.md` | Speaker notes for the Workbook 1 presentation |
| `meetings/` | One note per meeting, Aug 31 to Oct 19 |
| `README.md` | Setup, ingest, plot, run baselines and models |
| `data/processed/` (gitignored, rebuild with the scripts) | `fluid_responses.parquet`, baseline and model result tables, forgetting table |

Repo: `github.com/bhagyeshrathi07/LLM-Knowledge-Tracing`, branch `main`, head `1b032cf`.
