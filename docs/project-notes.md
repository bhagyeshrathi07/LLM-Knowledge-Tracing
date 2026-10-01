# Knowledge Tracing for LLMs — Project Notes

Working notes for the team. Covers what we are building, what prior work exists, the design problems we found, and the decisions made so far. Last updated 2026-09-30 (week 5 of semester 1).

**Status:** Decisions 1–5 adopted by the team on 2026-09-29 (Section 6); advisor sign-off on Decisions 1–2 still pending. Phase 1 data verified (Section 3). An independent review on 2026-09-30 corrected several empirical numbers, citations and method claims; see the dated corrections in Sections 3, 4 and 6 and `docs/workbook/Workbook1-review-2026-09-30.md`.

---

## 1. What we are building (plain-language version)

Large language models learn in stages. During training, developers save a snapshot of the model at many points along the way (often hundreds or thousands). Today, each snapshot is usually judged by a single score, like one exam grade. That score cannot tell you which specific skills the model has picked up, when it picked each one up, or whether it later lost some.

Educational software already solves this for human students. Tutoring apps use a technique called **knowledge tracing (KT)**. After every question a student answers, the app updates its estimate of how well the student knows each skill (fractions, algebra, etc.), uses that estimate to decide what to teach next, and notices when a student has forgotten something.

**Our project applies that same student-tracking method to AI models.** Each saved training snapshot is one "moment in time," and each test question is tagged with the skill it tests. The result is a per-skill progress report for the AI across its whole training: which skills appear first, which ones get forgotten, and which questions get easier or harder over time. We validate the tracker by asking it to predict how the *next* snapshot will do on questions it has not seen.

Per advisor feedback, the project also includes a **visualization tool** that lets a developer or researcher browse these per-skill progress reports. Target venue is **NeurIPS** (deadline historically mid-May; confirm 2027 dates).

### How knowledge tracing works (for those new to it)

Picture a math tutoring app. Each problem is tagged with a skill. The app keeps a hidden guess per skill: "60% chance this student has mastered adding fractions." After every answer, the guess moves up or down, but not all the way, because students sometimes guess right by luck or slip on something they know. The app checks its guesses by predicting the student's *next* answer before seeing it. Accurate predictions mean the mastery estimates are trustworthy.

Applied to an LLM, with a concrete example:

- **Observed:** on "adding fractions," checkpoint 1 got 5/10, checkpoint 2 got 7/10, checkpoint 3 got 6/10.
- **Hidden estimate:** mastery 48% → 69% → 71%. Smoothed, noise-adjusted. Not simply raw accuracy.
- **Prediction test:** before seeing checkpoint 3, the tracker predicted about 7/10. Actual was 6/10. Close enough; the tracker is working.

The mastery number differs from raw accuracy in three ways:

1. It smooths out noise (7/10 then 6/10 is probably not real forgetting).
2. It accounts for lucky guesses (25% chance on 4-option multiple choice) and careless slips.
3. It is tested by prediction, not just described. That is what makes it science rather than a chart.

---

## 2. Technical breakdown

**Data pipeline**
- Collect item-level correctness (right or wrong on each question) for every checkpoint.
- Phase 1: use already-published checkpoint responses from six open models (see Section 3).
- Phase 2: run OLMo-2 intermediate checkpoints ourselves on a skill-tagged math benchmark (MathCAMPS). This needs GPU time.
- Canonical table: `(model, checkpoint_step, tokens_seen, benchmark, item_id, skill_tags, correct)`.

**Models**
- Time step = training checkpoint. Each skill-tagged item = one noisy observation of mastery at that checkpoint.
- KT-family models with forgetting and temporal priors (see Decision 2).
- Baselines: static IRT per checkpoint, per-skill accuracy curves, threshold-based emergence.

**Evaluation (the paper's claims)**
- Next-checkpoint prediction: hold out checkpoint t+1, predict its responses, score with AUC and log-loss. Time-based split only.
- Item-difficulty invariance: does an item's difficulty stay stable across training or drift?
- Per-skill learning curves: emergence order, mastery timing, forgetting events.

**Visualization tool**
- Web app showing per-skill mastery over checkpoints with uncertainty; views for emergence order, forgetting events, difficulty drift; side-by-side model comparison.
- Likely stack: FastAPI backend serving fitted KT outputs, React frontend with charts. Built mainly in semester 2 once results exist.

---

## 3. The "six open models" (Phase 1 data)

Almost certainly the **Fluid Benchmarking** data release from AI2 (COLM 2025). Needs confirmation from the team that this is what the abstract meant.

| Model | Size | Trained by |
|---|---|---|
| Amber | 6.7B | LLM360 |
| K2 | 65B | LLM360 |
| OLMo-1 | 7B | AI2 |
| OLMo-2 | 7B | AI2 |
| Pythia | 2.8B | EleutherAI |
| Pythia | 6.9B | EleutherAI |

Each has 46–94 pretraining checkpoints evaluated on six benchmarks: ARC-Challenge, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande. Data lives at `allenai/fluid-benchmarking` on HuggingFace.

Note that OLMo-2 7B is both one of the six and the Phase 2 target. Phase 2 differs by benchmark (MathCAMPS, skill-tagged), not by model.

### Data verification (done 2026-09-29)

Inspected the HuggingFace dataset directly. **Per-item, per-checkpoint correctness is present.** Phase 1 needs no GPU.

Files (16 total):

- `data/lm_eval_results/{model}.csv` — one per model. Rows = 28,659 items, columns = checkpoints, cells = correctness. This is the table we need.
- `data/irt_models/{benchmark}.csv` — fitted 2PL IRT parameters (`a` discrimination, `b` difficulty) per item. A free static-IRT baseline.
- `data/id_to_item_map.json` — question text and answer choices for every item id.
- `data/open_llm_leaderboard_results.json` — 102 final models, not checkpoints. Not needed for now.

Per model:

| Model | Checkpoints | Column format |
|---|---|---|
| amber-7b | 73 | step |
| k2-65b | 46 | step |
| olmo1-7b | 83 | step |
| olmo2-7b | 94 | `stage1-step150-tokens1B` … `stage1-step928646-tokens3896B` |
| pythia-3b | 78 | `step0`, `step2`, `step8`, … `step143000` |
| pythia-7b | 78 | same as pythia-3b |

Total: 452 checkpoints × 28,659 items, about 13M cells, no missing correctness values. (Time metadata is incomplete: `step` is null for Amber and K2, `tokens_seen_b` is null for Amber, K2 and Pythia. The Fluid paper reports 467 checkpoints including 61 for K2; the release we inspected has 452 including 46 for K2. We report the release figures.)

Items per benchmark: ARC-Challenge 1,172; GSM8K 1,319; HellaSwag 10,042; MMLU 14,042; TruthfulQA 817; WinoGrande 1,267.

MMLU item ids embed the subject (e.g. `mmlu_abstract_algebra_0`). All 57 subjects present. Subjects are unbalanced: `professional_law` 1,534 items, `moral_scenarios` 895, smallest around 100.

Caveats:

1. **TruthfulQA is not binary.** Cells are continuous probabilities (mc2 metric). Every other benchmark is strictly 0/1. Drop it for Phase 1.
2. **GSM8K is near floor for most models.** Mean correctness: Pythia ~1%, Amber 2%, OLMo-1 8%, OLMo-2 13%, K2 36%. Little learning-curve signal except for K2.
3. OLMo-2 checkpoints are stage-1 pretraining only (ends at 3.9T tokens). Stage-2 annealing checkpoints are not included. Fine for Phase 1.

Local setup: `.venv/` at repo root with `huggingface_hub`, `datasets`, `pandas`, `pyarrow`. Data caches to `~/.cache/huggingface/`, not the repo.

---

## 4. Prior work (literature review, ~25 searches, 2026-09-22)

**Bottom line:** no prior work does exactly what the abstract proposes. The core claim ("no existing method maintains a per-skill knowledge state for an LLM, updated after each response and validated by predicting its next response") holds. But the abstract's sentence "skill emergence order, forgetting, and shifts in item difficulty go unmeasured" is too strong: these have been *described* (accuracy charts, thresholds, counting) but not *modeled* (a latent state that makes testable predictions).

### Closest work, ranked by overlap

1. **Mishra, Poesia & Goodman, "From Next-Token to Mathematics" (COLM 2025)** — closest overlap. Introduces MathCAMPS: math problems tagged with 44 Common Core standards. Tested Pythia, OLMo, Amber checkpoints; found skills emerge in roughly school-curriculum order; looked at which skills improve or degrade under instruction tuning. Only plots per-skill accuracy. No latent mastery state, no forgetting model, no next-checkpoint prediction. **This is the baseline our Phase 2 must beat, and MathCAMPS is likely our Phase 2 dataset.**

2. **Liu et al. (Neubig group), "The Implicit Curriculum Hypothesis" (April 2026)** — tracked 91 tasks across 9 models (OLMo-2, OLMo-3, Pythia, LLM360). Emergence order stable across models; composite skills emerge after components. Uses accuracy threshold, not latent state. No forgetting, no standard benchmarks. Directly contradicts our "emergence order goes unmeasured" claim.

3. **Fluid Benchmarking, AI2 (COLM 2025)** — fits IRT and adaptively selects items over ~450+ checkpoints of the six models above. IRT is static: one fixed difficulty per item, each checkpoint scored separately, no skill tags, no forgetting. Our natural baseline. Also observed that the most informative items change over training, which supports our difficulty-drift hypothesis.

4. **Harmon et al., "Mapping Post-Training Forgetting at Scale" (Oct 2025)** — counts item-level 1→0 transitions (forgetting) and 0→1 (backward transfer) across ~100 subdomains. Simple counting, no latent learner model, post-training only. Weakens our "forgetting goes unmeasured" claim.

5. **Item Response Scaling Laws (June 2026)** — IRT over 6,612 checkpoints and 37,682 questions to forecast performance vs scale. No skill tags, no sequential state.

### Related but not competing

- **Hu, Chen, Saphra & Cho, "Latent State Models of Training Dynamics" (TMLR 2023)** — fits an HMM (same family as BKT) to aggregate training metrics like loss. Not per skill or per item. Good methodological precedent to cite.
- **"Exploring Forgetting in LLM Pre-Training" (ACL 2025)** — forgetting of individual facts/entities, not skills.
- **Cognitive diagnosis for LLMs (FinCDM 2025; medical CDM, Sci. Reports 2026)** — per-skill profile of a single frozen model. No time dimension.
- **"Fast and Accurate Probing of In-Training LLMs" (April 2026)** — predicts later-checkpoint performance from internals, per task not per item. Another baseline for "predict the next checkpoint."
- **LLM-for-KT papers (LLM-KT, NTKT, CLST, LLMKT for dialogues)** — use an LLM to trace *human* students. Reverse direction from ours. Must state the distinction early in the paper so reviewers do not confuse them.
- **"DKT is an implicit dynamic multidimensional IRT" (2023)** — theory connecting the two model families we will use. Cite.
- **Visualization** — nothing found that shows per-skill mastery across training checkpoints. Existing tools are training dashboards or model comparators. This piece appears open.

### Implications for the abstract

- Keep the core novelty claim.
- Rephrase "go unmeasured" to "have been described but not modeled."
- Name baselines up front: static IRT (Fluid), per-skill accuracy curves (MathCAMPS), threshold emergence (Implicit Curriculum).
- Re-check arXiv and OpenReview before submission; this area moved fast in 2025–26.

### Deep literature review (2026-09-30)

A five-priority review by research agents is in `docs/lit-review-2026-09-30.md`, spot-verified against primary sources with four minor corrections applied. Headline consequences for this project:

1. **Core claim survives.** No prior work fits a sequential latent learner model to an LLM's checkpoint responses, validates it on held-out later checkpoints, or tests item-parameter drift within one model's training run.
2. **Item Response Scaling Laws (arXiv 2606.07616) already fits static IRT to 6,612 pretraining checkpoints.** It estimates each checkpoint's ability independently and fits a monotone scaling law through the estimates (the estimates themselves are not constrained). Our framing must say: they fit static per-checkpoint IRT with frozen items; we add the sequential state, test the invariance they assume, and validate forward in time at the item level. Forgetting is not something their scaling law represents.
3. **Per-item difficulty drift is probably below detectability at our data density.** A first-order application of the SLC bound gives δ_min ≈ 2.26 logits per item per checkpoint, but that bound assumes independent observations; a project-specific simulation-based power analysis is required before any claim. Plan to test at bank level or in early/mid/late thirds, and target discrimination, which prior evidence says is the parameter that drifts. (Corrected 2026-09-30.)
4. **MMLU subjects share most of their variance.** STEM and non-STEM abilities correlate at ρ = 0.965 across 1,000 final models (arXiv 2609.09372); the paper concludes they are nevertheless distinguishable. Measure inter-subject trajectory correlation early; if subjects separate across checkpoints where they barely do across final models, that is the headline. This is a question to test, not an assumption either way.
5. **Model choice updated.** Primary temporal-IRT model: a Kalman-filter dynamic IRT *inspired by* Martin-Quinn / `emIRT::dynIRT`. **Correction 2026-09-30:** dynIRT assigns each item to one session (`bill.session`), so sharing item parameters across checkpoints needs a real adaptation or a custom implementation, not a two-line patch; its manual also says its variational variances are "far too small and generally unusable," so uncertainty needs a separate procedure (bootstrap or MCMC refit). Martin-Quinn's data had changing cases per term, not a shared item bank. BKT via pyBKT: its internal format allows several observations per step, but the E-step multiplies raw emissions before normalising, which underflows for batches of hundreds of items, so a log-space implementation is required; with one guess/slip pair per skill BKT's within-skill AUC is 0.50 by construction, so it needs per-item emissions to compete on AUC. T-SKIRT (~50 lines; probit link, so Fluid's logistic 2PL parameters need conversion) as the first thing to run. VTIRT not adopted (one item per step). DAS3H added as a sixth trivial baseline.
6. **Forgetting prior.** Chang et al. (NeurIPS 2024) show pretraining forgetting of injected facts follows a power law, so a constant BKT forget rate is misspecified; use a gap-scaled forget rate or let dynIRT's random walk absorb it.
7. **More data exists.** `allenai/DataDecide-eval-instances` (ODC-BY, ~150k checkpoints, 122.9 GB, MMLU near chance at ≤1B) and `KaiserWhoLearns/ElementalTask` (MIT, per-item results for Amber, OLMo-2, Pythia-6.9B on 91 skill-structured tasks). Both are Phase 2 options that need no GPU.
8. **Statistics.** Cluster standard errors by subject; run the McNemar flip-rate test (arXiv 2602.10144) on the forgetting list before claiming forgetting; run a parametric-recovery simulation over 6 autocorrelated trajectories rather than arguing from 452 checkpoints.

New papers to add to the related-work list: Signal and Noise (2508.13144), Hidden Measurement Error (2604.11581), MMLU Psychometric Audit (2609.09372), Training on the Test Task (2407.07890), Factual Knowledge Acquisition (2406.11813), DeepTracker (1808.08531), Martin and Quinn 2002 / Imai et al. 2016, SLC (2606.14123), DataDecide.

### Sources

- [From Next-Token to Mathematics (arXiv 2407.00900)](https://arxiv.org/abs/2407.00900)
- [AI2: Watching an LLM learn math skills](https://allenai.org/blog/llm-math-skills)
- [Implicit Curriculum Hypothesis (arXiv 2604.08510)](https://arxiv.org/html/2604.08510)
- [Fluid Language Model Benchmarking (arXiv 2509.11106)](https://arxiv.org/html/2509.11106)
- [allenai/fluid-benchmarking GitHub](https://github.com/allenai/fluid-benchmarking)
- [Mapping Post-Training Forgetting (arXiv 2510.17776)](https://arxiv.org/abs/2510.17776)
- [Item Response Scaling Laws (arXiv 2606.07616)](https://arxiv.org/abs/2606.07616)
- [Latent State Models of Training Dynamics (arXiv 2308.09543)](https://arxiv.org/abs/2308.09543)
- [Exploring Forgetting in LLM Pre-Training (arXiv 2410.17018)](https://arxiv.org/abs/2410.17018)
- [FinCDM (arXiv 2508.13491)](https://arxiv.org/abs/2508.13491)
- [Fine-grained LLM evaluation in medicine via CDM (Sci. Reports)](https://www.nature.com/articles/s41598-026-36627-7)
- [Item Response Theory for AI Safety (arXiv 2608.05086)](https://arxiv.org/html/2608.05086v1)
- [Probing In-Training LLMs (arXiv 2604.01025)](https://arxiv.org/abs/2604.01025)
- [DKT is an implicit dynamic multidimensional IRT (arXiv 2309.12334)](https://arxiv.org/pdf/2309.12334)
- [LLM-KT (arXiv 2502.02945)](https://arxiv.org/abs/2502.02945)
- [Next Token Knowledge Tracing (arXiv 2511.02599)](https://arxiv.org/pdf/2511.02599)
- [Exploring KT in Tutor-Student Dialogues (arXiv 2409.16490)](https://arxiv.org/pdf/2409.16490)
- [CogLM (NAACL 2025)](https://aclanthology.org/2025.naacl-long.4/)
- [Skill-it! (NeurIPS 2023)](https://arxiv.org/abs/2307.14430)

---

## 5. Two design problems found

Both surfaced when comparing the abstract against the literature. Both must be resolved before writing code.

### Problem 1: KT assumes learning happens from practice. Here it does not.

**Plain version.** In a tutoring app, answering a problem *is* how the student learns; the test and the lesson are the same thing. For an LLM they are separate. The model learns by reading training data. At certain points someone pauses training, saves a snapshot, and runs a test. The model learns nothing from the test; it is a thermometer reading. All actual learning happened *between* snapshots.

So the abstract's phrase "each skill-tagged benchmark item as a practice opportunity" describes the test as if it were the lesson. The fix: mastery changes *between checkpoints*, and at each checkpoint we get a batch of test questions telling us where mastery currently sits. The math stays close to standard KT (there are existing variants built for "batch of observations at each time point"). Only the abstract sentence and the specific model variant change.

**Technical version.** Time step = checkpoint. Mastery transition happens once per step. All items at a checkpoint are simultaneous, conditionally independent noisy observations of the skill state. This is closer to temporal IRT (VTIRT, dynamic GP-IRT) or BKT with one transition per checkpoint and batched emissions.

### Problem 2: KT models train across thousands of students. We have six.

**Plain version.** KT tools were built for platforms with thousands of students. The neural-network variants learn their patterns by looking across all of them. Each AI model is one "student" for us, and we have six. A neural network trained on six examples will memorize them and be useless.

**Options.** (a) Treat each (model, benchmark) pair as a pseudo-learner. (b) Use the simpler per-skill models (BKT, temporal IRT) that fit one small model per skill and do not need a crowd. (c) Get more models: Pythia has 16 models at 8 sizes with 154 checkpoints each; OLMo-3 also releases checkpoints.

---

## 6. Decisions

### Decision 1: What is a "time step"? — DECIDED: Option B

**Option A — per item (abstract as written).** Every benchmark question is a step; mastery updates after each.
- Pro: matches standard KT exactly; off-the-shelf libraries (pyBKT, pyKT) work unchanged.
- Pro: easier to explain to KT reviewers.
- Con: conceptually wrong. The model does not learn from test items. Reviewers who know LLM training will flag it.
- Con: item order within a checkpoint is arbitrary, so the model learns a fake sequence.

**Option B — per checkpoint, batched observations.** Step = checkpoint. Mastery transitions once per step. All items at that checkpoint are simultaneous noisy readings.
- Pro: correct framing. Defensible in the paper.
- Pro: naturally handles uneven checkpoint spacing (tokens seen between steps becomes a feature).
- Con: off-the-shelf KT libraries do not support this directly. Custom HMM or temporal IRT code needed.
- Con: fewer time steps per learner (60–90 instead of hundreds of thousands).

**Agreed: Option B.** Cost is engineering; gain is correctness. Correctness is what gets NeurIPS acceptance. The abstract sentence about "practice opportunity" must be reworded.

### Decision 2: Which model family? — DECIDED: A + B, C as ablation

**Option A — BKT-style (per-skill hidden Markov model).**
- Pro: works with few learners; one small model per skill.
- Pro: interpretable: explicit learn rate, forget rate, guess, slip per skill. Those *are* the paper's findings.
- Pro: binary mastery state with probability; easy to visualize.
- Con: binary mastered/not is coarse for gradual skill growth.
- Con: no item difficulty; every item in a skill treated the same.

**Option B — temporal IRT (ability drifts over time, items have difficulty).**
- Pro: continuous ability per skill; item difficulty built in. Directly enables the "item difficulty invariance" test from the abstract.
- Pro: Fluid Benchmarking already fit static IRT on the same data. Natural extension, clear baseline.
- Con: more math; needs Bayesian inference (Stan/NumPyro or variational). Slower to build.
- Con: forgetting must be added explicitly; not native.

**Option C — deep KT (DKT, AKT, SAINT).**
- Pro: strongest predictive accuracy on human KT benchmarks.
- Pro: pyKT provides implementations.
- Con: needs thousands of learners. We have 6–20. Will overfit.
- Con: black box; per-skill learn/forget parameters not readable out.

**Agreed: A and B as primary models, C as ablation only.** A gives the interpretable forgetting story; B gives the difficulty story. The paper needs both. C shows we tried and explains why it fails with few learners, which is itself a finding.

### Decision 3: Phase 1 data source — ADOPTED 2026-09-29: Option A (data verified)

**Option A — Fluid Benchmarking release only.**
- Pro: free, immediate. 6 models × 6 benchmarks × 46–94 checkpoints. **Verified 2026-09-29: per-item, per-checkpoint correctness is included** (see Section 3).
- Pro: includes fitted static IRT parameters per item, which is a baseline for free.
- Con: 6 learners is thin.
- Con: TruthfulQA is non-binary and must be dropped; GSM8K is near floor for most models.

**Option B — run our own evals on the Pythia suite (16 models at 8 sizes × 154 checkpoints).**
- Pro: up to 16 learners, uniform checkpoint spacing, identical data order across sizes. Clean for cross-model comparison.
- Con: GPU cost. Weeks of compute unless subsetted hard.
- Con: delays first results to December or later.

**Proposed: Option A for Phase 1. No GPU needed.** Option B stays as an optional semester 2 expansion if more learners are needed. Needs team sign-off.

### Decision 4: Skill tags for Phase 1 — ADOPTED 2026-09-29: Option A

**Option A — MMLU subjects (57 subjects).**
- Pro: tags exist in the item ids; parsing is one regex. Zero labeling work. Confirmed present in the data.
- Pro: widely understood taxonomy.
- Con: subjects are topics ("high school chemistry"), not skills. In the paper, call them "knowledge components," KT's own term, which covers both.
- Con: unbalanced. Small subjects (~100 items) will give noisy mastery estimates. Mitigate by reporting uncertainty and by adding a rollup into MMLU's 4 standard categories (STEM, humanities, social sciences, other) as a coarse backup.
- Con: only covers MMLU.

**Option B — LLM auto-tag all items with skills.**
- Pro: `id_to_item_map.json` has the full text of all 28,659 items, so tagging is feasible offline.
- Pro: finer granularity, covers other benchmarks.
- Con: tags unvalidated. Reviewers will ask how we know they are right. Needs a human-labeled sample (~200 items) with reported agreement.
- Con: adds 2–3 weeks. HellaSwag and WinoGrande test commonsense and have no clean skill decomposition anyway.

**Option C — skip Phase 1 skills; go straight to MathCAMPS (44 Common Core standards).**
- Pro: cleanest skill taxonomy available. Already used by the COLM 2025 paper, so results are comparable.
- Con: only OLMo-2 (and Pythia if we run it). Requires our own GPU evals. No free data.

**Proposed: Option A for semester 1.** Use all 57 MMLU subjects as skills plus the 4-category rollup. Treat ARC-Challenge, GSM8K, HellaSwag and WinoGrande as single-skill benchmarks (one "skill" each) so they still feed the model and the difficulty-drift analysis. Drop TruthfulQA. Option C in semester 2. Option B only for ARC-Challenge, only if time allows. Needs team sign-off.

### Decision 5: Pseudo-learners? — ADOPTED 2026-09-29: Option (iii), ablation only

Context: the primary models (BKT, temporal IRT) fit per skill. For a skill like "high_school_chemistry" there are 6 sequences (one per model) of 46–94 checkpoints each. That is enough for models with 4–5 parameters per skill. Deep KT needs many more sequences.

**(i) No pseudo-learners.** 6 sequences. Deep KT unusable; skip it.
- Pro: honest, simple.
- Con: paper lacks the "we tried the modern method" ablation reviewers expect.

**(ii) (model, benchmark) pseudo-learners.** 6 × 5 = 30 sequences.
- Con: still too few for deep KT. Same independence problem. Little gained.

**(iii) (model, MMLU-subject) pseudo-learners.** 6 × 57 = 342 sequences of length 46–94, each step carrying ~100–1,500 item observations.
- Pro: enough to train a small DKT without immediate collapse. Makes the ablation meaningful.
- Con: sequences from the same model are correlated. Train/test split must be **leave-one-model-out**, never a random split over sequences, or leakage inflates results.
- Con: still far below human-KT scale (thousands of students). Expect deep KT to lose to BKT. That is itself a finding.

**Proposed: (iii), ablation only, leave-one-model-out split, clearly labeled.** Never for headline numbers. Headline numbers come from BKT and temporal IRT fit per skill on the real 6 learners. Needs team sign-off.

---

## 7. Next steps

### Immediate (weeks 5–8, through ~Oct 20)

1. **Revise the abstract.** Advisor's last-paragraph fix; soften "goes unmeasured"; fix the "practice opportunity" framing per Decision 1.
2. ~~Data feasibility spike.~~ Done 2026-09-29. Fluid data has per-item correctness. See Section 3.
3. **Team sign-off on Decisions 3–5; advisor sign-off on Decisions 1–2.**
4. **Compute plan (Phase 2 only).** OLMo-2 7B has ~900 checkpoints; pick ~50–100 log-spaced. MathCAMPS is generation-based eval. Estimate GPU-hours; source access (SJSU HPC, Colab Pro, Lambda/RunPod). Get a quote before the semester 1 report. No GPU needed for Phase 1.
5. **Related-work section.** Expand Section 4 into a formal write-up for the 295A report.
6. **Repo and schema.** One repo. Ingest script that converts the six Fluid CSVs into the canonical Parquet table per Section 2 (long format: one row per model × checkpoint × item, with `tokens_seen` parsed from column names where available, `skill` parsed from MMLU item ids, TruthfulQA dropped). Everything downstream reads it.
7. **Split work across three tracks:** data/infra; modeling; visualization + writing. Weekly advisor meeting at a fixed slot.

### Rest of semester 1 (weeks 9–15, ~Oct 20–Dec 12)

- Baselines: static 2PL IRT per checkpoint (Fluid), per-skill accuracy curves (MathCAMPS-style), threshold emergence (Implicit Curriculum).
- First KT model: BKT with forgetting, one transition per checkpoint, batched observations. pyBKT as a starting point; likely need a custom HMM.
- Eval harness: hold out checkpoint t+1, predict item responses, AUC + log-loss. Compare against a "copy last checkpoint" baseline, which will be hard to beat since adjacent checkpoints are near-identical; widen checkpoint gaps if so.
- First result on Fluid data with MMLU subjects: one figure, per-subject mastery curve with uncertainty for one model. Centerpiece of the 295A presentation.
- 295A deliverables: report (abstract, related work, method, preliminary results, 295B plan) and presentation.

### Semester 2 (Jan–May)

- Jan–Feb: OLMo-2 × MathCAMPS eval run. Deep KT (DKT/AKT via pyKT) as ablation.
- Feb–Mar: analyses: item-difficulty invariance (IRT per checkpoint window, drift test), per-skill learning curves, forgetting events, emergence order vs Common Core order.
- Mar–Apr: visualization dashboard (FastAPI + React).
- Apr: paper draft, internal review, arXiv preprint.
- May: NeurIPS submission, 295B report, defense.

### Biggest risks, in order

1. ~~Fluid data lacks per-item correctness.~~ Resolved: it is present.
2. Next-checkpoint prediction is trivially easy because adjacent checkpoints are near-identical. Mitigate with wider gaps and a copy-last baseline.
3. Skill tags limited to MMLU in Phase 1; small subjects give noisy estimates. Acceptable; state it and use the 4-category rollup.
4. GPU access for OLMo-2 (Phase 2). Secure before December.
