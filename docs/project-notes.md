# Knowledge Tracing for LLMs — Project Notes

Working notes for the team. Covers what we are building, what prior work exists, the design problems we found, and the decisions made so far. Last updated 2026-10-27 (week 10 of semester 1): the data tables in Sections 2 and 3 were corrected against the built Parquet table and completed items in Section 7 marked done. Design and roadmap text otherwise dates from 2026-09-30.

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
- Canonical table, as built by `src/llm_kt/ingest.py` (corrected 2026-10-27): `(model, checkpoint_idx, checkpoint_label, step, tokens_seen_b, benchmark, skill, item_id, correct)`, long format, one row per (model, checkpoint, item). `step` and `tokens_seen_b` are null where the source column names do not carry them.

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

| Model  | Size | Trained by | Id in our tables |
| ------ | ---- | ---------- | ---------------- |
| Amber  | 6.7B | LLM360     | `amber-7b`       |
| K2     | 65B  | LLM360     | `k2-65b`         |
| OLMo-1 | 7B   | AI2        | `olmo1-7b`       |
| OLMo-2 | 7B   | AI2        | `olmo2-7b`       |
| Pythia | 2.8B | EleutherAI | `pythia-3b`      |
| Pythia | 6.9B | EleutherAI | `pythia-7b`      |

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

| Model     | Checkpoints | Column format (raw column names, kept as `checkpoint_label`)                              | Parsed                  |
| --------- | ----------- | ----------------------------------------------------------------------------------------- | ----------------------- |
| amber-7b  | 73          | `ckpt_000`, `ckpt_005`, … `ckpt_358` (LLM360 checkpoint index)                            | neither step nor tokens |
| k2-65b    | 46          | `ckpt_093`, `ckpt_099`, … `ckpt_360` (LLM360 checkpoint index)                            | neither step nor tokens |
| olmo1-7b  | 83          | `step0-tokens0B`, `step5000-tokens20B`, … `step650650-tokens2729B`                        | step and tokens         |
| olmo2-7b  | 94          | `stage1-step150-tokens1B`, `stage1-step5000-tokens21B`, … `stage1-step928646-tokens3896B` | step and tokens         |
| pythia-3b | 78          | `step0`, `step2`, `step8`, … `step143000`                                                 | step only               |
| pythia-7b | 78          | same as pythia-3b                                                                         | step only               |

(Column formats corrected 2026-10-27 against `fluid_responses.parquet`; the earlier table listed Amber, K2 and OLMo-1 as plain `step` columns.)

Total: 452 checkpoints × 28,659 items, about 13M cells, no missing correctness values. (Time metadata is incomplete: `step` is null for Amber and K2, `tokens_seen_b` is null for Amber, K2 and Pythia. The Fluid paper reports 467 checkpoints including 61 for K2; the release we inspected has 452 including 46 for K2. We report the release figures.)

Items per benchmark:

| Benchmark     | Items  | Phase 1 treatment           |
| ------------- | ------ | --------------------------- |
| ARC-Challenge | 1,172  | one skill                   |
| GSM8K         | 1,319  | one skill                   |
| HellaSwag     | 10,042 | one skill                   |
| MMLU          | 14,042 | 57 subjects, one skill each |
| TruthfulQA    | 817    | dropped (continuous scores) |
| WinoGrande    | 1,267  | one skill                   |
| Total         | 28,659 | 27,842 kept                 |

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

1. ~~Revise the abstract.~~ Done: `docs/abstract-v2.md` (advisor's last-paragraph fix; "goes unmeasured" softened to "described but not modeled"; "practice opportunity" framing removed per Decision 1).
2. ~~Data feasibility spike.~~ Done 2026-09-29. Fluid data has per-item correctness. See Section 3.
3. ~~Team sign-off on Decisions 3–5~~ (Decisions 1–5 adopted by the team 2026-09-29); **advisor sign-off on Decisions 1–2 still to be recorded.**
4. **Compute plan (Phase 2 only).** OLMo-2 7B has ~900 checkpoints; pick ~50–100 log-spaced. MathCAMPS is generation-based eval. Estimate GPU-hours; source access (SJSU HPC, Colab Pro, Lambda/RunPod). Get a quote before the semester 1 report. No GPU needed for Phase 1.
5. **Related-work section.** Expand Section 4 into a formal write-up for the 295A report.
6. ~~Repo and schema.~~ Done (commit `5a4abcd`, 2026-09-30): `scripts/ingest_fluid.py` converts the six Fluid CSVs into `data/processed/fluid_responses.parquet` per Section 2 (long format: one row per model × checkpoint × item, `step` and `tokens_seen_b` parsed from column names where available, `skill` parsed from MMLU item ids, TruthfulQA dropped). Everything downstream reads it.
7. **Split work across three tracks:** data/infra; modeling; visualization + writing. Weekly advisor meeting at a fixed slot.

### Rest of semester 1 (weeks 9–15, ~Oct 20–Dec 12)

- Baselines: static 2PL IRT per checkpoint (Fluid), per-skill accuracy curves (MathCAMPS-style), threshold emergence (Implicit Curriculum).
- First KT model: BKT with forgetting, one transition per checkpoint, batched observations. pyBKT as a starting point; likely need a custom HMM. **Done 2026-10-27 as a custom log-space HMM with per-item emissions; see §9.**
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

---

## 8. Future work (not this semester)

### Knowledge tracing on chain-of-thought responses

Raised by the advisor on 2026-10-19 (`meetings/2026-10-19-advisor.md`) as a later direction. Recorded here so it is not lost; nothing in the semester-1 or semester-2 plan depends on it.

**The idea.** Same knowledge-tracing machinery, different clock. In the current project the learner is a model observed across training checkpoints. Here the learner is one fixed model observed across the turns of a single conversation, and the chain-of-thought it writes at each turn is the evidence. The quantity being traced is whether the model's working state is still grounded in the facts in play or has started to drift, and the question is what changes over a conversation that precedes a hallucination.

|              | Current project                                 | Chain-of-thought variant                                                                    |
| ------------ | ----------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Learner      | One model across training                       | One model across one conversation                                                           |
| Time step    | Training checkpoint                             | Conversation turn, or a step inside a long reasoning chain                                  |
| Observation  | Right or wrong on a skill-tagged benchmark item | Each factual claim in the chain, labeled correct or hallucinated                            |
| Hidden state | Per-skill mastery                               | Per-topic grounding (how far the model can still be trusted on that topic)                  |
| Validation   | Predict the next checkpoint's answers           | Predict whether the next step hallucinates, before seeing it                                |
| Covariates   | Tokens seen between checkpoints                 | Context length, an earlier wrong premise accepted, an unanswerable question, a topic switch |

**Why it fits.** Decisions 1 and 2 carry over unchanged: the time step is a point at which a batch of observations is taken (all claims in one turn), the state transitions between those points, and small interpretable models (BKT-style or temporal IRT) are the primary tools because there are few "students". The next-step prediction test is the same validation we use now.

**Data it would need.** Multi-turn conversations with (a) the model's full reasoning trace at every turn and (b) a correct / hallucinated label on each claim, plus the covariates above. Pieces of this exist separately; to our knowledge none combines all three, so a usable dataset would probably have to be assembled or labeled by us. Candidate ingredients, all to be verified before use:

- *Traces.* Open reasoning models publish their chains of thought (for example DeepSeek-R1 and OLMo-3 Think), so traces can be generated on any prompt set at the cost of inference.
- *Hallucination labels.* Existing benchmarks are mostly single-turn: TruthfulQA, HaluEval, FELM, FActScore (claim-level factuality for long-form answers), RAGTruth (grounding against retrieved context). HalluDial labels hallucination in dialogue but, as far as we know, without reasoning traces.
- *Claim labeling at scale.* Would need an automatic judge (an LLM checking each claim against a reference or retrieved evidence) with a human-labeled sample to report agreement, as planned for Option B of Decision 4.

**Open questions.** What counts as a "skill" or topic in free conversation (hand-tagged domains, clustered claims, or the retrieval source); whether a step-level clock inside one long chain is too fine to carry a state; and how to separate hallucination caused by the conversation's history from hallucination the model would have produced on the first turn anyway.

**Status.** Future work only. Revisit after the Phase 2 go/no-go (2026-12-12) if semester 2 has room.

---

## 9. BKT: design, why, results (2026-10-27)

Action item from the 2026-10-19 advisor meeting: "log-space BKT with forgetting and per-item emissions, scored against the item-mean bar." Code: `src/llm_kt/bkt.py` (25 tests in `tests/test_bkt.py`), `scripts/run_models.py --predictors bkt`, `scripts/analyze_bkt.py`. Design document: `docs/superpowers/specs/2026-10-27-bkt-design.md`; implementation plan: `docs/superpowers/plans/2026-10-27-bkt.md`. Every table below is also stored as a CSV under `docs/results/bkt/`, with the console report in `docs/results/bkt/analysis.txt`.

### What BKT is here

Bayesian Knowledge Tracing is the second model family adopted in Decision 2. For each MMLU subject the model has a hidden state with two values, *not learned* and *learned*. The state can switch once per checkpoint: up with probability `learn`, down with probability `forget`. At each checkpoint every question in the subject is one noisy reading of that state: a learned subject answers question `i` correctly with probability `1 - slip_i`, an unlearned one with probability `guess_i`. The readings at one checkpoint are combined by adding their log-probabilities, so a subject with 1,500 questions gives one very sharp reading per checkpoint instead of 1,500 practice opportunities. That is Decision 1 applied to BKT: the model learns between checkpoints, and the test is a thermometer.

To predict `h` checkpoints ahead, the filtered state at the last seen checkpoint is pushed through the transition `h` times, which pulls it toward its long-run value, and the prediction for question `i` is `P(learned) (1 - slip_i) + P(not learned) guess_i`. There is no separate forecast-uncertainty knob; longer horizons get softer on their own.

### The four design choices and why

- **Log space, own implementation.** pyBKT multiplies raw emission probabilities before normalising. With 100 to 1,534 items per checkpoint the product underflows to zero, so pyBKT cannot be used on batched checkpoints. We run forward-backward on log-probabilities with `logsumexp`; a test checks a 2,000-item checkpoint against brute force.
- **Forgetting.** A forget rate is native to BKT and is what lets the state go down. The item mean never forgets. The forgetting analysis (§5.6 of the briefing) is the paper's story, so the learner model must be able to express it.
- **Per-item guess and slip.** With one guess/slip pair per subject, every question in the subject gets the same probability at a checkpoint, and within-subject AUC is 0.50 by construction. The baseline ladder says per-question information is what matters, so each question gets its own guess and slip. We still run the per-subject version as a limit to show the gap.
- **Shrinkage with recency, two anchors.** 14,042 questions times two parameters, each seen at most 94 times, overfit if left free and trade off against the state. Each question's guess and slip start at an anchor and move away only as evidence accumulates, with pseudo-count `kappa`. Recent checkpoints can count more through `rho^(age)`. Anchor `skill`: the subject's own shared guess/slip, learned by EM. Anchor `fluid`: the guess and slip implied by Fluid's published 2PL parameters at two global ability levels, the "borrowed parameters" variant.

Fitting is MAP-EM per subject at every origin of the walk-forward harness, warm-started from the previous origin. The skill-level rates get a weak pseudo-count prior so a two-checkpoint history cannot push a rate to exactly 0 or 1. The item-level update with `rho < 1` is a forgetting-factor approximation, not exact EM. `kappa` at or above 1e5 pins every question to its anchor exactly, which is how the per-subject limit is produced; this guarantees the ties that give AUC 0.50 rather than leaving them to floating-point luck.

### The grid and the selection rule

Everything is OLMo-2 on MMLU (Decision 6: learner models on MMLU only), horizon 1, walk-forward with at least two checkpoints of history. The grid is `kappa` in {0.1, 0.3, 1, 5, 20, 100, 1e6} by `rho` in {1.0, 0.9, 0.7} for the skill anchor, and `kappa` in {1, 5, 20, 100, 1e6} by the same `rho` by `(theta_lo, theta_hi)` in {(-1, 1), (-2, 1), (-1, 2)} for the Fluid anchor. The `kappa` values 0.3 and 0.1 were added after the first pass put the optimum at the grid's lower edge. The best skill-anchored and best Fluid-anchored setting are chosen by validation log-loss (checkpoints 56 to 74) and only then scored on the test block (75 to 93) and at horizons 5 and 20. Test numbers are reported, never ranked.

The grid took one evening on a laptop: about 10 minutes per setting at horizon 1 (63 settings in eight parallel chunks), 70 minutes per setting at horizon 20.

### Results at horizon 1 (OLMo-2, MMLU)

Validation-block ranking, the one used to choose. Full table: `docs/results/bkt/validation_table.csv` (63 rows).

| Setting                                         | Val log-loss | Val AUC |
| ----------------------------------------------- | ------------ | ------- |
| BKT, skill anchor, `kappa` 0.1, `rho` 0.9       | 0.326        | 0.925   |
| BKT, skill anchor, `kappa` 0.3, `rho` 0.9       | 0.326        | 0.925   |
| BKT, Fluid anchor (−2, 1), `kappa` 1, `rho` 0.9 | 0.331        | 0.923   |
| Recency-weighted item mean, `rho` 0.9           | 0.332        | 0.927   |
| BKT, skill anchor, `kappa` 1, `rho` 0.9         | 0.351        | 0.925   |
| BKT, skill anchor, `kappa` 0.1, `rho` 1         | 0.375        | 0.910   |
| Item mean (the bar)                             | 0.386        | 0.912   |
| BKT, skill anchor, `kappa` 5, `rho` 1           | 0.414        | 0.908   |
| Skill mean                                      | 0.626        | 0.500   |
| BKT per-skill limit (`kappa` 1e6)               | 0.630        | 0.500   |

Selected: skill anchor with `kappa` 0.1 and `rho` 0.9; Fluid anchor with `theta_lo` −2, `theta_hi` 1, `kappa` 1, `rho` 0.9. Both optima sit at `rho` 0.9; `rho` 0.7 is always worse. The Fluid optimum is at its grid's lower `kappa` edge; that grid was not extended (each Fluid chunk took 3.5 hours, and the skill anchor already won at every shared `kappa`).

Scores for the chosen settings. "±" is a bootstrap standard error over the 57 subjects. Stored in `docs/results/bkt/h1_scores.csv`.

| Predictor                                       | Pooled log-loss | Pooled AUC | Test log-loss | Test AUC      |
| ----------------------------------------------- | --------------- | ---------- | ------------- | ------------- |
| Item mean (the bar)                             | 0.457           | 0.862      | 0.375 ± 0.015 | 0.916 ± 0.006 |
| Recency-weighted item mean, `rho` 0.9 (control) | 0.403           | 0.880      | 0.317 ± 0.019 | 0.933 ± 0.006 |
| BKT, skill anchor, `kappa` 0.1, `rho` 0.9       | 0.411           | 0.880      | 0.310 ± 0.023 | 0.933 ± 0.006 |
| BKT, Fluid anchor, `kappa` 1, `rho` 0.9         | 0.388           | 0.882      | 0.316 ± 0.022 | 0.929 ± 0.007 |
| BKT per-skill limit (`kappa` 1e6)               | 0.637           | 0.500      | 0.623 ± 0.011 | 0.500         |
| Skill mean                                      | 0.632           | 0.500      | 0.618 ± 0.013 | 0.500         |

Paired differences on the test block, BKT minus the other predictor, with 95% intervals. Negative log-loss and positive AUC favour BKT. Stored in `docs/results/bkt/h1_paired_differences.csv` and `extra_paired_differences.csv`.

| Comparison                                        | Log-loss                | AUC                     | Reading                               |
| ------------------------------------------------- | ----------------------- | ----------------------- | ------------------------------------- |
| BKT skill anchor minus item mean                  | −0.065 [−0.078, −0.045] | +0.017 [+0.014, +0.020] | BKT wins on both                      |
| BKT Fluid anchor minus item mean                  | −0.059 [−0.073, −0.039] | +0.014 [+0.009, +0.017] | BKT wins on both                      |
| BKT per-skill limit minus item mean               | +0.248 [+0.216, +0.270] | −0.416 [−0.425, −0.403] | the limit loses badly                 |
| BKT per-skill limit minus skill mean              | +0.005 [+0.004, +0.006] | 0 (both 0.50)           | the limit is the skill mean           |
| BKT skill anchor minus recency-weighted item mean | −0.007 [−0.013, +0.002] | −0.001 [−0.002, +0.000] | tie                                   |
| BKT Fluid anchor minus recency-weighted item mean | −0.000 [−0.008, +0.009] | −0.004 [−0.007, −0.002] | tie on log-loss, control ranks better |

### Further ahead

Test-block log-loss, pooled in brackets. Horizon 5 and 20 runs use the settings chosen at horizon 1. Stored in `docs/results/bkt/h5_scores.csv`, `h20_scores.csv` and the paired-difference files.

| Horizon | Item mean     | Recency item mean | BKT skill anchor | BKT Fluid anchor | BKT skill minus item mean | BKT skill minus recency item mean |
| ------- | ------------- | ----------------- | ---------------- | ---------------- | ------------------------- | --------------------------------- |
| 1       | 0.375 [0.457] | 0.317 [0.403]     | 0.310 [0.411]    | 0.316 [0.388]    | −0.065 [−0.078, −0.045]   | −0.007 [−0.013, +0.002], tie      |
| 5       | 0.382 [0.484] | 0.326 [0.430]     | 0.323 [0.450]    | 0.326 [0.405]    | −0.059 [−0.071, −0.041]   | −0.002 [−0.009, +0.008], tie      |
| 20      | 0.414 [0.545] | 0.355 [0.487]     | 0.359 [0.525]    | 0.356 [0.436]    | −0.055 [−0.067, −0.039]   | +0.003 [−0.002, +0.011], tie      |

Unlike T-SKIRT (§5.4 of the briefing), BKT keeps its lead over the item mean at horizon 20 on the test block, because its per-item part never spreads out with horizon the way a forecast variance does. Against the recency-weighted item mean it is a tie at every horizon on the test block, and on the pooled run the control is better (by 0.008, 0.020 and 0.038 log-loss at horizons 1, 5 and 20, all intervals excluding zero).

The Fluid anchor is the one setting that beats the control anywhere: pooled over the whole run it leads the recency-weighted item mean by 0.015, 0.025 and 0.051 log-loss at horizons 1, 5 and 20 (intervals exclude zero), while tying it on the late test block. The pooled run includes the early checkpoints, where an item has been seen a handful of times and a prior from Fluid's published parameters is worth something; by the test block every item has 75 or more observations and its own recent history is all that matters.

### What `kappa` and `rho` say

- **`kappa` 0.1** means each question's guess and slip move off the subject anchor after a fraction of one observation. The question-level parameters are essentially free, and the grid is flat below `kappa` 1 (0.326, 0.326, 0.327 at 0.1, 0.3, 0.03). The model wanted every question to carry its own numbers, which is the baseline-ladder finding again (§5.1 of the briefing).
- **`rho` 0.9** (half-life of about 6.6 checkpoints) is where the win comes from. With `rho` 1, no setting beats the item mean on the test block (0.364 and 0.372 against 0.375 for `kappa` 0.1 and 1), and on validation they tie it (0.375 and 0.380 against 0.386). Turning recency on moves the test log-loss from 0.364 to 0.310. This matches what Ronak reported for DriftIRT.
- With `kappa` near zero and `rho` below one, BKT's per-item emission for a question is a recency-weighted average of that question's own answers, split by the hidden state. When the state is constant, that is exactly the recency-weighted item mean. That is why the control was built and why it ties.

### The mastery figure

`docs/figures/bkt_mastery_olmo2-7b.png` is the selected setting fitted once on all 94 OLMo-2 checkpoints (tables: `docs/results/bkt/bkt_mastery_olmo2-7b.csv`, `bkt_skill_params_olmo2-7b.csv`). It should be read as a diagnostic, not as a learning curve:

- The filtered state is 0 or 1 (outside 0.01 to 0.99) on 98.5% of subject-checkpoints. With hundreds of readings per checkpoint a two-state model is always certain.
- The median subject switches state once; the worst switches 33 times. Seven subjects never leave the "not learned" state at all (`college_mathematics`, `college_physics`, `elementary_mathematics`, `global_facts`, `high_school_mathematics`, `high_school_physics`, `professional_accounting`): EM stopped after three iterations with `learn` at 0.001, and their per-question parameters carry all the information. Fifty of 57 subjects end in the learned state.
- For several subjects the two states are not "knows" and "does not know". `moral_scenarios` ends with `slip` 0.74 in the learned state and `guess` 0.21 in the other, which is two near-identical states that the per-question parameters then correct. This is the identifiability problem the design document predicted for small `kappa`: the per-question parameters explain the history and the state has nothing left to do.
- Learn rates are 0.05 to 0.86 per checkpoint (median 0.10); forget rates have median 0.001 and a tail to 0.35. The tail is not evidence of forgetting in the model: it belongs to subjects whose state flips back and forth to relabel noise.

**The per-skill limit is the figure to read for when subjects were learned.** `docs/figures/bkt_mastery_olmo2-7b_k1e+06_r1.png` is the same fit with `kappa` 1e6 and `rho` 1 (classic BKT, one guess/slip pair per subject; tables with the `_k1e+06_r1` suffix). There the state means what it says: curves are single steps, 45 of 57 subjects end learned, the median subject switches on at checkpoint 8 of 93 (about 9% of the way through training; earliest `virology` at 6, latest `machine_learning` at 29), and only nine subjects switch more than twice. Fitted forget rates are at most 0.047 per checkpoint (`professional_medicine`), with a median of zero, so this model sees no forgetting in OLMo-2 on MMLU, consistent with §5.6 of the briefing. Twelve subjects never leave "not learned": the mathematics and physics subjects, `abstract_algebra`, `formal_logic`, `econometrics`, `global_facts`, `moral_scenarios` and `professional_law`, which are the subjects where OLMo-2 stays near chance, so the learned state is never the better explanation. This is the mastery figure Decision 2 promised, and it costs 0.31 log-loss against the predictive setting.

### Honest verdict

The briefing's open question 1 asked whether BKT with per-item emissions would also only tie the item mean. It does better than that: on the late test block it beats the bar by 0.065 log-loss and 0.017 AUC at horizon 1 and holds the lead to horizon 20. But the control shows where the gain lives. A recency-weighted item mean, one line of code with the same `rho`, ties BKT on the test block at every horizon and beats it pooled. The per-skill state adds nothing measurable at the selected setting, and classic per-skill BKT (the `kappa` 1e6 limit) is the skill mean with AUC 0.50. The shared finding across T-SKIRT, DriftIRT and BKT is therefore sharper than the one proposed in the briefing: *recent per-item state is what matters*. The only place a prior beyond the question's own history helped was the Fluid anchor early in training, when histories are short.

What BKT still offers the project is the forgetting machinery and the interpretable state, and the figure shows that at a setting chosen for prediction the state is not interpretable. The two goals pull against each other: prediction wants small `kappa`, interpretation wants large.

### Limitations

- Checkpoint index is the clock; uneven spacing in steps or tokens is ignored (T-SKIRT and DriftIRT can use the other clocks).
- The recency-weighted item step is a forgetting-factor approximation inside EM, not exact EM; the fitted `loglik` is not a likelihood of the data under one model.
- Two of 57 subjects hit the 50-iteration cap in the full-history fit, and the seven never-learned subjects stopped at a cold-start fixed point. A second initialisation (start in the learned state) was not tried.
- One model (OLMo-2 7B), one benchmark (MMLU), per Decision 6. The per-skill limit's AUC of 0.50 is by construction and holds for any model.
- Selection used horizon 1 only; the horizon 5 and 20 runs reuse those settings. `kappa` 1 is better than `kappa` 0.1 pooled at horizons 5 and 20 (0.435 against 0.450; 0.477 against 0.525) but worse on the test block, so the choice does not change the verdict.

### Next steps

1. Decide with the team whether the recency-weighted item mean replaces the item mean as the bar. If it does, every learner model is re-scored against it; T-SKIRT and DriftIRT have not been.
2. Fit the per-skill limit with a second initialisation and report its mastery curves as the forgetting story's figure; that figure, not the selected setting's, is the one to show the advisor.
3. Pool across the six models at the skill level (shared `learn` and `forget` per subject, per-model emissions) as the one way a per-skill state could add information the item's own history does not have.
4. Try the Fluid anchor with `kappa` below 1 and with `rho` between 0.9 and 1, since both optima were at grid edges in the directions that matter least for the verdict.
