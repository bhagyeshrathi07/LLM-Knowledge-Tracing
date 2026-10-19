# Research Brief: Knowledge Tracing for Large Language Models

This brief is written for research assistants (human or AI agents) who will search the literature, read papers, and report back. It explains the project, what we already know, what we have already measured, and the specific questions we need answered. Read all of it before searching; the "What to report" section at the end defines the deliverable.

---

## 1. The project in one paragraph

Large language models (LLMs) are trained in stages, and developers save intermediate snapshots ("checkpoints") throughout training. Today each checkpoint is scored with a single benchmark accuracy number, which hides which skills the model has acquired, when, and whether any were later lost. Intelligent tutoring systems solved the analogous problem for human students decades ago with **knowledge tracing (KT)**: a learner model that maintains a per-skill estimate of mastery, updates it after every observed response, and is validated by predicting the learner's next responses. We are transferring KT to LLMs. Each training checkpoint is a moment in the learner's history; the model's answers to skill-tagged benchmark items at that checkpoint are evidence of its knowledge state at that moment. We fit KT-family models (Bayesian Knowledge Tracing with forgetting, temporal item response theory, and deep KT as an ablation) to item-level checkpoint responses, validate them by predicting held-out later checkpoints, test whether item difficulty stays stable over training, and derive per-skill learning and forgetting curves. We also build an interactive visualization for developers and researchers. Target venue: NeurIPS.

## 2. Context

- **Who:** three-person master's project (CMPE 295A/B, San José State University), advisor Dr. Mahima Suresh Agumbe. Two semesters, currently week 5 of semester 1 (late September 2026). NeurIPS deadline is historically mid-May; paper draft must exist by late April 2027.
- **Advisor feedback so far:** the idea is strong and appears unexplored; NeurIPS is the right target; add a visualization component; the abstract's last paragraph must be readable by engineers with no background in the field.
- **Current state:** abstract written and revised; Phase 1 data verified and ingested; per-skill accuracy curves plotted; next-checkpoint evaluation harness built with five trivial baselines scored. No KT model fit yet.

## 3. Vocabulary (so searches use the right terms)

| Term | Meaning here |
|---|---|
| Knowledge tracing (KT) | Sequential learner model estimating per-skill latent mastery from response correctness, updated per interaction. Classic: Bayesian Knowledge Tracing (BKT, Corbett & Anderson 1995). Deep: DKT (Piech et al. 2015), DKVMN, SAKT, AKT, SAINT. |
| Knowledge component (KC) / skill | The unit mastery is tracked over. In our Phase 1 data, MMLU subjects (57); in Phase 2, Common Core math standards (44). |
| Item response theory (IRT) | Psychometric model: P(correct) = f(learner ability − item difficulty). Static (one ability per learner). 2PL adds item discrimination. |
| Temporal / dynamic IRT | IRT where ability evolves over time (e.g. VTIRT, dynamic Gaussian-process IRT, Kalman-filter IRT). |
| Cognitive diagnosis models (CDM) | Per-skill binary mastery profile for a learner (DINA, NCDM). Static. |
| Item parameter drift / invariance | In IRT, whether an item's difficulty stays constant across administrations. We test it across training checkpoints. |
| Checkpoint | Saved model weights at a point during pretraining. Our "time step." |
| Forgetting | A skill or item answered correctly at an earlier checkpoint and incorrectly later. |
| Emergence | The checkpoint at which a skill's accuracy leaves chance. |
| Next-checkpoint prediction | Our validation: given responses at checkpoints ≤ t, predict item-level correctness at t + h. |

**Important distinction for searching:** there is a large recent literature on "LLMs *for* knowledge tracing" (using an LLM to trace *human* students). That is the reverse of our direction. We want work where the *LLM is the learner being traced*. Both should be reported but kept clearly separate.

## 4. Data we are using

**Phase 1 (verified, in hand):** AI2's Fluid Benchmarking release (`allenai/fluid-benchmarking`, HuggingFace). Per-item, per-checkpoint correctness for six models on six benchmarks.

| Model | Checkpoints | Family |
|---|---|---|
| Amber-6.7B | 73 | LLM360 |
| K2-65B | 46 | LLM360 |
| OLMo-1 7B | 83 | AI2 |
| OLMo-2 7B | 94 | AI2 |
| Pythia-2.8B | 78 | EleutherAI |
| Pythia-6.9B | 78 | EleutherAI |

Benchmarks: ARC-Challenge (1,172 items), GSM8K (1,319), HellaSwag (10,042), MMLU (14,042; 57 subjects), WinoGrande (1,267). TruthfulQA (817) is dropped because its score is continuous. Total: 452 checkpoints × 27,842 items ≈ 12.6M binary responses. The release also includes 2PL IRT item parameters fitted on 102 final models from the Open LLM Leaderboard.

**Phase 2 (planned, needs GPU):** OLMo-2 intermediate checkpoints evaluated on MathCAMPS (Mishra, Poesia & Goodman), ~10k synthetic math problems tagged with 44 Common Core standards K–8. Possibly Pythia's full suite (16 models at 8 parameter sizes, 154 checkpoints each) for more learners.

## 5. What we have already measured

These are our own results on the Phase 1 data. Research assistants should look for prior work that reports comparable numbers or contradicts these observations.

**Per-skill accuracy curves (OLMo-2 7B, MMLU):**
- Three curve shapes appear: sharp early rise then plateau (world_religions 0.32→0.82, high_school_biology 0.18→0.73); slow steady climb (professional_law 0.25→0.43); limited improvement that stays near chance (high_school_mathematics 0.21→0.31, college_mathematics 0.21→0.34, high_school_physics 0.20→0.38). Six subjects finish below 0.35, but under a pointwise Wilson interval every subject exceeds 0.25 at some checkpoint, so "never leaves chance" needs a sustained, multiplicity-adjusted definition before it can be claimed. (Corrected 2026-09-30 after independent review.)
- Largest gains: government_and_politics, management, sociology, psychology (+0.62 to +0.65). Smallest: moral_scenarios, abstract_algebra, econometrics, machine_learning (+0.04 to +0.07).
- Naive forgetting signal (peak accuracy minus final, strictly > 0.08): seven subjects: machine_learning (0.13), abstract_algebra (0.13), global_facts (0.12), public_relations (0.12), college_mathematics (0.10), econometrics (0.10), business_ethics (0.09). The peak is selected after seeing the whole curve and the subjects are small (100–200 items), so these are exploratory candidates, not evidence of forgetting. Any test must control for selecting the peak and screening 57 subjects.
- Across models on high_school_biology: Pythia-2.8B, Pythia-6.9B and Amber finish near chance (about 0.25) with small sustained gains; OLMo-1, OLMo-2 and K2 learn it (0.65 to 0.80). Two clear tiers. Dominguez-Olmedo et al. (ICLR 2025) offer a competing explanation (training on the test task) that must be addressed.

**Next-checkpoint prediction, trivial baselines, pooled over six models:**

| Predictor | h=1 log-loss | h=1 AUC | h=5 log-loss | h=5 AUC |
|---|---|---|---|---|
| global mean (last checkpoint accuracy) | 0.661 | 0.50 | 0.664 | 0.50 |
| skill mean (per-skill accuracy at last checkpoint) | 0.582 | 0.50 | 0.585 | 0.50 |
| copy last (item's last correctness, softened) | 0.432 | 0.73 | 0.473 | 0.70 |
| copy last, calibrated per skill (two-state Markov chain) | 0.378 | 0.73 | 0.404 | 0.70 |
| item mean (item's smoothed history average) | **0.360** | **0.80** | **0.388** | **0.78** |

(AUC computed within skill slices, so constant-within-skill predictors sit at 0.50 by construction.)

Findings: (a) per-skill averages are weak; per-item variance inside a skill dominates, so any learner model without item difficulty will lose; (b) smoothing an item's history beats copying its last value, so per-checkpoint readings are noisy; (c) adjacency inflation exists but is modest (baselines decay slightly from h=1 to h=5); (d) item difficulty is the dominant signal. Any KT model we build must beat item-mean: log-loss below 0.36 and AUC above 0.80 at h=1.

## 6. Design decisions already made

1. **Time step = checkpoint, not item.** Benchmark items are observations, not practice; the model does not learn from them. Mastery transitions once per checkpoint; all items at a checkpoint are simultaneous noisy readings. This is closer to temporal IRT or BKT with batched emissions than to textbook KT.
2. **Primary models: BKT with forgetting + temporal IRT with item difficulty. Deep KT (DKT/AKT) as ablation only,** because we have 6 learners, not thousands. For the ablation, (model, MMLU-subject) pairs serve as 342 pseudo-learners with a leave-one-model-out split.
3. **Phase 1 skills = MMLU's 57 subjects** (called "knowledge components" in the paper, since they are topics rather than skills), with a 4-category rollup for robustness. Other benchmarks enter as single-skill.
4. **Baselines the paper compares against:** static IRT per checkpoint (Fluid Benchmarking), per-skill accuracy curves with thresholds (MathCAMPS, Implicit Curriculum), item-level transition counting (post-training forgetting paper), plus our trivial predictors above.

## 7. Prior work we already know about

Do not re-discover these; do read them if assigned, and look for what cites them or what they missed.

**Closest (overlap with our contribution):**
- Mishra, Poesia & Goodman, "From Next-Token to Mathematics: The Learning Dynamics of Mathematical Reasoning in Language Models," COLM 2025, arXiv 2407.00900. MathCAMPS; skills emerge in curriculum order; only accuracy curves.
- Liu et al. (Neubig group), "What Do Language Models Learn and When? The Implicit Curriculum Hypothesis," April 2026, arXiv 2604.08510. 91 tasks, 9 models; threshold-based emergence; composite after component skills.
- Hofmann et al. (AI2), "Fluid Language Model Benchmarking," COLM 2025, arXiv 2509.11106. Static IRT + adaptive item selection over checkpoints. Our data source.
- Harmon, Hochlehnert, Bethge & Prabhu, "Mapping Post-Training Forgetting in Language Models at Scale," Oct 2025, arXiv 2510.17776. Item-level 1→0 / 0→1 transition counts; post-training only.
- "Item Response Scaling Laws," June 2026, arXiv 2606.07616. IRT over 6,612 checkpoints; no skills, no sequential state.

**Related, not competing:**
- Hu, Chen, Saphra & Cho, "Latent State Models of Training Dynamics," TMLR 2023, arXiv 2308.09543. HMM over aggregate training metrics.
- Liao et al., "Exploring Forgetting in Large Language Model Pre-Training," ACL 2025, arXiv 2410.17018. Entity/fact forgetting.
- FinCDM (arXiv 2508.13491) and a medical CDM paper (Sci. Reports 2026): cognitive diagnosis of a single frozen LLM.
- "Fast and Accurate Probing of In-Training LLMs' Downstream Performances," April 2026, arXiv 2604.01025. Predicts later checkpoint task performance from activations.
- "Deep Knowledge Tracing is an implicit dynamic multidimensional item response theory model," arXiv 2309.12334. Theory bridging DKT and IRT.
- Skill-it! (NeurIPS 2023, arXiv 2307.14430). Ordered skills for data selection.
- LLM-for-KT papers (reverse direction): LLM-KT (2502.02945), NTKT (2511.02599), CLST (2406.10296), LLMKT for dialogues (2409.16490).

## 8. Research questions for assistants

Work through these in order of priority. For each, find papers, read them (not just abstracts), and assess whether they threaten, support, or are irrelevant to our contribution.

**Priority 1 — Novelty threats.** Is there any paper, preprint, workshop paper, thesis, or blog post that:
- fits a sequential latent learner model (BKT, HMM, dynamic IRT, Kalman-filter IRT, state-space model, or deep KT) to an LLM's benchmark responses across training checkpoints?
- validates such a model by predicting held-out later checkpoints at the item level?
- reports per-skill or per-KC mastery trajectories for an LLM during pretraining with uncertainty?
- tests item parameter drift / difficulty invariance across training checkpoints of a neural network?
Search terms to try: "knowledge tracing" + "language model" + checkpoint; "dynamic item response theory" + "language model"; "learner model" + LLM + pretraining; "skill mastery" + checkpoints; "Bayesian knowledge tracing" + "neural network training"; "student model" + "language model as student"; "item parameter drift" + LLM; "psychometrics" + "training dynamics". Check arXiv (cs.CL, cs.LG, cs.AI), OpenReview (ICLR 2026/2027, NeurIPS 2026 submissions), ACL Anthology, EDM/AIED/LAK proceedings, and Semantic Scholar citations of the five "closest" papers above.

**Priority 2 — Methods we should adopt.** Which temporal IRT or KT variant fits our setting (few learners, many time steps, hundreds of observations per time step, uneven gaps between time steps)? Specifically compare: VTIRT (arXiv 2311.08594), dynamic GP-IRT (arXiv 2504.02643), Kalman/state-space IRT, BKT with forgetting (e.g. Khajah et al. 2016 "How deep is knowledge tracing?"), pyBKT capabilities, and any KT model designed for batched observations per time step. Report each one's assumptions, inference method, code availability, and whether it handles item difficulty.

**Priority 3 — Empirical claims to verify or contradict.** For each of our measured findings in Section 5, is there prior work that reports the same or the opposite? In particular: (a) has anyone shown that per-item variance dominates per-skill variance in LLM benchmarks; (b) has anyone measured forgetting of MMLU subjects during pretraining; (c) what emergence orders have been reported for OLMo or Pythia on MMLU; (d) is there evidence that MMLU subjects are a poor proxy for "skills" (e.g. low internal consistency, factor analysis results).

**Priority 4 — Data and benchmark landscape.** Are there other public releases of per-item, per-checkpoint LLM responses beyond Fluid Benchmarking? (Check: OLMo-3 evaluation releases, Pythia evaluation logs, LLM360 Analysis360, DataComp-LM, HF Open LLM Leaderboard checkpoint runs, "Item Response Scaling Laws" release.) Are there skill-tagged benchmarks besides MathCAMPS and MMLU suitable for pretraining-stage models (chance-level early on, non-trivial later)?

**Priority 5 — Visualization precedent.** Any interactive tool that shows per-skill or per-item performance across training checkpoints? (Known: none. Existing tools are loss dashboards, LLM Comparator, feature-dynamics viewers.) If something exists, report it.

## 9. What to report

For every paper you flag, give:
1. Full citation with arXiv ID or DOI and venue/date.
2. One paragraph: what they did (models, data, method, validation).
3. Verdict: **threat** (they did something we claim is new), **support** (they motivate or provide a piece we can build on), or **irrelevant** (looked promising, is not).
4. For threats: exactly which of our claims it undermines and what would still be novel after accounting for it.
5. For methods: assumptions, inference cost, code link, and whether it handles item difficulty and uneven time gaps.

Then a short summary: the three most important things we did not already know, and any change you recommend to the project's framing or abstract.

Be precise about what a paper actually does versus what its abstract implies. Several papers in this space have abstracts that sound like learner modeling but turn out to be accuracy curves with thresholds. Read the methods section.
