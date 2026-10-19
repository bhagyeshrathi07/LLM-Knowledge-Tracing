**Empirical, implementation, and method audit of Workbook 1**

Audit performed 2026-09-30. Workbook PDF page numbers count the cover as page 1. No workbook or implementation source was edited. Supporting numerical checks and reproduction logs are preserved in this review folder.

**Overall judgment**

**Agree that the repository has a functioning, tested initial data/curve/baseline pipeline. Push back on several claimed curve findings, on the description of the baselines, and on the assertion that the proposed learner models can be integrated with only trivial adaptations.** The baseline table itself matches the stored results, and its numbers have a clear implementation. The workbook currently overstates what those results establish about latent mastery, forgetting, and a validated future-prediction protocol for models that have not yet been implemented.

**Verified accomplishments**

- `.venv/bin/pytest -q`: **50 passed**. `.venv/bin/ruff check .`: **All checks passed**. Thus §§5.3, 5.6, and 8.1 accurately report these present local checks. This is not evidence that planned BKT/dynamic-IRT models work.
- The supposedly delivered Parquet files actually exist, although gitignored: `fluid_responses.parquet`, `skill_curves.parquet`, `baseline_results.parquet`, `baseline_results_h5.parquet`.
- Canonical data contains **12,584,584 rows = 452 × 27,842**, six models, 57 MMLU subjects, and strictly binary, nonmissing correctness. File size is **27,537,531 bytes** (27.54 MB or 26.26 MiB). Rounded “27 MB” is acceptable.
- Checkpoints per model: Amber 73, K2 46, OLMo-1 83, OLMo-2 94, Pythia-2.8B 78, Pythia-6.9B 78. Canonical names `pythia-3b`/`pythia-7b` are release aliases for 2.8B/6.9B, not necessarily a substantive discrepancy.
- Items by benchmark: ARC-Challenge 1,172; GSM8K 1,319; HellaSwag 10,042; MMLU 14,042; WinoGrande 1,267. There are **61** evaluation skill slices: 57 MMLU subjects plus four whole-benchmark slices.
- All six models have higher overall accuracy at the last retained checkpoint than at the first. OLMo-2 rises **0.241506 → 0.657999**, agreeing with §8.4's 0.24 → 0.66.
- Recomputing `skill_accuracy` over the canonical data reproduces all **27,572** saved curve rows exactly.
- In the current stateless baseline implementations, history is restricted to indices ≤ origin, and target response values are withheld. `src/llm_kt/evaluate.py:68–75` implements this correctly. Items recur across time intentionally; this supports prediction on previously observed items, not unseen-question generalization.
- The cached source release includes fitted MMLU item parameters (14,042 rows, columns `a`, `b`) and all six source CSVs. This supports the existence of inputs for a future IRT implementation. It does not show that a static-IRT ability baseline has been implemented; it has not.

**Baseline table: agree with numbers, clarify their estimand**

The following are recalculated aggregates of every row in the saved result files, using the repository's `summarize` function. Rounded values match §3.1 (PDF p.15).

| Predictor | h=1 log loss | h=1 AUC | h=5 log loss | h=5 AUC |
|---|---:|---:|---:|---:|
| global_mean | 0.661407 | 0.500000 | 0.664371 | 0.500000 |
| skill_mean | 0.582006 | 0.500000 | 0.584732 | 0.500000 |
| copy_last | 0.432347 | 0.730598 | 0.473154 | 0.704720 |
| copy_last_calibrated | 0.377524 | 0.728548 | 0.403830 | 0.704625 |
| item_mean | 0.360227 | 0.800198 | 0.388302 | 0.779227 |

h=1 scores 440 target checkpoints and 12,250,480 item outcomes per predictor; h=5 scores 416 target checkpoints and 11,582,272 outcomes. `min_history=2` means scoring starts at target index 2 for h=1, index 6 for h=5. It does not score 452 target checkpoints; 452 is the source count. Saved result row counts are 134,200 and 126,880 because each target produces 61 skill slices for each of five predictors.

**Missing qualifier:** Log loss/Brier/accuracy are weighted by item counts; AUC is an unweighted mean over defined (model, target, skill) slices. Models with more retained checkpoints have more weight. These are not equal-model macro-averages; AUC is not one pooled ROC AUC. See `src/llm_kt/evaluate.py:109–132`. Explicitly describe this beneath the table.

**MMLU scope:** The 0.360 bar is over all five retained benchmarks, although the project's stated skill focus is 57 MMLU subjects. On MMLU alone, item_mean has **0.483458 log loss / 0.792491 AUC at h=1**, and **0.507538 / 0.771217 at h=5**. Include both aggregates and predeclare which is primary. Using 0.360 to assess an MMLU-only learner would be an invalid comparison. `item_mean` is best here on log loss and AUC, but the calibrated copy baseline has slightly better h=1 Brier and accuracy; do not describe item_mean as universally best across all four metrics.

**Required corrections to concrete empirical claims**

**E1 — Claimed forgetting-subject count is wrong (high priority)**

**§8.4, PDF p.26; also §3.2's reference to “five candidate forgetting subjects.”** The saved curves give **seven** OLMo-2 MMLU subjects with a peak-minus-final drop strictly greater than 0.08:

| Subject | Peak minus final |
|---|---:|
| machine_learning | 0.133929 |
| abstract_algebra | 0.130000 |
| global_facts | 0.120000 |
| public_relations | 0.118182 |
| college_mathematics | 0.100000 |
| econometrics | 0.096491 |
| business_ethics | 0.090000 |

`medical_genetics` is exactly 0.08 (77/100 minus 69/100), and ordinary binary floating-point subtraction can misleadingly classify it as >0.08. Use integer-count comparisons or tolerance. The workbook's count of five does not reproduce. Correct to seven candidate subjects under this stated rule; **these remain exploratory drops, not established forgetting**.

**E2 — “6 of 57 never leave chance” is undefined and not supported by the data (high priority)**

**§8.4, p.26.** All 57 OLMo-2 subjects at some point have raw accuracy above .25; all 57 also have a pointwise 95% Wilson lower bound above .25 at some checkpoint. Thus neither natural reading gives six. Only **three final-checkpoint intervals** include .25: abstract_algebra, econometrics, global_facts. That is a different endpoint question and does not establish absence of learning. A sustained or multiplicity-adjusted rule might produce a different number, but no such rule appears in the implementation or workbook.

**Correction:** Remove the count pending a specified, implemented criterion. Say that several subjects remain near the chance reference line and that formal emergence tests are planned. Pointwise searches over 94 checkpoints are themselves multiple comparisons; these observations must not be turned into “all subjects demonstrably learned” either.

**E3 — Curve endpoints and interpretation are overstated (medium/high)**

**§8.4, p.26.** `world_religions` is **0.321637 → 0.824561**, not .25 → .82. `high_school_mathematics` is **0.211111 → 0.307407**, peaking at **.351852**, with maximum pointwise Wilson lower bound **.297330**. “Flat at chance for the whole run” is too categorical. Better: limited improvement relative to subjects such as world_religions, with formal interpretation pending.

For high_school_biology, Amber peaks .316129 (max Wilson lower .266888), Pythia-2.8B peaks .316129 (.266888), Pythia-6.9B peaks .319355 (.269937). They finish near .25 and lack the sustained gains of OLMo/K2; **“never leave chance” is literally false** under the displayed pointwise interval convention.

**E4 — “No missing values” requires restriction to response fields (medium)**

**§8.4 p.26; §5.1 p.20.** The table contains **3,313,198 missing `step` cells** (all Amber and K2 rows) and **7,656,550 missing `tokens_seen_b` cells** (all Amber, K2, and both Pythias). All item/skill/correctness identifiers are nonmissing. Say “no missing correctness values; time metadata remain incomplete.” The schedule plans to fill Pythia/Amber tokens but omits K2. Irregular-gap models and comparisons need a declared treatment for all four affected runs.

**E5 — item_mean is not Laplace smoothing (medium)**

**§3.1 p.15; `src/llm_kt/baselines.py:42–56`.** Code uses `(k + alpha * global_history_mean) / (n + alpha)`, with alpha=1. This is shrinkage toward an empirical global-history prior. Standard binary add-one/Laplace smoothing is `(k+1)/(n+2)`. Rename the existing predictor description to “item history mean, shrunk toward the global history mean (prior weight 1).” Its numbers remain valid. Calling it “Heineman-style” is acceptable only as a conceptual analogy; the item-level empirical-prior implementation is this team's adaptation.

**E6 — Cross-model x-axis is not fraction of training (medium)**

The plot script labels `checkpoint_idx / max(checkpoint_idx)` as fraction of the pretraining run (`scripts/plot_skill_curves.py:130–137`). Checkpoints are not uniformly spaced in tokens/steps, and K2's retained release starts at ckpt_093. This is **normalized index within the observed checkpoint sequence**, not elapsed training fraction. Relabel now, or use actual elapsed tokens once recovered. The workbook's planned uneven-gap adjustment already acknowledges this issue.

**Research-design and implementation issues to resolve**

**M1 — BKT as described cannot meet the required AUC bar (high)**

**§3.3 p.16 and §3.1 p.15.** With one mastery state and shared guess/slip per skill, every item in the same skill at the same target gets `P(correct)=P(mastered)*(1-slip)+(1-P(mastered))*guess`. Consequently within-skill AUC is **0.5**, exactly like skill_mean. It cannot exceed .80 regardless of how well it tracks the skill. This is a structural mismatch between the proposed model and success criterion.

Packing each item into a different pyBKT subpart permits item-specific emissions, but then the model has **2J emission parameters** plus transition/initial parameters per skill, not a simple four/five-parameter model. pyBKT's M-step estimates separate guess/slip for each subpart. Explicitly choose shared, fixed external, tied, or regularized item emissions and align evaluation with that choice. A fair AUC ablation should hold item heterogeneity constant across static and dynamic variants. [pyBKT M-step source](https://raw.githubusercontent.com/CAHLR/pyBKT/master/source-py/pyBKT/fit/M_step.py).

**M2 — “Only a packing shim; no algorithm change” is unproven and has a concrete numerical failure (high)**

**§3.3 p.16; §8.5 p.26.** The pure-Python pyBKT E-step does multiply simultaneous subpart emissions and advances state between columns, so batching exists. However it multiplies all emission probabilities before normalization, rather than accumulating in log space. On the actual first OLMo-2 professional_law checkpoint (1,534 items, 377 correct), reproducing this operation with plausible guess=.25 and slip=.10 gives likelihoods `[0.,0.]` and normalizer 0; their log likelihoods are about -855.48 and -2703.81. This causes nonfinite posteriors if passed through that normalization. This is a calculation using the source's algorithm; the library is not installed/executed here. The exact backend/version intended must be tested.

**Correction:** Schedule stable batched-emission validation, including largest subject and parameter tying. A log-space implementation or equivalent rescaling may be needed; do not promise integration is only data packing. [pyBKT Python E-step](https://raw.githubusercontent.com/CAHLR/pyBKT/master/source-py/pyBKT/fit/EM_fit.py) (likelihood multiplication and subsequent normalization).

**M3 — Forecast horizon and future elapsed time are absent from Predictor (high)**

**§§3.1/5.2/5.3; `evaluate.py:22–33,68–75`; `baselines.py:81–120`.** `target` exposes only item_id, benchmark, skill; `predict` is not told h, target index, or target token count. In particular, copy_last_calibrated estimates **one-step** transition probabilities for both h=1 and h=5. The h=5 table is a valid evaluation of unchanged one-step predictions against farther outcomes, but **not a five-step Markov-chain forecast**.

**Correction:** Supply forecast context/horizon and actual elapsed training amount, or explicitly configure horizon in each predictor and test that contract. For a homogeneous chain use the transition matrix power h, or estimate lag-h transitions using only observed history. For irregular gaps define time-scaled transitions. Recompute the comparison if claiming an h-step Markov baseline. This requirement is necessary for the planned gap-aware dynamic model too.

**M4 — Static target ability is not a future-prediction baseline (high)**

**§3.2 p.16; §8.5 p.26.** Fitting an independent ability to each checkpoint using that checkpoint's answers is descriptive/in-sample estimation. It cannot be scored as a held-out future forecast on those same answers. Define separate evaluation regimes: (a) fit last-observed-checkpoint ability and explicitly extrapolate/persist to t+h; (b) diagnostic item holdout within the target checkpoint; (c) temporal filtering forecast without target responses. Label each appropriately. Public item calibration may be a legitimate fixed external input, but its calibration population and potential overlap with evaluated model families must be documented.

**M5 — Current holdout protects baseline inputs, not the entire future model-selection process (medium/high)**

The code correctly withholds each target's responses from current stateless predictors. It has no `fit` protocol or reset per model/origin and does not guarantee that a future predictor's pretrained parameters exclude future outcomes. Results have already been inspected and used to design models. Calling the numerical threshold “pre-committed” does not reserve an independent test set.

Specify training/calibration data, chronological validation/test blocks, hyperparameter selection, state resets, and leave-one-model-out handling. Do not use full-run smoothing estimates in forward forecasts. Filtered past states and retrospectively smoothed trajectories should be clearly distinguished in the dashboard. Current results remain valid exploratory rolling-origin results.

**M6 — Transition-count test is off by one unless conventions are stated (medium)**

**§3.3 p.17.** For T observed checkpoints with an initial state at the first checkpoint, there are **T−1 transitions between observed checkpoints**, not T. pyBKT performs its first between-observation transition into the second column. Correct the proposed test to assert one transition per inter-checkpoint gap and no transitions between simultaneous items; separately specify any prior-to-first-observation transition convention.

**M7 — Uncertainty and forgetting need explicit statistical targets (medium/high)**

**§3.2, §§4.2/5.4, §8.4.** Existing bands are pointwise Wilson binomial intervals for raw item accuracy, not posterior credible intervals on mastery and not temporal variability across retraining or decoding draws. Dependence among related benchmark items also affects the simple binomial interpretation. Preserve the accurate “Wilson” description and define what population/uncertainty the bands represent.

Paired McNemar testing is sensible for a **fixed** checkpoint pair. Choosing each subject's maximum first and then testing max-versus-final incurs selection over time; testing 57 subjects introduces additional multiplicity. A prespecified max-drop procedure or appropriate resampling/multiplicity correction is needed. A significant response-accuracy decrease establishes a behavioral change under the given evaluation, not uniquely latent skill loss. Separate response flips, paired accuracy decline, and latent-state forgetting.

**M8 — Subject clustering alone does not establish valid inference (medium)**

The planned subject-clustered SEs address within-subject dependence under their assumptions, but many repeated checkpoints belong to the same six models and model-wide shocks can correlate subjects. The chosen generalization target matters: fixed six runs vs a population of model families, fixed item bank vs new items. Specify paired loss differences, weighting, temporal dependence, and model-level uncertainty/sensitivity. Six models are not enough to invoke ordinary large-cluster approximations blindly. This is an unresolved design choice, not proof that every subject-clustered analysis is invalid.

**M9 — Item drift and ability drift require identification constraints (medium/high)**

For an IRT model, shifting every ability and every difficulty by the same time-varying amount leaves response probabilities unchanged. An item-bank drift test must fix an anchor/reference, constrain mean drift, or otherwise separate ability movement from item movement. More observations or a bank-level test alone does not resolve this invariance. Define how discrimination scales and location are linked across time/models. Interpret external bank initialization as an assumption to assess, not evidence that items are invariant.

**M10 — Recovery of fitted parameters is useful but not complete validation (medium)**

Generating data from one's own fitted model and refitting can reveal finite-sample failure. It does not validate that the model describes the real data, and one generating parameter setting does not establish robustness. Include varied transition/emission strength, sample size, dependence/misspecification, parameter/state recovery, and interval coverage. Do not treat “six trajectories” as a universal effective sample size: information depends on time depth, dependence and which parameters are being estimated.

**Completeness and reproducibility details**

- §1.2/Ch.2 say the first two phases need no GPU; §§4.1/5.1/8.3/8.5 define Phase 2 as new OLMo-2 MathCAMPS evaluations requiring GPU. Reconcile as **Phase 1 needs none; preferred MathCAMPS Phase 2 needs GPU, with CPU-only fallback data**.
- Architecture distinguishes most planned models appropriately, but a curves table is not itself a `Predictor`; there is no forecast adapter in `curves.py`. No BKT/T-SKIRT/dynIRT, FastAPI, React, Docker, CI workflow, parameter-recovery or drift test code is currently in the repository. That is acceptable for a plan if stated as planned.
- `scripts/run_baselines.py:25,31,45` uses the same default output for all horizons: running `--horizon 5` alone overwrites h=1 output. Document `--out data/processed/baseline_results_h5.parquet` or use distinct default naming. The current two saved files are present and correct; this is a reproducibility hazard, not evidence they were overwritten.
- Ingestion fetches an unpinned Hugging Face revision (`ingest.py:129–134`), and dependencies are lower bounds only. Record the actual source snapshot (`e0a11a3be90972ffaa12c6e63957cfd5be3a3259` in this cache), package versions, commands, and model output manifests. The nominal schema alone does not guarantee future exact reproducibility.
- `predict` validates output shape only, not finiteness or [0,1] range (`evaluate.py:73–75`). Once real temporal models are added, reject invalid probabilities rather than allowing clipped log loss or NaN summaries to obscure failures.
- §5.6 places meeting notes under docs; actual notes are in top-level `meetings/`. Correct the layout description.
- Claimed 47-second ingestion and eight-minute-per-horizon runtime are plausible observed workstation timings but need hardware/version/command and a run log to be independently attributable. I did not rerun the entire original all-origin baseline pipeline merely to reproduce those timing claims.

**Recommended compact rewrite for §8.4 curve paragraph**

“OLMo-2 shows heterogeneous subject trajectories. World religions rises from 0.322 to 0.825, while professional law rises from 0.246 to 0.425 and high-school mathematics from 0.211 to 0.307. Seven MMLU subjects have an exploratory peak-to-final accuracy decrease greater than eight percentage points. These postselected decreases are candidates for paired and multiplicity-adjusted analysis, not established forgetting. On high-school biology, Amber and the two Pythia models finish near the .25 chance reference and show much smaller sustained gains than OLMo and K2. Formal emergence criteria remain to be specified.”

**Completed reproduction check**

The audit re-ingested **all six cached original source CSVs** using current `wide_to_long`; values and row ordering match every row of the saved canonical table. Parquet string-type normalization required ignoring the in-memory object/StringDtype distinction; no data differences were found. It also recalculated baseline predictions and all five scoring fields for **10,980 saved skill slices**: every predictor, each model, both horizons, at the first, middle and last available origin. Every numeric result matched to 1e-12. This is substantial sampled prediction reproduction plus complete aggregate verification, **not** a claim that all 261,080 saved slices were rerun. The combined re-ingest and sampling check took 86 seconds on the current environment. See [reproduction log](reproduction-log.txt).
