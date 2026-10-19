# Workbook 1 — verification of the revised submission

Reviewed September 30, 2026, against the updated `Workbook1.docx`, 35-page `Workbook1.pdf`, repository code and data, supporting documents, assignment rubric, research template, and primary sources. Page references count the PDF cover as page 1; section names identify the location independently of pagination.

**Verdict: agree with most of Claude's corrections, but push back on the implication that everything is now consistent and correct.** The empirical corrections, targeted bibliography repairs, and evaluation-interface changes are real. Several old claims survive in the PDF, its figures, and active project notes. Some newly acknowledged requirements remain unspecified or incomplete. This is a substantially improved workbook, with identifiable remaining submission work.

This is a review, not an implementation change. The workbook, code, meeting records, and previous audit were left intact. Detailed companion audits are in [the follow-up evidence folder](followup-review-2026-09-30/). The [original review](Workbook1-review-2026-09-30.md) remains the audit trail for unchanged references and earlier reproduction work.

## Verdict on each reported revision

| Claude's item | Verdict | Independently verified result and qualification |
|---|---|---|
| 1. Empirical findings | **Agree with the specified corrections; one other numerical interpretation remains wrong.** | Corrected endpoints, six final accuracies below .35, seven strict drops above .08, pointwise Wilson statement, and missing-metadata counts reproduce. The revised Pythia biology sentence still asserts sustained gains although both Pythias decline from their first retained checkpoint. |
| 2. Bibliography | **Agree with all eight targeted repairs and the five real additions.** | There are 49 unique references with no dangling numbers. Only 48 are cited outside References; [47] appears in the annotation to [20]. Add a body citation for [47]. Some annotations still overstate the papers' conclusions. |
| 3. dynIRT | **Agree with the methodological correction; disagree that it was rewritten everywhere.** | Main prose now acknowledges one-period-per-item, tied-parameter adaptation, unreliable variational uncertainty, custom inference, a Sprint 5 decision, and T-SKIRT fallback. Figure 5.1 still presents `emIRT::dynIRT` as the implementation. The custom binary-response inference algorithm remains unspecified. |
| 4. BKT | **Agree with the AUC and numerical-stability correction.** | Shared skill-level guess/slip implies identical probabilities and within-skill AUC .50. Log-space inference and item-specific emissions are now requirements. Their implementation and parameter-tying/shrinkage plan are future work; the figure still shows the old pyBKT description. |
| 5. Baselines and harness | **Agree with the claimed edits and reproduced metrics; contract remains incomplete for elapsed-time forecasts.** | Five signatures, spies, horizon forwarding, probability validation, and per-horizon filenames are correct. Fifty tests pass; ruff is clean. Future target step/token timing is absent, and nonpositive horizons are accepted. Saved positive-horizon results are unaffected. |
| 6. Novelty and statistics | **Partly fixed.** | Main discussion repairs Implicit Curriculum, Hu, IRSL, MMLU, selection control, and the unequal-bin issue. Other passages still deny held-out prediction, assert universal novelty, treat drift as known to be below threshold, and compare incompatible correlations. |
| 7. Scope, ownership, readiness | **Most listed edits verified; several contradictions remain.** | GPU scope, named ownership, pilot/decision dates, 2027, and removal of course deliverables are present. Ch8's “committed artifact” definition conflicts with repository history. Setup for two members is still scheduled after submission. Schedule contradictions are listed below. |
| Meeting log | **Agree that the cadence gap remains; qualify “nothing to change.”** | Preserve honest history and the dated correction. Publication/access and prompt historical posting remain unverified. README claims about prompt posting and individual ownership need to match evidence. |

## Changes that are genuinely resolved

The corrected OLMo-2 MMLU findings reproduce:

| Quantity | Reproduced value |
|---|---:|
| World religions, first → final | .321637 → .824561 |
| High-school mathematics, first → final; peak | .211111 → .307407; .351852 |
| Subjects with final accuracy below .35 | 6 |
| Subjects with peak-minus-final decline strictly greater than .08 | 7 |
| Medical genetics decline | .080 exactly; excluded from the strict count |
| Missing correctness values | 0 |
| Missing `step` cells | 3,313,198 |
| Missing `tokens_seen_b` cells | 7,656,550 |

Every MMLU subject has at least one checkpoint whose pointwise Wilson lower bound exceeds .25. That does not establish sustained improvement or a simultaneous significance claim; the new qualification is appropriate. The seven screened declines are exploratory candidates, as now stated. The September 29 meeting note preserves its original wording with a dated correction directly beneath it, which is appropriate.

Saved item-mean aggregates also reproduce:

| Evaluation subset | Horizon | Log-loss | Mean within-skill AUC |
|---|---:|---:|---:|
| All five benchmarks | 1 | .360227 | .800198 |
| All five benchmarks | 5 | .388302 | .779227 |
| MMLU only | 1 | .483458 | .792491 |
| MMLU only | 5 | .507538 | .771217 |

The revised workbook correctly distinguishes item-weighted losses from unweighted mean slice AUC, shrinkage toward the global mean from Laplace smoothing, and static-IRT forecasting from same-checkpoint diagnosis. It now admits that calibrated-copy reuses one-step probabilities at h=5. That is a legitimate fixed predictor evaluated farther ahead; it is not an implemented five-step Markov forecast. The existing results need not be discarded for that reason.

The eight repaired bibliography records [8], [13], [14], [15], [16], [37], [39], and [42] match the primary sources. The five additions are real and relevant. The removed unnamed medical-CDM/mechanistic-KT mentions are no longer a basis for criticism. [47] has a minor collective-author cleanup: the source uses **Team OLMo**. Its 2025 date is defensible for a 2025 version/venue despite the initial December 31, 2024 submission; specify the version rather than calling the year categorically wrong. Individual source links are in the [citation audit](followup-review-2026-09-30/citations.md).

## Remaining issues in the submission PDF

### 1. The architecture figure still promises the old implementation

**Ch5, Figure 5.1, p23:** the diagram says `Kalman dynamic IRT (emIRT::dynIRT)` and `BKT + forgetting (pyBKT, batched per step)`. These labels conflict with the custom NumPy and log-space adaptations in Ch3/Ch4. Update the source figure and the embedded image. Merely changing nearby prose does not repair the diagram required by the architecture rubric.

The revised dynIRT prose itself is right: the [official emIRT manual](https://cran.r-project.org/web/packages/emIRT/emIRT.pdf) gives each item one session and warns against its variational variance estimates. Custom tied-item inference and separately validated uncertainty are meaningful additional work, now honestly acknowledged.

### 2. Incorrect literature and novelty statements remain

**Ch1.2, p6:** a paragraph including Implicit Curriculum still ends with “there is no latent variable, no transition kernel, and no held-out prediction.” This contradicts Ch1.1 and the [Implicit Curriculum paper](https://arxiv.org/html/2604.08510v1), which predicts held-out task trajectories. Use the narrower distinction: the reviewed works do not jointly provide this project's proposed per-skill stochastic state and item-level forecasts of later checkpoints.

**Ch1.1, pp2–5, and annotation [42], p13:** “have not previously been combined,” “No existing tool shows,” and the absence of skill-grouped views remain universal claims. Bound them to reviewed sources and the specific combination of fitted state, uncertainty, and future-item prediction. The corrected novelty paragraph does not override incompatible assertions elsewhere.

**MathCAMPS, p4:** “report only per-skill accuracy curves” is still too broad. The [paper](https://arxiv.org/html/2407.00900v3) also studies instruction tuning and follow-up robustness. Describe the relevant pretraining comparison without reducing the whole paper to curves.

**Annotations [4], [32], [45], pp7/12/14:** “need thousands,” “assumes ... thousands,” and “needs large student populations” exceed what reported training/evaluation sizes establish. Say they were evaluated on larger learner populations and that behavior in this project's few-learner regime is unestablished. Similarly, ordered pretraining checkpoints do not imply a skewed ability distribution “by construction” (p3).

### 3. T-SKIRT conversion is reversed in one place and ambiguous in another

**Ch2 p15:** “probit-to-logistic conversion” runs in the opposite direction from the intended Fluid-logistic → T-SKIRT-probit initialization. **Ch3 p19:** “parameters ... divide by 1.7” could incorrectly be applied to both discrimination and difficulty.

For `sigmoid(a_L*(theta-b))` approximated by `Phi(a_P*(theta-b))` on a fixed ability scale, the conventional approximation uses `a_P ≈ a_L/1.7`, leaving `b` unchanged. State the actual parameterization and treat this as approximate, or refit. This follows from the link functions; [T-SKIRT Eq. 1](https://arxiv.org/html/1702.04282v1) uses the normal-ogive form.

### 4. Some statistical conclusions still outrun the proposed analysis

**Ch2 p15:** “per-item difficulty drift is below the first-order detectability threshold at our data density” is not established. Data density informs uncertainty; it does not supply the unknown drift magnitude. Say sensitivity may be poor and will be assessed by simulation. The acknowledgement of the [source's](https://arxiv.org/html/2606.14123v1) unequal-bin algebra error is correct: `n1=1,n2=100` gives `4.04 < 8`; equal-bin equality is unaffected. A significance cutoff also is not a minimum detectable effect at a specified power.

**Ch8.5, Sprint 5, p30:** “inter-subject trajectory correlation versus 0.965” still compares different quantities. The [MMLU audit's](https://arxiv.org/html/2609.09372v1) value describes two fitted latent factors across final models, not raw correlations among 57 subject trajectories over checkpoints. Use it as motivation; define the within-run analysis separately and account for common trends and dependence. Ch2's claim about most subject variance similarly should not be inferred from that two-factor statistic alone.

**Ch3.1 versus Ch3.3:** the blanket requirement that “any learner model” beat item_mean on both loss and AUC is inconsistent with retaining shared-emission BKT as a log-loss comparison that necessarily has AUC .50. Limit the joint success rule to candidate primary models; explicitly classify shared-emission BKT as an ablation/control.

### 5. Two empirical/visual statements still need repair

**Ch8.4, p29:** Pythia and Amber are described as having “small sustained gains” in high-school biology. Amber rises .183871 → .267742, but the two Pythias fall .316129 → .248387 and .316129 → .245161. “Finish near the chance reference, without the sustained gains seen in OLMo-1, OLMo-2 and K2” matches the data better.

The cross-model plot still labels `checkpoint_idx / last_index` as “fraction of pretraining run” ([plot code](../../scripts/plot_skill_curves.py), lines130–137). This is normalized position among retained checkpoints, not elapsed training fraction for irregular or truncated releases. K2's retained series starts at `ckpt_093`. Relabel and regenerate, or use genuine training-time metadata.

### 6. “Verified” tools and completed setup are different from a setup plan

**Ch8.2, p28:** “Verified” means a tool produced a **committed artifact** on the member's machine. The local history still contains only the initial commit; project source, documents, and meetings are untracked. That conflicts with this definition and with Claude's accurate “nothing committed” statement. Use “local artifact” unless the relevant commit exists. Local installations are observable; other people's usage cannot be proved from this checkout.

Bhagyesh and Aniket are scheduled to clone/install/run tests by October 7. That is clearer and more honest, but does not establish the rubric's tools-acquired condition at submission. Record completed member-specific setup checks if they actually occur. The custom models need not be finished for Workbook 1; their environments and clear implementation plans are the relevant distinction.

### 7. Schedule and design ambiguities remain

- **Fitted-state contract:** Ch8.5 places it in Sprint 7; Ch9 assigns it in Sprint 5. Choose one milestone or distinguish specification from later integration.
- **K2 time metadata:** Ch4 identifies missing K2 step/token data, while Sprint 3 repairs only Pythia/Amber. Assign K2 or explicitly choose an index-clock fallback.
- **Phase 2 leave-one-model-out:** Sprint 9 combines Phase 2 fitting with this ablation, but preferred Phase 2 names only OLMo-2. Identify the multi-model Phase 1 dataset for the split, add Phase 2 models, or change the experiment.
- **Binary-response dynamic inference:** “NumPy Kalman filter/smoother” does not specify the update for non-Gaussian binary observations. Identify an approximation or augmentation method; an ordinary linear-Gaussian filter is not the complete algorithm. This is an unresolved design choice, not evidence that the project is infeasible.
- **Per-item BKT estimation:** the workbook says a tying/shrinkage plan is needed; that plan is not yet given. Recognizing a requirement is progress, but does not make the design unambiguous.

The corrected GPU dependency, Aniket's November 25 pilot and December 12 decision, Sprint 5 dynamic-IRT decision, named comparator/reproducibility owners, explicit 2027 dates, and removal of coursework from Ch9 all check out. The remaining historical “Abstract submitted” progress entry is appropriate.

## Code verification and its limits

All 50 tests pass and ruff passes. Direct probes also verify horizon forwarding and rejection of NaN, infinities, and out-of-range probabilities. Per-horizon files exist with correct horizon columns. DOCX/PDF text comparison found no missing document words; the PDF is 35 pages, and the schedule renders without clipping.

Two narrower code issues remain:

1. **Future elapsed gap is unavailable.** `Predictor.predict(history, target, horizon)` supplies target item/benchmark/skill only. History timestamps plus a checkpoint count cannot determine target elapsed tokens on an irregular future schedule. Provide target time/gap without target answers, or state the extrapolation/index-clock assumption. Existing trivial baselines are unaffected.
2. **Nonpositive horizons are accepted.** Synthetic `h=0` produces `(origin,target)=(1,1),(2,2),...`; `h=-1` produces `(1,0),(2,1),...`. Target responses are already in history, contrary to the general future-only guarantee. Validate integer `h>=1` and minimum history at the evaluation boundary. This does not invalidate the recorded h=1/h=5 runs.

The runner's opening docstring also still names the old common output filename, despite the corrected implementation. See the [code audit](followup-review-2026-09-30/code-and-numerics.md) for exact locations and reproduction commands. This pass rechecked artifacts and relevant behavior; it did not repeat the prior audit's full raw-data ingestion or every expensive rolling-origin prediction. It does not replicate external papers' experiments or validate the unimplemented temporal models.

## Supporting documents still need consistent current guidance

- `docs/project-notes.md:153–160` retains deterministic/nondecreasing IRSL ability, undetectable drift, an absolute novelty claim, DAS3H as a “sixth trivial baseline,” and an overgeneralized forgetting argument. Its new dynIRT correction is real, but neighboring conclusions remain stale.
- The same file's opening says Decisions1–5 were adopted, while lines262/283/301 still label Decisions3–5 proposed/awaiting team sign-off.
- `docs/research-brief.md:73` retains the unconditional .36/.80 acceptance bar and causal interpretations not established by those baseline comparisons.
- `docs/abstract-v2.md` retains broad “no existing method”/described-but-not-modeled wording that no longer matches the bounded workbook claim.
- `docs/lit-review-2026-09-30.md` **does contain a substantial dated correction section, §9b**. Keeping earlier text as history is legitimate. Add a prominent superseded notice near its beginning and affected recommendations so readers do not follow the old “two-line patch,” monotonicity, or novelty advice. Do not pretend the erratum is absent, and do not silently rewrite historical meeting records.

## Claude's three pushbacks and the meeting URL

**“Whether this earns 21/21 is the advisor's call”: agree.** An independent review cannot award the grade. However, honesty about future work does not establish completed rubric requirements. The remaining diagram, setup, method, schedule, and template issues mean I still would not describe the present version as demonstrably satisfying every 3-point descriptor.

**“Keep two-page Ch2 because risks are rewarded”: push back.** The actual [research template](../../project-workbook-template-research-v3.docx) explicitly says: “Insert a one-page summary of the justification for your project.” Ch2 occupies pp15–16. The rubric does not waive that instruction or explicitly reward extra length. Keep a strong one-page justification and put detailed risks in Ch4. An advisor-granted exception would change this judgment; none is documented here.

**“Corbett & Anderson 1995 is acceptable”: agree with qualification.** There is a real publication-date discrepancy: the [authors' publication page](https://act-r.psy.cmu.edu/?post_type=publications&p=14344) explicitly uses 1995 and links the original scan, while the [publisher's current record](https://link.springer.com/article/10.1007/BF01099821) lists December 1994, including its issue date. Retaining 1995 consistently is defensible; calling it the unambiguous publisher issue date is not. This is minor bibliographic ambiguity, not a false source or submission blocker.

**Meeting URL: Claude's URL is correct, but it did not correct this audit.** The [original main review](Workbook1-review-2026-09-30.md) already gives `bhagyeshrathi07/LLM-Knowledge-Tracing` at line155 and explicitly warns against inferring ownership from the Ron5474 workspace path. The intended link is [the origin repository's meetings folder](https://github.com/bhagyeshrathi07/LLM-Knowledge-Tracing/tree/main/meetings), conditional on actual publication and advisor access.

Only August31 documents an advisor meeting including all three students. September1 omits Aniket; September3 is email. No later advisor meeting note exists. Team notes cover September8/15/23/29, leaving earlier coverage absent. A new push cannot establish prompt historical posting. The notes remain untracked locally; that alone does not prove they are absent from every remote copy. Preserve the honest cadence gap, verify the submission link/access, and align README's 24-hour posting/single-owner statements with demonstrated practice or label them intended policy.

## Assignment readiness against all seven workbook criteria

| Criterion | Current assessment |
|---|---|
| Ch1 literature/state of the art | Substantial breadth and repaired references; remaining false negatives and overclaims should be removed before claiming complete accuracy. |
| Ch2 justification | Credible project with more honest risks; compress to the required page and remove premature statistical conclusions. |
| Ch3 baseline approaches | Stronger protocol and comparator distinctions; resolve BKT success-rule exception, target timing, conversion, and inference specifications. |
| Ch4 dependencies/deliverables | Material improvement; GPU and model feasibility gates are concrete. Align K2 recovery and avoid universal software-absence claims. |
| Ch5 architecture | Diagram and functional/technology prose exist, but stale diagram labels directly contradict the selected implementation. |
| Ch8 implementation plan | Working local prototype is evidenced. Other members' environments are still future setup; “committed artifact” is unsupported. This is a concrete readiness gap, not merely grading subjectivity. |
| Ch9 schedule | Named ownership and deliverables are much stronger. Reconcile Sprint5/7, K2, and Phase2 validation before calling it complete and unambiguous. |

Meeting-log full credit is still not supported by the documented cadence and posting evidence. The correct response is accurate records and accessible submission, not retroactive invention. The workbook can be made consistent through focused corrections; the verified progress and positive-horizon results remain valid.
