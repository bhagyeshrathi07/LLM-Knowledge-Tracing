# Revision verification: literature, citations, and method wording

Reviewed updated DOCX/PDF text, original reference audit, and supporting literature/project notes on 2026-09-30. Page numbers below are physical PDF pages, counting the cover as page 1. The eight targeted metadata corrections were independently rechecked against primary records, as were the five additions. Unchanged references already covered in the original audit were not exhaustively reread.

## Verdict

Most substantive changes are real and correct. The claim that all literature errors were corrected everywhere is false: several old conclusions survive in the submitted PDF itself, including an explicit denial of held-out prediction immediately after correctly acknowledging it. This is incomplete propagation of revisions, not a disagreement with the revised research direction.

## Bibliography verification

The bibliography has exactly 49 uniquely numbered entries, [1] through [49], and there are no out-of-range/dangling numbered citations. Forty-eight entries are cited outside the References section. [47] is cross-referenced only in the annotation to [20], not in the chapter body. Thus 'all 49 cited in text' is defensible only if bibliography annotations count. Add [47] next to an OLMo-2 claim in the main text.

| Entries | Verdict | Primary verification |
|---|---|---|
| [8] Fluid | Corrected author list matches all ten authors. | https://arxiv.org/abs/2509.11106 |
| [13] Beyond Scores | Xu Zhang / Zhang, X. corrected correctly. | https://arxiv.org/abs/2604.12191 |
| [14] Laplace-PSN-IRT | Correct full-text byline, Mandujano Reyes, J. F., and title. | https://arxiv.org/html/2607.25257v1 |
| [15] FinCDM | Kuang-led eleven-author list correct. | https://arxiv.org/abs/2508.13491 |
| [16] CDS | Zhao-led six-author list and title correct. | https://arxiv.org/abs/2501.07674 |
| [37] NTKT | Norris, Gal, Bulathwela correct. | https://arxiv.org/abs/2511.02599 |
| [39] Dialogue KT | Scarlatos, Baker, Lan and LAK 2025 correct. | https://arxiv.org/abs/2409.16490 |
| [42] LLM Comparator | CHI Extended Abstracts 2024 correct; previous mixed venue removed. | https://research.google/pubs/llm-comparator-visual-analytics-for-side-by-side-evaluation-of-large-language-models/ |
| [45] SAINT | Nine-author list matches arXiv, including June 2020 v2; title and L@S 2020 description match. | https://arxiv.org/abs/2002.07033v2 |
| [46] Skill-it | Author list, title, NeurIPS 36 (2023) correct. | https://proceedings.neurips.cc/paper_files/paper/2023/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html |
| [47] OLMo 2 | Correct paper. Minor cleanup: credited collective name is **Team OLMo**, followed by named authors; use 'Team OLMo et al.' The first submission is Dec 31, 2024, but 2025 versions and a COLM 2025 version exist. **Do not call 2025 an unequivocal error**; identify the intended version/venue. | https://arxiv.org/abs/2501.00656 |
| [48] Kübler et al. | Seven authors, title, ICLR 2026 correct. | https://arxiv.org/abs/2602.10144 |
| [49] Yeung & Yeung | Names, title, L@S 2018 correct; reconstruction and prediction inconsistency summary appropriate. | https://arxiv.org/abs/1806.02180 |

## Changes I agree with

- Main discussion correctly distinguishes Fluid's 467 experimental checkpoints from the inspected release's 452.
- Main IRSL discussion correctly describes independent checkpoint estimates followed by a scaling fit. The old absolute monotonicity claim has been removed there.
- Main text now recognizes Implicit Curriculum's held-out task-trajectory prediction and Hu et al.'s aggregate latent states.
- Main MMLU discussion now acknowledges that the STEM/non-STEM constructs are distinguishable despite the 0.965 latent correlation.
- ATLAS is now primarily 3PL with 2PL comparison and no MMLU bank. Reuse is limited to overlapping items; correct direction, with eventual parameter-scale linking still needed.
- dynIRT's one-session-per-item data model and unusable variational variance warning are accurately reported. Recording custom inference as a feasibility risk with a go/no-go and T-SKIRT fallback is appropriate. See official manual pp.12–13: https://cran.r-project.org/web/packages/emIRT/emIRT.pdf .
- T-SKIRT temporal-only versus full-model gains and its normal-ogive link are now distinguished correctly. The source uses Phi(alpha*(theta-beta)), with a Wiener state prior: https://arxiv.org/pdf/1702.04282 .
- Selection-aware paired testing for the screened forgetting candidates is now explicitly required; this repairs the key inference claim.
- The unequal-bin bound objection is mathematically correct. In https://arxiv.org/html/2606.14123v1 §3.7, Proposition 2 incorrectly asserts 4/n1 + 4/n2 >= 8/min(n1,n2). n1=1,n2=100 is a counterexample; equal bins retain equality. Also, a Wald significance cutoff is not a specified-power minimum detectable effect. Treating 2.26 as a first-order indication and requiring simulation is an improvement.

## Remaining corrections in the actual workbook

1. **PDF p.6, Ch1.2: false 'no held-out prediction' remains.** The paragraph explicitly includes Implicit Curriculum and ends 'there is no latent variable, no transition kernel, and no held-out prediction.' This contradicts the corrected p.4. Implicit Curriculum predicts held-out composite task trajectories using function-vector similarity and kernel ridge regression; see §3 and §5, https://arxiv.org/html/2604.08510v1 . Replace the last sentence with: 'These works do not jointly provide the proposed per-skill stochastic state and item-level forecasts of later checkpoints; Implicit Curriculum does predict held-out task-level trajectories.'

2. **PDF p.2, p.5, p.13: universal novelty language remains.** Opening: four literatures 'have not previously been combined'; visualization passage: 'No existing tool shows...'; annotation [42]: 'skill-grouped views over training time do not [exist].' The last also conflicts with the workbook's own MathCAMPS and Zeno discussion. Scope each negative to the reviewed sources and the exact combination of fitted mastery plus uncertainty. The evidence-bounded paragraph on pp.6–7 is good, but does not cure incompatible assertions elsewhere.

3. **PDF p.4: MathCAMPS still 'report only per-skill accuracy curves.'** It also analyzes instruction tuning and follow-up-question robustness; https://arxiv.org/html/2407.00900v3 §§4.1–4.2. Say 'its relevant pretraining analysis reports skill-level performance rather than fitting our proposed sequential latent learner model.' The brief repeats the old 'only accuracy curves' wording at docs/research-brief.md:87. Also identify OLMo2 precisely when summarizing its checkpoint experiment.

4. **PDF p.7 [4], p.12 [32], p.14 [45]: hard learner-count necessity claims remain.** AKT annotation says models 'need thousands'; VTIRT says it 'assumes ... thousands'; SAINT says it 'needs large student populations.' Main p.2 correctly says their few-learner behavior is unestablished. Training/evaluation dataset sizes do not prove a minimum sample-size requirement. Propagate that cautious wording to the annotations. This is an **unsupported necessity claim**, not evidence that these models are known to work with six learners. AKT primary paper: https://arxiv.org/abs/2007.12324 ; VTIRT proceedings: https://educationaldatamining.org/edm2023/proceedings/2023.EDM-short-papers.24/2023.EDM-short-papers.24.pdf .

5. **PDF p.15/p.19: T-SKIRT conversion direction and rule need precision.** Ch2 says 'probit-to-logistic conversion', while the intended initialization goes **Fluid logistic to T-SKIRT probit**. Ch3 says logistic 'item parameters' should be divided by 1.7, which is ambiguous. For p=logistic(a_L*(theta-b)) and common theta/b scale, use the approximation a_P≈a_L/1.7, b_P=b; do not divide both difficulty and discrimination. If using slope/intercept form, transform the complete linear predictor consistently. State the exact Fluid parameterization and validate approximation error or refit. This conclusion follows from the stated link functions and the conventional logistic/normal approximation; the factor is approximate, not exact.

6. **PDF p.30, Sprint 5: 'inter-subject trajectory correlation versus 0.965' remains.** 0.965 is a fitted two-factor latent correlation across final models, not the null/reference value for raw subject accuracy correlations across autocorrelated checkpoints. Use it as motivation; define a within-run separation analysis on its own scale and handle common trends/dependence.

7. **Method specification remains incomplete, although honestly labeled planned.** 'NumPy Kalman filter/smoother' does not identify how binary logistic/probit emissions are handled: an ordinary linear-Gaussian Kalman filter is not exact for these emissions. Choose/identify an approximation (e.g., Laplace/iterated Gaussian approximation) or augmentation/inference method before treating the implementation as fully specified. A 200–400-line estimate is a planning guess, not feasibility evidence. Likewise the per-item BKT entry correctly says it still 'needs a parameter-tying or shrinkage plan'; that requirement has been acknowledged, not actually supplied. This does not make a Workbook-1 plan invalid, but does prevent claiming the method design is complete and unambiguous.

8. **PDF p.21: 'No off-the-shelf implementation fits our design' remains absolute.** Scope to implementations reviewed. Similarly, the p.3 phrase that a pretraining run has a skewed ability distribution 'by construction' is too categorical: ordered observations do not mathematically imply distributional skew. Say it 'may yield a restricted or skewed sampled ability distribution.'

9. **Minor annotation consistency:** [2] still attributes the roughly-half improvement jointly to forgetting/ability/skill discovery rather than the specific 50.6% Assistments forgetting result explained correctly in the main text; [3] saying DKT 'establishes' next-response prediction can imply priority that the original BKT literature already has. Use 'uses the prediction protocol we adopt.'

## Supporting documents and audit history

Keeping dated historical corrections is legitimate; no need to silently rewrite meeting history. But current project guidance still contradicts the corrected workbook:

- docs/project-notes.md:153–158 still says no prior work exists, IRSL ability cannot decrease, drift is undetectable, DAS3H is a 'sixth trivial baseline', and power-law forgetting proves constant BKT forgetting misspecified. These conclusions need qualification/replacement in active guidance, not only the dynIRT correction in the middle.
- docs/project-notes.md:202,244,247 still predicts inevitable deep-KT failure ('will ... be useless', 'will overfit', 'explains why it fails') before an ablation has been run.
- docs/research-brief.md:73 still imposes pooled 0.36/.80 as the bar for any model without the updated matching-subset qualification, and :87 still says MathCAMPS only reports curves.
- docs/lit-review-2026-09-30.md §9b explicitly preserves preceding statements as history and corrects many of them. That audit approach is acceptable **if the beginning and affected sections prominently state that those conclusions are superseded by §9b**. Currently its opening 'Everything else ... held', old action items, summary 'dynIRT fits our shape', categorical novelty, and old five-subject/never-leave-chance wording remain easy to mistake for current advice. Its verification-log tail also restates full T-SKIRT gains without the ablation distinction. An erratum at the end is real evidence of correction, but not consistent active documentation.

## Corbett–Anderson year pushback

Agree that retaining 1995 is defensible and should not be counted as a factual-error correction outstanding. Corbett's own publication list explicitly records 1995: https://pact.cs.cmu.edu/corbett/corbettpubs.htm ; the authors' ACT-R individual publication page does too: https://act-r.psy.cmu.edu/?post_type=publications&p=14344 . That author-hosted page links the original scan: https://act-r.psy.cmu.edu/wordpress/wp-content/uploads/2012/12/893CorbettAnderson1995.pdf . Springer currently labels the paper 1994 / December 1994: https://link.springer.com/article/10.1007/BF01099821 . The exact claim '1995 is the journal issue date' should not pretend the publisher's present metadata agrees. Say '1995 following the authors' citation convention; publisher metadata lists 1994', or simply use one convention consistently with DOI 10.1007/BF01099821. The original audit already treated this as a bibliographic discrepancy, not a fabricated or nonexistent source. In this follow-up, the linked scan is confirmed accessible as a 26-page PDF but the browser tool did not expose its screenshot image to inspect; the 1995 verification rests on the authors' explicit bibliographic entry, not a newly inspected masthead.

No user source files were edited by this verification.
