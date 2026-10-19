# Revision audit: code, numerical findings, and model specifications

Reviewed 2026-09-30. Scope: actual source, tests, saved canonical/curve/baseline artifacts, updated 35-page Workbook1 PDF (cover is PDF p.1), DOCX text extraction, and supporting project documents. No workbook or implementation source edited. Full ingestion and the expensive all-origin baseline recomputation were not repeated; the previous audit had already reproduced those, and the current predictor changes do not alter the five baseline probability formulas.

## Verified corrections: agree

1. **All claimed API edits exist.** `src/llm_kt/evaluate.py:28` declares `predict(history, target, horizon=1)` and line 76 forwards `horizon=horizon`. All five implementations in `baselines.py` accept the argument (lines 25, 35, 56, 78, 98). Both existing spy implementations in `tests/test_evaluate.py:83,94` were updated. A separate synthetic h=2 probe observed `(origin,horizon)=(1,2),(2,2)` and target columns only `(item_id,benchmark,skill)`.
2. **Probability checks work.** `evaluate.py:77–80` checks shape, finiteness and range. Direct probes rejected NaN, positive infinity, negative infinity, −0.001 and 1.001, and accepted endpoints 0 and 1. These checks are new working behavior, although the existing 50-test suite has no dedicated regression tests for these cases or for forwarding the nondefault horizon.
3. **Output overwrite hazard fixed.** `scripts/run_baselines.py:36–37` now defaults to `baseline_results_h{h}.parquet`. Both h1/h5 files actually exist with the correct horizon column; h1 contains 134,200 slice rows, h5 126,880. Minor stale documentation remains in that script's opening docstring (lines 4–5), which still names `baseline_results.parquet`.
4. **Current checks pass:** `.venv/bin/pytest -q` → `50 passed in 0.49s`; `.venv/bin/ruff check .` → `All checks passed!`.
5. **Item-mean interpretation fixed.** `baselines.py:43–48` correctly describes empirical shrinkage `(k+alpha*prior)/(n+alpha)`, not Laplace smoothing. Workbook §3.1 (PDF p.17) distinguishes pooled versus MMLU results, item-weighted loss/Brier versus unweighted mean within-skill AUC, and one-step calibrated-copy predictions reused at h=5.
6. **No false claim of an implemented multi-step calibrated-copy forecast.** The actual baseline remains horizon-invariant (`baselines.py:87–90,98–126`). Synthetic h1/h5 calls on identical history produce identical outputs for every baseline. The revised workbook explicitly admits this, so the old h5 numbers remain valid as evaluation of fixed predictions at a farther target, without claiming to be five-step transition forecasts.
7. **Saved metric aggregates reproduce exactly.** For item_mean: pooled h1 loss `0.360227`, AUC `0.800198`; pooled h5 `0.388302`, `0.779227`; MMLU h1 `0.483458`, `0.792491`; MMLU h5 `0.507538`, `0.771217`. All five predictors' displayed rounded table entries match recomputed aggregates.
8. **Corrected empirical claims reproduce.** Updated §8.4 (PDF p.29) and the brief correctly give world_religions `.321637→.824561`, high_school_mathematics `.211111→.307407`, peak `.351852`; exactly six final subject accuracies are below .35; zero subjects remain at/below .25 at every checkpoint under the pointwise Wilson-lower-bound check. Seven peak-minus-final declines strictly exceed .08, after guarding floating-point equality: abstract_algebra, business_ethics, college_mathematics, econometrics, global_facts, machine_learning, public_relations. medical_genetics equals .080 and is excluded.
9. **Missing-value qualifier fixed.** Canonical Parquet metadata gives 12,584,584 rows; missing step cells `3,313,198`; missing token cells `7,656,550`; missing correctness cells `0`. These agree with the revised §8.4.
10. **BKT correction is theoretically right and honestly framed as planned.** Shared guess/slip per skill gives identical item probabilities within a slice and AUC .50. §3.3 (PDF pp.18–19) explicitly requires item-specific emissions to compete on AUC and acknowledges that tying/shrinkage remains to be specified. It explicitly describes raw-product underflow as a source-based finding, not an executed pyBKT test. No BKT implementation has been supplied, but the workbook no longer claims one is complete. Log-space accumulation addresses the identified numerical problem; it does not itself solve identifiability or emission-estimation decisions.
11. **Dynamic IRT feasibility framing is materially improved.** §§3.3/4.1 (PDF pp.19,21) remove the drop-in/two-line-patch claim, acknowledge tied-bank adaptation and uncertainty work, assign a Sprint 5 decision, and make T-SKIRT the fallback. These are defensible revisions to a project plan, not completed validation.
12. **Static forecast versus diagnostic fitting is correctly separated.** §3.2 (PDF p.18) now limits forecasts to ability estimated from history and uses within-checkpoint held-out items for diagnosis. §3.3 also corrects T observations to T−1 inter-checkpoint state transitions, distinguishes Wilson bands from mastery uncertainty, and admits dependence concerns. §8.4 calls present results exploratory.
13. **Historical meeting correction exists.** `meetings/2026-09-29-team.md:21–22` preserves the original statement and adds a dated correction immediately underneath; this is appropriate audit-trail handling.

## Remaining issues: push back on "everything is now fixed"

### R1. Future elapsed training time is still missing from the predictor contract (method completeness; medium/high)

Workbook §5.2, PDF p.24; `src/llm_kt/evaluate.py:22,28–36,74–76`.

Passing the checkpoint count h fixes the discrete-horizon omission, **not the future elapsed-gap omission**. Target has only item ID, benchmark and skill. History can provide past step/token timestamps, but cannot establish the time of an unobserved target on an irregular future checkpoint schedule. Two futures with the same history and h but different target token counts are indistinguishable to this interface. The docstring currently suggests using h plus history's step/tokens to scale transitions, without specifying an extrapolation assumption.

Add explicit target step/tokens or elapsed-token gap as forecast context, available without target answers, and define the missing-metadata/index-clock fallback. If using a history-based gap estimate instead, state it as an assumption. This is especially relevant to the scheduled gap-scaled dynamic IRT and T-SKIRT. Existing h1/h5 baseline results are unaffected.

### R2. Nonpositive horizons silently violate the claimed future-only evaluation (code correctness; medium)

`rolling_origins` (`evaluate.py:41–46`) and the public evaluation entry points do not validate horizon. CLI accepts any integer (`scripts/run_baselines.py:33`). Direct synthetic calls with h=0 and h=−1 succeed: h=0 produces origin/target pairs `(1,1),(2,2),(3,3),(4,4)`; h=−1 produces `(1,0),(2,1),(3,2),(4,3),(5,4)`. In each case target answers already occur in supplied history. This does not invalidate the recorded h=1/h=5 runs, but the API's general strict-future guarantee is false outside its intended inputs.

Require integer h≥1 at the evaluation boundary and validate the minimum history. Add focused tests for invalid h and forwarding a nondefault h.

### R3. The normalized-index plot still calls its x-axis elapsed training fraction (existing error; medium)

`scripts/plot_skill_curves.py:130–137` computes `checkpoint_idx / max(checkpoint_idx)` and labels it `fraction of pretraining run (checkpoint index / last index)`. Those values are normalized position within retained checkpoints, not training fractions when spacing is irregular or the first retained point is mid-run (K2 starts at ckpt_093). Rename the axis/comment and regenerate the affected figure, or use genuine elapsed-training metadata. This was explicitly flagged in the previous audit and remains unchanged.

### R4. The revised biology sentence still overstates Pythia learning (factual wording; medium)

Workbook §8.4, PDF p.29, says Pythia and Amber finish near chance "with small sustained gains." Amber goes `.183871→.267742`, but Pythia-2.8B goes `.316129→.248387` and Pythia-6.9B `.316129→.245161`. Both Pythias end **below their first retained checkpoint**, so grouping them under sustained gains is not supported.

Use: "Pythia and Amber finish near the chance reference, without the sustained gains seen in OLMo-1, OLMo-2 and K2." The main corrected endpoint/drop statistics themselves are right.

### R5. Time metadata dependency and scheduled recovery still do not fully align (completeness; medium)

Chapter 4 correctly names missing K2 metadata (PDF p.22). Sprint 3 in §§8.5/9.1 (PDF pp.30,33) still schedules filling tokens only for Pythia and Amber. Include K2 in the owned recovery task, or state explicitly that K2 uses an index-clock fallback and how such fits will be compared. Merely noting K2's published schedule elsewhere does not assign the recovery work.

### R6. "Divide by 1.7" needs a parameterization, not a blanket parameter conversion (methods ambiguity; medium)

§3.3 T-SKIRT row (PDF p.19): "Fluid's logistic 2PL parameters need conversion (divide by 1.7) or refitting." [T-SKIRT Eq.1](https://arxiv.org/html/1702.04282v1) uses `Phi[a(theta−b)]`. For the standard logistic form `sigmoid[a(theta−b)]`, the usual approximation on a fixed ability scale replaces the discrimination by `a/1.7` while leaving difficulty b unchanged. Dividing both a and b is not that conversion. If source parameters instead represent slope and intercept, state the corresponding conversion explicitly. Label it approximate and validate/refit as planned.

### R7. The new custom dynamic-IRT plan needs a specified inference approximation (methods ambiguity; medium)

§§3.3,4.1,5.2 now call it a NumPy Kalman filter/smoother, with an estimated 200–400 lines. Binary probit/logistic item observations do not have a linear-Gaussian observation model; an ordinary Kalman smoother is not sufficient as the full inference method. State the intended latent-variable augmentation, variational updates, Laplace/extended-filter approximation, or other binary-observation inference scheme. The new feasibility gate is good, but it does not settle this central design choice or validate the line estimate. This is unresolved planned work, not a fabricated current implementation.

### R8. Some supporting documents remain inconsistent with the workbook (medium/low)

- `docs/research-brief.md:73` still states the unconditional `.36/.80` acceptance threshold without the matching-subset qualifier and claims the averages establish noisy readings and dominant item difficulty. The workbook now correctly limits the comparison rule and exploratory interpretation.
- `docs/project-notes.md:157` calls DAS3H a "sixth trivial baseline"; the revised workbook correctly calls it a learned non-latent baseline.
- `docs/lit-review-2026-09-30.md` retains its earlier absolute novelty, two-line dynIRT, and BKT-no-algorithm-change statements and appends a dated corrections section at lines 172+. That is defensible as a historical record if clearly marked as superseded at the top; it is not literally "rewritten everywhere." Readers encounter the superseded claims first.

## Calibration of the verdict

Agree with the great majority of Claude's reported changes and with treating existing tests/baseline data as real. Do not re-raise already corrected .25→.82, five-versus-seven, Laplace, h5-Markov, or dynIRT-drop-in claims as current workbook errors. Remaining concerns above are narrower: incomplete forecast context and model specification, two unrepaired presentation/data claims, schedule completeness, and consistency of supporting documents. Honesty about work being planned supports Workbook 1, but does not automatically remove ambiguity under Chapter 3's completeness criterion.

## Minimal reproduction commands

Existing checks:

```sh
.venv/bin/pytest -q
.venv/bin/ruff check .
```

Nonpositive-horizon acceptance, with recorded origin and target indices (no source edits):

```sh
.venv/bin/python - <<'PY'
from llm_kt.baselines import GlobalMean
from llm_kt.evaluate import evaluate_model
from tests.test_evaluate import make_responses
x = make_responses('m', {'a': [0, 1, 1, 0, 1]}, {'a': 's'})
for h in [0, -1]:
    out = evaluate_model(x, GlobalMean(), horizon=h)
    pairs = list(zip(out.origin_idx, out.target_idx))
    print(h, pairs)
    assert all(target <= origin for origin, target in pairs)
PY
```

The input `history = responses[checkpoint_idx <= origin]` at `evaluate.py:72` therefore includes each reported target checkpoint for these accepted inputs. A comparison of item IDs alone would not prove leakage because repeated item identity is intentional; the checkpoint-index comparison is the relevant evidence.
