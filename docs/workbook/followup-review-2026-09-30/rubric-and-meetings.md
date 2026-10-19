# Updated workbook: scope, rubric, schedule, meetings and pushbacks

Reviewed 2026-09-30. Files: updated `docs/workbook/Workbook1.docx` and `Workbook1.pdf`, research template, original review, supporting notes, repository status. No user files changed. Page numbers below are the updated PDF's printed pages.

## What is verified and resolved

- PDF is **35 pages**, per `pdfinfo`. DOCX/PDF word-token multiset comparison finds **no DOCX words missing from PDF**; PDF extras are page numbers and repeated table headers. The p23 architecture and p33–34 schedule were rendered and inspected; tables are readable and not clipped. The diagram remains small but present.
- Ch9 pp33–35 has no `All` owners and removes Workbook 1, 295A/295B reports, defense and `Abstract submitted` as schedule deliverables. Abstract *drafting* remains legitimate research framing; actual course submission remains only in the historical Ch8 progress log.
- Named owners now exist for DAS3H (Ronak, Sprint4), static 2PL forecast/diagnosis (Aniket, Sprint4), static profiles (Aniket, Sprint9), validation blocks/bootstrap (Ronak, Sprint3), and reproducibility package (Ronak, Sprint13). Semester-2 dates explicitly say 2027.
- Ch1, Ch2 p15, Ch4 p21 and Ch8 p29 distinguish no-GPU Phase1 from preferred GPU Phase2 with no-inference fallback. Ch4 specifies Aniket, a Nov25 pilot/cost estimate and Dec12 decision; Sprints6–7 repeat these milestones.
- Ch4 pp21–22 now lists dynamic-IRT adaptation, BKT underflow, incomplete time metadata and identifiability as relevant dependencies; dynamic IRT has an end-Sprint5 go/no-go with T-SKIRT fallback.
- Ch5 p24 adds the horizon argument and names filtered/smoothed state, uncertainty and item-parameter artifacts. The `/meetings/` location is corrected in §5.6 p25.
- Ch8 p28 distinguishes Ronak's installed tools from Bhagyesh/Aniket's setup planned by Oct7. In this checkout, Python3.12.3, llm-kt0.1.0, listed pipeline packages, pytest/ruff, Node22.16.0, Git, VSCode command and a WeasyPrint executable really exist. This supports local installation, not independent verification of each person's usage.

## Remaining concrete problems

### High: architecture image still contradicts the revised implementation

Figure5.1 p23 still says **`Kalman dynamic IRT (emIRT::dynIRT)`** and **`BKT + forgetting (pyBKT, batched per step)`**. Its old technology labels were not updated when the surrounding prose changed to custom NumPy / log-space inference. Thus “rewritten everywhere” is false. Update `docs/figures/architecture.png` and re-embed/rebuild the workbook. Include the custom dynamic-IRT method and the log-space BKT requirement; the current diagram's prediction/results flow also does not explicitly show calibration parameters or scoring truth. This does not mean the architecture is absent, only that the image is stale.

### High: definition of “Verified” is false for this checkout; setup is still incomplete

§8.2 p28 defines verified as installed and used to produce a **“committed artifact”** on the member's machine. `git log --all` still contains only `d81e8fb Initial commit`, and `git ls-tree HEAD` contains only `.gitignore` and `LICENSE`. `docs/`, `src/`, `tests/`, `meetings/`, README and packaging are untracked. Replace “committed artifact” with “local artifact” unless actual relevant commits are created/identified. The user's statement “nothing committed” directly conflicts with the workbook's definition.

More importantly for the rubric: documenting Oct7 setup honestly does not turn that setup into something acquired by the Oct1 submission. The supplied Ch8 criterion explicitly includes tools acquired for each member who requires them, while Ch8/Ch9 each have a 3-point threshold. An advisor may accept a staged plan, but the text alone does not establish all current requirements completed. Run and record the relevant members' environment smoke checks if those are in fact completed before submission. Do not assert another person's use from this checkout.

### Medium: Ch2 length pushback is not justified by the instructions

Updated Ch2 still occupies pp15–16. The research template explicitly says **“Insert a one-page summary of the justification for your project.”** The rubric rewards strong justification; it does not waive the one-page instruction or explicitly award length/risk discussion. Agree that the extra risk content is relevant; push back on treating relevance as compliance. Compress Ch2 to one page and place detailed risks in Ch4/Ch3, where they already belong. An explicit advisor exception could override that advice; no exception is documented here.

### Medium: remaining schedule and cross-document inconsistencies

1. **Fitted-state artifact contract:** §8.5 p30 schedules it in Sprint7; Ch9 p34 assigns Ronak in Sprint5. Pick one date or distinguish first contract from later serving integration.
2. **K2 time metadata:** Ch4 p22 identifies missing K2 steps/tokens and promises published schedules, but Sprint3 (§8.5 p30 and Ch9 p33) repairs only Pythia/Amber. Give K2 an owner/date or explicitly adopt checkpoint-index time for K2 with limitations.
3. **Phase2 leave-one-model-out:** Sprint9 p34 says fit all models on Phase2 data and perform a leave-one-model-out deep-KT ablation. Preferred Phase2 is OLMo-2 only (Ch4 p21; Sprint8 p34). A one-model dataset cannot provide this split. State that the ablation uses the six-model Phase1 dataset, add other Phase2 learners, or change the validation design. This is a scope ambiguity that affects whether the scheduled experiment is executable.
4. **Supporting decision status:** `docs/project-notes.md:5` now says Decisions1–5 adopted Sep29, but lines262,283,301 still label Decisions3–5 “Proposed”/“Needs team sign-off,” and line311 still calls for that sign-off. Mark those sections adopted with date; retain only genuine pending advisor approval.
5. **API ownership:** Ronak's FastAPI backend Sprint7 and Bhagyesh's data endpoints Sprint11 may be a sensible handoff, but it remains unspecified. “So that tasks do not overlap” is still stronger than the documented boundary; describe skeleton versus endpoint completion.
6. **Timing and hardware evidence:** 47 seconds/eight minutes remain in §8.3 p29 without laptop hardware/cache/run context. Server availability is asserted without status. These are team-reported timings/plans, not independently established benchmark or provisioned-server facts.
7. **Historical note vs current plan:** Sep29 meeting action to add horizons10/20 is due Oct7; current schedule evaluates them Sprint5. This can be implementation versus evaluation, but explicitly recording that distinction would preserve traceability.

## Meeting submission and Claude's URL claim

- Actual origin remains `git@github.com:bhagyeshrathi07/LLM-Knowledge-Tracing.git`; correct candidate folder URL is `https://github.com/bhagyeshrathi07/LLM-Knowledge-Tracing/tree/main/meetings`.
- **Claude's URL is correct, but it is not a correction to the existing audit.** The original main review line155 and rubric audit line19 already explicitly give that exact URL and warn against using the Ron5474 workspace path as owner evidence.
- Cadence is unchanged: only Aug31 documents a synchronous advisor meeting including all three students; Sep1 omits Aniket and Sep3 is email. No later advisor note exists. Team notes cover Sep8,15,23,29; earlier team-only coverage remains absent. Notes do not repair the historical cadence gap, and should not be altered to imply meetings that did not occur.
- All meeting files remain untracked in this checkout. This does not prove nothing exists remotely, but live publication/advisor access and prompt historical posting remain unverified. An eventual catch-up push cannot prove prompt past posting.
- Push back mildly on “nothing to change”: no invented retrospective meeting is appropriate, but `meetings/README.md:5` still asserts notes are written within24 hours without supporting Git history; line7 says every action has one owner while Aug31 and Sep29 use `All`. Phrase timely posting as intended policy if it cannot be demonstrated, clarify historical collective tasks, and record closure/status for the repeated overdue recurring-advisor-meeting request.
- Meeting dates/index are plain text, not direct note links; hyperlinks would improve discoverability but are not a missing-content issue.

## Verdict on the three stated pushbacks

1. **21/21:** Agree that only the advisor awards a grade and that a corrected honest workbook is better than the original. Push back on using “honest about planned work” as evidence that the required work is completed. There remain specific Ch8 readiness gaps and diagram/schedule/template issues. Do not guarantee a score.
2. **Ch2 two pages:** Push back for the explicit template reason above. Relevant risks can stay in the workbook without making the one-page justification two pages.
3. **Corbett1994/1995:** Bibliographic-source agent should supply the source-specific verdict; this rubric audit does not independently adjudicate publication dates.

## Updated assignment readiness

The schedule is materially stronger and plausibly near the top descriptor after its remaining contradictions are fixed; the architecture has a real diagram and substantive technological description but the diagram must match the new methods. Ch8 has a credible local prototype and now clearer ownership, yet the other members' tools remain future work. Meeting-log full credit is still not supported by the available cadence/publication evidence. The submission is much improved, but “everything is now consistent and correct” is not supportable.
