# Workbook 1 Presentation Script

Target 10–12 minutes at a normal speaking pace (about 150 words per minute); the current script is about 1788 words, roughly 12 minutes. Each section below is also embedded as the speaker notes of the matching slide in `Workbook1_Presentation.pptx`. Timing marks are cumulative.

Split: Bhagyesh slides 1–6, Ronak slides 7–12 (Aniket is not presenting). The hand-off is written into the script at the end of slide 6.

---

## Slide 1 — Title (0:00–0:41)

Good morning. I am Bhagyesh, this is Ronak, and our third teammate Aniket could not be here today. Our advisor is Dr. Mahima Agumbe. Our project is Knowledge Tracing for Large Language Models.

The one-sentence version: tutoring software already knows how to track what a student understands, skill by skill, from the answers they give. We are pointing that same idea at an AI language model while it is being trained.

In the next ten minutes: what others have done, why this is worth two semesters, how we will know if it works, what we have built in five weeks, and what comes next.

## Slide 2 — One score hides four stories (0:41–2:01)

Here is the project in one picture: real numbers from one open AI model whose developers saved 94 snapshots during training and gave each one the same exam.

Normally each snapshot gets one score. That one number hides four very different stories. Some subjects are learned early and then level off. Some improve slowly the whole way. Some never really get off the ground. And some peak and then dip, which might be forgetting, or might just be noise.

Our idea is to treat the model as a student. Each snapshot is that student on a different day of class. Each exam question is tagged with a skill, and the answer is evidence of what the student knows that day. Knowledge tracing keeps a running estimate of how well each skill is known and, this is the important part, proves it means something by predicting the next day's exam before seeing it.

One thing we had to get right early: the model does not learn from the exam. The exam is a thermometer, not a lesson. So our unit of time is the snapshot, not the question. A small change from the tutoring version, but it decides which methods fit.

## Slide 3 — Three literatures, none yet combined (2:01–3:29)

We read about 45 papers in full and ended with 49 references, most peer-reviewed. They fall into three groups.

The first group measures what a model knows at one point in time, using the statistics behind standardized tests. One of these groups published the data we use. But every snapshot is scored on its own; nothing carries forward, so there is no notion of a skill being learned or lost.

The second group watches models during training. They show that skills appear in a consistent order and count how often a question flips from right to wrong. But they describe these patterns with charts and counts; none builds a model of the student or tests it by predicting what comes next.

The third group is the student-tracking methods themselves, from education research and, oddly enough, from political science, where the same math tracks judges' positions over decades. Built for exactly our situation, a few learners watched over many periods, and never applied to an AI model's snapshots.

So the gap is the combination. In everything we reviewed, no one keeps a running skill-by-skill estimate across a model's training and checks it by predicting the next snapshot. We say "in everything we reviewed" on purpose: a search can show a question is open, not that it has never been asked.

## Slide 4 — Why feasible and publishable (3:29–4:32)

Three reasons this is worth two semesters.

First, the question is open, as we just showed.

Second, the hard part is already solved. When we wrote the abstract we did not know whether anyone had published right-or-wrong answers for every question at every snapshot. One research group had. We downloaded it, checked it, and converted it: six models, 452 snapshots, nearly 28,000 questions, 12.6 million answers. The first phase needs no special hardware; it all runs on a laptop.

Third, we decided in advance how we will know if it works. Before building any learning model, we built the best predictor that has no learning in it at all, and its score is on the slide. Anything we propose has to beat it on the same data, and if it does not, we will say so. The target is a conference submission in 2027.

So how will we know if it works? That is the next slide.

## Slide 5 — The bar (4:32–5:42)

Before building anything clever, we need to know what "good" looks like. So we started with five very simple ways of guessing whether the model gets a question right at the next checkpoint. None has any idea of learning inside; they just look at the past.

The simplest uses overall accuracy so far. Another uses how the model did on that subject. Another repeats what it did last time on that exact question. The best of the five averages everything the model has done on that exact question across all earlier checkpoints.

Each predictor forecasts a checkpoint it has not seen, and we compare with what happened. Lower is better in the first column, higher in the second.

Two things stand out. Knowing the subject helps little; knowing the specific question helps a lot. And averaging a question's history beats copying its last answer, so any single checkpoint is a noisy snapshot.

That best simple guess is our bar. Any learning model has to beat it on the same data, or it has not earned its keep.

## Slide 6 — Learner models under study (5:42–7:18)

These are the three learning models we will build, simplest first.

The first takes an existing difficulty score for every question and adds the idea that the model's ability drifts over training, so older answers count a little less. It is an afternoon's work, and it answers one question before we invest further: does adding time help at all?

The second is the classic model from tutoring software, Bayesian Knowledge Tracing. For every skill it keeps a running estimate of whether the skill has been learned, and it produces numbers a person can read: how fast the skill is picked up, how fast it is forgotten, how often the model guesses right or slips. That readability is why we want it.

The third is our main model. It tracks an ability for every skill that moves a little between checkpoints, and it learns how hard each question is for this particular model. Nothing ready-made fits our situation, so we write it ourselves and test it on simulated data first.

We will also try a neural-network version, but only as a comparison: those need thousands of students, and we have six.

Every model goes through the same checks so we cannot fool ourselves: a simulation test, checkpoints held back for the final score, and a check that any uncertainty we report is honest.

Ronak will take it from here with the risks, the system, and where we are today.

## Slide 7 — Showstoppers and deliverables (7:18–8:10)

Thanks, Bhagyesh. We list only the things that could stop the project.

The biggest is resolved: we needed right-or-wrong answers for every question at every checkpoint, and a public release provides them. The one open item is GPU access next semester, to evaluate a model ourselves on a math benchmark with proper skill labels. Aniket owns the cost estimate and the December go-or-no-go; if we cannot get GPU time, two public datasets work instead. The rest are engineering risks with a fallback named.

On the right is what we hand over: the dataset and pipeline, the model library, the evaluation harness, the skill-by-skill learning curves with uncertainty, the visualization our advisor asked for, the analyses of when skills appear and whether they are forgotten, and a conference submission with a public repository.

## Slide 8 — Architecture (8:10–8:54)

The system is five stages in a line.

Public evaluation data goes in one end and becomes a single table: one row per model, checkpoint and question. Every predictor plugs into the same slot, so the evaluation stage scores all of them the same way. The analysis stage turns those scores into what we care about: when each skill appears, whether it is later lost, and how models compare. The last stage is a small web app that lets a developer browse all of that without reading our code.

Each stage writes files, so any stage can be re-run alone and everything rebuilds from the raw public data.

## Slide 9 — Five weeks in (8:54–9:59)

Here is where we are, five weeks in.

The pipeline is built and fast: the full dataset converts in under a minute, and the simple predictors score all 452 checkpoints in about eight minutes. Everything has automated tests, and all three of us can set it up from scratch.

We also have first results, all exploratory. On one model we see three kinds of learning curve: skills picked up early that level off, skills that improve slowly throughout, and skills that never get off the ground. Seven subjects dip after a peak; those are candidates we will test properly for forgetting, not claims we make today. And the smaller Pythia models barely move on this benchmark, matching published results, while the larger models reach 65 to 80 percent on biology.

One more thing. We had the workbook independently reviewed, and several numbers and citations were wrong. They are fixed and the review is in the repository; we would rather show that than hide it.

## Slide 10 — Sprints (9:59–10:51)

We work in two-week sprints with one owner per track, so nothing falls between two people. I own the data and the tooling, including the visualization. Bhagyesh owns the learning models. Aniket owns the analyses, the statistics, and next semester's evaluation run.

This semester goes from the first, simplest model in Sprint 3, through the two heavier models in Sprints 4 and 5, to the forgetting and drift analyses and the GPU pilot in Sprint 6, ending with a write-up and the decision about next semester's data in December.

Spring is the second dataset, refitting every model on it, the final analyses, the dashboard, and the paper. We have set ourselves an internal finish line at the end of April so that a May conference deadline is not a surprise.

## Slide 11 — Who did what (10:51–11:22)

Briefly, who did what so far. I wrote the method section of the abstract, ran the literature search, verified the data, and built the pipeline and evaluation tooling. Bhagyesh created the repository, wrote the abstract's background and gap, led the first literature survey and the design write-ups, and is our advisor liaison. Aniket wrote the problem statement that seeded the abstract, surveyed which model releases publish question-level results, drafted the data layout, and keeps the meeting notes.

## Slide 12 — Next two weeks (11:22–11:55)

In the next two weeks: the first learning model gets scored against the bar. The forgetting candidates get a proper statistical test, with the safeguards needed because we picked them after looking at the curves. And our evaluation tooling gains error bars and a locked-away set of snapshots for final scoring, so from here on every comparison comes with a margin of error and nothing is tuned on the data it is judged on.

Thank you. We are happy to take questions.
