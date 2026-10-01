# Abstract — Draft v2

Revision of the September 2026 abstract. Changes address (1) the advisor's request that the last paragraph be readable by engineers with no background in the field, (2) the literature review finding that skill emergence and forgetting have been *described* by prior work even though they have not been *modeled*, and (3) the reframing of the time step from "benchmark item" to "training checkpoint" (Decision 1 in `project-notes.md`).

Paragraphs 1 and 2 are lightly edited. Paragraph 3 is rewritten.

---

## Knowledge Tracing for Large Language Models

Knowledge tracing (KT) estimates a learner's latent mastery of individual skills from observed responses, updating it after every interaction. KT is the learner model inside intelligent tutoring systems, deciding what to teach next and when a skill is mastered or forgotten. Large language models (LLMs) are now learners too: they acquire skills across thousands of pretraining checkpoints, gain and lose competence during fine-tuning, and revise answers as corrections arrive in conversation.

Existing methods characterize what an LLM knows at a single point in time. Item response theory and cognitive diagnosis estimate one ability profile for a frozen model. Studies of training dynamics have observed when skills first appear and which facts are later lost, but they rely on accuracy curves and fixed thresholds rather than a learner model. No existing method maintains a per-skill knowledge state for an LLM that is updated at each checkpoint and validated by predicting the model's next responses. As a result, skill emergence, forgetting, and shifts in item difficulty during training are described but not modeled, and mastery estimates cannot reliably guide training or instruction.

This project adapts knowledge tracing to LLMs. We treat each saved training checkpoint as one moment in a learner's history, and the model's answers to skill-tagged test questions at that checkpoint as evidence of what it knows at that moment. We fit sequential KT models, which allow for forgetting and for uneven gaps between checkpoints, to these answers. We begin with published checkpoint results for six openly trained models, then extend to a model whose complete training history is public (OLMo-2), tracking its grade-school mathematics skills in detail. We validate the models by holding out later checkpoints and predicting their answers, test whether question difficulty stays stable over the course of training, and derive per-skill learning curves. We also build an interactive visualization so that developers and researchers can see which skills a model learned, when it learned them, and what it forgot. The result is a validated learner model for LLMs that supports skill-level evaluation and mastery-driven training curricula.

---

## Change log against v1

**Paragraph 1.** Unchanged.

**Paragraph 2.**
- "Methods for characterizing what an LLM knows remain static" → "Existing methods characterize what an LLM knows at a single point in time." Same meaning, plainer.
- Added a sentence acknowledging that training-dynamics studies have observed emergence and forgetting, so that reviewers familiar with MathCAMPS, Fluid Benchmarking, the Implicit Curriculum paper, or the post-training forgetting paper do not read the abstract as unaware of them.
- "go unmeasured" → "are described but not modeled." The literature review showed the original claim was too strong.
- "updated after each response" → "updated at each checkpoint." Matches Decision 1.

**Paragraph 3.**
- Removed "each skill-tagged benchmark item as a practice opportunity." The model does not learn from benchmark items; they are observations, not practice. Replaced with "evidence of what it knows at that moment."
- "forgetting and temporal priors" → "allow for forgetting and for uneven gaps between checkpoints." Same content, no jargon.
- "released checkpoint responses for six open models" → "published checkpoint results for six openly trained models."
- OLMo-2 sentence rewritten so a reader who has never heard of OLMo-2 still understands why it was chosen: its complete training history is public. The name stays in parentheses so specialists can identify it. If the advisor prefers, the parenthetical can be dropped with no loss of meaning.
- "skill-annotated mathematics" → "grade-school mathematics skills."
- "held-out next-checkpoint prediction, test item-difficulty invariance" → spelled out in plain words.
- Added the visualization sentence per advisor feedback.
- "skill-aware evaluation and mastery-driven curricula" → "skill-level evaluation and mastery-driven training curricula."

## Open questions for the advisor

1. Keep "(OLMo-2)" in parentheses, or remove the name entirely?
2. Is "knowledge state" acceptable in paragraph 2, or should it be "estimate of mastery"?
3. Word count is now ~330 vs ~290 in v1. If there is a limit, the visualization sentence or the "Studies of training dynamics" sentence can be cut.
