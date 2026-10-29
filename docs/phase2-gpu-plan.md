# Phase 2 compute plan: GPU options, throughput and cost

Owner: Aniket Sarrin (data analyses, statistical tests, Phase 2 evaluation). Action items from
`meetings/2026-09-03-advisor.md` (identify GPU options and rough cost) and the Sprint 6 plan
(pilot throughput and cost estimate due 2026-11-25; go/no-go memo due 2026-12-12).

Status: first estimate, 2026-10-28. Prices are from provider pages and price trackers consulted
on October 28, 2026 and must be re-checked on the provider's own page before any money is
spent. Throughput numbers are assumptions until the pilot in Section 4 has been run.

## 1. What Phase 2 has to compute

Phase 2 evaluates OLMo-2 7B pretraining checkpoints on MathCAMPS (Mishra, Poesia & Goodman,
COLM 2025), the one public benchmark whose problems carry fine-grained skill tags (44 Common
Core standards). Unlike Phase 1, where AI2 published the per-item correctness, we have to run the
model ourselves.

| Quantity | Value | Source |
|---|---|---|
| Checkpoints | 50 to 100, log-spaced from the ~900 OLMo-2 7B Stage-1 checkpoints on HuggingFace (`allenai/OLMo-2-1124-7B`, revisions `stage1-step*`) | Workbook 1 §4; OLMo-2 model card |
| Problems | 4,900 initial problems; 9,607 with the incremental and counterfactual follow-ups | MathCAMPS paper (arXiv 2407.00900) |
| Evaluation type | Generation: the model writes a worked answer, the final number is parsed and compared; MathCAMPS ships the grader | MathCAMPS repository, github.com/gpoesia/mathcamps |
| Prompt | Few-shot (the MathCAMPS default), roughly 600 to 900 input tokens per problem | MathCAMPS prompts |
| Output | Capped at 512 new tokens per problem; typical answers are 150 to 300 tokens | assumption, to measure in the pilot |
| Model | 7B parameters, bf16, about 15 GB of weights per checkpoint | OLMo-2 model card |

Base case: 100 checkpoints x 9,607 problems = 960,700 generations. Small case (initial problems
only, 50 checkpoints): 245,000 generations.

## 2. Throughput assumptions

Batched generation with vLLM on one 80 GB GPU for a 7B bf16 model, with 9,607 prompts of
~800 input tokens and ~250 output tokens each:

| GPU | Assumed decode throughput (batched, tokens/s) | Time per checkpoint (9,607 problems) | Time per checkpoint (4,900 problems) |
|---|---|---|---|
| A100 80 GB | 1,500 to 2,500 | 20 to 35 min | 10 to 18 min |
| H100 80 GB | 3,000 to 5,000 | 10 to 18 min | 5 to 9 min |
| L4 / A10G 24 GB | 400 to 700 | 60 to 100 min | 30 to 50 min |

Add about 2 minutes per checkpoint to download and load 15 GB of weights (network-bound; keep
checkpoints on local NVMe and delete after scoring). The decode numbers are typical published
vLLM figures for 7B models at high batch sizes; the pilot replaces them with measurements.

GPU-hours for the base case (100 checkpoints, 9,607 problems): A100 35 to 60 h; H100 18 to 32 h.
Small case (50 checkpoints, 4,900 problems): A100 9 to 16 h; H100 5 to 8 h.

## 3. Options and rough cost

Prices per GPU-hour, on demand, as listed on October 28, 2026 (re-check before use):

| Option | Price | Base case (A100 35–60 h or H100 18–32 h) | Small case | Notes |
|---|---|---|---|---|
| SJSU College of Engineering HPC (coe-hpc.sjsu.edu) | $0 | $0 | $0 | Faculty must request access, so the advisor files the request; GPU types and queue limits unknown until we have an account; condo partition is shared with lab owners. First choice if the advisor can sponsor the account by mid-November. |
| Google Colab Pro / Pro+ | $9.99/month for 100 compute units; $49.99/month for Pro+ (500 to 600 units); A100 burns roughly 5 to 15 units per hour | 2 to 4 months of Pro+ or pay-as-you-go top-ups: about $50 to $200 | about $30 to $80 | A100 not guaranteed; sessions time out (24 h max on Pro+), so the run must checkpoint its own progress per model checkpoint. Good for the pilot, poor for the full run. |
| Lambda Cloud | A100 80 GB $2.79/h (8x nodes); H100 $3.99 to $4.29/h | A100 $100 to $170; H100 $70 to $140 | A100 $25 to $45 | Reliable, simple SSH box, pay per minute. Single-GPU A100 availability varies. |
| RunPod | H100 $3.99/h secure, $2.69/h community; A100 80 GB about $1.79/h | A100 $65 to $110; H100 community $50 to $85 | A100 $16 to $30 | Community cloud is cheaper but can be pre-empted; use a persistent volume for checkpoints and results. |
| Vast.ai | A100 80 GB median about $1.00 to $1.16/h; H100 median about $2.40/h | A100 $35 to $70 | A100 $9 to $19 | Marketplace; host quality varies; cheapest if the job is restartable, which ours is (one checkpoint = one unit of work). |

Recommendation: run the pilot on Colab Pro or a single RunPod A100 (under $20 total); if SJSU HPC
access is granted, run the full job there; otherwise budget about $100 (A100 on RunPod or Vast.ai,
restartable per checkpoint) for the base case, or $30 for the small case. The whole Phase 2
compute fits in a student budget; the decision is about reliability and time, not money.

## 4. Pilot (due 2026-11-25)

1. One checkpoint (the final `stage1` checkpoint) on the 4,900 initial problems with vLLM, bf16,
   max 512 new tokens, batch as large as memory allows. Record: wall-clock time, tokens/s,
   accuracy per Common Core standard (compare with the MathCAMPS paper's OLMo numbers as a
   sanity check), GPU memory.
2. Same checkpoint with the 4,707 follow-up problems to measure the cost of the full set.
3. One early checkpoint (`stage1-step5000` or similar) to check that near-chance outputs are
   parsed sensibly by the grader (empty or runaway generations).
4. Multiply up to 50 and 100 checkpoints; write the cost table with measured numbers; propose
   the checkpoint grid (log-spaced, denser early where Phase 1 curves change fastest).

## 5. Go/no-go (due 2026-12-12)

Go if a GPU source is confirmed and the pilot's per-checkpoint time is under 45 minutes on the
available hardware. No-go switches Phase 2 to the no-GPU fallbacks already verified in Workbook 1
§4: `allenai/DataDecide-eval-instances` (about 150,000 checkpoints of models up to 1B, item-level
correctness, ODC-BY) or the Implicit Curriculum repository (`KaiserWhoLearns/ElementalTask`,
per-item results for Amber, OLMo-2 and Pythia-6.9B on 91 skill-structured tasks, MIT). Both lose
the Common Core taxonomy but keep a skill-decomposed structure.

## 6. Sources consulted (October 28, 2026)

- MathCAMPS: arXiv 2407.00900; github.com/gpoesia/mathcamps.
- OLMo-2 checkpoints: huggingface.co/allenai/OLMo-2-1124-7B (revisions).
- SJSU CoE HPC: sjsu.edu/cmpe/resources/hpc.php (access by faculty request; condo partition).
- Lambda pricing: lambda.ai/pricing. RunPod, Vast.ai, Thunder Compute: price trackers
  thundercompute.com/blog/cheapest-cloud-gpu-providers, gpuperhour.com, dev.to
  "GPU rental prices compared (Oct 2026)", vast.ai live pricing guide.
- Colab: Google Colab pricing page (Pro $9.99/100 units; Pro+ $49.99); A100 unit burn rate
  from mccormickml.com and thundercompute.com comparisons (5 to 15 units per hour, varies).
