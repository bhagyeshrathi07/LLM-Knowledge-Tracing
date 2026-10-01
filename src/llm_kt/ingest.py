"""Convert Fluid Benchmarking per-model CSVs into one long-format response table.

Source: https://huggingface.co/datasets/allenai/fluid-benchmarking

Each source CSV is wide: one row per benchmark item, one column per pretraining
checkpoint, cell = correctness. Checkpoint column names differ by model family:

    pythia-*   step143000
    olmo1-7b   step5000-tokens20B
    olmo2-7b   stage1-step928646-tokens3896B
    amber-7b   ckpt_093        (LLM360 checkpoint index; no step or token count)
    k2-65b     ckpt_093

The canonical long table has one row per (model, checkpoint, item):

    model            str    e.g. "olmo2-7b"
    checkpoint_idx   int    0-based position of the checkpoint in the source file
    checkpoint_label str    raw column name from the source file
    step             Int64  optimizer step, null when the source does not give one
    tokens_seen_b    Int64  billions of tokens seen, null when not given
    benchmark        str    "arc_challenge", "gsm8k", "hellaswag", "mmlu", "winogrande"
    skill            str    MMLU subject for MMLU items; benchmark name otherwise
    item_id          str    e.g. "mmlu_abstract_algebra_0"
    correct          int8   0 or 1
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HF_REPO = "allenai/fluid-benchmarking"
MODELS = ("amber-7b", "k2-65b", "olmo1-7b", "olmo2-7b", "pythia-3b", "pythia-7b")

# TruthfulQA mc2 scores are continuous probabilities, not 0/1, so they do not fit
# a correctness-based learner model. Dropped by default; if kept, thresholded at 0.5.
CONTINUOUS_BENCHMARKS = ("truthfulqa_mc2",)
DEFAULT_DROP_BENCHMARKS = CONTINUOUS_BENCHMARKS
CONTINUOUS_THRESHOLD = 0.5

COLUMNS = [
    "model",
    "checkpoint_idx",
    "checkpoint_label",
    "step",
    "tokens_seen_b",
    "benchmark",
    "skill",
    "item_id",
    "correct",
]

_STEP_RE = re.compile(r"^(?:stage\d+-)?step(\d+)(?:-tokens(\d+)B)?$")
_CKPT_RE = re.compile(r"^ckpt_\d+$")
_ITEM_RE = re.compile(r"^(.*)_(\d+)$")


def parse_checkpoint(label: str) -> tuple[int | None, int | None]:
    """Return (step, tokens_seen_b) from a checkpoint column name. Either may be None."""
    m = _STEP_RE.match(label)
    if m:
        step = int(m.group(1))
        tokens = int(m.group(2)) if m.group(2) is not None else None
        return step, tokens
    if _CKPT_RE.match(label):
        return None, None
    raise ValueError(f"Unrecognised checkpoint label: {label!r}")


def parse_item_id(item_id: str) -> tuple[str, str]:
    """Return (benchmark, skill) from an item id like 'mmlu_abstract_algebra_0'."""
    m = _ITEM_RE.match(item_id)
    if not m:
        raise ValueError(f"Unrecognised item id: {item_id!r}")
    prefix = m.group(1)
    if prefix.startswith("mmlu_"):
        return "mmlu", prefix[len("mmlu_") :]
    return prefix, prefix


def wide_to_long(
    wide: pd.DataFrame,
    model: str,
    drop_benchmarks: tuple[str, ...] = DEFAULT_DROP_BENCHMARKS,
) -> pd.DataFrame:
    """Melt one model's wide CSV into the canonical long format."""
    wide = wide.rename(columns={wide.columns[0]: "item_id"})
    checkpoint_labels = list(wide.columns[1:])

    long = wide.melt(id_vars="item_id", var_name="checkpoint_label", value_name="correct")

    parsed_items = long["item_id"].map(parse_item_id)
    long["benchmark"] = parsed_items.str[0]
    long["skill"] = parsed_items.str[1]
    long = long[~long["benchmark"].isin(drop_benchmarks)]

    is_continuous = long["benchmark"].isin(CONTINUOUS_BENCHMARKS)
    long.loc[is_continuous, "correct"] = (
        long.loc[is_continuous, "correct"] >= CONTINUOUS_THRESHOLD
    ).astype(float)

    bad = long["correct"][~long["correct"].isin([0.0, 1.0])]
    if not bad.empty:
        raise ValueError(
            f"{model}: {len(bad)} non-binary correctness values in kept benchmarks "
            f"(first: {bad.iloc[0]!r})"
        )
    long["correct"] = long["correct"].astype("int8")

    label_to_idx = {label: i for i, label in enumerate(checkpoint_labels)}
    long["checkpoint_idx"] = long["checkpoint_label"].map(label_to_idx).astype("int32")

    parsed_ckpts = {label: parse_checkpoint(label) for label in checkpoint_labels}
    long["step"] = long["checkpoint_label"].map(lambda c: parsed_ckpts[c][0]).astype("Int64")
    long["tokens_seen_b"] = (
        long["checkpoint_label"].map(lambda c: parsed_ckpts[c][1]).astype("Int64")
    )

    long["model"] = model
    return long[COLUMNS].reset_index(drop=True)


def download_model_csv(model: str, cache_dir: Path | None = None) -> Path:
    """Download one model's eval-results CSV from HuggingFace and return the local path."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        HF_REPO,
        f"data/lm_eval_results/{model}.csv",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return Path(path)


def ingest_all(
    models: tuple[str, ...] = MODELS,
    drop_benchmarks: tuple[str, ...] = DEFAULT_DROP_BENCHMARKS,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Download and convert every model, returning one concatenated long table."""
    frames = []
    for model in models:
        path = download_model_csv(model, cache_dir=cache_dir)
        wide = pd.read_csv(path)
        frames.append(wide_to_long(wide, model=model, drop_benchmarks=drop_benchmarks))
    return pd.concat(frames, ignore_index=True)
