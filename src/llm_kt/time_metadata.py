"""Fill the training-time clock (tokens seen) for models whose Fluid labels do not carry it.

The Fluid Benchmarking release names checkpoints differently per model family (see
``llm_kt.ingest``). OLMo labels carry both the optimizer step and the tokens seen; Pythia
labels carry the step only; the LLM360 models (Amber, K2) carry only a checkpoint index. The
temporal models (T-SKIRT, DriftIRT) accept any monotone time column, so a tokens-seen clock
lets the elapsed-training gap between checkpoints be measured in the same unit for every
model instead of in checkpoint positions.

Three sources, recorded per row in ``tokens_seen_source``:

``published``
    The label itself gives the count (OLMo-1, OLMo-2). Left unchanged.
``derived``
    Pythia: every step consumes a fixed batch of 1,024 sequences x 2,048 tokens =
    2,097,152 tokens (Biderman et al., 2023, Section 2), so tokens = step x 2,097,152.
    143,000 steps gives 299.9B tokens, matching the published 300B. Exact up to rounding.
``estimated``
    Amber: 360 checkpoints (ckpt_000 to ckpt_359), one per data chunk, over 1,259.13B
    tokens; the maintainers state the checkpoints are equally spaced in steps, so checkpoint
    k is taken to sit after (k + 1) chunks of 1259.13 / 360 = 3.497B tokens.
    K2: 360 Stage-1 checkpoints (ckpt_001 to ckpt_360) over a 1.3T-token Stage-1 mix, so
    checkpoint k is taken to sit at k / 360 of 1,300B = 3.611B tokens per checkpoint. The
    equal-spacing assumption for K2 is not documented by LLM360 and is the weaker of the two.

Both estimates are linear in the checkpoint index, so for Amber and K2 a tokens-seen clock
is the checkpoint index rescaled; they matter only when comparing gaps across models.
Verification steps and sources are in docs/phase2-gpu-plan.md's companion note in the
handoff README. Values are integers in billions, like the published column.
"""

from __future__ import annotations

import re

import pandas as pd

# Pythia: batch size 1,024 sequences x sequence length 2,048 tokens per optimizer step.
PYTHIA_TOKENS_PER_STEP = 1_024 * 2_048  # 2,097,152

# LLM360 checkpoint schedules: (tokens in billions per checkpoint, index offset).
# tokens_b(k) = (k + offset) * per_ckpt_b
LLM360_SCHEDULE = {
    "amber-7b": (1259.13 / 360, 1),  # ckpt_000 is after the first chunk
    "k2-65b": (1300.0 / 360, 0),  # ckpt_360 is the end of Stage 1
}

TOKENS_PER_STEP = {"pythia-3b": PYTHIA_TOKENS_PER_STEP, "pythia-7b": PYTHIA_TOKENS_PER_STEP}

SOURCE_PUBLISHED = "published"
SOURCE_DERIVED = "derived"
SOURCE_ESTIMATED = "estimated"

_CKPT_RE = re.compile(r"^ckpt_(\d+)$")


def tokens_from_step(model: str, step: int) -> int:
    """Billions of tokens seen after ``step`` optimizer steps, for models with a fixed batch."""
    per_step = TOKENS_PER_STEP[model]
    return round(step * per_step / 1e9)


def tokens_from_ckpt_label(model: str, label: str) -> int:
    """Estimated billions of tokens seen at an LLM360 checkpoint label such as ``ckpt_093``."""
    m = _CKPT_RE.match(label)
    if not m:
        raise ValueError(f"{model}: not an LLM360 checkpoint label: {label!r}")
    per_ckpt_b, offset = LLM360_SCHEDULE[model]
    return round((int(m.group(1)) + offset) * per_ckpt_b)


def fill_time_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the long response table with ``tokens_seen_b`` filled where it was null.

    Adds a ``tokens_seen_source`` column with one of ``published``, ``derived`` or
    ``estimated`` per row. Rows whose model has no rule and no published value keep a null
    token count and a null source. Works on the unique (model, checkpoint_label) pairs and
    maps back, so it costs seconds on the full 12.6-million-row table.
    """
    out = df.copy()
    keys = out[["model", "checkpoint_label", "step", "tokens_seen_b"]].drop_duplicates()

    tokens: list[object] = []
    source: list[object] = []
    for model, label, step, published in keys.itertuples(index=False):
        if pd.notna(published):
            tokens.append(int(published))
            source.append(SOURCE_PUBLISHED)
        elif model in TOKENS_PER_STEP and pd.notna(step):
            tokens.append(tokens_from_step(model, int(step)))
            source.append(SOURCE_DERIVED)
        elif model in LLM360_SCHEDULE:
            tokens.append(tokens_from_ckpt_label(model, label))
            source.append(SOURCE_ESTIMATED)
        else:
            tokens.append(pd.NA)
            source.append(pd.NA)
    keys = keys.assign(_tokens=tokens, _source=source)

    merged = out.merge(
        keys[["model", "checkpoint_label", "_tokens", "_source"]],
        on=["model", "checkpoint_label"],
        how="left",
        validate="many_to_one",
    )
    out["tokens_seen_b"] = pd.array(merged["_tokens"].tolist(), dtype="Int64")
    out["tokens_seen_source"] = pd.array(merged["_source"].tolist(), dtype="string")
    return out
