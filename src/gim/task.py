# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""GIM (Grounded Integration Measure) evaluation task for Inspect AI.

Usage (run from the project root with ``uv run``):

    # Evaluate all modalities (default)
    uv run inspect eval gim/v3 --model anthropic/claude-sonnet-4-20250514

    # Text-only samples (no attachments)
    uv run inspect eval gim/v3 --model openai/gpt-4o -T modality=text_only

    # Image-bearing samples only
    uv run inspect eval gim/v3 --model openai/gpt-4o -T modality=image

    # Document (PDF) samples only
    uv run inspect eval gim/v3 --model openai/gpt-4o -T modality=docs

    # All attachment-bearing samples (images + PDFs)
    uv run inspect eval gim/v3 --model openai/gpt-4o -T modality=media

    # Use GCS for media (Google models only)
    uv run inspect eval gim/v3 --model google/gemini-3-flash-preview-genai -M api_version=v1 \\
        -T media_base=gs://tbd-evals/gim/v3.0.0

    # Override grader model
    uv run inspect eval gim/v3 --model openai/gpt-4o -T grader_model=openai/gpt-4o

    # Multiple epochs for statistical robustness
    uv run inspect eval gim/v3 --model openai/gpt-4o -T epochs=5

    # Select a specific split from a DatasetDict
    uv run inspect eval gim/v3 --model openai/gpt-4o -T split=private
"""

import asyncio
import logging

from inspect_ai import Epochs, Task, task
from inspect_ai._util.content import ContentDocument
from inspect_ai.solver import Generate, generate, Solver, solver, TaskState

from .dataset import gim_dataset
from .scorers import gim_scorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error classification (mirrors TBR SamplerFailureAsStaticRewardActor)
# ---------------------------------------------------------------------------


def _classify_error(exc: Exception) -> str:
    """Classify a generation exception for error-type metrics."""
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg or "deadline" in msg:
        return "timeout"
    if "rate limit" in msg or "429" in msg or "too many tokens" in msg:
        return "rate_limit"
    if "filenotfounderror" in msg or "no such file" in msg:
        return "file_not_found"
    if "bad gateway" in msg or "502" in msg or "503" in msg:
        return "server_error"
    if "context length" in msg or "max_tokens" in msg or "too long" in msg:
        return "context_length"
    return "other"


# ---------------------------------------------------------------------------
# Generation wrapper
# ---------------------------------------------------------------------------


@solver
def _generate_or_zero(
    timeout: float | None = 300, skip_documents: bool = False
) -> Solver:
    """generate() wrapper that converts generation failures into empty completions.

    Inspect AI only calls the scorer when the solver returns without error
    (run.py: ``if error is None: ... score_result = await scorer(...)``).n    When generation raises, Inspect marks the sample as errored and skips
    scoring entirely, excluding it from the metric denominator.

    This wrapper catches any exception from generate(), logs it with error
    classification, and returns the TaskState with its default empty
    ModelOutput so the scorer IS called. The scorer then detects the empty
    completion and returns Score(value=0.0), counting the failure in the
    denominator.

    Error classification and metadata are stored on the state so downstream
    metrics can aggregate failure types.

    Args:
        timeout: Per-sample generation timeout in seconds. ``None`` disables
            the timeout. Defaults to 300 s (5 min).
        skip_documents: If True, skip samples with document/PDF content (for
            models that cannot process documents). Defaults to False.
    """
    if isinstance(timeout, str):
        timeout = None if timeout.lower() == "none" else float(timeout)

    gen = generate()

    def _has_documents(state: TaskState) -> bool:
        """Check if any message contains ContentDocument (e.g. PDF)."""
        for msg in state.messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, ContentDocument):
                        return True
        return False

    async def solve(state: TaskState, fn: Generate) -> TaskState:
        if skip_documents and _has_documents(state):
            logger.warning(
                "Sample %s contains document content (PDF) that the model "
                "cannot process \u2014 scoring as 0.0",
                state.sample_id,
            )
            state.metadata["solved"] = False
            state.metadata["generation_error_type"] = "unsupported_content"
            return state

        try:
            if timeout is not None:
                state = await asyncio.wait_for(gen(state, fn), timeout=timeout)
            else:
                state = await gen(state, fn)
            state.metadata["solved"] = True
            return state
        except asyncio.TimeoutError:
            logger.warning(
                "Generation timed out for sample %s after %ss (counted as 0.0)",
                state.sample_id,
                timeout,
            )
            state.metadata["solved"] = False
            state.metadata["generation_error_type"] = "timeout"
            return state
        except Exception as exc:
            error_type = _classify_error(exc)
            logger.warning(
                "Generation failed for sample %s [%s] (counted as 0.0): %s",
                state.sample_id,
                error_type,
                exc,
            )
            state.metadata["solved"] = False
            state.metadata["generation_error_type"] = error_type
            return state

    return solve


_MODALITY_PRESETS: dict[str, tuple[list[str] | None, bool]] = {
    "all": (None, False),
    "text_only": ([], False),
    "image": (["image"], True),
    "docs": (["document"], True),
    "media": (["image", "document"], True),
}


@task
def v3(
    grader_model: str | None = "google/gemini-3-flash-preview",
    dataset_path: str = "",
    epochs: int = 1,
    media_base: str = "",
    split: str = "public",
    modality: str = "all",
    generation_timeout: float | None = 600,
    no_score: bool = False,
    skip_documents: bool = False,
) -> Task:
    """GIM v3 — all modalities (text, image, document).

    Args:
        grader_model: Model for LLM-as-judge scoring. Defaults to google/gemini-3-flash-preview.
        dataset_path: HuggingFace dataset directory. Defaults to data/gim_v3_dataset.
        epochs: Runs per sample (use 5 for paper results).
        media_base: Base path/URI for resolving attachment paths. Defaults to the
            dataset directory. Use "gs://bucket/prefix" for GCS (Google models only).
        split: Which split to load from a DatasetDict. Use "public" (default),
            "private", or "all" to concatenate all splits. Ignored when the
            on-disk dataset is a plain Dataset (not a DatasetDict).
        modality: Which samples to include. One of:
            "all" (default) — all modalities (text, image, document).
            "text_only" — text-only samples (no attachments).
            "image" — image-bearing samples only (no PDFs).
            "docs" — document (PDF) bearing samples only.
            "media" — all attachment-bearing samples (images + PDFs).
        generation_timeout: Per-sample generation timeout in seconds. ``None``
            disables the timeout. Defaults to 600 s (10 min).
        no_score: If True, run inference only (no LLM-as-judge scoring).
        skip_documents: If True, skip samples with document/PDF content (for
            models that cannot process documents). Defaults to False.
    """
    if modality not in _MODALITY_PRESETS:
        raise ValueError(
            f"Unknown modality={modality!r}. "
            f"Choose from: {', '.join(_MODALITY_PRESETS)}"
        )
    modalities, require_attachment = _MODALITY_PRESETS[modality]
    return Task(
        dataset=gim_dataset(
            path=dataset_path or None,
            media_base=media_base or None,
            modalities=modalities,
            require_attachment=require_attachment,
            split=split,
        ),
        solver=_generate_or_zero(
            timeout=generation_timeout, skip_documents=skip_documents
        ),
        scorer=None if no_score else gim_scorer(grader_model=grader_model),
        epochs=Epochs(epochs, reducer="mean") if epochs > 1 else None,
        fail_on_error=False,
    )
