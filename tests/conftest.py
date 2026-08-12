# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared fixtures for GIM tests."""

import pytest
from inspect_ai.dataset import Sample
from inspect_ai.scorer import SampleScore, Score, CORRECT, INCORRECT


@pytest.fixture
def sample_hf_record():
    """A minimal HuggingFace dataset row (TBR schema)."""
    return {
        "prompt_id": "gim_test0001",
        "prompt": "What is the capital of France?",
        "answer_gtfa": "Paris",
        "labels": "reasoning, knowledge",
        "rubrics": ["Identifies Paris as the capital", "Provides geographic context"],
        "attachments": [],
        "solution_reasoning": "Direct factual recall",
        "citations": None,
    }


@pytest.fixture
def sample_hf_record_no_rubrics():
    """An HF record with no rubrics (exact-answer only)."""
    return {
        "prompt_id": "gim_test0002",
        "prompt": "Solve 2+2",
        "answer_gtfa": "4",
        "labels": "reasoning",
        "rubrics": None,
        "attachments": [],
        "solution_reasoning": "",
        "citations": None,
    }


@pytest.fixture
def sample_hf_record_nan_answer():
    """An HF record where the answer is null/nan."""
    return {
        "prompt_id": "gim_test0003",
        "prompt": "Open-ended question",
        "answer_gtfa": None,
        "labels": "language, reasoning",
        "rubrics": ["Demonstrates creative thinking", "Uses examples"],
        "attachments": ["media/gim_test0003/image.png"],
        "solution_reasoning": "",
        "citations": None,
    }


def make_sample_score(
    value,
    labels: list[str] | None = None,
    sample_id: str = "test",
    modality: str = "text",
    score_metadata: dict | None = None,
) -> SampleScore:
    """Helper to create SampleScore objects for metric tests."""
    return SampleScore(
        score=Score(value=value, metadata=score_metadata),
        sample_id=sample_id,
        sample_metadata={"labels": labels or [], "modality": modality},
    )
