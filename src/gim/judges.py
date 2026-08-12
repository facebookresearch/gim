# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Canonical calibrated GIM judge registry.

The GIM item bank is calibrated with an additive judge fixed effect, so every
score must be attributable to one of the judges that entered the joint fit.
This module holds those judge identities and resolves them from provider model
routes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeSpec:
    """Stable judge identity plus default model route."""

    id: str
    display_name: str
    default_model: str
    description: str = ""


PERMITTED_JUDGES: dict[str, JudgeSpec] = {
    "gemini-3-flash-preview": JudgeSpec(
        id="gemini-3-flash-preview",
        display_name="Gemini 3 Flash Preview",
        default_model="google/gemini-3-flash-preview",
    ),
    "gemini-3-5-flash": JudgeSpec(
        id="gemini-3-5-flash",
        display_name="Gemini 3.5 Flash",
        default_model="google/gemini-3-5-flash",
    ),
    "gpt-5.4-mini": JudgeSpec(
        id="gpt-5.4-mini",
        display_name="GPT-5.4 Mini",
        default_model="openai/gpt-5.4-mini",
    ),
    "claude-4.5-haiku": JudgeSpec(
        id="claude-4.5-haiku",
        display_name="Claude 4.5 Haiku",
        default_model="anthropic/claude-4.5-haiku",
    ),
    "gemma-4-31b-it": JudgeSpec(
        id="gemma-4-31b-it",
        display_name="Gemma 4 31B IT",
        default_model="vllm/google/gemma-4-31b-it",
    ),
}

_DEFAULT_MODEL_TO_JUDGE_ID: dict[str, str] = {
    judge.default_model: judge.id for judge in PERMITTED_JUDGES.values()
}


def validate_judge_id(judge_id: str) -> JudgeSpec:
    """Return the judge spec for a known calibrated judge ID."""
    try:
        return PERMITTED_JUDGES[judge_id]
    except KeyError as exc:
        choices = ", ".join(sorted(PERMITTED_JUDGES))
        raise ValueError(
            f"Unknown judge_id={judge_id!r}. Choose one of: {choices}."
        ) from exc


def infer_judge_id_from_model(judge_model: str) -> str:
    """Infer the calibrated judge ID from an exact known model route."""
    try:
        return _DEFAULT_MODEL_TO_JUDGE_ID[judge_model]
    except KeyError as exc:
        raise ValueError(
            "Cannot infer judge_id from judge_model="
            f"{judge_model!r}. Pass judge_id explicitly when using a custom "
            "judge_model route."
        ) from exc


def resolve_judge(
    judge_id: str | None,
    judge_model: str | None = None,
) -> tuple[JudgeSpec, str]:
    """Resolve canonical judge ID and model route.

    The canonical judge ID keys IRT calibration. The model route is only the
    provider string used to call the judge.
    """
    if judge_id is None:
        if judge_model is None:
            raise ValueError(
                "Official GIM scoring requires judge_id, or a judge_model that "
                "exactly matches a calibrated default judge model."
            )
        judge_id = infer_judge_id_from_model(judge_model)

    spec = validate_judge_id(judge_id)
    return spec, judge_model or spec.default_model


def judge_metadata(judge: JudgeSpec, judge_model: str) -> dict[str, str]:
    """Metadata fields persisted on every score for IRT aggregation."""
    return {
        "judge_id": judge.id,
        "judge_model": judge_model,
        "judge_display_name": judge.display_name,
    }
