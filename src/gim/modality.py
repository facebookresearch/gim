# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Modality tagging helpers for GIM evaluation.

Provides consistent modality classification and inclusive superset tags
for per-modality metric aggregation, matching the TBR implementation.
"""


def modality_tag(sample_modalities: set[str]) -> str:
    """Return a single tag describing the modality mix of a sample."""
    if not sample_modalities:
        return "text"
    if sample_modalities == {"image"}:
        return "image"
    if sample_modalities == {"document"}:
        return "document"
    return "mixed"


def modality_supersets(tag: str) -> list[str]:
    """Return inclusive superset tags this sample belongs to.

    Used for aggregating metrics across related modality groups. For example,
    an "image" sample contributes to both "text+image" and "attachment" groups.
    """
    supersets = []
    if tag in ("text", "image"):
        supersets.append("text+image")
    if tag in ("text", "document"):
        supersets.append("text+document")
    if tag != "text":
        supersets.append("attachment")
    return supersets
