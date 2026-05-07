# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""GIM-specific metrics for Inspect AI.

Provides:
- ``gim_score``: IRT ability (θ) computed from the pre-calibrated item bank.
- ``raw_mean``: Simple mean score across all samples.
- ``gim_per_label``: Per-label score breakdown for multi-label samples.
- ``gim_per_modality``: Per-modality score breakdown with superset aggregation.
"""

import logging
from collections import defaultdict

from inspect_ai.scorer import metric, SampleScore, Value

from .modality import modality_supersets

logger = logging.getLogger(__name__)

# Minimum number of samples per label to report a per-label metric.
DEFAULT_MIN_SAMPLES = 3


@metric
def gim_score() -> ...:
    """IRT ability estimate (θ) from the pre-calibrated GIM item bank.

    Computes a single ability score using 2PL IRT closed-form WLS, weighting
    each prompt by its discrimination and difficulty parameters.  Returns a
    dict with ``gim_score``, ``gim_score_se``, ``gim_score_ci_lower``,
    ``gim_score_ci_upper``, ``gim_score_n_items``, and ``gim_score_coverage``.
    """

    def compute(scores: list[SampleScore]) -> Value:
        # Lazy import to avoid loading numpy/item bank when not needed.
        from .irt import GIMScorer

        score_dict: dict[str, float] = {}
        for s in scores:
            if s.sample_id:
                score_dict[str(s.sample_id)] = s.score.as_float()

        try:
            scorer = GIMScorer()
            result = scorer.score(score_dict)
        except Exception:
            logger.warning("IRT scoring failed; returning empty metrics", exc_info=True)
            return {}

        return {
            "gim_score": result.theta,
            "gim_score_se": result.se,
            "gim_score_ci_lower": result.ci_95_lower,
            "gim_score_ci_upper": result.ci_95_upper,
            "gim_score_n_items": result.n_items_scored,
            "gim_score_coverage": result.coverage,
        }

    return compute


@metric
def raw_mean() -> ...:
    """Simple mean GIM score across all samples."""

    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        return sum(s.score.as_float() for s in scores) / len(scores)

    return compute


@metric
def gim_per_label(min_samples: int = DEFAULT_MIN_SAMPLES) -> ...:
    """Per-label mean scores.

    Explodes multi-label samples so each label gets its own aggregate.
    Labels with fewer than ``min_samples`` samples are excluded.

    Returns a dict mapping ``label_name`` → mean score.
    """

    def compute(scores: list[SampleScore]) -> Value:
        label_totals: dict[str, float] = defaultdict(float)
        label_counts: dict[str, int] = defaultdict(int)

        for s in scores:
            labels = (s.sample_metadata or {}).get("labels", [])
            val = s.score.as_float()
            for label in labels:
                label_totals[label] += val
                label_counts[label] += 1

        return {
            label: label_totals[label] / count
            for label, count in sorted(label_counts.items())
            if count >= min_samples
        }

    return compute


@metric
def gim_per_modality() -> ...:
    """Per-modality mean scores with inclusive superset aggregation.

    Uses the ``modality`` tag from sample metadata (text, image, document,
    mixed) and also aggregates into superset groups (text+image,
    text+document, attachment) matching the TBR implementation.

    Returns a dict mapping ``modality/{tag}`` -> mean score.
    """

    def compute(scores: list[SampleScore]) -> Value:
        modality_totals: dict[str, float] = defaultdict(float)
        modality_counts: dict[str, int] = defaultdict(int)

        for s in scores:
            tag = (s.sample_metadata or {}).get("modality", "unknown")
            val = s.score.as_float()

            modality_totals[tag] += val
            modality_counts[tag] += 1

            for superset in modality_supersets(tag):
                modality_totals[superset] += val
                modality_counts[superset] += 1

        return {
            f"modality/{tag}": modality_totals[tag] / count
            for tag, count in sorted(modality_counts.items())
            if count > 0
        }

    return compute
