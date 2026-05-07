# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for gim.metrics — raw_mean, gim_per_label, and gim_per_modality."""

import pytest

from gim.metrics import raw_mean, gim_per_label, gim_per_modality, DEFAULT_MIN_SAMPLES
from tests.conftest import make_sample_score


class TestRawMean:
    """Tests for the raw_mean metric."""

    def test_empty_scores(self):
        compute = raw_mean()
        assert compute([]) == 0.0

    def test_all_correct(self):
        compute = raw_mean()
        scores = [make_sample_score(1.0) for _ in range(5)]
        assert compute(scores) == 1.0

    def test_all_incorrect(self):
        compute = raw_mean()
        scores = [make_sample_score(0.0) for _ in range(5)]
        assert compute(scores) == 0.0

    def test_mixed_scores(self):
        compute = raw_mean()
        scores = [
            make_sample_score(1.0),
            make_sample_score(0.0),
        ]
        assert compute(scores) == 0.5

    def test_fractional_scores(self):
        compute = raw_mean()
        scores = [
            make_sample_score(0.8),
            make_sample_score(0.6),
            make_sample_score(0.4),
        ]
        result = compute(scores)
        assert abs(result - 0.6) < 1e-9

    def test_single_score(self):
        compute = raw_mean()
        scores = [make_sample_score(0.75)]
        assert compute(scores) == 0.75


class TestGimPerLabel:
    """Tests for the gim_per_label metric."""

    def test_empty_scores(self):
        compute = gim_per_label(min_samples=1)
        result = compute([])
        assert result == {}

    def test_single_label_above_threshold(self):
        compute = gim_per_label(min_samples=2)
        scores = [
            make_sample_score(1.0, labels=["reasoning"]),
            make_sample_score(0.5, labels=["reasoning"]),
            make_sample_score(0.0, labels=["reasoning"]),
        ]
        result = compute(scores)
        assert "reasoning" in result
        assert abs(result["reasoning"] - 0.5) < 1e-9

    def test_label_below_threshold_excluded(self):
        compute = gim_per_label(min_samples=5)
        scores = [
            make_sample_score(1.0, labels=["rare_label"]),
            make_sample_score(1.0, labels=["rare_label"]),
        ]
        result = compute(scores)
        assert "rare_label" not in result

    def test_multi_label_samples_exploded(self):
        compute = gim_per_label(min_samples=1)
        scores = [
            make_sample_score(1.0, labels=["reasoning", "knowledge"]),
            make_sample_score(0.0, labels=["reasoning"]),
        ]
        result = compute(scores)
        assert abs(result["reasoning"] - 0.5) < 1e-9
        assert abs(result["knowledge"] - 1.0) < 1e-9

    def test_default_min_samples(self):
        assert DEFAULT_MIN_SAMPLES == 3

    def test_labels_sorted_alphabetically(self):
        compute = gim_per_label(min_samples=1)
        scores = [
            make_sample_score(1.0, labels=["zebra"]),
            make_sample_score(1.0, labels=["alpha"]),
            make_sample_score(1.0, labels=["middle"]),
        ]
        result = compute(scores)
        assert list(result.keys()) == ["alpha", "middle", "zebra"]

    def test_no_labels_metadata(self):
        compute = gim_per_label(min_samples=1)
        scores = [make_sample_score(1.0, labels=[])]
        result = compute(scores)
        assert result == {}

    def test_missing_labels_key_in_metadata(self):
        """Handles samples with no 'labels' in metadata gracefully."""
        from inspect_ai.scorer import SampleScore, Score

        compute = gim_per_label(min_samples=1)
        scores = [
            SampleScore(
                score=Score(value=1.0),
                sample_id="no-meta",
                sample_metadata={},
            ),
        ]
        result = compute(scores)
        assert result == {}

    def test_boundary_min_samples(self):
        """Exactly min_samples is included."""
        compute = gim_per_label(min_samples=3)
        scores = [
            make_sample_score(1.0, labels=["exact"]),
            make_sample_score(0.5, labels=["exact"]),
            make_sample_score(0.0, labels=["exact"]),
        ]
        result = compute(scores)
        assert "exact" in result
        assert abs(result["exact"] - 0.5) < 1e-9


class TestGimPerModality:
    """Tests for the gim_per_modality metric."""

    def test_empty_scores(self):
        compute = gim_per_modality()
        assert compute([]) == {}

    def test_single_modality(self):
        compute = gim_per_modality()
        scores = [
            make_sample_score(1.0, modality="text"),
            make_sample_score(0.5, modality="text"),
        ]
        result = compute(scores)
        assert "modality/text" in result
        assert abs(result["modality/text"] - 0.75) < 1e-9

    def test_multiple_modalities(self):
        compute = gim_per_modality()
        scores = [
            make_sample_score(1.0, modality="text"),
            make_sample_score(0.5, modality="image"),
        ]
        result = compute(scores)
        assert "modality/text" in result
        assert "modality/image" in result
        assert result["modality/text"] == 1.0
        assert result["modality/image"] == 0.5

    def test_superset_aggregation(self):
        """text+image superset should aggregate text and image samples."""
        compute = gim_per_modality()
        scores = [
            make_sample_score(1.0, modality="text"),
            make_sample_score(0.0, modality="image"),
        ]
        result = compute(scores)
        # text+image contains both text and image
        assert "modality/text+image" in result
        assert abs(result["modality/text+image"] - 0.5) < 1e-9

    def test_attachment_superset(self):
        """attachment superset should include image and document but not text."""
        compute = gim_per_modality()
        scores = [
            make_sample_score(1.0, modality="image"),
            make_sample_score(0.0, modality="document"),
        ]
        result = compute(scores)
        assert "modality/attachment" in result
        assert abs(result["modality/attachment"] - 0.5) < 1e-9

    def test_text_not_in_attachment(self):
        compute = gim_per_modality()
        scores = [make_sample_score(1.0, modality="text")]
        result = compute(scores)
        # text+image and text+document supersets should be present
        assert "modality/text+image" in result
        assert "modality/text+document" in result
        # but not attachment
        assert "modality/attachment" not in result
