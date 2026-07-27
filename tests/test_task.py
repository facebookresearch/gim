# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for gim.task — task construction and parameter handling."""

from unittest.mock import patch

import pytest
from inspect_ai import Task, Epochs
from inspect_ai.dataset import MemoryDataset, Sample

from gim.task import _classify_error, _generate_or_missing, v3


def _fake_dataset():
    return MemoryDataset(
        samples=[Sample(input="test prompt", id="test-1")],
        name="test",
    )


# ---------------------------------------------------------------------------
# Core task construction
# ---------------------------------------------------------------------------


class TestV3Task:
    @patch("gim.task.gim_dataset", return_value=_fake_dataset())
    def test_returns_task(self, mock_dataset):
        assert isinstance(v3(), Task)

    @patch("gim.task.gim_dataset", return_value=_fake_dataset())
    def test_default_no_epochs(self, mock_dataset):
        assert v3().epochs is None

    @patch("gim.task.gim_dataset", return_value=_fake_dataset())
    def test_epochs_greater_than_one(self, mock_dataset):
        result = v3(epochs=5)
        if isinstance(result.epochs, Epochs):
            assert result.epochs.epochs == 5
        else:
            assert result.epochs == 5

    @patch("gim.task.gim_dataset", return_value=_fake_dataset())
    def test_passes_dataset_path(self, mock_dataset):
        v3(dataset_path="/custom/gim_v3_dataset")
        mock_dataset.assert_called_once_with(
            path="/custom/gim_v3_dataset",
            media_base=None,
            modalities=None,
            require_attachment=False,
            split="public",
        )

    @patch("gim.task.gim_dataset", return_value=_fake_dataset())
    def test_default_dataset_path_is_none(self, mock_dataset):
        v3()
        mock_dataset.assert_called_once_with(
            path=None,
            media_base=None,
            modalities=None,
            require_attachment=False,
            split="public",
        )

    @patch("gim.task.gim_dataset", return_value=_fake_dataset())
    @patch("gim.task.gim_scorer")
    def test_passes_grader_model(self, mock_scorer, mock_dataset):
        from gim.scorers import gim_scorer as real_scorer
        mock_scorer.side_effect = real_scorer
        v3(grader_model="openai/gpt-4o")
        mock_scorer.assert_called_once_with(grader_model="openai/gpt-4o")

    @patch("gim.task.gim_dataset", return_value=_fake_dataset())
    def test_fail_on_error_is_false(self, mock_dataset):
        assert v3().fail_on_error is False

    @patch("gim.task.gim_dataset", return_value=_fake_dataset())
    def test_media_base_forwarded(self, mock_dataset):
        v3(media_base="gs://bucket/v3")
        mock_dataset.assert_called_once_with(
            path=None,
            media_base="gs://bucket/v3",
            modalities=None,
            require_attachment=False,
            split="public",
        )

    @patch("gim.task.gim_dataset", return_value=_fake_dataset())
    def test_empty_media_base_becomes_none(self, mock_dataset):
        v3(media_base="")
        mock_dataset.assert_called_once_with(
            path=None,
            media_base=None,
            modalities=None,
            require_attachment=False,
            split="public",
        )

    @patch("gim.task.gim_dataset", return_value=_fake_dataset())
    def test_split_parameter_forwarded(self, mock_dataset):
        v3(split="private")
        mock_dataset.assert_called_once_with(
            path=None,
            media_base=None,
            modalities=None,
            require_attachment=False,
            split="private",
        )

    @patch("gim.task.gim_dataset", return_value=_fake_dataset())
    def test_split_all(self, mock_dataset):
        v3(split="all")
        _, kwargs = mock_dataset.call_args
        assert kwargs["split"] == "all"


# ---------------------------------------------------------------------------
# Modality variant tasks
# ---------------------------------------------------------------------------


class TestGenerateOrMissing:
    async def test_successful_generation_returned(self):
        """When generate() succeeds the state is passed through."""
        from unittest.mock import AsyncMock, MagicMock
        from inspect_ai.solver import TaskState

        state = MagicMock(spec=TaskState)
        state.output = MagicMock()
        state.output.completion = "some answer"

        mock_gen = AsyncMock(return_value=state)

        with patch("gim.task.generate", return_value=mock_gen):
            solver_fn = _generate_or_missing()
            result = await solver_fn(state, MagicMock())

        assert result.output.completion == "some answer"

    async def test_generation_failure_returns_state_with_empty_completion(self):
        """When generate() raises, state is returned with its default empty output."""
        from unittest.mock import AsyncMock, MagicMock
        from inspect_ai.solver import TaskState

        state = MagicMock(spec=TaskState)
        state.sample_id = "gim_test"
        state.output = MagicMock()
        state.output.completion = ""  # default empty output
        state.metadata = {}

        mock_gen = AsyncMock(side_effect=RuntimeError("API error"))

        with patch("gim.task.generate", return_value=mock_gen):
            solver_fn = _generate_or_missing()
            result = await solver_fn(state, MagicMock())

        # The scorer treats the unchanged empty completion as missing.
        assert result is state
        assert state.metadata["solved"] is False
        assert state.metadata["generation_error_type"] == "other"

    async def test_successful_generation_sets_sampled_true(self):
        """When generate() succeeds, state.metadata['sampled'] is set to True."""
        from unittest.mock import AsyncMock, MagicMock
        from inspect_ai.solver import TaskState

        state = MagicMock(spec=TaskState)
        state.output = MagicMock()
        state.output.completion = "some answer"
        state.metadata = {}

        mock_gen = AsyncMock(return_value=state)

        with patch("gim.task.generate", return_value=mock_gen):
            solver_fn = _generate_or_missing()
            result = await solver_fn(state, MagicMock())

        assert result.metadata["solved"] is True


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_timeout(self):
        assert _classify_error(RuntimeError("Request timed out")) == "timeout"

    def test_timeout_deadline(self):
        assert _classify_error(RuntimeError("deadline exceeded")) == "timeout"

    def test_rate_limit(self):
        assert _classify_error(RuntimeError("rate limit exceeded")) == "rate_limit"

    def test_rate_limit_429(self):
        assert _classify_error(RuntimeError("HTTP 429")) == "rate_limit"

    def test_file_not_found(self):
        assert _classify_error(FileNotFoundError("FileNotFoundError: /path")) == "file_not_found"

    def test_server_error_502(self):
        assert _classify_error(RuntimeError("502 Bad Gateway")) == "server_error"

    def test_server_error_503(self):
        assert _classify_error(RuntimeError("503 service unavailable")) == "server_error"

    def test_context_length(self):
        assert _classify_error(RuntimeError("context length exceeded")) == "context_length"

    def test_max_tokens(self):
        assert _classify_error(RuntimeError("max_tokens reached")) == "context_length"

    def test_unknown_error(self):
        assert _classify_error(RuntimeError("something weird")) == "other"
