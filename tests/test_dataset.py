# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for gim.dataset — HuggingFace record parsing, filtering, and loading."""

from pathlib import Path
from unittest.mock import patch

import pytest
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser

from gim.dataset import (
    _deduplicate_ids,
    get_sample_modalities,
    gim_dataset,
    record_to_sample,
    resolve_attachment,
    should_include,
)


# ---------------------------------------------------------------------------
# Minimal record factory
# ---------------------------------------------------------------------------


def _record(**overrides) -> dict:
    """Build a minimal HF record with optional field overrides."""
    base = {
        "prompt_id": "gim_test0001",
        "prompt": "What is the capital of France?",
        "answer_gtfa": "Paris",
        "labels": "reasoning, knowledge",
        "rubrics": None,
        "attachments": [],
        "solution_reasoning": "Direct factual recall",
        "citations": None,
    }
    base.update(overrides)
    return base


def _make(media_base="/base", modalities=None, require_attachment=False, **overrides):
    """record_to_sample with convenient defaults."""
    return record_to_sample(
        _record(**overrides),
        media_base=media_base,
        modalities=modalities,
        require_attachment=require_attachment,
    )


# ---------------------------------------------------------------------------
# get_sample_modalities
# ---------------------------------------------------------------------------


class TestGetSampleModalities:
    def test_empty_attachments(self):
        assert get_sample_modalities([]) == set()

    def test_image_extensions(self):
        for ext in ("file.png", "file.jpg", "file.jpeg", "file.webp"):
            assert get_sample_modalities([ext]) == {"image"}

    def test_pdf_extension(self):
        assert get_sample_modalities(["file.pdf"]) == {"document"}

    def test_mixed(self):
        assert get_sample_modalities(["a.png", "b.pdf"]) == {"image", "document"}

    def test_unknown_extension_ignored(self):
        assert get_sample_modalities(["file.xyz"]) == set()

    def test_case_insensitive(self):
        assert get_sample_modalities(["FILE.PNG"]) == {"image"}


# ---------------------------------------------------------------------------
# should_include
# ---------------------------------------------------------------------------


class TestShouldInclude:
    def test_none_modalities_includes_all(self):
        assert should_include({"image"}, None, False) is True
        assert should_include(set(), None, False) is True

    def test_empty_modalities_text_only(self):
        assert should_include(set(), [], False) is True
        assert should_include({"image"}, [], False) is False

    def test_image_modality(self):
        assert should_include({"image"}, ["image"], False) is True
        assert should_include({"document"}, ["image"], False) is False
        assert should_include(set(), ["image"], False) is True  # text passes

    def test_mixed_modalities(self):
        assert should_include({"image", "document"}, ["image", "document"], False) is True
        assert should_include({"image"}, ["image", "document"], False) is True
        assert should_include(set(), ["image", "document"], False) is True

    def test_require_attachment_excludes_text_only(self):
        assert should_include(set(), None, True) is False
        assert should_include({"image"}, None, True) is True

    def test_require_attachment_with_modality_filter(self):
        assert should_include(set(), ["image"], True) is False
        assert should_include({"image"}, ["image"], True) is True


# ---------------------------------------------------------------------------
# resolve_attachment
# ---------------------------------------------------------------------------


class TestResolveAttachment:
    def test_local_path(self):
        result = resolve_attachment("media/gim_xxx/file.jpg", "/data/gim_v3")
        assert result == str(Path("/data/gim_v3/media/gim_xxx/file.jpg"))

    def test_gcs_uri(self):
        result = resolve_attachment("media/gim_xxx/file.jpg", "gs://bucket/prefix")
        assert result == "gs://bucket/prefix/media/gim_xxx/file.jpg"

    def test_gcs_uri_trailing_slash_stripped(self):
        result = resolve_attachment("media/gim_xxx/file.jpg", "gs://bucket/prefix/")
        assert result == "gs://bucket/prefix/media/gim_xxx/file.jpg"

    def test_gs_direct_uri(self):
        result = resolve_attachment("media/gim_xxx/file.jpg", "gs-direct://bucket/prefix")
        assert result == "gs-direct://bucket/prefix/media/gim_xxx/file.jpg"

    def test_https_uri(self):
        result = resolve_attachment("media/gim_xxx/file.jpg", "https://host/base")
        assert result == "https://host/base/media/gim_xxx/file.jpg"


# ---------------------------------------------------------------------------
# record_to_sample — basic fields
# ---------------------------------------------------------------------------


class TestRecordToSampleBasic:
    def test_basic_conversion(self):
        sample = _make()
        assert isinstance(sample, Sample)
        assert sample.id == "gim_test0001"
        assert sample.target == "Paris"

    def test_text_only_input_is_string(self):
        sample = _make()
        assert isinstance(sample.input, str)
        assert sample.input == "What is the capital of France?"

    def test_labels_parsed_as_list(self):
        sample = _make()
        assert sample.metadata["labels"] == ["reasoning", "knowledge"]

    def test_empty_labels(self):
        sample = _make(labels="")
        assert sample.metadata["labels"] == []

    def test_null_rubrics_gives_empty_list(self):
        sample = _make(rubrics=None)
        assert sample.metadata["rubrics"] == []

    def test_rubrics_parsed_from_list(self):
        sample = _make(rubrics=["First criterion", "Second criterion"])
        assert sample.metadata["rubrics"] == ["First criterion", "Second criterion"]

    def test_empty_rubric_entries_filtered(self):
        sample = _make(rubrics=["valid", "", "also valid"])
        assert sample.metadata["rubrics"] == ["valid", "also valid"]

    def test_none_answer_is_empty(self):
        sample = _make(answer_gtfa=None)
        assert sample.target == ""

    def test_whitespace_answer_is_empty(self):
        sample = _make(answer_gtfa="   ")
        assert sample.target == ""

    def test_attachments_in_metadata(self):
        sample = _make(attachments=["media/gim_xxx/file.jpg"])
        assert sample.metadata["attachments"] == ["media/gim_xxx/file.jpg"]

    def test_citations_none_becomes_empty_string(self):
        sample = _make()
        assert sample.metadata["citations"] == ""

    def test_missing_optional_fields_default_gracefully(self):
        sample = record_to_sample(
            {"prompt_id": "gim_999", "prompt": "Q?", "answer_gtfa": "42", "labels": ""},
            media_base="/base",
            modalities=None,
            require_attachment=False,
        )
        assert sample.metadata["rubrics"] == []
        assert sample.metadata["attachments"] == []


# ---------------------------------------------------------------------------
# record_to_sample — multimodal input construction
# ---------------------------------------------------------------------------


class TestRecordToSampleMultimodal:
    def test_image_attachment_builds_multimodal_input(self):
        from inspect_ai._util.content import ContentImage, ContentText

        sample = _make(
            attachments=["media/gim_xxx/photo.jpg"],
            media_base="/data",
        )
        assert isinstance(sample.input, list)
        assert len(sample.input) == 1
        msg = sample.input[0]
        assert isinstance(msg, ChatMessageUser)
        assert isinstance(msg.content, list)
        assert any(isinstance(c, ContentImage) for c in msg.content)
        assert any(isinstance(c, ContentText) for c in msg.content)

    def test_pdf_attachment_uses_content_document(self):
        from inspect_ai._util.content import ContentDocument

        sample = _make(
            attachments=["media/gim_xxx/doc.pdf"],
            media_base="/data",
        )
        msg = sample.input[0]
        assert any(isinstance(c, ContentDocument) for c in msg.content)

    def test_attachments_before_text(self):
        from inspect_ai._util.content import ContentText

        sample = _make(
            attachments=["media/gim_xxx/photo.jpg"],
            media_base="/data",
        )
        content = sample.input[0].content
        assert isinstance(content[-1], ContentText)

    def test_multiple_attachments_all_included(self):
        sample = _make(
            attachments=["media/gim_xxx/a.jpg", "media/gim_xxx/b.png"],
            media_base="/data",
        )
        content = sample.input[0].content
        # 2 images + 1 text
        assert len(content) == 3

    def test_image_path_resolved_against_media_base(self):
        from inspect_ai._util.content import ContentImage

        sample = _make(
            attachments=["media/gim_xxx/photo.jpg"],
            media_base="/data/gim_v3",
        )
        images = [c for c in sample.input[0].content if isinstance(c, ContentImage)]
        assert images[0].image == str(Path("/data/gim_v3/media/gim_xxx/photo.jpg"))

    def test_gs_direct_media_base_produces_uri(self):
        from inspect_ai._util.content import ContentImage

        sample = _make(
            attachments=["media/gim_xxx/photo.jpg"],
            media_base="gs-direct://bucket/v3",
        )
        images = [c for c in sample.input[0].content if isinstance(c, ContentImage)]
        assert images[0].image == "gs-direct://bucket/v3/media/gim_xxx/photo.jpg"

    def test_unknown_extension_skipped(self):
        sample = _make(
            attachments=["media/gim_xxx/file.xyz"],
            media_base="/data",
        )
        # Unknown extension: content list should just be the text
        content = sample.input[0].content
        assert len(content) == 1  # only ContentText

    def test_unknown_extension_only_returns_multimodal_with_text(self):
        """Even if all attachments are skipped, input is still multimodal list."""
        from inspect_ai._util.content import ContentText
        sample = _make(
            attachments=["media/gim_xxx/file.xyz"],
            media_base="/data",
        )
        assert isinstance(sample.input, list)
        assert isinstance(sample.input[0].content[-1], ContentText)


# ---------------------------------------------------------------------------
# record_to_sample — modality filtering
# ---------------------------------------------------------------------------


class TestRecordToSampleFiltering:
    def test_text_only_filter_excludes_image_sample(self):
        result = record_to_sample(
            _record(attachments=["file.jpg"]),
            media_base="/base",
            modalities=[],
            require_attachment=False,
        )
        assert result is None

    def test_text_only_filter_passes_text_sample(self):
        result = record_to_sample(
            _record(attachments=[]),
            media_base="/base",
            modalities=[],
            require_attachment=False,
        )
        assert result is not None

    def test_image_filter_excludes_pdf(self):
        result = record_to_sample(
            _record(attachments=["file.pdf"]),
            media_base="/base",
            modalities=["image"],
            require_attachment=False,
        )
        assert result is None

    def test_image_filter_passes_image(self):
        result = record_to_sample(
            _record(attachments=["file.jpg"]),
            media_base="/base",
            modalities=["image"],
            require_attachment=False,
        )
        assert result is not None

    def test_require_attachment_excludes_text_only(self):
        result = record_to_sample(
            _record(attachments=[]),
            media_base="/base",
            modalities=["image", "document"],
            require_attachment=True,
        )
        assert result is None

    def test_require_attachment_passes_sample_with_attachment(self):
        result = record_to_sample(
            _record(attachments=["file.jpg"]),
            media_base="/base",
            modalities=["image", "document"],
            require_attachment=True,
        )
        assert result is not None

    def test_none_modalities_passes_everything(self):
        result = record_to_sample(
            _record(attachments=["file.pdf"]),
            media_base="/base",
            modalities=None,
            require_attachment=False,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# _deduplicate_ids
# ---------------------------------------------------------------------------


class TestDeduplicateIds:
    def test_no_duplicates_unchanged(self):
        samples = [Sample(input="a", id="1"), Sample(input="b", id="2")]
        assert [s.id for s in _deduplicate_ids(samples)] == ["1", "2"]

    def test_duplicates_get_suffixes(self):
        samples = [Sample(input="a", id="dup"), Sample(input="b", id="dup")]
        result = _deduplicate_ids(samples)
        assert result[0].id == "dup"
        assert result[1].id == "dup_2"

    def test_empty_list(self):
        assert _deduplicate_ids([]) == []


# ---------------------------------------------------------------------------
# gim_dataset
# ---------------------------------------------------------------------------


class TestRecordToSampleModality:
    """Tests for modality tag in sample metadata."""

    def test_text_only_sample_has_text_modality(self):
        sample = _make()
        assert sample.metadata["modality"] == "text"

    def test_image_sample_has_image_modality(self):
        sample = _make(attachments=["media/gim_xxx/photo.jpg"])
        assert sample.metadata["modality"] == "image"

    def test_pdf_sample_has_document_modality(self):
        sample = _make(attachments=["media/gim_xxx/doc.pdf"])
        assert sample.metadata["modality"] == "document"

    def test_mixed_sample_has_mixed_modality(self):
        sample = _make(attachments=["media/gim_xxx/photo.jpg", "media/gim_xxx/doc.pdf"])
        assert sample.metadata["modality"] == "mixed"


# ---------------------------------------------------------------------------
# gim_dataset
# ---------------------------------------------------------------------------


class TestGimDataset:
    @patch("gim.dataset.load_dataset")
    def test_loads_from_path(self, mock_load):
        mock_load.return_value = [_record()]
        ds = gim_dataset("/some/path")
        mock_load.assert_called_once_with("/some/path")
        assert len(ds.samples) == 1

    @patch("gim.dataset.load_dataset")
    def test_default_media_base_is_dataset_dir(self, mock_load):
        """When media_base is None, attachments resolve relative to the dataset dir."""
        from inspect_ai._util.content import ContentImage

        mock_load.return_value = [_record(attachments=["media/gim_xxx/a.jpg"])]
        ds = gim_dataset("/some/dataset")
        msg = ds.samples[0].input[0]
        images = [c for c in msg.content if isinstance(c, ContentImage)]
        assert images[0].image == str(Path("/some/dataset/media/gim_xxx/a.jpg"))

    @patch("gim.dataset.load_dataset")
    def test_text_only_filter_applied(self, mock_load):
        mock_load.return_value = [
            _record(prompt_id="txt", attachments=[]),
            _record(prompt_id="img", attachments=["file.jpg"]),
        ]
        ds = gim_dataset("/path", modalities=[])
        assert len(ds.samples) == 1
        assert ds.samples[0].id == "txt"

    @patch("gim.dataset.load_dataset")
    def test_deduplicates_ids(self, mock_load):
        mock_load.return_value = [_record(prompt_id="dup"), _record(prompt_id="dup")]
        ds = gim_dataset("/path")
        assert ds.samples[0].id == "dup"
        assert ds.samples[1].id == "dup_2"

    @patch("gim.dataset.load_dataset")
    def test_dataset_name_is_gim(self, mock_load):
        mock_load.return_value = []
        ds = gim_dataset("/path")
        assert ds.name == "gim"


# ---------------------------------------------------------------------------
# gim_dataset — DatasetDict support
# ---------------------------------------------------------------------------


class TestGimDatasetDict:
    @patch("gim.dataset.load_dataset")
    def test_plain_dataset_loads_all_rows(self, mock_load):
        """When on-disk data is a plain Dataset (not DatasetDict), all rows are loaded."""
        mock_load.return_value = [_record(prompt_id="a"), _record(prompt_id="b")]
        ds = gim_dataset("/path", split="public")
        assert len(ds.samples) == 2

    @patch("gim.dataset.load_dataset")
    def test_plain_dataset_split_param_ignored(self, mock_load):
        """split parameter has no effect on a plain Dataset."""
        mock_load.return_value = [_record()]
        ds_public = gim_dataset("/path", split="public")
        mock_load.return_value = [_record()]
        ds_private = gim_dataset("/path", split="private")
        assert len(ds_public.samples) == len(ds_private.samples)

    @patch("gim.dataset.load_dataset")
    def test_dataset_dict_selects_split(self, mock_load):
        """When on-disk data is a DatasetDict, the requested split is selected."""
        mock_dict = {
            "public": [_record(prompt_id="pub1")],
            "private": [_record(prompt_id="priv1")],
        }
        mock_load.return_value = mock_dict
        ds = gim_dataset("/path", split="public")
        assert len(ds.samples) == 1
        assert ds.samples[0].id == "pub1"

    @patch("gim.dataset.load_dataset")
    def test_dataset_dict_private_split(self, mock_load):
        mock_dict = {
            "public": [_record(prompt_id="pub1")],
            "private": [_record(prompt_id="priv1")],
        }
        mock_load.return_value = mock_dict
        ds = gim_dataset("/path", split="private")
        assert len(ds.samples) == 1
        assert ds.samples[0].id == "priv1"

    @patch("gim.dataset.load_dataset")
    @patch("gim.dataset.concatenate_datasets")
    def test_dataset_dict_all_concatenates(self, mock_concat, mock_load):
        """split='all' concatenates all splits in a DatasetDict."""
        public_data = [_record(prompt_id="pub1")]
        private_data = [_record(prompt_id="priv1")]
        mock_dict = {"public": public_data, "private": private_data}
        mock_load.return_value = mock_dict
        mock_concat.return_value = public_data + private_data
        ds = gim_dataset("/path", split="all")
        mock_concat.assert_called_once()
        assert len(ds.samples) == 2


