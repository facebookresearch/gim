# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for gim.system_registry — gs-direct:// passthrough patches."""

from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai._util.content import ContentDocument, ContentImage, ContentText

import gim.system_registry as sr


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------


class TestIsGsDirect:
    def test_gs_direct_uri(self):
        assert sr._is_gs_direct("gs-direct://bucket/path/file.jpg") is True

    def test_plain_gs_uri_is_not_direct(self):
        assert sr._is_gs_direct("gs://bucket/path/file.jpg") is False

    def test_https_uri(self):
        assert sr._is_gs_direct("https://example.com/file.jpg") is False

    def test_local_path(self):
        assert sr._is_gs_direct("/data/file.jpg") is False

    def test_empty_string(self):
        assert sr._is_gs_direct("") is False


class TestToGsUri:
    def test_basic_conversion(self):
        assert sr._to_gs_uri("gs-direct://bucket/path/file.jpg") == "gs://bucket/path/file.jpg"

    def test_preserves_nested_path(self):
        assert (
            sr._to_gs_uri("gs-direct://b/p/q/r/file.pdf") == "gs://b/p/q/r/file.pdf"
        )

    def test_no_path_after_bucket(self):
        assert sr._to_gs_uri("gs-direct://bucket") == "gs://bucket"


# ---------------------------------------------------------------------------
# Stage 1 patch — base64 conversion bypass for gs-direct content
# ---------------------------------------------------------------------------


class TestStage1Passthrough:
    async def test_gs_direct_image_returned_unchanged(self):
        original = AsyncMock()
        with patch.object(sr, "_original_chat_content", original):
            content = ContentImage(image="gs-direct://bucket/photo.jpg")
            result = await sr._gcs_passthrough_chat_content(content)
        assert result is content
        original.assert_not_called()

    async def test_gs_direct_document_returned_unchanged(self):
        original = AsyncMock()
        with patch.object(sr, "_original_chat_content", original):
            content = ContentDocument(document="gs-direct://bucket/doc.pdf")
            result = await sr._gcs_passthrough_chat_content(content)
        assert result is content
        original.assert_not_called()

    async def test_local_image_delegates_to_original(self):
        sentinel = object()
        original = AsyncMock(return_value=sentinel)
        with patch.object(sr, "_original_chat_content", original):
            content = ContentImage(image="/data/photo.jpg")
            result = await sr._gcs_passthrough_chat_content(content)
        assert result is sentinel
        original.assert_awaited_once_with(content)

    async def test_https_image_delegates_to_original(self):
        sentinel = object()
        original = AsyncMock(return_value=sentinel)
        with patch.object(sr, "_original_chat_content", original):
            content = ContentImage(image="https://example.com/photo.jpg")
            result = await sr._gcs_passthrough_chat_content(content)
        assert result is sentinel
        original.assert_awaited_once_with(content)

    async def test_plain_gs_image_delegates_to_original(self):
        """Plain gs:// (not gs-direct://) goes through the normal path."""
        sentinel = object()
        original = AsyncMock(return_value=sentinel)
        with patch.object(sr, "_original_chat_content", original):
            content = ContentImage(image="gs://bucket/photo.jpg")
            result = await sr._gcs_passthrough_chat_content(content)
        assert result is sentinel
        original.assert_awaited_once_with(content)

    async def test_text_content_delegates_to_original(self):
        sentinel = object()
        original = AsyncMock(return_value=sentinel)
        with patch.object(sr, "_original_chat_content", original):
            content = ContentText(text="hello")
            result = await sr._gcs_passthrough_chat_content(content)
        assert result is sentinel
        original.assert_awaited_once_with(content)


# ---------------------------------------------------------------------------
# Stage 2 patch — Google provider Part conversion
# ---------------------------------------------------------------------------


class TestStage2GoogleProvider:
    async def test_gs_direct_image_creates_part_from_uri(self):
        original = AsyncMock()
        with patch.object(sr, "_original_chat_content_to_part", original):
            content = ContentImage(image="gs-direct://bucket/photo.jpg")
            client = object()
            result = await sr._gcs_passthrough_chat_content_to_part(client, content)
        original.assert_not_called()
        # Part.from_uri returns a google.genai.types.Part — check the URI was rewritten
        assert hasattr(result, "file_data")
        assert result.file_data.file_uri == "gs://bucket/photo.jpg"

    async def test_gs_direct_image_infers_mime_from_extension(self):
        original = AsyncMock()
        with patch.object(sr, "_original_chat_content_to_part", original):
            content = ContentImage(image="gs-direct://bucket/photo.png")
            result = await sr._gcs_passthrough_chat_content_to_part(object(), content)
        assert result.file_data.mime_type == "image/png"

    async def test_gs_direct_image_unknown_ext_falls_back_to_jpeg(self):
        original = AsyncMock()
        with patch.object(sr, "_original_chat_content_to_part", original):
            content = ContentImage(image="gs-direct://bucket/file.weirdext")
            result = await sr._gcs_passthrough_chat_content_to_part(object(), content)
        assert result.file_data.mime_type == "image/jpeg"

    async def test_gs_direct_document_uses_part_from_uri(self):
        original = AsyncMock()
        with patch.object(sr, "_original_chat_content_to_part", original):
            content = ContentDocument(
                document="gs-direct://bucket/doc.pdf",
                mime_type="application/pdf",
            )
            result = await sr._gcs_passthrough_chat_content_to_part(object(), content)
        original.assert_not_called()
        assert result.file_data.file_uri == "gs://bucket/doc.pdf"
        assert result.file_data.mime_type == "application/pdf"

    async def test_gs_direct_document_default_mime(self):
        """Document without explicit mime_type falls back to octet-stream."""
        original = AsyncMock()
        content = ContentDocument(document="gs-direct://bucket/doc.pdf")
        # Force mime_type to be falsy regardless of inspect_ai's default behavior
        with patch.object(content, "mime_type", None), patch.object(
            sr, "_original_chat_content_to_part", original
        ):
            result = await sr._gcs_passthrough_chat_content_to_part(object(), content)
        assert result.file_data.mime_type == "application/octet-stream"

    async def test_local_image_delegates_to_original(self):
        sentinel = object()
        original = AsyncMock(return_value=sentinel)
        with patch.object(sr, "_original_chat_content_to_part", original):
            content = ContentImage(image="/data/photo.jpg")
            client = object()
            result = await sr._gcs_passthrough_chat_content_to_part(client, content)
        assert result is sentinel
        original.assert_awaited_once_with(client, content)

    async def test_plain_gs_image_delegates_to_original(self):
        sentinel = object()
        original = AsyncMock(return_value=sentinel)
        with patch.object(sr, "_original_chat_content_to_part", original):
            content = ContentImage(image="gs://bucket/photo.jpg")
            client = object()
            result = await sr._gcs_passthrough_chat_content_to_part(client, content)
        assert result is sentinel
        original.assert_awaited_once_with(client, content)

    async def test_text_content_delegates_to_original(self):
        sentinel = object()
        original = AsyncMock(return_value=sentinel)
        with patch.object(sr, "_original_chat_content_to_part", original):
            content = ContentText(text="hello")
            client = object()
            result = await sr._gcs_passthrough_chat_content_to_part(client, content)
        assert result is sentinel
        original.assert_awaited_once_with(client, content)


# ---------------------------------------------------------------------------
# Module import wires the patches into inspect_ai
# ---------------------------------------------------------------------------


class TestModulePatchesInstalled:
    def test_stage1_patch_installed_on_images_module(self):
        import inspect_ai._eval.task.images as images_module

        assert (
            images_module.chat_content_with_base64_content
            is sr._gcs_passthrough_chat_content
        )

    def test_stage2_patch_installed_on_google_module(self):
        import inspect_ai.model._providers.google as google_module

        assert (
            google_module.chat_content_to_part
            is sr._gcs_passthrough_chat_content_to_part
        )
