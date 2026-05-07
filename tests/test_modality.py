# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for gim.modality — modality tagging and superset helpers."""

from gim.modality import modality_supersets, modality_tag


class TestModalityTag:
    def test_text(self):
        assert modality_tag(set()) == "text"

    def test_image(self):
        assert modality_tag({"image"}) == "image"

    def test_document(self):
        assert modality_tag({"document"}) == "document"

    def test_mixed(self):
        assert modality_tag({"image", "document"}) == "mixed"


class TestModalitySupersets:
    def test_text(self):
        supersets = modality_supersets("text")
        assert "text+image" in supersets
        assert "text+document" in supersets
        assert "attachment" not in supersets

    def test_image(self):
        supersets = modality_supersets("image")
        assert "text+image" in supersets
        assert "text+document" not in supersets
        assert "attachment" in supersets

    def test_document(self):
        supersets = modality_supersets("document")
        assert "text+image" not in supersets
        assert "text+document" in supersets
        assert "attachment" in supersets

    def test_mixed(self):
        supersets = modality_supersets("mixed")
        assert "text+image" not in supersets
        assert "text+document" not in supersets
        assert "attachment" in supersets
