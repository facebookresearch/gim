# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Custom model API registrations for GIM evaluation.

Each ``@modelapi`` entry registers a named system configuration that can be
used directly with ``--model <name>/<model>``.  This lets us pin
provider-specific parameters (thinking level, timeouts, etc.) without
relying on Inspect's unified abstraction layer.

Usage examples:
    uv run inspect eval gim/v3 --model google/gemini-3-pro --reasoning-effort high
    uv run inspect eval gim/v3 --model google/gemini-3-flash-preview-genai --reasoning-effort low

gs-direct:// passthrough
------------------------
Use ``media_base=gs-direct://bucket/prefix`` to pass GCS media to the
Gemini API as fileData parts (``Part.from_uri``) rather than reading the
bytes locally.  Inspect AI's pipeline normally converts all media content to
base64 before any provider code runs; the two patches below intercept that
pipeline for ``gs-direct://`` URIs only, leaving all other content
(local files, HTTPS, data URIs) completely unchanged.

How it works:
  - dataset.py produces ``ContentImage(image="gs-direct://bucket/.../file.jpg")``
  - Stage 1 (images.py): base64 conversion is skipped for gs-direct:// URIs
  - Stage 2 (google.py): gs-direct:// is rewritten to gs:// and passed as
    Part.from_uri so the Gemini API reads the file from GCS directly

Both patched functions are module-level and called by name within their
respective modules, so replacing the module attribute is sufficient for all
internal callers to pick up the patch.
"""

import mimetypes

import inspect_ai._eval.task.images as _images_module
import inspect_ai.model._providers.google as _google_module
from google.genai.types import Part
from inspect_ai._util.content import ContentDocument, ContentImage

_GS_DIRECT = "gs-direct://"
_GS = "gs://"


def _is_gs_direct(uri: str) -> bool:
    return uri.startswith(_GS_DIRECT)


def _to_gs_uri(uri: str) -> str:
    """Convert a gs-direct:// URI to a gs:// URI."""
    return _GS + uri[len(_GS_DIRECT) :]


# ---------------------------------------------------------------------------
# Stage 1: skip base64 conversion for gs-direct:// content
# ---------------------------------------------------------------------------

_original_chat_content = _images_module.chat_content_with_base64_content


async def _gcs_passthrough_chat_content(content):
    if isinstance(content, ContentImage) and _is_gs_direct(content.image):
        return content
    if isinstance(content, ContentDocument) and _is_gs_direct(content.document):
        return content
    return await _original_chat_content(content)


_images_module.chat_content_with_base64_content = _gcs_passthrough_chat_content


# ---------------------------------------------------------------------------
# Stage 2: convert gs-direct:// to Part.from_uri(gs://) in the Google provider
# ---------------------------------------------------------------------------

_original_chat_content_to_part = _google_module.chat_content_to_part


async def _gcs_passthrough_chat_content_to_part(client, content):
    if isinstance(content, ContentImage) and _is_gs_direct(content.image):
        mime_type, _ = mimetypes.guess_type(content.image, strict=False)
        return Part.from_uri(
            file_uri=_to_gs_uri(content.image), mime_type=mime_type or "image/jpeg"
        )
    if isinstance(content, ContentDocument) and _is_gs_direct(content.document):
        mime_type = content.mime_type or "application/octet-stream"
        return Part.from_uri(file_uri=_to_gs_uri(content.document), mime_type=mime_type)
    return await _original_chat_content_to_part(client, content)


_google_module.chat_content_to_part = _gcs_passthrough_chat_content_to_part
