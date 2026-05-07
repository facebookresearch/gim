# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""GIM dataset loader.

Loads GIM prompts from a HuggingFace dataset saved to disk, as produced by
build_gim_eval.py. The on-disk schema matches the TBR implementation:

    prompt_id, labels, prompt, attachments,
    answer_gtfa, rubrics, solution_reasoning, citations

Attachment paths in the dataset are relative to the dataset root (e.g.
"media/gim_xxx/file.jpg"). Pass ``media_base`` to resolve them to absolute
local paths or remote URIs:

    - None (default): resolved to the dataset directory itself, so the media/
      subdirectory produced by build_gim_eval.py is found automatically.
    - "/abs/path": Path(media_base) / relative_path
    - "gs-direct://bucket/prefix": passes gs://bucket/prefix/... URIs directly
      to the Gemini API as fileData parts (Part.from_uri), bypassing local I/O.
      Requires the GCS passthrough patch in system_registry.py to be active.
    - "https://host/prefix": f"https://host/prefix/{relative_path}"

Modality filtering mirrors the TBR implementation:
    - modalities=None          no filtering (all samples pass)
    - modalities=[]            text-only (skip any sample with attachments)
    - modalities=["image"]     text + image (skip PDFs)
    - modalities=["document"]  text + PDF only
    - modalities=["image", "document"]  all attachment types
    require_attachment=True    additionally exclude text-only samples
"""

import logging
from collections import Counter
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai._util.content import ContentDocument, ContentImage, ContentText

from .modality import modality_tag

logger = logging.getLogger(__name__)

# Default dataset directory (relative to project root).
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data"

# Maps file extension → modality label and Inspect content type.
_EXT_TO_MODALITY: dict[str, str] = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".pdf": "document",
}


# ---------------------------------------------------------------------------
# Modality helpers
# ---------------------------------------------------------------------------


def get_sample_modalities(attachments: list[str]) -> set[str]:
    """Return the set of modality labels present in an attachment list."""
    modalities: set[str] = set()
    for path in attachments:
        ext = Path(path).suffix.lower()
        modality = _EXT_TO_MODALITY.get(ext)
        if modality:
            modalities.add(modality)
    return modalities


def should_include(
    sample_modalities: set[str],
    modalities: list[str] | None,
    require_attachment: bool,
) -> bool:
    """Return True if a sample passes the modality filter.

    Args:
        sample_modalities: Modality labels present in the sample's attachments.
        modalities: Allowed modality labels. None means no filtering.
            [] means text-only (sample_modalities must be empty).
        require_attachment: If True, samples with no attachments are excluded.
    """
    if modalities is not None:
        allowed = set(modalities)
        if not sample_modalities.issubset(allowed):
            return False
    if require_attachment and not sample_modalities:
        return False
    return True


# ---------------------------------------------------------------------------
# Attachment path resolution
# ---------------------------------------------------------------------------


def resolve_attachment(relative_path: str, media_base: str) -> str:
    """Resolve a relative attachment path to an absolute path or URI.

    Args:
        relative_path: Path as stored in the HF dataset (e.g.
            "media/gim_xxx/file.jpg").
        media_base: Base to prepend. Remote prefixes (gs://, https://, http://)
            are joined with "/"; everything else is treated as a local directory.
    """
    if media_base.startswith(("gs-direct://", "gs://", "https://", "http://")):
        return f"{media_base.rstrip('/')}/{relative_path}"
    return str(Path(media_base) / relative_path)


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------


def record_to_sample(
    record: dict,
    media_base: str,
    modalities: list[str] | None,
    require_attachment: bool,
) -> Sample | None:
    """Convert a HuggingFace dataset row into an Inspect Sample.

    Returns None when the sample is filtered out by the modality settings.

    Each sample carries structured metadata so scorers can access the golden
    answer, rubrics, labels, and attachments without re-parsing.
    """
    # Parse comma-separated labels into a list
    raw_labels = record.get("labels") or ""
    labels = [lbl.strip() for lbl in str(raw_labels).split(",") if lbl.strip()]

    # Rubrics arrive as list[str] from the HuggingFace dataset
    rubrics_raw = record.get("rubrics")
    rubrics: list[str] = []
    if isinstance(rubrics_raw, list):
        rubrics = [str(r) for r in rubrics_raw if r]

    # Attachments: list of relative media paths (e.g. "media/gim_xxx/file.pdf")
    attachments_raw = record.get("attachments")
    if isinstance(attachments_raw, list):
        attachments = [str(a) for a in attachments_raw if a]
    else:
        attachments = []

    # Modality filtering
    sample_modalities = get_sample_modalities(attachments)
    if not should_include(sample_modalities, modalities, require_attachment):
        return None

    # Golden answer — treat empty / "nan" / None as absent
    raw_answer = record.get("answer_gtfa")
    answer = str(raw_answer).strip() if raw_answer is not None else ""
    if answer.lower() in ("", "nan", "none"):
        answer = ""

    # Build the user message.  When attachments are present, construct a
    # multimodal ChatMessageUser; otherwise use a plain string for efficiency.
    prompt = str(record["prompt"])
    if attachments:
        contents: list[ContentImage | ContentDocument | ContentText] = []
        for rel_path in attachments:
            ext = Path(rel_path).suffix.lower()
            resolved = resolve_attachment(rel_path, media_base)
            if ext in (".png", ".jpg", ".jpeg", ".webp"):
                contents.append(ContentImage(image=resolved))
            elif ext == ".pdf":
                contents.append(ContentDocument(document=resolved))
            else:
                logger.warning("Skipping attachment with unknown extension: %s", rel_path)
        contents.append(ContentText(text=prompt))
        sample_input = [ChatMessageUser(content=contents)]
    else:
        sample_input = prompt

    return Sample(
        input=sample_input,
        target=answer,
        id=str(record["prompt_id"]),
        metadata={
            "labels": labels,
            "rubrics": rubrics,
            "answer": answer,
            "attachments": attachments,
            "modality": modality_tag(sample_modalities),
            "solution_reasoning": record.get("solution_reasoning") or "",
            "citations": record.get("citations") or "",
        },
    )


def _deduplicate_ids(samples: list[Sample]) -> list[Sample]:
    """Append suffixes to duplicate sample IDs to ensure uniqueness."""
    counts: Counter[str | int | None] = Counter()
    result: list[Sample] = []
    for s in samples:
        counts[s.id] += 1
        if counts[s.id] > 1:
            s = s.model_copy(update={"id": f"{s.id}_{counts[s.id]}"})
        result.append(s)
    return result


def gim_dataset(
    path: str | Path | None = None,
    media_base: str | None = None,
    modalities: list[str] | None = None,
    require_attachment: bool = False,
    split: str = "public",
) -> Dataset:
    """Load the GIM evaluation dataset from a HuggingFace dataset on disk.

    The dataset directory is produced by build_gim_eval.py with
    ``--output-format huggingface``.

    Args:
        path: Path to the HuggingFace dataset directory. Defaults to
            data/gim_v3_dataset relative to the project root.
        media_base: Base path or URI for resolving attachment paths. Defaults
            to the dataset directory itself (so the media/ subdirectory created
            by build_gim_eval.py is found automatically). Use "gs://bucket/..."
            to point at GCS (Google models only).
        modalities: Allowed attachment modality labels. None means no filtering.
            [] means text-only. ["image"] includes only image attachments.
            ["image", "document"] includes all supported types.
        require_attachment: When True, samples with no attachments are excluded
            even if their empty modality set is a subset of ``modalities``.
        split: Which split to load from a DatasetDict. Use "public" for the
            public subset (default), "private" for the private subset, or
            "all" to concatenate all splits. When the on-disk dataset is a
            plain Dataset (not a DatasetDict), all rows are loaded regardless
            of this parameter.
    """
    hf_path = Path(path or DEFAULT_DATA_PATH)
    resolved_media_base = media_base or str(hf_path)

    ds = load_dataset(str(hf_path))

    # DatasetDict: select or concatenate splits.
    # Plain Dataset: load all rows (split parameter is ignored).
    if hasattr(ds, "keys"):
        if split == "all":
            ds = concatenate_datasets(list(ds.values()))
        else:
            ds = ds[split]
        logger.info("Loaded %d rows from GIM DatasetDict (split=%s)", len(ds), split)
    else:
        logger.info("Loaded %d rows from GIM Dataset (plain, no splits)", len(ds))

    samples: list[Sample] = []
    for row in ds:
        sample = record_to_sample(
            dict(row),
            media_base=resolved_media_base,
            modalities=modalities,
            require_attachment=require_attachment,
        )
        if sample is not None:
            samples.append(sample)

    return MemoryDataset(
        samples=_deduplicate_ids(samples),
        name="gim",
        location=str(hf_path),
    )
