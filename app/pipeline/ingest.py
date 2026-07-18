"""Ingest layer: turn raw uploads into a flat list of candidate documents.

Each plain file = one document. A ZIP is exploded **safely** (no path traversal,
no absolute paths, caps enforced) and each supported entry becomes its own
document. By default 1 document = 1 candidate downstream (the most reliable
separation); multi-CV-in-one-PDF splitting is handled later in segmentation.
"""

from __future__ import annotations

import os
import zipfile
from typing import Any

SUPPORTED_DOC = (".pdf",)
SUPPORTED_OFFICE = (".docx",)  # Word (XML) - clean text layer, no OCR/VLM needed
SUPPORTED_IMG = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")
SUPPORTED = SUPPORTED_DOC + SUPPORTED_OFFICE + SUPPORTED_IMG


def kind_of(name: str) -> str:
    low = name.lower()
    if low.endswith(SUPPORTED_DOC):
        return "pdf"
    if low.endswith(SUPPORTED_OFFICE):
        return "docx"
    if low.endswith(SUPPORTED_IMG):
        return "image"
    return "unsupported"


def _safe_member(name: str) -> bool:
    """Reject path traversal, absolute paths, hidden macOS cruft, directories."""
    if not name or name.endswith("/"):
        return False
    norm = name.replace("\\", "/")
    if norm.startswith("/") or ".." in norm.split("/"):
        return False
    base = os.path.basename(norm)
    if not base or base.startswith(".") or "__MACOSX" in norm:
        return False
    return kind_of(base) != "unsupported"


def expand_uploads(raw: list[tuple[str, str]], dest_dir: str, settings) -> list[dict[str, Any]]:
    """raw = [(original_name, saved_path), ...]. Returns ordered
    [{name, path, kind, container}] with caps from settings enforced."""
    out: list[dict[str, Any]] = []
    extract_root = os.path.join(dest_dir, "_unzipped")

    for original_name, path in raw:
        if original_name.lower().endswith(".zip"):
            out.extend(_explode_zip(original_name, path, extract_root, settings))
        elif kind_of(original_name) != "unsupported":
            out.append(
                {
                    "name": os.path.basename(original_name),
                    "path": path,
                    "kind": kind_of(original_name),
                    "container": None,
                }
            )
        # silently skip other unsupported top-level files; caller reports the count
        if len(out) >= settings.max_files:
            out = out[: settings.max_files]
            break
    return out


def _explode_zip(zip_name: str, zip_path: str, extract_root: str, settings) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    target = os.path.join(extract_root, os.path.basename(zip_name) + "_x")
    os.makedirs(target, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if _safe_member(m)]
            for m in members[: settings.max_zip_entries]:
                try:
                    dest = os.path.join(target, os.path.basename(m))
                    # de-collide duplicate basenames from different zip folders
                    if os.path.exists(dest):
                        stem, ext = os.path.splitext(dest)
                        dest = f"{stem}_{len(found)}{ext}"
                    with zf.open(m) as src, open(dest, "wb") as fh:
                        fh.write(src.read())
                    found.append(
                        {
                            "name": os.path.basename(m),
                            "path": dest,
                            "kind": kind_of(m),
                            "container": os.path.basename(zip_name),
                        }
                    )
                except Exception:
                    continue
    except zipfile.BadZipFile:
        pass
    return found
