from __future__ import annotations

from pathlib import Path
import os
import re
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def split_photo_raw(raw):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s = str(raw).replace("\\", "/")
    parts = re.split(r"[;|\n\r]+", s)
    return [
        p.strip()
        for p in parts
        if p.strip() and Path(p.strip()).suffix.lower() in IMAGE_EXTS
    ]


def build_basename_index(root):
    idx = {}
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            idx.setdefault(p.name, []).append(p)
    return idx


def resolve_image(raw_path, dataset_root, source_file=None, basename_index=None):
    if not raw_path:
        return None

    root = Path(dataset_root)
    s = str(raw_path).replace("\\", os.sep).replace("/", os.sep)
    p = Path(s)

    if p.is_absolute() and p.exists():
        return str(p.resolve())

    direct = root / p
    if direct.exists():
        return str(direct.resolve())

    if source_file:
        source = Path(source_file)
        for parent in list(source.parents)[:9]:
            cand = parent / p
            try:
                cand = cand.resolve()
            except Exception:
                pass
            if cand.exists():
                return str(cand)

    name = p.name
    candidates = (basename_index or {}).get(name, [])
    if not candidates:
        return None

    if source_file and len(candidates) > 1:
        tokens = set(Path(source_file).parts)
        candidates = sorted(
            candidates,
            key=lambda q: -len(tokens.intersection(set(q.parts))),
        )

    return str(Path(candidates[0]).resolve())


def resolved_image_paths_for_manifest(manifest, dataset_root):
    basename_index = build_basename_index(dataset_root)
    resolved = []
    declared = 0

    for row in manifest.itertuples():
        raws = split_photo_raw(getattr(row, "photo_raw", None))
        declared += len(raws)
        for raw in raws:
            p = resolve_image(
                raw,
                dataset_root,
                getattr(row, "manual_csv", None),
                basename_index,
            )
            if p:
                resolved.append(p)

    return list(dict.fromkeys(resolved)), declared, basename_index
