from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Hashable, Optional, Sequence

import numpy as np

from ..utils import utc_ts


def grouped_split(
    groups: Sequence[Hashable],
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
    names: Sequence[str] = ("train", "val", "test"),
    seed: int = 7,
) -> Dict[str, np.ndarray]:
    """Split row indices so that every group lands in exactly one split."""
    if len(fractions) != len(names) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must match names and sum to 1")
    uniq = sorted(set(groups), key=repr)
    order = np.random.default_rng(seed).permutation(len(uniq))
    cuts = np.floor(np.cumsum(fractions) * len(uniq) + 1e-9).astype(int)
    cuts[-1] = len(uniq)
    assign: Dict[Hashable, str] = {}
    lo = 0
    for name, hi in zip(names, cuts):
        for j in order[lo:hi]:
            assign[uniq[int(j)]] = name
        lo = int(hi)
    out = {name: [] for name in names}
    for i, g in enumerate(groups):
        out[assign[g]].append(i)
    return {name: np.asarray(idx, dtype=np.int64) for name, idx in out.items()}


def assert_group_disjoint(groups: Sequence[Hashable], split: Dict[str, np.ndarray]) -> None:
    seen: Dict[Hashable, str] = {}
    for name, idx in split.items():
        for i in idx:
            g = groups[int(i)]
            if seen.setdefault(g, name) != name:
                raise AssertionError(f"group {g!r} appears in both {seen[g]} and {name}")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    path: str | Path,
    files: Dict[str, str | Path],
    sealed: Sequence[str] = (),
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record the content hash of each split file. Sealed files are evaluated once per analysis."""
    path = Path(path)
    entries: Dict[str, Any] = {}
    for name, file_path in files.items():
        fp = Path(file_path)
        with fp.open("r", encoding="utf-8") as f:
            n_records = sum(1 for row in f if row.strip())
        entries[name] = {
            "path": fp.name,
            "sha256": sha256_file(fp),
            "records": n_records,
            "sealed": name in sealed,
            "evaluations": [],
        }
    manifest = {"created": utc_ts(), "meta": meta or {}, "files": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def open_split(manifest_path: str | Path, name: str, analysis_id: Optional[str] = None) -> Path:
    """Return the path of a split file after verifying its hash.

    A sealed split needs an `analysis_id`, and each id may open it once; the
    evaluation is logged in the manifest.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"][name]
    fp = manifest_path.parent / entry["path"]
    if sha256_file(fp) != entry["sha256"]:
        raise RuntimeError(f"{fp} does not match the hash recorded in the manifest")
    if entry["sealed"]:
        if not analysis_id:
            raise PermissionError(f"split {name!r} is sealed; pass the pre-registered analysis id")
        if any(e["analysis_id"] == analysis_id for e in entry["evaluations"]):
            raise PermissionError(f"analysis {analysis_id!r} has already been evaluated on {name!r}")
        entry["evaluations"].append({"analysis_id": analysis_id, "ts": utc_ts()})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return fp
