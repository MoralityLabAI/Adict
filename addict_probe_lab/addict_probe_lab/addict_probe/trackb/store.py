from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


class ActivationStore:
    """Float16 activation shards with a JSONL index.

    A key names one stream of activations, e.g. ("base", 12, "tstar"): a variant,
    a residual site and a position set. Each `write` appends one shard for a
    key and one index row per record; `read` returns all records of a key in
    the order they were written. Activations are recomputable from the stored
    prefixes, so the store is a cache.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    @staticmethod
    def _key(variant: str, site: int, position: str) -> str:
        for part in (variant, position):
            if "__" in part or "/" in part or "\\" in part:
                raise ValueError(f"invalid key part: {part!r}")
        return f"{variant}__s{int(site)}__{position}"

    def _rows(self) -> List[Dict[str, Any]]:
        if not self.index_path.exists():
            return []
        with self.index_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def write(self, variant: str, site: int, position: str, records: List[Dict[str, Any]], acts: np.ndarray) -> Path:
        if len(records) != acts.shape[0]:
            raise ValueError("one record per activation row is required")
        key = self._key(variant, site, position)
        part = sum(1 for _ in self.root.glob(f"{key}__*.npy"))
        shard = self.root / f"{key}__{part:05d}.npy"
        np.save(shard, acts.astype(np.float16))
        with self.index_path.open("a", encoding="utf-8") as f:
            for row, rec in enumerate(records):
                f.write(json.dumps({**rec, "key": key, "shard": shard.name, "row": row}) + "\n")
        return shard

    def read(self, variant: str, site: int, position: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        key = self._key(variant, site, position)
        rows = [r for r in self._rows() if r["key"] == key]
        if not rows:
            raise KeyError(f"nothing stored under {key}")
        shards: Dict[str, np.ndarray] = {}
        out = []
        for r in rows:
            if r["shard"] not in shards:
                shards[r["shard"]] = np.load(self.root / r["shard"])
            out.append(shards[r["shard"]][r["row"]])
        return rows, np.stack(out, axis=0)

    def keys(self) -> List[str]:
        return sorted({r["key"] for r in self._rows()})

    def stored_prefix_ids(self, variant: str, site: int, position: str) -> set:
        key = self._key(variant, site, position)
        return {r["prefix_id"] for r in self._rows() if r["key"] == key}
