from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LexicaRuntime:
    """Runtime scorer for Addict lexica diagnostics during harness runs."""

    dictionary: np.ndarray
    head_w: np.ndarray
    head_b: float
    top_k: int = 4

    @staticmethod
    def from_npz(path: str | Path) -> "LexicaRuntime":
        data = np.load(path)
        return LexicaRuntime(
            dictionary=data["dictionary"].astype(np.float32),
            head_w=data["head_w"].astype(np.float32).reshape(-1),
            head_b=float(data["head_b"].reshape(-1)[0]),
            top_k=int(data["top_k"].reshape(-1)[0]),
        )

    def score(self, feat: np.ndarray) -> dict:
        x = np.asarray(feat, dtype=np.float32).reshape(-1)
        z = np.maximum(x @ self.dictionary.T, 0.0)
        y = float(z @ self.head_w + self.head_b)
        atom_idx = np.argsort(z)[-self.top_k :][::-1]
        return {
            "lexica_score": y,
            "lexica_top_atoms": atom_idx.astype(int).tolist(),
            "lexica_top_values": z[atom_idx].astype(float).tolist(),
        }

