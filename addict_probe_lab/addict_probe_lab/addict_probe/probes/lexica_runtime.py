from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class LexicaRuntime:
    """Runtime scorer for Addict lexica diagnostics during harness runs."""

    dictionary: np.ndarray
    head_w: np.ndarray
    head_b: float
    top_k: int = 4
    guardrail_atoms: list[int] | None = None
    suppress_atoms: list[int] | None = None
    positive_weight: float = 0.05
    negative_weight: float = 0.07

    @staticmethod
    def from_npz(path: str | Path) -> "LexicaRuntime":
        data = np.load(path)
        return LexicaRuntime(
            dictionary=data["dictionary"].astype(np.float32),
            head_w=data["head_w"].astype(np.float32).reshape(-1),
            head_b=float(data["head_b"].reshape(-1)[0]),
            top_k=int(data["top_k"].reshape(-1)[0]),
        )

    def configure_guardrails(
        self,
        report_path: str | Path,
        positive_weight: float = 0.05,
        negative_weight: float = 0.07,
    ) -> "LexicaRuntime":
        report = json.loads(Path(report_path).read_text(encoding="utf-8-sig"))
        self.guardrail_atoms = [int(x) for x in report.get("guardrail_atoms", [])]
        self.suppress_atoms = [int(x) for x in report.get("suppress_atoms", [])]
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)
        return self

    def score(self, feat: np.ndarray) -> dict:
        x = np.asarray(feat, dtype=np.float32).reshape(-1)
        z = np.maximum(x @ self.dictionary.T, 0.0)
        y = float(z @ self.head_w + self.head_b)
        atom_idx = np.argsort(z)[-self.top_k :][::-1]
        guardrail_atoms = self.guardrail_atoms or []
        suppress_atoms = self.suppress_atoms or []
        guardrail_mass = float(np.sum(z[np.asarray(guardrail_atoms, dtype=np.int32)])) if guardrail_atoms else 0.0
        suppress_mass = float(np.sum(z[np.asarray(suppress_atoms, dtype=np.int32)])) if suppress_atoms else 0.0
        guardrail_score = self.positive_weight * guardrail_mass - self.negative_weight * suppress_mass
        return {
            "lexica_score": y,
            "lexica_top_atoms": atom_idx.astype(int).tolist(),
            "lexica_top_values": z[atom_idx].astype(float).tolist(),
            "guardrail_atoms": [int(v) for v in guardrail_atoms],
            "suppress_atoms": [int(v) for v in suppress_atoms],
            "guardrail_mass": guardrail_mass,
            "suppress_mass": suppress_mass,
            "guardrail_score": guardrail_score,
        }
