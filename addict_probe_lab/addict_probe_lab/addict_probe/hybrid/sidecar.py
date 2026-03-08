from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .types import LayerSliceBatch, SidecarScores


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


@dataclass
class ANESidecarProbeRunner:
    """
    ANE-friendly sidecar scaffold:
    - fixed-size linear projections
    - no dynamic control flow in core score pass
    """

    probe_dim: int = 6
    capability_dim: int = 3
    sae_dim: int = 8
    seed: int = 7

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._weights: Dict[str, Dict[str, np.ndarray]] = {}

    def _init_layer(self, key: str, in_dim: int) -> None:
        self._weights[key] = {
            "probe_w": self.rng.normal(0.0, 0.08, size=(in_dim, self.probe_dim)).astype(np.float32),
            "probe_b": np.zeros((self.probe_dim,), dtype=np.float32),
            "cap_w": self.rng.normal(0.0, 0.08, size=(in_dim, self.capability_dim)).astype(np.float32),
            "cap_b": np.zeros((self.capability_dim,), dtype=np.float32),
            "sae_w": self.rng.normal(0.0, 0.06, size=(in_dim, self.sae_dim)).astype(np.float32),
            "sae_b": np.zeros((self.sae_dim,), dtype=np.float32),
            "conf_w": self.rng.normal(0.0, 0.08, size=(in_dim, 1)).astype(np.float32),
            "conf_b": np.zeros((1,), dtype=np.float32),
        }

    def score(self, batch: LayerSliceBatch) -> SidecarScores:
        if batch.batch_size == 0 or not batch.slices:
            z0 = np.zeros((0, self.probe_dim), dtype=np.float32)
            return SidecarScores(
                probe_scores=z0,
                conflict_score=np.zeros((0, 1), dtype=np.float32),
                capability_scores=np.zeros((0, self.capability_dim), dtype=np.float32),
                sae_salience=np.zeros((0, self.sae_dim), dtype=np.float32),
            )

        probe_acc = []
        cap_acc = []
        sae_acc = []
        conf_acc = []
        for key, x in batch.slices.items():
            # x: [batch, time, dim] -> mean pool over time
            pooled = np.mean(x, axis=1).astype(np.float32)
            in_dim = int(pooled.shape[1])
            if key not in self._weights:
                self._init_layer(key, in_dim)
            w = self._weights[key]
            probe_acc.append(_sigmoid(pooled @ w["probe_w"] + w["probe_b"]))
            cap_acc.append(_sigmoid(pooled @ w["cap_w"] + w["cap_b"]))
            sae_acc.append(np.abs(pooled @ w["sae_w"] + w["sae_b"]))
            conf_acc.append(_sigmoid(pooled @ w["conf_w"] + w["conf_b"]))

        probe = np.mean(np.stack(probe_acc, axis=0), axis=0).astype(np.float32)
        cap = np.mean(np.stack(cap_acc, axis=0), axis=0).astype(np.float32)
        sae = np.mean(np.stack(sae_acc, axis=0), axis=0).astype(np.float32)
        conf = np.mean(np.stack(conf_acc, axis=0), axis=0).astype(np.float32)
        return SidecarScores(
            probe_scores=probe,
            conflict_score=conf,
            capability_scores=cap,
            sae_salience=sae,
        )

