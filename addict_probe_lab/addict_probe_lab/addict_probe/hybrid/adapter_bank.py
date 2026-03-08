from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TinyLoraAdapterBank:
    """
    Lightweight adapter composition scaffold.
    Stores per-adapter vectors and returns weighted mixed adapter signal.
    """

    adapter_names: List[str]
    adapter_dim: int
    seed: int = 7

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.matrix = rng.normal(0.0, 0.05, size=(len(self.adapter_names), self.adapter_dim)).astype(np.float32)

    @property
    def count(self) -> int:
        return int(self.matrix.shape[0])

    def compose(self, route_weights: np.ndarray) -> np.ndarray:
        """
        route_weights: [batch, n_adapters]
        returns mixed adapter vectors: [batch, adapter_dim]
        """
        if route_weights.size == 0:
            return np.zeros((0, self.adapter_dim), dtype=np.float32)
        return (route_weights @ self.matrix).astype(np.float32)

