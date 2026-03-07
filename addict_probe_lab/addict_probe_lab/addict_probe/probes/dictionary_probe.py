from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..starter import SparseDictionaryProbe
from .linear_probe import LinearProbe


@dataclass
class DictionaryProbeModel:
    encoder: SparseDictionaryProbe
    head: LinearProbe

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.encoder.encode(x.astype(np.float32))
        return self.head.predict(z)


def train_dictionary_probe(
    x: np.ndarray, y: np.ndarray, dict_size: int = 64, top_k: int = 4, seed: int = 7
) -> DictionaryProbeModel:
    encoder = SparseDictionaryProbe(feat_dim=x.shape[1], dict_size=dict_size, top_k=top_k, seed=seed)
    z = encoder.encode(x.astype(np.float32))
    head = LinearProbe.fit(z, y)
    return DictionaryProbeModel(encoder=encoder, head=head)
