from __future__ import annotations
import numpy as np

class SparseDictionaryProbe:
    def __init__(self, feat_dim=128, dict_size=64, top_k=4, seed=7):
        rng = np.random.default_rng(seed)
        self.dictionary = rng.normal(size=(dict_size, feat_dim)).astype(np.float32)
        self.top_k = top_k

    def encode(self, x: np.ndarray) -> np.ndarray:
        raw = x @ self.dictionary.T
        raw = np.maximum(raw, 0.0)
        mask = np.zeros_like(raw)
        idx = np.argpartition(raw, -self.top_k, axis=1)[:, -self.top_k:]
        np.put_along_axis(mask, idx, 1.0, axis=1)
        return raw * mask
