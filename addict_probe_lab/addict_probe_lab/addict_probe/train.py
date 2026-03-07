from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .dataset import TraceDataset


@dataclass
class DenseBaseline:
    """Very small dense baseline for quick sanity checks."""

    w: np.ndarray
    b: np.ndarray

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2) -> "DenseBaseline":
        x2 = x.reshape(x.shape[0], -1).astype(np.float32)
        y2 = y.astype(np.float32)
        x_aug = np.concatenate([x2, np.ones((x2.shape[0], 1), dtype=np.float32)], axis=1)
        a = x_aug.T @ x_aug + ridge * np.eye(x_aug.shape[1], dtype=np.float32)
        w_all = np.linalg.solve(a, x_aug.T @ y2).astype(np.float32)
        return DenseBaseline(w=w_all[:-1], b=w_all[-1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        x2 = x.reshape(x.shape[0], -1).astype(np.float32)
        return x2 @ self.w + self.b


def train_dense_baseline(ds: TraceDataset) -> Tuple[DenseBaseline, Dict[str, float]]:
    tr = ds.splits["train"]
    va = ds.splits["val"]
    model = DenseBaseline.fit(ds.x_windows[tr], ds.y_next[tr])
    pred = model.predict(ds.x_windows[va])
    mse = float(np.mean((pred - ds.y_next[va]) ** 2)) if len(va) else float("nan")
    return model, {"val_mse": mse}
